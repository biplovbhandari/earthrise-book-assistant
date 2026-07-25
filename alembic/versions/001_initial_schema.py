"""Initial schema: 15 tables + 7 views.

Revision ID: 001
Revises:
Create Date: 2026-07-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS earthrise")
    op.execute("SET search_path TO earthrise, public")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")

    # --- conversations ---
    op.create_table(
        "conversations",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("visitor_id", UUID, nullable=True),
        sa.Column("ga4_client_id", sa.Text, nullable=True),
        sa.Column("title", sa.Text, nullable=True),
        sa.Column("summary", sa.Text, nullable=True),
        sa.Column("topic", sa.Text, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("idx_conversations_visitor_id", "conversations", ["visitor_id"])
    op.create_index("idx_conversations_created_at", "conversations", ["created_at"])

    # --- prompt_versions ---
    op.create_table(
        "prompt_versions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("hash", sa.Text, nullable=False, unique=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("label", sa.Text, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    # --- index_runs ---
    op.create_table(
        "index_runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("commit_sha", sa.Text, nullable=False),
        sa.Column("config", JSONB, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("idx_index_runs_created_at", "index_runs", ["created_at"])

    # --- deployments ---
    op.create_table(
        "deployments",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "prompt_version_id",
            sa.Integer,
            sa.ForeignKey("prompt_versions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "index_run_id",
            sa.Integer,
            sa.ForeignKey("index_runs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("model_name", sa.Text, nullable=False),
        sa.Column("temperature", sa.Float, nullable=False),
        sa.Column("retrieval_strategy", sa.Text, nullable=False),
        sa.Column("label", sa.Text, nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "retrieval_strategy IN ('dense', 'hybrid')", name=op.f("ck_deployments_strategy_valid")
        ),
    )
    op.create_index(
        "idx_deployments_is_active",
        "deployments",
        ["is_active"],
        unique=True,
        postgresql_where=sa.text("is_active IS TRUE"),
    )

    # --- interactions ---
    op.create_table(
        "interactions",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "conversation_id",
            UUID,
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("response_text", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("latency_ms", sa.Integer, nullable=False),
        sa.Column("query_embedding", Vector(1024), nullable=False),
        sa.Column(
            "deployment_id",
            sa.Integer,
            sa.ForeignKey("deployments.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("token_count >= 0", name=op.f("ck_interactions_token_count_nonneg")),
        sa.CheckConstraint("latency_ms >= 0", name=op.f("ck_interactions_latency_nonneg")),
    )
    op.create_index("idx_interactions_conversation_id", "interactions", ["conversation_id"])
    op.create_index("idx_interactions_deployment_id", "interactions", ["deployment_id"])
    op.create_index("idx_interactions_created_at", "interactions", ["created_at"])

    # --- chunks ---
    op.create_table(
        "chunks",
        sa.Column("chunk_id", sa.Text, primary_key=True),
        sa.Column(
            "index_run_id",
            sa.Integer,
            sa.ForeignKey("index_runs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("source_path", sa.Text, nullable=False),
        sa.Column("chapter", sa.Text, nullable=True),
        sa.Column("section", sa.Text, nullable=True),
        sa.Column("url", sa.Text, nullable=True),
        sa.Column("display_label", sa.Text, nullable=True),
    )
    op.create_index("idx_chunks_index_run_id", "chunks", ["index_run_id"])
    op.create_index("idx_chunks_source_path", "chunks", ["source_path"])

    # --- interaction_citations ---
    op.create_table(
        "interaction_citations",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "interaction_id",
            UUID,
            sa.ForeignKey("interactions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("citation_index", sa.Integer, nullable=False),
        sa.Column(
            "chunk_id",
            sa.Text,
            sa.ForeignKey("chunks.chunk_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("ranking_method", sa.Text, nullable=False),
        sa.UniqueConstraint(
            "interaction_id",
            "citation_index",
            name=op.f("uq_interaction_citations_interaction_id_citation_index"),
        ),
        sa.CheckConstraint(
            "citation_index >= 0", name=op.f("ck_interaction_citations_citation_index_nonneg")
        ),
    )
    op.create_index(
        "idx_interaction_citations_interaction_id", "interaction_citations", ["interaction_id"]
    )
    op.create_index("idx_interaction_citations_chunk_id", "interaction_citations", ["chunk_id"])

    # --- interaction_traces ---
    op.create_table(
        "interaction_traces",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "interaction_id",
            UUID,
            sa.ForeignKey("interactions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stage", sa.Text, nullable=False),
        sa.Column("data", JSONB, nullable=False),
        sa.Column("latency_ms", sa.Integer, nullable=False),
        sa.UniqueConstraint(
            "interaction_id", "stage", name=op.f("uq_interaction_traces_interaction_id_stage")
        ),
        sa.CheckConstraint("latency_ms >= 0", name=op.f("ck_interaction_traces_latency_nonneg")),
    )
    op.create_index(
        "idx_interaction_traces_interaction_id", "interaction_traces", ["interaction_id"]
    )
    op.create_index("idx_interaction_traces_stage", "interaction_traces", ["stage"])

    # --- feedback ---
    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "interaction_id",
            UUID,
            sa.ForeignKey("interactions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("rating", sa.Text, nullable=False),
        sa.Column("comment", sa.Text, nullable=True),
        sa.Column("admin_tags", JSONB, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("interaction_id", name=op.f("uq_feedback_interaction_id")),
        sa.CheckConstraint("rating IN ('up', 'down')", name=op.f("ck_feedback_rating_valid")),
    )
    op.create_index("idx_feedback_interaction_id", "feedback", ["interaction_id"])
    op.create_index("idx_feedback_rating", "feedback", ["rating"])

    # --- shared_responses ---
    op.create_table(
        "shared_responses",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "interaction_id",
            UUID,
            sa.ForeignKey("interactions.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )

    # --- feedback_rate_limits ---
    op.create_table(
        "feedback_rate_limits",
        sa.Column("ip_hash", sa.Text, primary_key=True),
        sa.Column("window_start", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("count", sa.Integer, nullable=False),
        sa.CheckConstraint("count >= 0", name=op.f("ck_feedback_rate_limits_count_nonneg")),
    )

    # --- eval_sets ---
    op.create_table(
        "eval_sets",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    # --- eval_questions ---
    op.create_table(
        "eval_questions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "eval_set_id",
            sa.Integer,
            sa.ForeignKey("eval_sets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("expected_answer", sa.Text, nullable=True),
        sa.Column("expected_sources", JSONB, nullable=False),
        sa.Column("tags", JSONB, nullable=True),
    )
    op.create_index("idx_eval_questions_eval_set_id", "eval_questions", ["eval_set_id"])

    # --- eval_runs ---
    op.create_table(
        "eval_runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "eval_set_id",
            sa.Integer,
            sa.ForeignKey("eval_sets.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "deployment_id",
            sa.Integer,
            sa.ForeignKey("deployments.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("config", JSONB, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name=op.f("ck_eval_runs_status_valid"),
        ),
    )
    op.create_index("idx_eval_runs_eval_set_id", "eval_runs", ["eval_set_id"])
    op.create_index("idx_eval_runs_deployment_id", "eval_runs", ["deployment_id"])

    # --- eval_results ---
    op.create_table(
        "eval_results",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "eval_run_id",
            sa.Integer,
            sa.ForeignKey("eval_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "question_id",
            sa.Integer,
            sa.ForeignKey("eval_questions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("question_snapshot", sa.Text, nullable=False),
        sa.Column("expected_answer_snapshot", sa.Text, nullable=True),
        sa.Column("expected_sources_snapshot", JSONB, nullable=False),
        sa.Column("response", sa.Text, nullable=False),
        sa.Column("citations", JSONB, nullable=False),
        sa.Column("trace", JSONB, nullable=False),
        sa.Column("scores", JSONB, nullable=False),
        sa.UniqueConstraint(
            "eval_run_id", "question_id", name=op.f("uq_eval_results_eval_run_id_question_id")
        ),
    )
    op.create_index("idx_eval_results_eval_run_id", "eval_results", ["eval_run_id"])
    op.create_index("idx_eval_results_question_id", "eval_results", ["question_id"])

    # --- 7 views ---

    op.execute("""
        CREATE VIEW v_interaction_summary AS
        SELECT i.id, i.question, i.response_text, i.latency_ms, i.token_count,
               i.created_at, f.rating, f.comment, f.admin_tags,
               COUNT(DISTINCT c.id) AS citation_count,
               d.label AS deployment_label,
               d.model_name,
               d.temperature
        FROM interactions i
        LEFT JOIN feedback f ON f.interaction_id = i.id
        LEFT JOIN interaction_citations c ON c.interaction_id = i.id
        JOIN deployments d ON d.id = i.deployment_id
        GROUP BY i.id, f.id, d.id
    """)

    op.execute("""
        CREATE VIEW v_citation_heatmap AS
        SELECT ch.source_path, ch.display_label,
               COUNT(*) AS cite_count,
               AVG(CASE WHEN f.rating = 'up' THEN 1.0
                        WHEN f.rating = 'down' THEN 0.0 END) AS thumbs_up_pct
        FROM interaction_citations c
        JOIN chunks ch ON ch.chunk_id = c.chunk_id
        JOIN interactions i ON i.id = c.interaction_id
        LEFT JOIN feedback f ON f.interaction_id = i.id
        GROUP BY ch.source_path, ch.display_label
        ORDER BY cite_count DESC
    """)

    op.execute("""
        CREATE VIEW v_retrieval_gaps AS
        SELECT i.question,
               AVG(top_score) AS avg_top_score,
               COUNT(*) AS thumbs_down_count
        FROM interactions i
        JOIN feedback f ON f.interaction_id = i.id AND f.rating = 'down'
        LEFT JOIN LATERAL (
            SELECT MAX(c.score) AS top_score
            FROM interaction_citations c
            WHERE c.interaction_id = i.id
        ) cs ON true
        GROUP BY i.question
        ORDER BY avg_top_score ASC
    """)

    op.execute("""
        CREATE VIEW v_deployment_metrics AS
        SELECT d.id AS deployment_id, d.label, d.model_name, d.retrieval_strategy,
               COUNT(i.id) AS interaction_count,
               AVG(CASE WHEN f.rating = 'up' THEN 1.0
                        WHEN f.rating = 'down' THEN 0.0 END) AS satisfaction_rate,
               AVG(i.latency_ms) AS avg_latency_ms
        FROM deployments d
        LEFT JOIN interactions i ON i.deployment_id = d.id
        LEFT JOIN feedback f ON f.interaction_id = i.id
        GROUP BY d.id
    """)

    op.execute("""
        CREATE VIEW v_daily_stats AS
        SELECT i.created_at::date AS day,
               COUNT(*) AS total_queries,
               COUNT(*) FILTER (WHERE f.rating = 'up') AS thumbs_up,
               COUNT(*) FILTER (WHERE f.rating = 'down') AS thumbs_down,
               AVG(i.token_count) AS avg_tokens,
               AVG(i.latency_ms) AS avg_latency_ms
        FROM interactions i
        LEFT JOIN feedback f ON f.interaction_id = i.id
        GROUP BY day
        ORDER BY day DESC
    """)

    op.execute("""
        CREATE VIEW v_conversation_summary AS
        SELECT conv.id, conv.title, conv.topic, conv.visitor_id, conv.created_at,
               COUNT(i.id) AS interaction_count,
               MAX(i.created_at) AS last_activity,
               MAX(i.created_at) - conv.created_at AS duration
        FROM conversations conv
        LEFT JOIN interactions i ON i.conversation_id = conv.id
        GROUP BY conv.id
    """)

    op.execute("""
        CREATE VIEW v_eval_run_summary AS
        SELECT er.id AS eval_run_id, er.eval_set_id, er.deployment_id, er.created_at,
               d.label AS deployment_label,
               COUNT(res.id) AS questions_evaluated,
               AVG((res.scores->>'retrieval_precision')::float) AS avg_retrieval_precision,
               AVG((res.scores->>'retrieval_recall')::float) AS avg_retrieval_recall,
               AVG((res.scores->>'answer_similarity')::float) AS avg_answer_similarity
        FROM eval_runs er
        JOIN deployments d ON d.id = er.deployment_id
        LEFT JOIN eval_results res ON res.eval_run_id = er.id
        GROUP BY er.id, d.id
    """)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS v_eval_run_summary")
    op.execute("DROP VIEW IF EXISTS v_conversation_summary")
    op.execute("DROP VIEW IF EXISTS v_daily_stats")
    op.execute("DROP VIEW IF EXISTS v_deployment_metrics")
    op.execute("DROP VIEW IF EXISTS v_retrieval_gaps")
    op.execute("DROP VIEW IF EXISTS v_citation_heatmap")
    op.execute("DROP VIEW IF EXISTS v_interaction_summary")

    op.drop_table("eval_results")
    op.drop_table("eval_runs")
    op.drop_table("eval_questions")
    op.drop_table("eval_sets")
    op.drop_table("feedback_rate_limits")
    op.drop_table("shared_responses")
    op.drop_table("feedback")
    op.drop_table("interaction_traces")
    op.drop_table("interaction_citations")
    op.drop_table("chunks")
    op.drop_table("interactions")
    op.drop_table("deployments")
    op.drop_table("index_runs")
    op.drop_table("prompt_versions")
    op.drop_table("conversations")
