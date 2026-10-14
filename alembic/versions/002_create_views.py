"""Create database views for analytics and evaluation.

Views are hand-written SQL - they contain JOIN/GROUP BY logic
that cannot be expressed as ORM models. The ORM models in
db/models/views.py map to these views for typed query access.

Revision ID: 002
Revises: 001
Create Date: 2026-10-14
"""

from collections.abc import Sequence

from alembic import op

revision: str = "002"
down_revision: str = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create all analytics and evaluation views."""
    op.execute("SET search_path TO earthrise, public")

    # --- Analytics views ---

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
        CREATE VIEW v_citation_heatmap AS
        SELECT ch.source_path, ch.display_label,
               COUNT(*) AS cite_count,
               AVG(CASE WHEN f.rating = 'up' THEN 1.0
                        WHEN f.rating = 'down' THEN 0.0 END) AS thumbs_up_pct,
               MAX(i.created_at) AS last_cited_at,
               STRING_AGG(DISTINCT c.ranking_method, ', ' ORDER BY c.ranking_method) AS ranking_methods
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
               COUNT(*) AS thumbs_down_count,
               STRING_AGG(DISTINCT cs.ranking_method, ', ' ORDER BY cs.ranking_method) AS ranking_methods
        FROM interactions i
        JOIN feedback f ON f.interaction_id = i.id AND f.rating = 'down'
        LEFT JOIN LATERAL (
            SELECT MAX(c.score) AS top_score, c.ranking_method
            FROM interaction_citations c
            WHERE c.interaction_id = i.id
            GROUP BY c.ranking_method
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
        CREATE VIEW v_conversation_summary AS
        SELECT conv.id, conv.title, conv.topic, conv.visitor_id, conv.created_at,
               COUNT(i.id) AS interaction_count,
               MAX(i.created_at) AS last_activity,
               MAX(i.created_at) - conv.created_at AS duration
        FROM conversations conv
        LEFT JOIN interactions i ON i.conversation_id = conv.id
        GROUP BY conv.id
    """)

    # --- Evaluation views ---

    op.execute("""
        CREATE VIEW v_eval_run_summary AS
        SELECT er.id AS eval_run_id, er.eval_set_id, er.deployment_id, er.created_at,
               d.label AS deployment_label,
               COUNT(res.id) AS questions_evaluated,
               AVG((res.scores->>'retrieval_precision')::float) AS avg_retrieval_precision,
               AVG((res.scores->>'retrieval_recall')::float) AS avg_retrieval_recall,
               AVG((res.scores->>'answer_similarity')::float) AS avg_answer_similarity,
               AVG((res.scores->>'faithfulness')::float) AS avg_faithfulness
        FROM eval_runs er
        JOIN deployments d ON d.id = er.deployment_id
        LEFT JOIN eval_results res ON res.eval_run_id = er.id
        GROUP BY er.id, d.id
    """)


def downgrade() -> None:
    """Drop all views."""
    op.execute("SET search_path TO earthrise, public")

    op.execute("DROP VIEW IF EXISTS v_eval_run_summary")
    op.execute("DROP VIEW IF EXISTS v_conversation_summary")
    op.execute("DROP VIEW IF EXISTS v_deployment_metrics")
    op.execute("DROP VIEW IF EXISTS v_retrieval_gaps")
    op.execute("DROP VIEW IF EXISTS v_citation_heatmap")
    op.execute("DROP VIEW IF EXISTS v_daily_stats")
    op.execute("DROP VIEW IF EXISTS v_interaction_summary")
