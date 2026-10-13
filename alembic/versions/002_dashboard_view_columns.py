"""Add last_cited_at and ranking_method to dashboard views.

Revision ID: 002
Revises: 001
Create Date: 2026-10-13
"""

from collections.abc import Sequence

from alembic import op

revision: str = "002"
down_revision: str = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("SET search_path TO earthrise, public")

    op.execute("""
        CREATE OR REPLACE VIEW v_citation_heatmap AS
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
        CREATE OR REPLACE VIEW v_retrieval_gaps AS
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


def downgrade() -> None:
    op.execute("SET search_path TO earthrise, public")

    op.execute("""
        CREATE OR REPLACE VIEW v_citation_heatmap AS
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
        CREATE OR REPLACE VIEW v_retrieval_gaps AS
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
