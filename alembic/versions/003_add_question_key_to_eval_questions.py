"""Add question_key to eval_questions for stable YAML sync.

Autogenerate detected only false positives (FK schema re-qualification)
for this change, so the migration is written manually.

Revision ID: 003
Revises: 002
Create Date: 2026-10-15
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add question_key column and unique constraint to eval_questions."""
    op.add_column(
        "eval_questions",
        sa.Column("question_key", sa.Text(), nullable=True),
        schema="earthrise",
    )
    op.create_unique_constraint(
        "uq_eval_questions_set_key",
        "eval_questions",
        ["eval_set_id", "question_key"],
        schema="earthrise",
    )


def downgrade() -> None:
    """Remove question_key column and its unique constraint."""
    op.drop_constraint(
        "uq_eval_questions_set_key",
        "eval_questions",
        schema="earthrise",
        type_="unique",
    )
    op.drop_column("eval_questions", "question_key", schema="earthrise")
