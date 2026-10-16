"""Generate a Mermaid ER diagram from SQLAlchemy ORM models.

Usage:
    uv run python scripts/generate_schema_diagram.py
    uv run python scripts/generate_schema_diagram.py -o system-design/db-schema.md
"""

import argparse
from pathlib import Path

from geoai_rag.db import Base

DOMAIN_GROUPS = {
    "User Interaction": [
        "conversations",
        "interactions",
        "interaction_citations",
        "interaction_traces",
        "feedback",
    ],
    "Infrastructure": [
        "prompt_versions",
        "index_runs",
        "deployments",
        "chunks",
    ],
    "Evaluation": [
        "eval_sets",
        "eval_questions",
        "eval_runs",
        "eval_results",
    ],
    "Sharing and Limits": [
        "shared_responses",
        "feedback_rate_limits",
    ],
}

VIEW_GROUPS = {
    "Analytics": [
        "v_interaction_summary",
        "v_daily_stats",
        "v_citation_heatmap",
        "v_retrieval_gaps",
        "v_conversation_summary",
        "v_deployment_metrics",
    ],
    "Evaluation": [
        "v_eval_run_summary",
    ],
}


def _col_type(col) -> str:
    """Map SQLAlchemy column type to a short label."""
    type_name = type(col.type).__name__.upper()
    type_map = {
        "UUID": "UUID",
        "INTEGER": "INT",
        "TEXT": "TEXT",
        "FLOAT": "FLOAT",
        "BOOLEAN": "BOOL",
        "DATETIME": "TIMESTAMPTZ",
        "JSONB": "JSONB",
        "VARCHAR": "TEXT",
    }
    if type_name == "VECTOR":
        return f"VECTOR({col.type.dim})"
    return type_map.get(type_name, type_name)


def _col_markers(col) -> str:
    """Build PK/FK/nullable markers for a column."""
    markers = []
    if col.primary_key:
        markers.append("PK")
    if col.foreign_keys:
        markers.append("FK")
    if col.nullable and not col.primary_key:
        markers.append("nullable")
    return " ".join(markers)


def _ondelete_label(ondelete: str) -> str:
    """Format ON DELETE behavior as a readable label."""
    labels = {
        "CASCADE": "deletes children",
        "RESTRICT": "blocks delete",
        "SET NULL": "nullifies FK",
    }
    return labels.get(ondelete, ondelete)


def _is_view(table) -> bool:
    """Check if a SQLAlchemy table is a view ORM model."""
    return table.info.get("is_view", False)


def generate_mermaid() -> str:
    """Generate Mermaid ER diagram for tables (not views) from Base.metadata."""
    lines = ["erDiagram"]
    tables = {t.name: t for t in Base.metadata.sorted_tables if not _is_view(t)}
    fk_relationships = []

    for group_name, table_names in DOMAIN_GROUPS.items():
        lines.append(f"    %% --- {group_name} ---")
        for name in table_names:
            table = tables[name]
            lines.append(f"    {name} {{")
            for col in table.columns:
                col_type = _col_type(col)
                markers = _col_markers(col)
                comment = f' "{markers}"' if markers else ""
                lines.append(f"        {col_type} {col.name}{comment}")

                for fk in col.foreign_keys:
                    ref_table = fk.column.table.name
                    ondelete = fk.ondelete or "NO ACTION"
                    cardinality = "||--o{" if col.nullable else "||--|{"
                    label = _ondelete_label(ondelete)
                    fk_relationships.append(f'    {ref_table} {cardinality} {name} : "{label}"')
            lines.append("    }")

    lines.append("")
    lines.append("    %% --- Relationships ---")
    lines.extend(fk_relationships)

    return "\n".join(lines)


def generate_views_mermaid() -> str:
    """Generate a separate Mermaid ER diagram for views from Base.metadata."""
    lines = ["erDiagram"]
    views = {t.name: t for t in Base.metadata.sorted_tables if _is_view(t)}

    for group_name, view_names in VIEW_GROUPS.items():
        lines.append(f"    %% --- {group_name} ---")
        for name in view_names:
            view = views[name]
            lines.append(f"    {name} {{")
            for col in view.columns:
                col_type = _col_type(col)
                lines.append(f"        {col_type} {col.name}")
            lines.append("    }")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Mermaid ER diagram from ORM models.")
    parser.add_argument(
        "-o",
        "--output",
        default="system-design/db-schema.md",
        help="Output file path (default: system-design/db-schema.md)",
    )
    args = parser.parse_args()

    tables_mermaid = generate_mermaid()
    views_mermaid = generate_views_mermaid()
    schema = Base.metadata.schema or "public"
    all_tables = Base.metadata.sorted_tables
    table_count = sum(1 for t in all_tables if not _is_view(t))
    view_count = sum(1 for t in all_tables if _is_view(t))

    content = (
        f"# Database Schema\n\n"
        f"PostgreSQL 17 + pgvector.\n"
        f"{table_count} tables and {view_count} views in the `{schema}` schema.\n"
        f"Auto-generated from SQLAlchemy ORM models.\n"
        f"Regenerate with `uv run python scripts/generate_schema_diagram.py`.\n\n"
        f"## Tables\n\n"
        f"```mermaid\n"
        f"{tables_mermaid}\n"
        f"```\n\n"
        f"## Views\n\n"
        f"Pre-aggregated analytics and evaluation summaries.\n"
        f"Defined in Alembic migrations as raw SQL; mapped by read-only ORM models.\n\n"
        f"```mermaid\n"
        f"{views_mermaid}\n"
        f"```\n"
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)
    print(f"Written to {output}")


if __name__ == "__main__":
    main()
