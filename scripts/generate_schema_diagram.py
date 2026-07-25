"""Generate a Mermaid ER diagram from SQLAlchemy ORM models.

Usage:
    uv run python scripts/generate_schema_diagram.py
    uv run python scripts/generate_schema_diagram.py -o system-design/db-schema.md
"""

import argparse
from pathlib import Path

from earthrise_rag.db import Base

DOMAIN_GROUPS = {
    "User Interaction": [
        "conversations", "interactions", "interaction_citations",
        "interaction_traces", "feedback",
    ],
    "Infrastructure": [
        "prompt_versions", "index_runs", "deployments", "chunks",
    ],
    "Evaluation": [
        "eval_sets", "eval_questions", "eval_runs", "eval_results",
    ],
    "Sharing and Limits": [
        "shared_responses", "feedback_rate_limits",
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


def generate_mermaid() -> str:
    """Generate Mermaid ER diagram from Base.metadata."""
    lines = ["erDiagram"]
    tables = {t.name: t for t in Base.metadata.sorted_tables}
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
                    fk_relationships.append(
                        f'    {ref_table} {cardinality} {name} : "{label}"'
                    )
            lines.append("    }")

    lines.append("")
    lines.append("    %% --- Relationships ---")
    for rel in fk_relationships:
        lines.append(rel)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Mermaid ER diagram from ORM models.")
    parser.add_argument(
        "-o", "--output",
        default="system-design/db-schema.md",
        help="Output file path (default: system-design/db-schema.md)",
    )
    args = parser.parse_args()

    mermaid = generate_mermaid()
    schema = Base.metadata.schema or "public"
    table_count = len(Base.metadata.sorted_tables)

    content = (
        f"# Database Schema\n\n"
        f"PostgreSQL 17 + pgvector.\n"
        f"{table_count} tables in the `{schema}` schema.\n"
        f"Auto-generated from SQLAlchemy ORM models.\n"
        f"Regenerate with `uv run python scripts/generate_schema_diagram.py`.\n\n"
        f"```mermaid\n"
        f"{mermaid}\n"
        f"```\n"
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)
    print(f"Written to {output}")


if __name__ == "__main__":
    main()
