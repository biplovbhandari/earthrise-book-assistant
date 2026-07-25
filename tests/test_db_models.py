"""Metadata introspection tests for SQLAlchemy ORM models.

Tests verify schema properties that matter for correctness and performance:
referential integrity (FK targets and cascade behavior), domain constraints
(CHECK, UNIQUE), performance contracts (indexes), and non-obvious design
decisions (composite PK, nullable FKs, partial unique index).

Column sets are not tested separately - the model definitions are the source
of truth, and critical columns are already exercised by the constraint,
FK, and index tests.
"""

from sqlalchemy import CheckConstraint, UniqueConstraint

from earthrise_rag.db import Base


def _table(name: str):
    """Look up a table by unqualified name in the project schema."""
    return Base.metadata.tables[f"{Base.metadata.schema}.{name}"]


def test_all_15_tables_registered():
    """All 15 tables are present in Base.metadata."""
    table_names = {t.name for t in Base.metadata.tables.values()}
    assert table_names == {
        "conversations",
        "interactions",
        "chunks",
        "interaction_citations",
        "interaction_traces",
        "feedback",
        "prompt_versions",
        "index_runs",
        "deployments",
        "eval_sets",
        "eval_questions",
        "eval_runs",
        "eval_results",
        "shared_responses",
        "feedback_rate_limits",
    }


def _get_fk_map():
    """Build a map of (table, column) -> (target_table, ondelete)."""
    fk_map = {}
    for table in Base.metadata.tables.values():
        for col in table.columns:
            for fk in col.foreign_keys:
                fk_map[(table.name, col.name)] = (fk.column.table.name, fk.ondelete)
    return fk_map


def test_all_15_foreign_keys():
    """All 15 FK relationships have correct targets and ON DELETE behavior."""
    fk_map = _get_fk_map()
    assert len(fk_map) == 15

    expected = {
        ("interactions", "conversation_id"): ("conversations", "CASCADE"),
        ("interactions", "deployment_id"): ("deployments", "RESTRICT"),
        ("interaction_citations", "interaction_id"): ("interactions", "CASCADE"),
        ("interaction_citations", "chunk_id"): ("chunks", "RESTRICT"),
        ("interaction_traces", "interaction_id"): ("interactions", "CASCADE"),
        ("feedback", "interaction_id"): ("interactions", "CASCADE"),
        ("shared_responses", "interaction_id"): ("interactions", "CASCADE"),
        ("chunks", "index_run_id"): ("index_runs", "RESTRICT"),
        ("deployments", "prompt_version_id"): ("prompt_versions", "RESTRICT"),
        ("deployments", "index_run_id"): ("index_runs", "RESTRICT"),
        ("eval_questions", "eval_set_id"): ("eval_sets", "CASCADE"),
        ("eval_runs", "eval_set_id"): ("eval_sets", "RESTRICT"),
        ("eval_runs", "deployment_id"): ("deployments", "RESTRICT"),
        ("eval_results", "eval_run_id"): ("eval_runs", "CASCADE"),
        ("eval_results", "question_id"): ("eval_questions", "SET NULL"),
    }

    for key, (target, ondelete) in expected.items():
        assert fk_map[key] == (target, ondelete), f"FK {key} mismatch"


def _get_all_indexes():
    """Collect all Index objects from all tables, keyed by name."""
    indexes = {}
    for table in Base.metadata.tables.values():
        for idx in table.indexes:
            indexes[idx.name] = idx
    return indexes


def test_all_20_indexes():
    """All 20 named indexes are present."""
    expected_names = {
        "idx_conversations_visitor_id",
        "idx_conversations_created_at",
        "idx_interactions_conversation_id",
        "idx_interactions_deployment_id",
        "idx_interactions_created_at",
        "idx_chunks_index_run_id",
        "idx_chunks_source_path",
        "idx_interaction_citations_interaction_id",
        "idx_interaction_citations_chunk_id",
        "idx_interaction_traces_interaction_id",
        "idx_interaction_traces_stage",
        "idx_feedback_interaction_id",
        "idx_feedback_rating",
        "idx_index_runs_created_at",
        "idx_deployments_is_active",
        "idx_eval_questions_eval_set_id",
        "idx_eval_runs_eval_set_id",
        "idx_eval_runs_deployment_id",
        "idx_eval_results_eval_run_id",
        "idx_eval_results_question_id",
    }

    assert set(_get_all_indexes().keys()) == expected_names


def test_partial_unique_index_on_deployments():
    """At most one active deployment, enforced by partial unique index."""
    idx = _get_all_indexes()["idx_deployments_is_active"]
    assert idx.unique
    assert [c.name for c in idx.columns] == ["is_active"]
    where = str(idx.dialect_kwargs["postgresql_where"])
    assert "is_active" in where


def test_all_6_unique_constraints():
    """All 6 UNIQUE constraints are present with correct columns."""
    expected = {
        ("feedback", frozenset(["interaction_id"])),
        ("prompt_versions", frozenset(["hash"])),
        ("shared_responses", frozenset(["interaction_id"])),
        ("interaction_citations", frozenset(["interaction_id", "citation_index"])),
        ("interaction_traces", frozenset(["interaction_id", "stage"])),
        ("eval_results", frozenset(["eval_run_id", "question_id"])),
    }

    found = set()
    for table in Base.metadata.tables.values():
        for constraint in table.constraints:
            if isinstance(constraint, UniqueConstraint):
                cols = frozenset(c.name for c in constraint.columns)
                found.add((table.name, cols))

    assert found == expected


def test_all_8_check_constraints():
    """All 8 named CHECK constraints enforce domain rules with correct expressions."""
    expected = {
        "ck_feedback_rating_valid": "rating",
        "ck_deployments_strategy_valid": "retrieval_strategy",
        "ck_eval_runs_status_valid": "status",
        "ck_interactions_token_count_nonneg": "token_count",
        "ck_interactions_latency_nonneg": "latency_ms",
        "ck_interaction_citations_citation_index_nonneg": "citation_index",
        "ck_interaction_traces_latency_nonneg": "latency_ms",
        "ck_feedback_rate_limits_count_nonneg": "count",
    }

    found = {}
    for table in Base.metadata.tables.values():
        for constraint in table.constraints:
            if isinstance(constraint, CheckConstraint) and constraint.name:
                found[constraint.name] = str(constraint.sqltext)

    assert set(found.keys()) == set(expected.keys())
    for name, column in expected.items():
        assert column in found[name], f"{name} doesn't reference {column}"


def test_feedback_rate_limits_composite_pk():
    """Composite PK on (ip_hash, window_start) - order matters for lookups."""
    table = _table("feedback_rate_limits")
    pk_cols = [c.name for c in table.primary_key.columns]
    assert pk_cols == ["ip_hash", "window_start"]


def test_eval_results_question_id_nullable():
    """question_id is nullable for ON DELETE SET NULL from eval_questions."""
    table = _table("eval_results")
    assert table.columns["question_id"].nullable


def test_query_embedding_not_null():
    """query_embedding is NOT NULL - always computed at interaction time."""
    table = _table("interactions")
    assert not table.columns["query_embedding"].nullable
