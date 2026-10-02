"""Tests for the read-only analytics endpoints."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from conftest import create_test_client

from api.dependencies import Pipelines, require_db_session

_LIST_PATHS = [
    "/analytics/daily",
    "/analytics/citations",
    "/analytics/gaps",
    "/analytics/conversations",
]
_ALL_PATHS = ["/analytics/overview", *_LIST_PATHS]


@pytest.fixture
def session():
    """Mock AsyncSession; each test decides what session.execute returns."""
    return AsyncMock()


@pytest.fixture
def client(monkeypatch, session):
    """TestClient whose database dependency hands out the mock session."""
    test_client = create_test_client(monkeypatch, Pipelines())
    from api.main import app

    async def mock_session():
        """Stand in for require_db_session so no real database is needed."""
        return session

    monkeypatch.setitem(app.dependency_overrides, require_db_session, mock_session)
    with test_client:
        yield test_client


@pytest.fixture
def client_no_db(monkeypatch):
    """TestClient with the database disabled and the real require_db_session in place."""
    with create_test_client(monkeypatch, Pipelines()) as test_client:
        yield test_client


def _returns_rows(session, rows):
    """Make session.execute resolve to a result whose .mappings().all() yields rows."""
    result = MagicMock()
    result.mappings.return_value.all.return_value = rows
    session.execute.return_value = result


def _returns_row(session, row):
    """Make session.execute resolve to a result whose .mappings().one() yields row."""
    result = MagicMock()
    result.mappings.return_value.one.return_value = row
    session.execute.return_value = result


def _executed_sql(session):
    """Return the SQL text of the last statement awaited on the mock session.

    The database is mocked, so this is the only way to confirm that an endpoint reads the view
    it is documented to read.
    """
    return str(session.execute.await_args.args[0])


def test_overview_returns_totals(client, session):
    """Totals and date range come through, and a Postgres numeric average becomes a float."""
    first = datetime(2026, 9, 1, 8, 30, tzinfo=UTC)
    last = datetime(2026, 10, 1, 17, 45, tzinfo=UTC)
    row = {
        "total_interactions": 42,
        "total_conversations": 15,
        "unique_visitors": 9,
        "first_interaction_at": first,
        "last_interaction_at": last,
        "avg_latency_ms": Decimal("1234.5"),
    }
    _returns_row(session, row)

    resp = client.get("/analytics/overview")

    assert resp.status_code == 200
    body = resp.json()
    first_seen = datetime.fromisoformat(body.pop("first_interaction_at"))
    last_seen = datetime.fromisoformat(body.pop("last_interaction_at"))
    assert first_seen == first
    assert last_seen == last
    assert body == {
        "total_interactions": 42,
        "total_conversations": 15,
        "unique_visitors": 9,
        "avg_latency_ms": 1234.5,
    }


def test_overview_with_empty_tables_returns_zero_counts(client, session):
    """An empty database gives zero totals and null dates and latency, not an error."""
    row = {
        "total_interactions": 0,
        "total_conversations": 0,
        "unique_visitors": 0,
        "first_interaction_at": None,
        "last_interaction_at": None,
        "avg_latency_ms": None,
    }
    _returns_row(session, row)

    resp = client.get("/analytics/overview")

    assert resp.status_code == 200
    assert resp.json() == row


def test_daily_returns_rows(client, session):
    """Rows keep the view's order and fields, and Postgres numeric averages become floats."""
    rows = [
        {
            "day": date(2026, 9, 30),
            "total_queries": 12,
            "thumbs_up": 5,
            "thumbs_down": 1,
            "avg_tokens": Decimal("350.5"),
            "avg_latency_ms": Decimal("1200.25"),
        },
        {
            "day": date(2026, 9, 29),
            "total_queries": 3,
            "thumbs_up": 0,
            "thumbs_down": 0,
            "avg_tokens": Decimal(200),
            "avg_latency_ms": Decimal(900),
        },
    ]
    _returns_rows(session, rows)

    resp = client.get("/analytics/daily")

    assert resp.status_code == 200
    assert resp.json() == [
        {
            "day": "2026-09-30",
            "total_queries": 12,
            "thumbs_up": 5,
            "thumbs_down": 1,
            "avg_tokens": 350.5,
            "avg_latency_ms": 1200.25,
        },
        {
            "day": "2026-09-29",
            "total_queries": 3,
            "thumbs_up": 0,
            "thumbs_down": 0,
            "avg_tokens": 200.0,
            "avg_latency_ms": 900.0,
        },
    ]
    assert "earthrise.v_daily_stats" in _executed_sql(session)


def test_citations_returns_rows(client, session):
    """An unrated source has a null thumbs_up_pct, and a source may have no display label."""
    rows = [
        {
            "source_path": "book/03_Segmentation/index.qmd",
            "display_label": "03 - U-Net",
            "cite_count": 17,
            "thumbs_up_pct": Decimal("0.75"),
        },
        {
            "source_path": "book/01_Intro/index.qmd",
            "display_label": None,
            "cite_count": 4,
            "thumbs_up_pct": None,
        },
    ]
    _returns_rows(session, rows)

    resp = client.get("/analytics/citations")

    assert resp.status_code == 200
    assert resp.json() == [
        {
            "source_path": "book/03_Segmentation/index.qmd",
            "display_label": "03 - U-Net",
            "cite_count": 17,
            "thumbs_up_pct": 0.75,
        },
        {
            "source_path": "book/01_Intro/index.qmd",
            "display_label": None,
            "cite_count": 4,
            "thumbs_up_pct": None,
        },
    ]
    assert "earthrise.v_citation_heatmap" in _executed_sql(session)


def test_gaps_returns_rows(client, session):
    """A question whose answers cited nothing has a null score instead of failing validation."""
    rows = [
        {
            "question": "How do I fine-tune on Sentinel-1?",
            "avg_top_score": 0.21,
            "thumbs_down_count": 3,
        },
        {
            "question": "What is SAR speckle?",
            "avg_top_score": None,
            "thumbs_down_count": 1,
        },
    ]
    _returns_rows(session, rows)

    resp = client.get("/analytics/gaps")

    assert resp.status_code == 200
    assert resp.json() == rows
    assert "earthrise.v_retrieval_gaps" in _executed_sql(session)


def test_conversations_returns_rows(client, session):
    """UUIDs are returned as strings and the interval as seconds; empty conversations get nulls."""
    conversation_id = UUID("11111111-1111-4111-8111-111111111111")
    visitor_id = UUID("22222222-2222-4222-8222-222222222222")
    empty_id = UUID("33333333-3333-4333-8333-333333333333")
    started = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
    rows = [
        {
            "id": conversation_id,
            "title": "U-Net basics",
            "topic": "segmentation",
            "visitor_id": visitor_id,
            "created_at": started,
            "interaction_count": 3,
            "last_activity": started + timedelta(minutes=5),
            "duration": timedelta(minutes=5),
        },
        {
            "id": empty_id,
            "title": None,
            "topic": None,
            "visitor_id": None,
            "created_at": started - timedelta(hours=1),
            "interaction_count": 0,
            "last_activity": None,
            "duration": None,
        },
    ]
    _returns_rows(session, rows)

    resp = client.get("/analytics/conversations")

    assert resp.status_code == 200
    first, second = resp.json()
    assert first["id"] == str(conversation_id)
    assert first["title"] == "U-Net basics"
    assert first["topic"] == "segmentation"
    assert first["visitor_id"] == str(visitor_id)
    assert first["interaction_count"] == 3
    assert first["duration_seconds"] == 300.0
    assert datetime.fromisoformat(first["created_at"]) == started
    assert datetime.fromisoformat(first["last_activity"]) == started + timedelta(minutes=5)
    assert second["id"] == str(empty_id)
    assert second["visitor_id"] is None
    assert second["interaction_count"] == 0
    assert second["last_activity"] is None
    assert second["duration_seconds"] is None
    assert "earthrise.v_conversation_summary" in _executed_sql(session)


@pytest.mark.parametrize("path", _LIST_PATHS)
def test_list_endpoint_returns_empty_list_for_empty_view(client, session, path):
    """A view with no rows is a normal empty result, not an error."""
    _returns_rows(session, [])

    resp = client.get(path)

    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.parametrize(
    ("path", "param", "default", "supplied"),
    [
        ("/analytics/daily", "days", 30, 7),
        ("/analytics/citations", "limit", 20, 5),
        ("/analytics/gaps", "limit", 20, 5),
        ("/analytics/conversations", "limit", 50, 5),
    ],
)
def test_param_uses_default_then_supplied_value(client, session, path, param, default, supplied):
    """The value bound into the query is the documented default, or the caller's when given."""
    _returns_rows(session, [])

    assert client.get(path).status_code == 200
    assert session.execute.await_args.args[1] == {param: default}

    assert client.get(path, params={param: supplied}).status_code == 200
    assert session.execute.await_args.args[1] == {param: supplied}


@pytest.mark.parametrize("path", _ALL_PATHS)
def test_returns_503_when_database_is_unavailable(client_no_db, path):
    """Without a database session factory every analytics endpoint answers 503."""
    resp = client_no_db.get(path)

    assert resp.status_code == 503
    assert resp.json() == {"detail": "Database unavailable"}
