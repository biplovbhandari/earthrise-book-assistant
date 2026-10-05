"""Tests for POST /feedback.

The database is mocked, so these tests cover the HTTP behavior and the statements the endpoint
sends: their order and the values bound into them.
They cannot show that the SQL is valid or that the rate limit holds under concurrent requests.
That needs a real PostgreSQL.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from conftest import create_test_client

from api.dependencies import Pipelines, require_db_session
from api.routes.feedback import MAX_COMMENT_LENGTH, _window_start

_INTERACTION_ID = UUID("11111111-1111-4111-8111-111111111111")
_VALID = {"interaction_id": str(_INTERACTION_ID), "rating": "up"}

# The endpoint sends three statements through session.scalar: the interaction lookup (call 0),
# the rate-limit upsert (call 1) and the feedback insert (call 2).
_RATE_LIMIT_CALL = 1
_INSERT_CALL = 2


@pytest.fixture
def session():
    """Mock AsyncSession; each test scripts the results of the statements the endpoint sends."""
    return AsyncMock()


@pytest.fixture
def client(monkeypatch, session):
    """TestClient whose database dependency is the mock session."""
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
    """TestClient with the database disabled, so require_db_session answers 503."""
    with create_test_client(monkeypatch, Pipelines()) as test_client:
        yield test_client


def _script_db(session, *, interaction_exists=True, within_limit=True, inserted=True):
    """Script the three session.scalar results the endpoint reads, in the order it reads them.

    The calls are the interaction lookup (None when it does not exist), the rate-limit upsert
    (None when the IP's window is already full) and the feedback insert (None when feedback
    already exists, so the endpoint updates it instead).
    """
    session.scalar.side_effect = [
        _INTERACTION_ID if interaction_exists else None,
        1 if within_limit else None,
        1 if inserted else None,
    ]


def _bound_params(statement):
    """Return the values bound into a SQLAlchemy statement, keyed by parameter name."""
    return statement.compile().params


def _scalar_params(session, call_index):
    """Return the values bound into the statement of the nth session.scalar call."""
    return _bound_params(session.scalar.await_args_list[call_index].args[0])


def _post(client, **overrides):
    """POST a valid thumbs-up for the test interaction, with some fields replaced or added."""
    return client.post("/feedback", json={**_VALID, **overrides})


@pytest.mark.parametrize(
    "comment",
    [
        pytest.param(None, id="no-comment"),
        pytest.param("The answer ignored my question about U-Net.", id="with-comment"),
    ],
)
def test_first_rating_is_stored_and_returns_201(client, session, comment):
    """A first rating inserts one row with exactly the submitted values and answers 201."""
    _script_db(session)

    resp = _post(client, comment=comment)

    assert resp.status_code == 201
    assert resp.json() == {
        "interaction_id": str(_INTERACTION_ID),
        "rating": "up",
        "comment": comment,
    }
    assert _scalar_params(session, _INSERT_CALL) == {
        "interaction_id": _INTERACTION_ID,
        "rating": "up",
        "comment": comment,
    }
    session.execute.assert_not_awaited()
    session.commit.assert_awaited_once()


@pytest.mark.parametrize(
    "comment",
    [
        pytest.param(None, id="clears-comment"),
        pytest.param("It cited the wrong chapter.", id="replaces-comment"),
    ],
)
def test_resubmitting_replaces_the_stored_rating_and_returns_200(client, session, comment):
    """A later submission overwrites rating and comment, so leaving the comment out clears it."""
    _script_db(session, inserted=False)

    resp = _post(client, rating="down", comment=comment)

    assert resp.status_code == 200
    assert resp.json() == {
        "interaction_id": str(_INTERACTION_ID),
        "rating": "down",
        "comment": comment,
    }
    session.execute.assert_awaited_once()
    params = _bound_params(session.execute.await_args.args[0])
    assert params["rating"] == "down"
    assert params["comment"] == comment
    assert _INTERACTION_ID in params.values(), "the update must target the rated interaction"
    session.commit.assert_awaited_once()


def test_unknown_interaction_returns_404_and_writes_nothing(client, session):
    """Rating an interaction that does not exist is a 404 that uses none of the rate limit."""
    _script_db(session, interaction_exists=False)

    resp = _post(client)

    assert resp.status_code == 404
    assert resp.json() == {"detail": "Interaction not found"}
    assert session.scalar.await_count == 1, "only the lookup should run"
    session.execute.assert_not_awaited()
    session.commit.assert_not_awaited()


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({**_VALID, "rating": "sideways"}, id="unknown-rating"),
        pytest.param({"rating": "up"}, id="missing-interaction-id"),
        pytest.param({**_VALID, "interaction_id": "not-a-uuid"}, id="malformed-interaction-id"),
        pytest.param({**_VALID, "comment": "x" * (MAX_COMMENT_LENGTH + 1)}, id="oversized-comment"),
    ],
)
def test_invalid_payload_returns_422_without_touching_the_database(client, session, payload):
    """Bad input is rejected at the boundary, before any statement is sent."""
    resp = client.post("/feedback", json=payload)

    assert resp.status_code == 422
    session.scalar.assert_not_awaited()
    session.commit.assert_not_awaited()


def test_returns_503_when_database_is_unavailable(client_no_db):
    """Without a database session factory the endpoint answers 503."""
    resp = _post(client_no_db)

    assert resp.status_code == 503
    assert resp.json() == {"detail": "Database unavailable"}


def test_full_rate_limit_window_returns_429_and_writes_nothing(client, session):
    """An IP that has used its window's quota is rejected and its feedback is not stored."""
    _script_db(session, within_limit=False)

    resp = _post(client)

    assert resp.status_code == 429
    assert session.scalar.await_count == 2, "the feedback insert should not run"
    session.execute.assert_not_awaited()
    session.commit.assert_not_awaited()


def test_rate_limit_row_is_keyed_by_hashed_ip_and_window_start(client, session):
    """The raw IP is never stored: the row is keyed by its SHA-256 digest and a window start."""
    _script_db(session)

    _post(client)

    params = _scalar_params(session, _RATE_LIMIT_CALL)
    # Starlette's TestClient reports its peer address as "testclient".
    assert params["ip_hash"] == hashlib.sha256(b"testclient").hexdigest()
    assert "testclient" not in params.values()
    assert params["window_start"] == _window_start(params["window_start"])


@pytest.mark.parametrize("inserted", [True, False], ids=["insert", "update"])
def test_client_supplied_admin_tags_are_ignored(client, session, inserted):
    """admin_tags is set only by admins, so a client's value is dropped on insert and update."""
    _script_db(session, inserted=inserted)

    resp = _post(client, admin_tags=["spam"])

    assert resp.status_code == (201 if inserted else 200)
    bound = {}
    for call in [*session.scalar.await_args_list, *session.execute.await_args_list]:
        bound.update(_bound_params(call.args[0]))
    assert "admin_tags" not in bound
    assert ["spam"] not in bound.values()


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        pytest.param(
            datetime(2026, 10, 5, 12, 5, 0, tzinfo=UTC),
            datetime(2026, 10, 5, 12, 5, tzinfo=UTC),
            id="first-instant-of-window",
        ),
        pytest.param(
            datetime(2026, 10, 5, 12, 4, 59, 999999, tzinfo=UTC),
            datetime(2026, 10, 5, 12, 0, tzinfo=UTC),
            id="last-instant-of-previous-window",
        ),
        pytest.param(
            datetime(2026, 10, 5, 12, 59, 59, tzinfo=UTC),
            datetime(2026, 10, 5, 12, 55, tzinfo=UTC),
            id="last-window-of-the-hour",
        ),
    ],
)
def test_window_start_rounds_down_to_the_five_minute_boundary(now, expected):
    """Every instant in a five-minute window maps to that window's start."""
    assert _window_start(now) == expected
