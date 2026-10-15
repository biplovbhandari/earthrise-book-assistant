import pytest

from earthrise_rag.eval.scorer import (
    build_faithfulness_prompt,
    build_similarity_prompt,
    parse_judge_score,
    retrieval_precision,
    retrieval_recall,
)

# Expected-source prefixes in the two shapes data/eval/golden_questions.yml uses:
# a folder for book content and an exact file path for a video transcript.
_CROP_MAPPING = "book/03_Semantic_Segmentation/01__Crop_Mapping"
_ETHICS = "book/11_Ethics"
_CROP_VIDEO = "data/transcripts/l3Tv5UBlw9I.json"

# (builder, positional args) for both judge prompts, shared by the prompt structure tests.
_JUDGE_PROMPTS = [
    pytest.param(
        build_similarity_prompt,
        ("What is U-Net?", "A patch-based encoder-decoder network.", "A kind of pizza."),
        id="similarity",
    ),
    pytest.param(
        build_faithfulness_prompt,
        ("U-Net is a CNN.", "[1] book/03/index.qmd\nU-Net is an encoder-decoder network."),
        id="faithfulness",
    ),
]


def test_precision_is_one_when_every_chunk_matches_an_expected_source():
    """A chunk is relevant if it matches any one expected source, folder or file."""
    retrieved = [f"{_CROP_MAPPING}/notebooks/rice.ipynb", _CROP_VIDEO]

    assert retrieval_precision(retrieved, [_CROP_MAPPING, _CROP_VIDEO]) == 1.0


def test_precision_is_zero_when_no_chunk_matches():
    """Chunks from unrelated sources give zero precision."""
    retrieved = ["book/index.md", f"{_ETHICS}/index.qmd"]

    assert retrieval_precision(retrieved, [_CROP_MAPPING]) == 0.0


def test_precision_counts_chunks_not_files():
    """Three matching chunks out of eight is 0.375, even when two of them share one file."""
    retrieved = [
        f"{_CROP_MAPPING}/notebooks/rice.ipynb",
        f"{_CROP_MAPPING}/notebooks/rice.ipynb",
        f"{_CROP_MAPPING}/slides.pdf",
        "book/index.md",
        "book/index.md",
        f"{_ETHICS}/index.qmd",
        "book/05_Time_Series/01__Soybean_Yield_Prediction/notebooks/yield.ipynb",
        "data/transcripts/c7_GxZ6apqY.json",
    ]

    precision = retrieval_precision(retrieved, [_CROP_MAPPING])

    assert precision == 0.375, "each retrieved chunk counts once, even when files repeat"


def test_precision_is_zero_when_nothing_was_retrieved():
    """No retrieved chunks gives 0.0 rather than dividing by zero."""
    assert retrieval_precision([], [_CROP_MAPPING]) == 0.0


def test_recall_is_one_when_every_expected_source_is_retrieved():
    """Once every expected source has a hit, unrelated extra chunks do not lower recall."""
    retrieved = [
        f"{_ETHICS}/index.qmd",
        "book/index.md",
        f"{_CROP_MAPPING}/slides.pdf",
    ]

    assert retrieval_recall(retrieved, [_CROP_MAPPING, _ETHICS]) == 1.0


def test_recall_is_zero_when_no_expected_source_is_retrieved():
    """Unrelated chunks, or no chunks at all, find none of the expected sources."""
    assert retrieval_recall(["book/index.md"], [_CROP_MAPPING, _ETHICS]) == 0.0
    assert retrieval_recall([], [_CROP_MAPPING, _ETHICS]) == 0.0


def test_recall_counts_sources_not_chunks():
    """Several chunks under one expected source count once, so 1 of 2 sources is 0.5."""
    retrieved = [
        f"{_CROP_MAPPING}/notebooks/rice.ipynb",
        f"{_CROP_MAPPING}/notebooks/rice.ipynb",
        f"{_CROP_MAPPING}/slides.pdf",
    ]

    assert retrieval_recall(retrieved, [_CROP_MAPPING, _ETHICS]) == 0.5


def test_recall_is_zero_when_no_sources_are_expected():
    """An empty expected list gives 0.0 rather than dividing by zero."""
    assert retrieval_recall([f"{_CROP_MAPPING}/slides.pdf"], []) == 0.0


@pytest.mark.parametrize(
    "metric",
    [
        pytest.param(retrieval_precision, id="precision"),
        pytest.param(retrieval_recall, id="recall"),
    ],
)
def test_expected_source_matches_as_a_prefix_of_the_chunk_path(metric):
    """The expected source must be a prefix of the chunk path, never the other way round."""
    assert metric(["book/03/sub/file.py"], ["book/03"]) == 1.0
    assert metric(["book/03"], ["book/03/sub/file.py"]) == 0.0, "a parent folder is no match"


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        pytest.param("0.8", 0.8, id="bare-decimal"),
        pytest.param("  0.75\n", 0.75, id="surrounding-whitespace"),
        pytest.param("1", 1.0, id="bare-integer-one"),
        pytest.param("0", 0.0, id="bare-integer-zero"),
        pytest.param("Score: 0.85", 0.85, id="label-before-number"),
        pytest.param("I think 0.75 is appropriate", 0.75, id="number-inside-sentence"),
        pytest.param("0.6, or maybe 0.9", 0.6, id="first-number-wins"),
    ],
)
def test_parse_judge_score_reads_the_first_number(response, expected):
    """The score is the first number in the reply, whatever text surrounds it."""
    assert parse_judge_score(response) == expected


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        pytest.param("1.5", 1.0, id="above-range"),
        pytest.param("-0.3", 0.0, id="below-range"),
    ],
)
def test_parse_judge_score_clamps_to_the_unit_range(response, expected):
    """Scores outside 0.0-1.0 are clamped, and a leading minus sign counts as negative."""
    assert parse_judge_score(response) == expected


@pytest.mark.parametrize(
    "response",
    [
        pytest.param("", id="empty"),
        pytest.param("I cannot rate", id="no-number"),
    ],
)
def test_parse_judge_score_defaults_to_zero_without_a_number(response):
    """An empty or number-free reply scores 0.0 instead of raising."""
    assert parse_judge_score(response) == 0.0


@pytest.mark.parametrize(("builder", "args"), _JUDGE_PROMPTS)
def test_judge_prompt_is_a_system_then_user_message_list(builder, args):
    """Each prompt is a [system, user] list of role/content dicts for a chat API."""
    messages = builder(*args)

    assert [message["role"] for message in messages] == ["system", "user"]
    for message in messages:
        role = message["role"]
        assert set(message) == {"role", "content"}
        assert message["content"].strip(), f"{role} message must not be empty"


@pytest.mark.parametrize(("builder", "args"), _JUDGE_PROMPTS)
def test_judge_prompt_ends_by_asking_for_a_bare_score(builder, args):
    """Both prompts end by asking for a bare number on the 0.0-1.0 scale the parser expects."""
    user_message = builder(*args)[1]["content"]

    assert user_message.rstrip().endswith("Respond with only a number between 0.0 and 1.0.")


def test_similarity_prompt_puts_each_input_under_its_own_label():
    """A swapped argument would invert the grading, so each input must land on its label."""
    messages = build_similarity_prompt("What is U-Net?", "A CNN encoder-decoder.", "A pizza.")

    user_message = messages[1]["content"]
    assert "Question: What is U-Net?" in user_message
    assert "Expected answer: A CNN encoder-decoder." in user_message
    assert "Actual answer: A pizza." in user_message


def test_faithfulness_prompt_puts_chunks_and_answer_under_their_own_labels():
    """The judge must see the chunks as the context and the answer as the claim to check."""
    chunks_text = "[1] book/03/index.qmd\nU-Net is an encoder-decoder network."

    messages = build_faithfulness_prompt("U-Net won a pizza contest.", chunks_text)

    user_message = messages[1]["content"]
    assert f"Context chunks:\n{chunks_text}" in user_message
    assert "Answer:\nU-Net won a pizza contest." in user_message


def test_prompts_embed_braces_in_answers_and_chunks_verbatim():
    """Code or LaTeX braces in answers and chunks must not be read as template fields."""
    text = r"loss = {a: 1} and \frac{a}{b} and {0} and {}"

    similarity = build_similarity_prompt(text, text, text)[1]["content"]
    faithfulness = build_faithfulness_prompt(text, text)[1]["content"]

    assert similarity.count(text) == 3, "all three inputs must survive verbatim"
    assert faithfulness.count(text) == 2, "both inputs must survive verbatim"
