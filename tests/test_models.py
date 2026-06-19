from earthrise_rag.models import Chunk


def test_chunk_round_trip_serialization():
    chunk = Chunk(
        content="U-Net is a convolutional neural network",
        content_hash="abc123",
        source_type="book_text",
        content_type="concept",
        metadata={"chapter": "03", "section": "Semantic Segmentation"},
    )
    data = chunk.model_dump()
    restored = Chunk.model_validate(data)
    assert restored == chunk
