from pydantic import BaseModel


class Citation(BaseModel):
    """A citation linking an answer reference to a source chunk."""

    chunk_id: str
    source_path: str
    chapter: str
    section: str
    url: str = ""
    display_label: str = ""
