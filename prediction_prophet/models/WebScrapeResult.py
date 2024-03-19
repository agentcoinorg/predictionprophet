import typing as t
from pydantic import BaseModel

class WebScrapeResult(BaseModel):
    query: str
    url: str
    title: str
    content: str

    def __getitem__(self, item: t.Any) -> str:
        return t.cast(str, getattr(self, item))  # Cast as all fields have str type.
