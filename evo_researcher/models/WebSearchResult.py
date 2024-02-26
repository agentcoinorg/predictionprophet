import typing as t
from pydantic import BaseModel


class WebSearchResult(BaseModel):
    title: str
    url: str
    description: str
    raw_content: str | None
    relevancy: float
    query: str
    
    def __getitem__(self, item: t.Any) -> t.Union[str, float]:
        return t.cast(t.Union[str, float], getattr(self, item))  # Cast because fields on this model have either str or float.
