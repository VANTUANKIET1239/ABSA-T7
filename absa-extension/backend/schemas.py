from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictStr
try:
    from .config import ASPECTS
except ImportError:
    from config import ASPECTS


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviews: list[StrictStr] = Field(max_length=1000)


class AspectPrediction(BaseModel):
    code: str
    aspect: str
    name: str
    sentiment: Literal["Positive", "Negative", "Neutral"]
    confidence: float = Field(ge=0, le=1)


class ReviewResult(BaseModel):
    review_id: int
    text: str
    aspects: list[AspectPrediction]
    status: Literal["analyzed", "empty"] = "analyzed"


class AspectSummary(BaseModel):
    aspect: str
    name: str
    positive: int = 0
    negative: int = 0
    neutral: int = 0
    mentions: int = 0


class AnalyzeResponse(BaseModel):
    total_reviews: int
    results: list[ReviewResult]
    summary: dict[str, AspectSummary]
    model_version: str
    device: str


def summarize(results):
    summary = {f"A{i}": AspectSummary(aspect=aspect, name=name)
               for i, (aspect, name) in enumerate(ASPECTS, 1)}
    for result in results:
        seen = set()
        for prediction in result.aspects:
            pair = (prediction.code, prediction.sentiment)
            if pair in seen:
                continue
            seen.add(pair)
            row = summary[prediction.code]
            key = prediction.sentiment.lower()
            setattr(row, key, getattr(row, key) + 1)
        for code in {code for code, _ in seen}:
            summary[code].mentions += 1
    return summary
