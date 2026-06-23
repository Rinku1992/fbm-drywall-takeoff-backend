"""
Pydantic schemas for the classifier service.

The plan_type strings returned match exactly the planType Literal in
drywall-takeoff-3d/prompts.py:116-123. This means the caller can drop our
plan_type straight into the existing Pydantic ArchitecturalDrawingClassifier-
Response without any conversion.
"""
from typing import List, Literal
from pydantic import BaseModel, Field


# Must match drywall-takeoff-3d/prompts.py:planType exactly.
PlanTypeLiteral = Literal[
    "FLOOR_PLAN",
    "ROOF_PLAN",
    "ELECTRICAL_PLAN",
    "FOUNDATION_PLAN",
    "ELEVATION_PLAN",
    "NOT_ARCHITECTURAL_PLAN",
]


class PagePrediction(BaseModel):
    """One page's classification result."""
    page_number: int = Field(..., description="Page number from the metadata input")
    plan_type: PlanTypeLiteral = Field(
        ...,
        description="Final predicted class (after threshold applied).",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence in the final predicted class.",
    )
    model_prediction: PlanTypeLiteral = Field(
        ...,
        description="Raw argmax before threshold. Useful for logging/A-B comparison.",
    )
    threshold_demoted: bool = Field(
        default=False,
        description="True if model said floor_plan but conf was below threshold.",
    )


class ClassifyResponse(BaseModel):
    """Body of the response from POST /classify_pages."""
    pages: List[PagePrediction]
    inference_time_seconds: float
    total_time_seconds: float


class RequestMetadata(BaseModel):
    """JSON body in the 'metadata' multipart field."""
    page_numbers: List[int] = Field(
        ...,
        description="0-indexed page numbers, same order as uploaded image files.",
        min_length=1,
    )


class ClassifyPagesRequest(BaseModel):
    """JSON body for POST /classify_pages."""
    project_id: str = Field(..., min_length=1, description="Project ID; lowercased for GCS path lookup")
    plan_id: str = Field(..., min_length=1, description="Plan ID; lowercased for GCS path lookup")
    user_id: str = Field(..., min_length=1, description="User ID; logged but not used for path resolution")
    organization_slug: str = Field(..., min_length=1, description="Organization Slug; used for GCS path segregation")
