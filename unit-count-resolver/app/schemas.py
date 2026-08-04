"""Request/response schemas for the unit-count-resolver service."""
from pydantic import BaseModel, Field


class ResolveUnitCountsRequest(BaseModel):
    """Body for POST /resolve_unit_counts.

    Exactly the three fields the build spec calls for. `user_id` is the caller's
    email; it is used to derive the organization_slug for the GCS path (see
    gcs_client.py) and is logged.
    """
    project_id: str = Field(..., min_length=1)
    plan_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)


class ResolveUnitCountsAccepted(BaseModel):
    """202 body — returned BEFORE the work runs (D5a: 202 immediately)."""
    status: str = "accepted"
    project_id: str
    plan_id: str
