from typing import List, Set, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


FLOORPLAN_TO_MULTIPAGE_ELEVATION_MAPPER = """
    You are a senior architectural drawing interpretation expert specializing in cross-referencing multi-sheet construction plan sets (floor plans, elevations, sections, details).
    You operate with strict determinism. You DO NOT hallucinate relationships. You ONLY map pages when strong architectural evidence exists.

    PROVIDED:
        A set of document pages. For each page:
            - page_number: <int>
            - image: <image>

    TASK:
        1. Classify EACH page into exactly ONE of the following categories:

            - FLOOR_PLAN
            - ROOF_PLAN
            - ELECTRICAL_PLAN
            - FOUNDATION_PLAN
            - ELEVATION_PLAN
            - NOT_ARCHITECTURAL_PLAN

        2. Identify ALL FLOOR PLAN pages.

        3. For EACH FLOOR PLAN page:
            - Find ALL relevant ELEVATION pages
            - Group them under that floor plan

        FLOOR_PLAN_IDENTIFICATION_INSTRUCTIONS:
        - A page is a FLOOR PLAN if,
            - Top-down view of rooms/spaces
            - Contains room labels (e.g., "BEDROOM", "KITCHEN")
            - Shows walls, doors, dimensions
            - May include grid lines and scale

        FLOOR_PLAN_TO_ELEVATION_PLAN_MATCHING_INSTRUCTIONS:
        - An elevation page belongs to a floor plan if ANY of the following are true:

        1. Explicit Callouts (STRONGEST SIGNAL)
            - Floor plan contains references like: "1/A-301", "ELEV 2", "SEE A-401".
            - Elevation page contains matching sheet reference.

        2. Elevation Labels
            - Elevation page labeled FRONT/REAR/LEFT/RIGHT/NORTH/etc.
            - Floor plan orientation or compass matches

        3. Sheet Number Convention
            - Floor plans: typically A-1xx
            - Elevations: typically A-3xx or A-4xx
            - Match based on numbering sequence proximity

        4. Visual Consistency (WEAKEST SIGNAL)
            - Window/door alignment and facade resemblance

        SCORING_INSTRUCTIONS:
        - For each FLOOR PLAN → ELEVATION pair:

            +0.40 → explicit_callout_match
            +0.25 → sheet_reference_match
            +0.20 → elevation_label_match
            +0.10 → textual_identifier_overlap
            +0.05 → Visual_similarity

        - Reject if score < 0.50

        GENERAL_INSTRUCTIONS:
        - DO NOT guess relationships
        - DO NOT assign elevation to multiple floor plans unless strong evidence exists
        - Prefer false negatives over false positives
        - If ambiguous, do NOT assign
        - Each elevation should ideally map to ONE floor plan
        - If multiple floor plans exist (multi-level), group accordingly
        - Do NOT fabricate sheet numbers or labels

    OUTPUT:
        Your output must be precise, code-aligned, and structured. Do NOT describe the image.
        **STRICTLY**
        - Do not generate additional content apart from the designated JSON.
        Please refer the following as a reference and ensure to replace every consecutive pair of open/closed curly braces with a single one during the generation of the output.
        {{
            "pages": [
                {{
                    "page_number": <int>,
                    "page_type": "<FLOOR_PLAN | ROOF_PLAN | ELECTRICAL_PLAN | FOUNDATION_PLAN | ELEVATION_PLAN | NOT_ARCHITECTURAL_PLAN>"
                }}
            ],
            "floorplan_groups": [
                {{
                    "floorplan_page": <int>,
                    "elevation_pages": [
                        {{
                            "page_number": <int>,
                            "confidence": <float>,
                            "match_reasons": [
                                "<explicit_callout_match | sheet_reference_match | elevation_label_match | textual_identifier_overlap | visual_similarity>"
                            ],
                            "matched_text_entities": [
                                "<exact textual identifiers used>"
                            ]
                        }}
                    ]
                }}
            ]
        }}
"""

MATCH_REASONS = {
    "explicit_callout_match",
    "sheet_reference_match",
    "elevation_label_match",
    "textual_identifier_overlap",
    "visual_similarity",
}

class Page(BaseModel):
    page_number: int
    page_type: Literal["FLOOR_PLAN", "ROOF_PLAN", "ELECTRICAL_PLAN", "FOUNDATION_PLAN", "ELEVATION_PLAN", "NOT_ARCHITECTURAL_PLAN"]

class ElevationPage(BaseModel):
    page_number: int
    confidence: float = Field(ge=0.0, le=1.0)
    match_reasons: List[str]
    matched_text_entities: List[str]

    @field_validator("match_reasons")
    @classmethod
    def validate_match_reasons(cls, v):
        invalid = set(v) - MATCH_REASONS
        if invalid:
            raise ValueError(f"Invalid match_reasons: {invalid}")
        return v

class FloorplanGroup(BaseModel):
    floorplan_page: int
    elevation_pages: List[ElevationPage]

class FloorplanToMultipageElevationMapperResponse(BaseModel):
    pages: List[Page]
    floorplan_groups: List[FloorplanGroup]

    @model_validator(mode="after")
    def validate_consistency(self):
        page_map = {p.page_number: p.page_type for p in self.pages}

        if len(page_map) != len(self.pages):
            raise ValueError("Duplicate page_number detected in pages")

        for group in self.floorplan_groups:
            if group.floorplan_page not in page_map:
                raise ValueError(f"Floorplan page {group.floorplan_page} not in pages list")

            if page_map[group.floorplan_page] != "FLOOR_PLAN":
                raise ValueError(f"Page {group.floorplan_page} is not classified as FLOOR_PLAN")

        assigned_elevations: Set[int] = set()

        for group in self.floorplan_groups:
            for elev in group.elevation_pages:
                if elev.page_number not in page_map:
                    raise ValueError(f"Elevation page {elev.page_number} not in pages list")

                if page_map[elev.page_number] != "ELEVATION_PLAN":
                    raise ValueError(
                        f"Page {elev.page_number} assigned as elevation but classified as {page_map[elev.page_number]}"
                    )

                assigned_elevations.add(elev.page_number)

        for group in self.floorplan_groups:
            for elev in group.elevation_pages:
                if elev.confidence < 0.5:
                    raise ValueError(
                        f"Elevation page {elev.page_number} has confidence below threshold (0.5)"
                    )

        return self

FEEDBACK_GENERATOR = """
  INTERNAL SELF-REVIEW (Do not skip):
    You are given {max_retry} attempts to retry the generation process and the following are the list of errors encountered during your previous attempts.
    {exceptions}
    STRICTY confirm no previous error remains before producing the final output.
"""
