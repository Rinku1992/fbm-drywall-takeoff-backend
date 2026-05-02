from typing import List, Set, Literal, Tuple
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

planType = Literal[
    "FLOOR_PLAN",
    "ROOF_PLAN",
    "ELECTRICAL_PLAN",
    "FOUNDATION_PLAN",
    "ELEVATION_PLAN",
    "NOT_ARCHITECTURAL_PLAN"
]

class Page(BaseModel):
    page_number: int
    page_type: planType

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

ARCHITECTURAL_DRAWING_CLASSIFIER = """
    You are an expert architectural drawing classifier.
 
    PROVIDED:
        A LIST of pages extracted from an architectural construction plan project document.
        Each page is provided in PNG format and is associated with a unique `page_number`.
 
    TASK:
        For EACH page independently, classify the construction drawing into ONE or MORE of the following categories:

            - FLOOR_PLAN
            - ROOF_PLAN
            - ELECTRICAL_PLAN
            - FOUNDATION_PLAN
            - ELEVATION_PLAN
            - NOT_ARCHITECTURAL_PLAN

    INSTRUCTIONS:
        - Process EACH page independently. Do NOT mix information across pages.
        - For every page:
            - Choose one or more categories from the allowed list.
            - Use architectural conventions (symbols, annotations, layout, views).
            - Consider labels, dimensions, symbols, and drawing orientation.

        MASK FACTOR:
            - A page containing architectural plans usually has metadata text in stray sections (right and bottom).
            - Compute:
                - `horizontal`: fraction [0, 1] of width occupied by right-side metadata
                - `vertical`: fraction [0, 1] of height occupied by bottom metadata
                - Generate the horizontal `mask_factor` (boundary -> [0, 1]) of the image by determining the fraction of the total width of the image that contains text information and isolated from the architecture drawing on the page on the right-most section of the page.
                - Generate the vertical `mask_factor` (boundary -> [0, 1]) of the image by determining the fraction of the total height of the image that contains text information and isolated from the architecture drawing on the page on the bottom-most section of the page.

        MULTIPLE DRAWINGS PER PAGE:
            - A page may contain one or more architectural drawings.
            - For EACH drawing:
                - Compute bounding box offsets:
                    - `offset_top_left` (x, y) in [0, 1]
                    - `offset_bottom_right` (x, y) in [0, 1]
                - Origin is TOP-LEFT of the page.
                - Bounding boxes must tightly enclose the drawing + relevant annotations (excluding page borders).
                - REMEMBER, `TOPMOST-LEFTMOST` of the page is considered as the origin to compute the bounding box offset from.
                - The offsets should be computed in fraction [0, 1] to represent the bounding box. If a `TOP-LEFT` of a bounding box lies in a point which is at a distance of 0.5 of the total width of the page from the origin in X-direction (towards `RIGHT`) and at a distance of 0.25 of the total height of the page from the origin in Y-direction (towards `DOWN`), then the `TOP-LEFT` offset of the bounding box should be (0.5, 0.25). In a SIMILAR way, If a `BOTTOM-RIGHT` of a bounding box lies in a point which is at a distance of 0.85 of the total width of the page from the origin in X-direction (towards `RIGHT`) and at a distance of 0.75 of the total height of the page from the origin in Y-direction (towards `DOWN`), then the `BOTTOM-RIGHT` offset of the bounding box should be (0.85, 0.75).
                - STRICTLY REMEMBER, the visual grounding computed for each of the drawings should be precise, such that it tightly encloses that drawing while capturing its relevant outermost lines (dimension lines, wall lines or any extended artifact but excluding the margin-lines / outermost bounding-box lines) and the title of the drawing (if present in the neighborhood).
                - Identify the title of each of the available/identified architecture drawings, typically found at the bottom of each of the drawings and associate it to the respective visual-grounding/bounding-box offsets.
                    - If respective drawing titles cannot be identified, use the title as `FLOOR_PLAN_<unique_identification_number>` with unique identification number for each of the identified architecture drawings.
                    - STRICTLY REMEMBER, the titles must be unique. If duplicate titles are observed across the identified architecture drawings, add alpha-numerical suffixes to ensure they are unique.
                - Identify the plan category for each of the identified plans.

        TITLES:
            - Identify title of each drawing (typically bottom of drawing).
            - If not found:
                → use `FLOOR_PLAN_<unique_id>`
            - Titles MUST be unique per page (append suffix if needed).

        PLAN TYPE PER DRAWING:
            - Assign a plan_type for EACH bounding box independently.
 
        GENERAL RULES:
            - Base decisions ONLY on visible evidence.
            - Do NOT hallucinate.
            - If uncertain → choose most defensible category.
            - If no architectural content → NOT_ARCHITECTURAL_PLAN.

    OUTPUT:
        Return a LIST of results, one per page.

        **STRICTLY**
        - Do NOT generate any text outside JSON.
        - The number of output objects MUST equal the number of input pages.
        - Each page MUST have a non-empty `bounding_box_offsets`.
        - Refer the `FORMAT` as a template for reference and ensure to replace every consecutive pair of open/closed curly braces with a single one during the generation of the output.

        FORMAT:
            {{
                "pages": [
                    {{
                        "page_number": <int>,
                        "plan_type": [
                            "<FLOOR_PLAN|ROOF_PLAN|ELECTRICAL_PLAN|FOUNDATION_PLAN|ELEVATION_PLAN|NOT_ARCHITECTURAL_PLAN>"
                        ],
                        "mask_factor": {{
                            "horizontal": <float>,
                            "vertical": <float>
                        }},
                        "bounding_box_offsets": [
                            {{
                                "offset_top_left": [<float>, <float>],
                                "offset_bottom_right": [<float>, <float>],
                                "title": "<string>",
                                "plan_type": "<FLOOR_PLAN|ROOF_PLAN|ELECTRICAL_PLAN|FOUNDATION_PLAN|ELEVATION_PLAN|NOT_ARCHITECTURAL_PLAN>"
                            }}
                        ]
                    }}
                ]
            }}
"""

class MaskFactor(BaseModel):
    horizontal: float = Field(..., ge=0, le=1)
    vertical: float = Field(..., ge=0, le=1)

class BoundingBox(BaseModel):
    offset_top_left: List[float]
    offset_bottom_right: List[float]
    title: str
    plan_type: planType
    @field_validator("offset_top_left", "offset_bottom_right")
    @classmethod
    def validate_offsets(cls, v):
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError("Offsets must be [x, y]")
        if not all(isinstance(i, (int, float)) for i in v):
            raise ValueError("Offsets must be numeric")
        if not all(0 <= i <= 1 for i in v):
            raise ValueError("Offsets must be within [0, 1]")
        return v
    @model_validator(mode="after")
    def validate_box_geometry(self):
        x1, y1 = self.offset_top_left
        x2, y2 = self.offset_bottom_right
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bounding box: bottom_right must be greater than top_left")
        return self

class Pages(BaseModel):
    page_number: int
    plan_type: List[planType]
    mask_factor: MaskFactor
    bounding_box_offsets: List[BoundingBox]
    @field_validator("plan_type")
    @classmethod
    def validate_plan_type_list(cls, v):
        if not v:
            raise ValueError("plan_type must contain at least one value")
        return v
    @model_validator(mode="after")
    def validate_page(self):
        # bounding boxes must exist
        if not self.bounding_box_offsets:
            raise ValueError("bounding_box_offsets must be non-empty")
        # ensure unique titles per page
        titles = [box.title for box in self.bounding_box_offsets]
        if len(titles) != len(set(titles)):
            raise ValueError("Bounding box titles must be unique per page")
        return self

class ArchitecturalDrawingClassifierResponse(BaseModel):
    pages: List[Pages]
    @model_validator(mode="after")
    def validate_pages(self):
        if not self.pages:
            raise ValueError("At least one page must be present")
        page_numbers = [p.page_number for p in self.pages]
        if len(page_numbers) != len(set(page_numbers)):
            raise ValueError("Duplicate page_number detected")
        return self

FEEDBACK_GENERATOR = """
  INTERNAL SELF-REVIEW (Do not skip):
    You are given {max_retry} attempts to retry the generation process and the following are the list of errors encountered during your previous attempts.
    {exceptions}
    STRICTY confirm no previous error remains before producing the final output.
"""
