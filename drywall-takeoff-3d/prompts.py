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
        A single page extracted from an architectural construction plan project document entitled to a planned residence in `PNG` format.

    TASK:
        Classify the construction drawing into ONE or MORE of the following categories:

        - FLOOR_PLAN
        - ROOF_PLAN
        - ELECTRICAL_PLAN
        - FOUNDATION_PLAN
        - ELEVATION_PLAN
        - NOT_ARCHITECTURAL_PLAN

        INSTRUCTIONS:
        - Choose one or more category from the allowed list to classify the single-page plan(s).
        - Use architectural conventions (symbols, annotations, layout, views).
        - Consider labels, dimensions, symbols, and drawing orientation.
        - A page containing architecture plan will contain the architecture metadata information in text at the stray sections of the image, usually at the right and bottom section of the image. Generate a mask factor containing the information on the stray section of the image following the below instrution,
            - Generate the horizontal `mask_factor` (boundary -> [0, 1]) of the image by determining the fraction of the total width of the image that contains text information and isolated from the architecture drawing on the page on the right-most section of the page.
            - Generate the vertical `mask_factor` (boundary -> [0, 1]) of the image by determining the fraction of the total height of the image that contains text information and isolated from the architecture drawing on the page on the bottom-most section of the page.
        - A page may contain one or more architecture drawings. Compute bounding box or visual grounding offset for each of the available drawings with offset containing `TOP-LEFT` and `BOTTOM-RIGHT` corner of the bounding box) and produce a list of bounding box offsets.
            - REMEMBER, `TOPMOST-LEFTMOST` of the page is considered as the origin to compute the bounding box offset from.
            - The offsets should be computed in fraction [0, 1] to represent the bounding box. If a `TOP-LEFT` of a bounding box lies in a point which is at a distance of 0.5 of the total width of the page from the origin in X-direction (towards `RIGHT`) and at a distance of 0.25 of the total height of the page from the origin in Y-direction (towards `DOWN`), then the `TOP-LEFT` offset of the bounding box should be (0.5, 0.25). In a SIMILAR way, If a `BOTTOM-RIGHT` of a bounding box lies in a point which is at a distance of 0.85 of the total width of the page from the origin in X-direction (towards `RIGHT`) and at a distance of 0.75 of the total height of the page from the origin in Y-direction (towards `DOWN`), then the `BOTTOM-RIGHT` offset of the bounding box should be (0.85, 0.75).
            - STRICTLY REMEMBER, the visual grounding computed for each of the drawings should be precise, such that it tightly encloses that drawing while capturing its relevant outermost lines (dimension lines, wall lines or any extended artifact but excluding the margin-lines / outermost bounding-box lines) and the title of the drawing (if present in the neighborhood).
        - Identify the title of each of the available/identified architecture drawings, typically found at the bottom of each of the drawings and associate it to the respective visual-grounding/bounding-box offsets.
            - If respective drawing titles cannot be identified, use the title as `FLOOR_PLAN_<unique_identification_number>` with unique identification number for each of the identified architecture drawings.
            - STRICTLY REMEMBER, the titles must be unique. If duplicate titles are observed across the identified architecture drawings, add alpha-numerical suffixes to ensure they are unique.
        - Identify the plan category for each of the identified plans.

        Base your decision only on visual and textual evidence present in the drawing.

        If the drawing does not clearly represent an architectural or construction plan, classify it as NOT_ARCHITECTURAL_PLAN.

        Do not guess.
        Do not invent details.
        If uncertain, choose the most defensible category based on evidence.

    OUTPUT:
        Your output must be precise, code-aligned, and structured. You must reason spatially and geometrically. Do NOT describe the image.
        **STRICTLY**
        - Do not generate additional content apart from the designated JSON.
        Please refer the following as a reference and ensure to replace every consecutive pair of open/closed curly braces with a single one during the generation of the output.
        **STRICTLY FOLLOW**
          - The `bounding_box_offsets` field value should always be a non-empty list. 
        {{
            "plan_type": ["category of plan 1 (<FLOOR_PLAN>/<ROOF_PLAN>/<ELECTRICAL_PLAN>/<FOUNDATION_PLAN>/<ELEVATION_PLAN>/<NOT_ARCHITECTURAL_PLAN>)", "category of plan 2 (<FLOOR_PLAN>/<ROOF_PLAN>/<ELECTRICAL_PLAN>/<FOUNDATION_PLAN>/<ELEVATION_PLAN>/<NOT_ARCHITECTURAL_PLAN>)"],
            "mask_factor":
                {{
                    "horizontal": <mask factor for the width of the image in float rounded upto 2 decimal places>,
                    "vertical": <mask factor for the height of the image in float rounded upto 2 decimal places>
                }}
            "bounding_box_offsets":
                [
                    {{"offset_top_left": <`TOP-LEFT` offset of the bounding-box for architecture drawing 1>, "offset_bottom_right": <`BOTTOM-RIGHT` offset of the bounding-box for architecture drawing 1>, "title": "<identified title of the architecture drawing 1>", "plan_type": "type of architectural drawing 1 (<FLOOR_PLAN>/<ROOF_PLAN>/<ELECTRICAL_PLAN>/<FOUNDATION_PLAN>/<ELEVATION_PLAN>/<NOT_ARCHITECTURAL_PLAN>)"}},
                    {{"offset_top_left": <`TOP-LEFT` offset of the bounding-box for architecture drawing 2>, "offset_bottom_right": <`BOTTOM-RIGHT` offset of the bounding-box for architecture drawing 2>, "title": "<identified title of the architecture drawing 2>", "plan_type": "type of architectural drawing 2 (<FLOOR_PLAN>/<ROOF_PLAN>/<ELECTRICAL_PLAN>/<FOUNDATION_PLAN>/<ELEVATION_PLAN>/<NOT_ARCHITECTURAL_PLAN>)"}}
                ]
        }}
"""

class MaskFactor(BaseModel):
    horizontal: float = Field(..., ge=0)
    vertical: float = Field(..., ge=0)

    @field_validator("horizontal", "vertical")
    @classmethod
    def round_two_decimals(cls, v):
        return round(v, 2)

class BoundingBoxOffset(BaseModel):
    offset_top_left: Tuple[float, float]
    offset_bottom_right: Tuple[float, float]
    title: str
    plan_type: planType

    @field_validator("offset_top_left", "offset_bottom_right")
    @classmethod
    def validate_coordinates(cls, v):
        if len(v) != 2:
            raise ValueError("Offset must be a tuple of (x, y)")
        return v

    @model_validator(mode="after")
    def check_box_validity(self):
        x1, y1 = self.offset_top_left
        x2, y2 = self.offset_bottom_right
 
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bounding box: bottom_right must be greater than top_left")
 
        return self

class ArchitecturalDrawingClassifierResponse(BaseModel):
    plan_type: List[planType]
    mask_factor: MaskFactor
    bounding_box_offsets: List[BoundingBoxOffset]

    @field_validator("bounding_box_offsets")
    @classmethod
    def validate_non_empty_offsets(cls, v):
        if not v or len(v) == 0:
            raise ValueError("bounding_box_offsets must be a non-empty list")
        return v

    @field_validator("plan_type")
    @classmethod
    def validate_plan_type_list(cls, v):
        if not v:
            raise ValueError("plan_type cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_plan_consistency(self):
        bbox_plan_types = {bbox.plan_type for bbox in self.bounding_box_offsets}

        if not bbox_plan_types.issubset(set(self.plan_type)):
            raise ValueError("Mismatch between plan_type and bounding_box_offsets.plan_type")

        return self

FEEDBACK_GENERATOR = """
  INTERNAL SELF-REVIEW (Do not skip):
    You are given {max_retry} attempts to retry the generation process and the following are the list of errors encountered during your previous attempts.
    {exceptions}
    STRICTY confirm no previous error remains before producing the final output.
"""
