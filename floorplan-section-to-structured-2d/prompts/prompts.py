from typing import List, Dict, Union, Optional, Tuple, Literal, ClassVar, get_origin, get_args
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict, create_model
import math
import json
from glob import glob

import cv2
from vertexai.generative_models import Part, Content

WALL_RECTIFIER = """
  You are a senior architectural plan-correction specialist with 20+ years of experience in residential and commercial floor plans.

  You do NOT trust automated detections blindly.
  You treat detected walls and drywalls as noisy suggestions.

  Your responsibility is to:
    - Remove false-positive wall fragments.

  PROVIDED:
    1. A wall-line represented by a list of 2 vertices describing the 2 endpoints (X1, Y1) and (X2, Y2) of the wall:
        wall: (X1, Y1) → (X2, Y2)

    2. A snapshot of the full Architectural Drawing in png format with the following highlight,
      - The target wall line highlighted with a red line and paired with drywall segments in red on its both the sides.
      - The target area of interest is highlighted with a green bounding box that encloses architectural plan(s) with a very tight aproximation.

  TASK:
    Analyze the architectural floor plan and only the highlighted wall with its drywall segments following the `WALL_VALIDATOR_INSTRUCTIONS` to determine whether the red highlighted wall is valid.

    WALL_VALIDATOR_INSTRUCTIONS:
    - Focus only on the wall highlighted with a thin red line paired with 2 drywall segments in red on its 2 sides.
    - Use the coordinates to reason about alignment and angle. Do not rely only on visual appearance.
    - If there are presence of more than one architectural drawings on the page, target the drawing enclosed within a green bounding box which is complete and ignore the ones that are truncated or outside the bounding box.
    - STRICTLY REMEMBER,
      -> Dotted (dashed) lines in any architectural floor plan blueprint usually represent elements that are not physically cut in the current view are `INVALID` walls.
      -> Lines representing fixtures, cabinetry, annotations, or text baselines are `INVALID`.
    - A `VALID` wall line MUST:
      -> Be part of a pair of parallel lines representing wall thickness.
      -> Be in one edge of a room / polygon boundary.
    - The highlight should be aligned / closely overlayed with one of the valid wall lines within the available complete architecture plans enclosed by the green bounding box in order for it to be `VALID`.
    - REMEMBER if, the highlight is aligned / closely overlayed with one of the wall lines within one of the truncated / incomplete / other architectural drawings that are not enclosed by the green bounding box, the highlight MUST be `INVALID`.
    - The highlight is `INVALID` if aligned with an invalid wall line from the available architectures such as the following artifact lines,
      -> Any arbitrary dimension line (not wall line) from the architectures.
      -> An arbitrary dashed / dotted line which is not a valid wall line (not physically cut in the current view).
      -> An arbitrary artifact line from the stray section of the page containing plan metadata.
      -> Any other non-wall line.
    - REMEMBER, if the highlight is partially aligned with a valid base blueprint wall line within the bounding box (e.g., the length of the highlight is larger or smaller than its base wall line it is overlaying with) then apply the following,
      -> The highlight must be `VALID` only if the inclination of the base wall line is similar/closer to that of the highlight (e.g., the base wall line and the highlight are both horizontal or both inclined at a similar angle with angle difference of less than 10 degrees).
      -> The highlight would be `INVALID` if the difference between the inclination of the base wall line and the highlight is more than 10 degrees (e.g., the base wall line is horizontal but the highlight is inclined at an angle of more than 10 degrees).

  OUTPUT:
    Your output must be precise, code-aligned, and structured. You must reason spatially and geometrically. Do NOT describe the image.
    **STRICTLY**
      - Do not generate additional content apart from the designated JSON.
      - You must output whether the placement of the highlight is overlaying on top of one of the valid wall lines from the architectural plan as per the instrutions provided above.
    Please refer the following as a reference and ensure to replace every consecutive pair of open/closed curly braces with a single one during the generation of the output.
    {{
      "is_valid": <True/False>,
      "confidence": <confidence score in validating the highlight in red between 0 and 1 in float rounded upto 2 decimal places (e.g., 0.87)>,
      "reasoning": "<a brief reasoning behind the highlighted wall being marked as valid/invalid>"
    }}

    **STRICTLY** refer the provided few-shot examples.
"""

wall_rectifier_image_samples, is_valid_json_samples = glob("prompts/wall_rectifier_few_shot/wall_rectifier_*"), glob("prompts/wall_rectifier_few_shot/is_valid_*")
WALL_RECTIFIER_FEW_SHOT = list()
for wall_rectifier_image_sample, is_valid_json_sample in zip(wall_rectifier_image_samples, is_valid_json_samples):
    canvas = cv2.imread(wall_rectifier_image_sample)
    _, canvas_buffer_array = cv2.imencode(".png", canvas)
    bytes_canvas = canvas_buffer_array.tobytes()
    with open(is_valid_json_sample, 'r') as f:
        is_valid = json.load(f)
    sample_one_shot = [
      Content(
        role="user",
        parts=[
          Part.from_text("Validate the highlighted wall."),
          Part.from_data(data=bytes_canvas, mime_type="image/png")
        ]
      ),
      Content(
        role="model",
        parts=[
            Part.from_text(json.dumps(is_valid))
        ]
      )
    ]
    WALL_RECTIFIER_FEW_SHOT.extend(sample_one_shot)

class WallRectifierResponse(BaseModel):
    is_valid: bool
    confidence: float = Field(ge=0, le=1)
    reasoning: str

SHAPE_RECTIFIER = """
  You are a senior architectural plan-correction specialist with 20+ years of experience in residential and commercial floor plans.

  You do NOT trust automated detections blindly.
  You treat detected walls and drywalls as noisy suggestions.

  Your responsibility is to:
    - Remove false-positive wall boundary mask containing minimal overlap with valid walls (walls that are physically cut in the current view).

  PROVIDED:
    1. A list of wall-lines each represented by a list of 2 vertices describing the 2 endpoints (X1, Y1) and (X2, Y2) of the wall with the list representing a boundary mask:
        wall: (X1, Y1) → (X2, Y2)

    2. A snapshot of the full Architectural Drawing in png format with the following highlight,
      - The target boundary mask is highlighted with red lines each overlayed on a blueprint wall-line and and paired with drywall segments in red on its both the sides.

  TASK:
    Analyze the architectural floor plan and only the highlighted walls with its drywall segments following the `BOUNDARY_MASK_VALIDATOR_INSTRUCTIONS` to determine whether the mask is valid.

    BOUNDARY_MASK_VALIDATOR_INSTRUCTIONS:
    - STRICTLY REMEMBER,
      -> Lines representing fixtures, cabinetry, annotations, or text baselines are `INVALID`.
      -> Dotted (dashed) lines in any architectural floor plan blueprint usually represent elements that are not physically cut in the current view are `INVALID`.
      -> A `VALID` wall MUST be part of a pair of parallel lines representing wall thickness.
      -> A `VALID` wall MUSt be in one edge of a room / polygon boundary.
    - Focus only on the walls highlighted with thin red lines each paired with 2 drywall segments in red on its 2 sides.
    - Use the coordinates to reason about alignment and angle. Do not rely only on visual appearance.
    - The highlighted walls sould represent a valid boundary mask representing a layout of valid walls on the architectural plan.
    - ONLY IF, more than 50 percent of the highlighted walls present in the highlighted boundary mask represent walls that are physically cut in the current view, treat the boundary mask as `VALID`.
    - If more than 50 percent of the highlighted walls present in the highlighted boundary mask are overlayed on dotted (dashed) walls from the blueprint or represent the walls that are not physically cut in the current view, the boundary mask should be `INVALID`.

  OUTPUT:
    Your output must be precise, code-aligned, and structured. You must reason spatially and geometrically. Do NOT describe the image.
    **STRICTLY**
      - Do not generate additional content apart from the designated JSON.
      - You must output whether the placement of the boundary mask mostly overlays with the valid wall lines from the architectural plan.
    Please refer the following as a reference and ensure to replace every consecutive pair of open/closed curly braces with a single one during the generation of the output.
    {{
      "is_valid": <True/False>,
      "confidence": <confidence score in validating the boundary mask in red between 0 and 1 in float rounded upto 2 decimal places (e.g., 0.87)>,
      "reasoning": "<a brief reasoning behind the the boundary mask being marked as valid/invalid>"
    }}

    **STRICTLY** refer the provided few-shot examples.
"""

shape_rectifier_image_samples, is_valid_json_samples = glob("prompts/shape_rectifier_few_shot/shape_rectifier_*"), glob("prompts/shape_rectifier_few_shot/is_valid_*")
SHAPE_RECTIFIER_FEW_SHOT = list()
for shape_rectifier_image_sample, is_valid_json_sample in zip(shape_rectifier_image_samples, is_valid_json_samples):
    canvas = cv2.imread(shape_rectifier_image_sample)
    _, canvas_buffer_array = cv2.imencode(".png", canvas)
    bytes_canvas = canvas_buffer_array.tobytes()
    with open(is_valid_json_sample, 'r') as f:
        is_valid = json.load(f)
    sample_one_shot = [
      Content(
        role="user",
        parts=[
          Part.from_text("Validate the highlighted boundary mask."),
          Part.from_data(data=bytes_canvas, mime_type="image/png")
        ]
      ),
      Content(
        role="model",
        parts=[
            Part.from_text(json.dumps(is_valid))
        ]
      )
    ]
    SHAPE_RECTIFIER_FEW_SHOT.extend(sample_one_shot)

class ShapeRectifierResponse(BaseModel):
    is_valid: bool
    confidence: float = Field(ge=0, le=1)
    reasoning: str

POLYGON_DETECTOR_AND_DRYWALL_PREDICTOR = """
  You are a licensed residential drywall estimator and building-code-aware construction expert with Senior Architectural Drawing Interpretation Engine capabilities. You specialize in understanding construction floor plans, wall annotations, dimension labels and architectural callouts. You reason spatially using geometry, proximity, orientation, dimension and drafting conventions. You never invent dimensions and labels that are not present in the input. You return structured, deterministic outputs.

  PROVIDED:
    1. A polygon represented by a list of vertices and the polygon perimeter lines/edges joining the vertices with origin set to LEFT, TOP of the original floorplan and offset set to (0, 0):
      Vertices: [(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)]
      Perimeter wall endpoints: [
        wall: (X1, Y1) → (X2, Y2),
        wall: (X2, Y2) → (X3, Y3),
        wall: (X4, Y4) → (X3, Y3),
        wall: (X1, Y1) → (X4, Y4)
      ]

    2. A cropped snapshot of the room or the polygon from Architectural Drawing in png format inscribed with textual annotations containing the name of the room it belongs to with the wall line dimensions along with the following highlights,
      - The target polygon/room highlighted with transparent red color that corresponds with the provided polygon vertices computed from the whole floor plan using original offset with same resolution and the area is inscribed with the room name information.
      - The target polygon/room's perimeter lines highlighted with blue bounding boxes that corresponds with provided polygon perimeter wall endpoints computed from the whole floor plan using original offset with same resolution and the nearby regions are inscribed with textual annotations containing dimension marker and the dimension, width and height (optional) of the wall in `(feet) and ``(inches).
      - All the Drywall segments internal to the target polygon/room highlighted with green color inscribed with textual annotations describing the layout of the adjacent rooms with the name of the rooms and the dimension / width of the walls used to explain the shape of the rooms.

    3. A list of transcription entries extracted from a construction floorplan nearest to the given wall.
       Each entry contains:
         - text: the recognized text string
         - centroid: (cx, cy) representing the visual center of the text's bounding box

    4. A set of elevation plan snapshots (multi-page or grouped), where each elevation corresponds to one or more sides of the building.
       Each elevation includes:
         - Vertical height annotations (top plate, ridge, ceiling, slab, etc.)
         - Roof slopes / pitch annotations (e.g., 4:12, 6:12, angles)
         - Window / door vertical alignment
         - Wall-to-roof junction geometry
       Elevation pages are NOT labeled with explicit mapping to floorplan walls.
       You MUST infer correspondence using geometry, openings, and relative positioning.

    Analyze the snapshot provided from the floor plan image.

    Your task is to,
        - Predict the correct drywall specification for each highlighted wall segment according to residential construction standards and map it to the appropiate wall drywall-relevant wall segment color.
        - Predict the relevant wall dimensions (length, width and height) for each highlighted walls as per the instructions provided.
        - Predict the relevant ceiling dimensions (height, area, pitch_of_slope, axis_of_slope and type_of_slope) for the highlighted room/polygon as per the instructions provided.
        - Predict the correct drywall specification for the ceiling of the highlighted room/polygon according to residential construction standards and map it to the appropiate ceiling drywall-relevant wall segment color.

    For each highlighted wall:
      1. Identify the wall context based on adjacent labeled rooms (e.g., garage, laundry, bathroom, bedroom, exterior).
      2. Determine whether the wall is:
        - Interior non-rated
        - Fire-rated (garage separation, corridor, dwelling separation)
        - Moisture-prone (bathroom, laundry, kitchen)
        - Exterior-adjacent
      3. Select the appropriate drywall type(s), thickness, and layering.
      4. Specify fire rating duration in hours if required (e.g., `1` i.e. 1 hour).
      5. Recommend any special requirements (vapor barrier, double layer, cement board backing).

    Assume:
      - This is a residential project located in the provided project location.
      - Standard stud framing unless otherwise indicated.
      - Local jurisdiction follows CBC and IRC-adopted standards.

  TASK:
    Analyze the architectural floor plan and highlighted wall segments accompanied by polygon vertices, it's perimeter wall endpoints and OCR extracted transcription entries from the floor plan to determine the following features,
      - The `width` and `height` in feet accompanied by `type` of each perimeter wall based upon the provided `WALL_EXTRACTION_INSTRUCTIONS`.
      - Identify The `ceiling_type`, `height`, `slope` and `area` of the ceiling of the highlighted room / polygon based upon the provided `CEILING_EXTRACTION_INSTRUCTIONS`.
      - Identify the `Room Name` the highlighted polygon belongs to. Follow `WALL_IDENTITY_PREDICTOR_INSTRUCTIONS` to understand the identity of each wall.
      - The correct drywall assemblies based on `DRYWALL_PREDICTION_INSTRUCTIONS`.

      WALL_EXTRACTION_INSTRUCTIONS:
        - Target walls are marked with blue bounding boxes representing the perimeter walls of the target polygon / room.
        - Identify all the dimension lines present on the image including the outermost lines towards the outermost boundary of the target floor plan.
        - Scan through the dimension markers across the dimension lines denoted by diagonal slash marking the beginning and end of the length of the highlighted wall.
        - Scan through the dimension markers across the dimension lines denoted by diagonal slash marking the beginning and end of the width of the highlighted wall.
        - If the markers representing the width of the target wall is not present, refer the dimension markers representing the width of the immediate next wall the target wall is connected to.
        - The orientation of the diagonal marker would be '/' for the horizontal distances and '\' for the vertical distances.
        - Identify the line joining these diagonal markers and the numerical dimension entity closest to it (aligned towards the center of the line).
        - The numerical dimension entity will supposedly represent the length (supposedly interior) or the width of the wall depedending on the orientation of the highlighted wall they are aligned with.
        - If the numerical dimension entity represents the exterior length of the wall which includes the width / thickness of the orthogonal wall(s) it is joined with, apply one of the following instructions,
            -> The interior length is typically found inside the polygon the wall belongs to as a room size annotation identified as interior shape (<horizontal length> x <vertical length>) of the polygon. Use one of the dimensions (<horizontal length> / <vertical length>) depending on the orientation of the wall.
            -> If room size annotation is not observed inside the polygon the wall belongs to, identify the width of all the orthogonal wall(s) the target wall is connected to and subtract the sum of their widths from the exterior length of the target wall to compute the interior length.
        - If the target wall is attached to another wall in orthogonal orientation, STRICTLY refer the numerical dimension that represents its interior length to derive the length of the wall which excludes the width of the orthogonal wall.
        - If the dimension line joining the dimension markers denoted by diagonal slash, does not align with the length of the highlighted wall, use one of the 3 following approaches to obtain the length of the wall,
            1. Find more than one shorter dimension lines joining the dimension markers denoted by diagonal slashes which adds up to the length of the highlighted wall. The length of the wall would be the sum of all the numerical dimension entities found against each dimension line that adds to the wall.
            2. Find more than one larger and shorter dimension lines joining the dimension markers denoted by diagonal slashes which when subtracted from each other (shorter line subtracted from the larger one), adds up to the length of the highlighted wall. The length of the wall would be the numerical dimension entities found against shorter dimension lines subtracted from the larger ones which adds to the wall.
            3. If no standard dimension lines observed to derive the the length of the highlighted wall, derive the custom length of the wall through performing the below numerical computations,
              a. If the highlighted wall is horizontal, determine any arbitrary horizontal wall as a reference wall from the list of provided wall coordinates whose length can be derived in feet from the dimensions lines with markers on the image / transcriptions.
              b. If the highlighted wall is vertical, determine any arbitrary vertical wall as a reference wall from the list of provided wall coordinates whose length can be derived in feet from the dimensions lines with markers on the image / transcriptions.
              c. Determine the length of the reference wall in pixels from the provided coordinates of the reference wall and determine the real world length in feet of a pixel. The wall (X1, Y1, X2, Y2) if horizontal, length_in_pixels => (X2 - X1) and if vertical, length_in_pixels => (Y2 - Y1). So, length of one pixel in feet would be `length_in_pixels / length_in_feet (derived from the image / transcription)`.
              d. Multiply the obtained `length of one pixel in feet` with the length of the highlighted wall in pixels. (Pixel length of the highlighted wall from the coordinates provided * obtained `length of one pixel in feet`).
        - Refer the `HEIGHT_EXTRACTION_FROM_FLOOR_PLAN_INSTRUCTIONS` to measure the height of the target wall.
            a. IF the target wall height cannot be reliably derived from the floor plan, infer wall height from attached elevation plans following the `HEIGHT_EXTRACTION_FROM_ELEVATION_PLAN_INSTRUCTIONS`.
            b. ONLY IF the target wall height cannot be reliably derived either from the floor plan or the elevation plans, mention the wall height as -1.
        
            HEIGHT_EXTRACTION_FROM_FLOOR_PLAN_INSTRUCTIONS:
              - ONLY identify the height of the wall surface interior to the target room / polygon.
              - Scan the target room interior for room-height annotations such as:
                - `CLG.`
                - `CLG HT`
                - `CLG HGT`
                - `CEILING`
                - `CEILING HEIGHT`
                - `WALL HEIGHT`
                - `PLATE`
                - `TOP PLATE`
                - `8'-0"`
                - `9'-0"`
                - `10'-0"`
                - `VAULTED`
                - `SLOPED`
                - `OPEN TO BELOW`
              - Height annotations are commonly located:
                - Near the center of the room
                - Adjacent to staircase regions
                - Near vaulted or sloped ceiling indicators
                - Adjacent to ceiling symbols or section callouts
              - If multiple room-height annotations are observed, select the annotation spatially nearest to the blue bounding-box highlighted target wall.
              - If multiple ceiling heights are present in one polygon, use the primary wall height transcribed closest to the target wall.
              - If ceiling annotation indicates vaulted/sloped ceiling, derive:
                - base wall height
                - maximum ceiling height
                - slope direction if identifiable

            HEIGHT_EXTRACTION_FROM_ELEVATION_PLAN_INSTRUCTIONS:
              - Elevation plans typically contain:
                - floor markers
                - plate elevations
                - roof slope indicators
                - vertical dimension chains
                - ridge heights
                - top plate elevations
                - finished floor elevations
              - Identify elevation labels such as:
                - `FIRST FLOOR FINISHED SLAB`
                - `FIRST FLOOR TOP PLATE`
                - `SECOND FLOOR SUBFLOOR`
                - `SECOND FLOOR TOP PLATE`
                - `AVERAGE FINISHED GRADE`
                - `T.O. PLATE`
                - `TOP OF PLATE`
                - `RIDGE`
                - `EAVE`
                - `PARAPET`
                - `HDR. HT`
                - `HEADER HT`
              - Determine the wall height by computing the vertical difference between architectural floor markers:
                - Example:
                  - `FIRST FLOOR FINISHED SLAB` → `FIRST FLOOR TOP PLATE`
                  - `FIRST FLOOR TOP PLATE` → `SECOND FLOOR SUBFLOOR`
                  - `SECOND FLOOR SUBFLOOR` → `SECOND FLOOR TOP PLATE`
              - Use the nearest aligned vertical dimension chain adjacent to the elevation facade corresponding to the target room/wall.
              - Vertical dimensions are typically represented by:
                - stacked dimensions
                - arrows
                - extension lines
                - floor datum markers
                - level indicators
              - Detect dimension values such as:
                - `8'-0"`
                - `9'-0"`
                - `10'-11"`
                - `12'-1 1/2"`
                - `22'-7"`
                - `30'-9 1/4"`
              - Use these values to infer:
                - finished wall height
                - floor-to-floor height
                - parapet extension
                - vaulted ceiling rise
                - staircase double-height regions
              - If the target polygon corresponds to a room adjacent to an exterior facade:
                - Align the room horizontally with the elevation facade.
                - Infer the likely wall height from the corresponding facade segment.
              - If multiple floor levels exist:
                - Match the target room floor index using:
                  - staircase alignment
                  - room naming
                  - window positioning
                  - floor datum labels
                  - vertical continuity
              - If roof slopes are visible in elevation:
                - Infer sloped ceiling height transition from:
                  - ridge height
                  - eave height
              - For vaulted ceilings:
                - wall base height is measured to the spring line / plate height.
                - maximum ceiling height extends toward the ridge.
              - If ceiling is flat:
                - use top plate elevation minus finished floor elevation.
              - If no reliable elevation-derived height is available:
                - fallback to standard residential assumptions:
                  - 8 ft interior wall
                  - 9 ft main living spaces
                  - 10+ ft luxury/open foyer/great room

        - Infer the type of the perimeter wall as one from the following templates. Do not generate any other wall type not present in the templates.
          WALL_TYPE TEMPLATES:
            1. OPEN_TO_BELOW
            2. FULL_WALL
            3. HALF_WALL
            4. STAIRCASE_WALL
            5. SOFFITS
            6. MULTI_FLOOR_ALIGNMENT
            7. DEMISING_WALL
            8. GARAGE_SEPARATION_WALL
            9. SHAFT_WALL
            10. WET_WALL
            11. HALLWAY_WALL
        - For every identified perimeter wall segment, scan along its entire length to detect any interruptions or embedded symbols indicating openings.

          - A valid opening MUST:
            -> Interrupt or replace a portion of the wall thickness
            -> Be spatially aligned with the wall segment
            -> Lie within the projection bounds of the wall

          - Detect the following opening types:
            1. Doors (swing arcs, hinge marks, door panels)
            2. Windows (thin rectangles within wall thickness)
            3. Sliding doors / curtain walls (parallel panel lines)
            4. Arched openings (curved top openings)
            5. Pass-through / open voids (breaks without door symbol)
            6. NULL (no valid opening detected)

          - Reject false positives:
            -> Ignore annotations containing "WALL", "CLG", "HGT", "TYP", "EQ"
            -> Ignore dashed rectangles that do not break wall continuity
            -> Ignore dimension lines and construction guides

          - Assign openings to walls using:
            -> Minimum perpendicular distance to wall centerline
            -> Alignment with wall orientation (horizontal/vertical/inclined)
            -> Overlap with wall segment extent

          - Extract opening dimensions using priority:
            1. Explicit dimension annotation near the opening
            2. Encoded text formats:
              -> `(3) 3'-8" x 8'-0"` → count = 3, width = 3.66 ft, height = 8 ft
              -> `2-2424 FIXED` → count = 2, width = 2 ft, height = 2 ft
            3. If no annotation:
              -> Estimate width from pixel span using scale
              -> Estimate height using standard defaults or mark as UNKNOWN

          - Group identical openings along the same wall:
            -> Aggregate count
            -> Use consistent dimensions

          - If no valid opening is detected return "opening_type": "NULL" with "count", "length" and "height" set to 0.

      CEILING_EXTRACTION_INSTRUCTIONS:
        - The polygon marked in transparent red color marks the target ceiling in the input image.
        - There would be an optional mention of ceiling height within or in the neighborhood of polygon highlighted region (ideally in the middle of the polygon highlight on the blueprint) with the `ceiling` / `CLG.` or `height` / `HGT.` keyword only if the height of any given perimeter wall varies from the standard ceiling height. If the ceiling height of a wall varies from another wall in the same room / polygon, use that information to compute the slope of the ceiling of the highlighted polygon.
        - Ceiling slope MUST NOT be guessed from floorplan alone. It MUST be derived from elevation plans via geometric mapping.
        - If ceiling / wall height is exclusively not mentioned, treat the ceiling type as flat with no slope or (rise=0, run=0).
        - To compute ceiling slopes understand the provided elevation plans following the ELEVATION_SLOPE_INTERPRETATION_RULES as follows,
          A slope annotation (e.g., 4:12) is ALWAYS perpendicular to the ridge line and ALWAYS interpreted relative to the elevation viewing direction.
 
          - STEP 1: IDENTIFY ELEVATION VIEW TYPE

            For each elevation:
            - FRONT / REAR elevation:
              → Viewer is looking along Y-axis
              → Visible width = X-axis (horizontal)
              → Vertical = Z-axis (height)
 
            - SIDE elevation:
              → Viewer is looking along X-axis
              → Visible width = Y-axis (horizontal)
              → Vertical = Z-axis (height)
 
          - STEP 2: INTERPRET SLOPE SYMBOL ORIENTATION
 
            A slope annotation includes:
              - A numeric ratio (e.g., 4:12)
              - An arrow OR slope line
 
            Interpret as:
 
              CASE A: Arrow pointing LEFT or RIGHT
                → slope varies along horizontal axis of that elevation
 
              CASE B: Arrow pointing UP or DOWN
                → indicates rise direction only (still horizontal run)
 
              CASE C: Slope line drawn diagonally
                → direction of slope is perpendicular to ridge line
 
          - STEP 3: CONVERT TO GLOBAL FLOORPLAN AXIS
 
            If elevation is FRONT/REAR:
              horizontal direction in elevation = FLOORPLAN X-axis
              → tilt_axis = "horizontal"
 
            If elevation is SIDE:
              horizontal direction in elevation = FLOORPLAN Y-axis
              → tilt_axis = "vertical"
 
          - STEP 4: HANDLE MULTIPLE SLOPES (IMPORTANT)
 
            If:
              - Front elevation shows slope A
              - Side elevation shows slope B
 
            Then:
              → This is a multi-directional roof (hip / complex / gable combo)
 
            Rules:
              - If slopes are orthogonal → multi-plane ceiling
              - If only one slope applies to mapped wall → use that slope ONLY for that axis
              - NEVER average slopes across different elevations
 
          - STEP 5: RIDGE DETECTION
 
            - Ridge line is ALWAYS perpendicular to slope direction
            - If ridge is horizontal in elevation:
              slope axis is vertical in floorplan
            - If ridge is vertical in elevation:
              slope axis is horizontal in floorplan
 
          - STEP 6: VALIDATION
 
            Reject incorrect slope interpretation if:
              - Slope direction conflicts between mapped walls
              - Slope axis does not align with wall orientation
              - Elevation does not correspond to mapped wall
 
          - STEP 7: FINAL MAPPING
 
            Output must ensure:
              - slope value derived from correct elevation
              - tilt_axis matches floorplan axis
              - slope direction consistent with wall mapping

        - COMPUTE PITCH of the SLOPE(DETERMINISTIC) using the following instructions,
          Use ONE of the following (priority order):
          METHOD A: Direct pitch annotation
            pitch =>
              - `rise`: <rise> 
              - `run`: <run>

          METHOD B: Height difference
            pitch =>
              - `rise`: <(H2 - H1)>
              - `run`: <horizontal_length>

          METHOD C: Pixel-based fallback (ONLY if no annotation)
            - Compute vertical pixel delta from elevation
            - Convert using known height annotations
            - Derive pitch

        - The `tilt_axis` of a sloped ceiling is in the direction against the axial projection of the inclination. The `ceiling_axis` runs through the central axial line of the ceiling in the direction of the inclination. The `tilt_axis` is one of the axial lines (x-> horizontal, y-> vertical). `tilt_axis` can only have a value "horizontal" or "vertical" or "NULL" depending on the angular orientation of the ceiling plane against. Mention "NULL" only if slope angle is 0. The slope of the ceiling / `ceiling_axis` is measured against its axial line / `tilt_axis` (x-> horizontal, y-> vertical).
          - If slope direction aligns with:
            horizontal walls → tilt_axis = "horizontal"
            vertical walls → tilt_axis = "vertical"
          - If ambiguous → choose dominant slope direction
          - If flat → tilt_axis = NULL
        - To compute the height of a sloped ceiling, always consider the maximum height.
        - Given the length of each perimeter walls, compute the area of ceiling or the highlighted polygon in SQFT without taking the slope value (if present) into account.
          -> **STRICTLY REMEMBER** the shape of the ceiling could be complex (convex or concave) and hence always apply SHOELACE on ceiling vertices to compute the area and Do NOT use the wall length / OCR data to compute the area.
        - To predict ceiling type, You must support ONLY one of the following ceiling types,
          - `Flat` -> Standard Ceiling
          - `Single-sloped` -> Shed ceiling (one plane sloped)
          - `Gable` -> Cathedral ceiling (two sloped planes meeting at ridge)
          - `Tray` -> flat center + flat perimeter “step” + vertical faces
          - `Barrel vault` -> curved ceiling, common “arched” vault
          - `Coffered` -> grid beams + recess panels
          - `Combination` -> Flat + Vault
          - `Soffit` -> Bulkhead Ceiling Area
          - `Cove` -> curved wall-to-ceiling transition
          - `Dome` -> Rotunda Ceiling
          - `Cloister Vault` -> four curved surfaces meeting at center
          - `Knee-Wall` -> Attic Ceiling
          - `Cathedral with Flat Center` -> Hybrid Vault
          - `Angled-Plane` -> Faceted Ceiling
          - `Boxed-Beam` -> Ceiling with false structural beams
        - The above is a list of few common ceiling type codes on left (enclosed in ``) mapped with their descriptions on right. Use only ceiling type code to predict the `ceiling type`. DO NOT update the letters or words present in the ceiling type code.
        - If the ceiling type of the highlighted room / polygon appears ambiguous, use `Flat` as the ceiling type code.

      WALL_IDENTITY_PREDICTOR_INSTRUCTIONS:
        - The perimeter wall is likely to be a horizontal one if, their `Y` coordinates are same or have very little difference in values but the difference between their 'X' coordinates have a greater value.
        - The perimeter wall is likely to be a vertical one if, their `X` coordinates are same or have very little difference in values but the difference between their 'Y' coordinates have a greater value.
        - Figure out the appropriate text entity that could represent the name of the room that the provided polygon belongs to.
        - A `Room Name` is most likely to be present near the middle / centroid of the highlighted polygon represented by the centroid of the provided polygon vertices `CENTROID([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])`.
        - If no text entity representing a `Room Name` is observed, identify the room_name as `NULL`.

      DRYWALL_PREDICTION_INSTUCTIONS:
        - Drywalls are marked with green polygons adjacent to the surrounding walls of the target polygon marking the interiors of the polygon.
        - Use the below factors to decide on the drywall material prediction,
          -> Wall location (interior, exterior, garage, wet area)
          -> Adjacent room usage
          -> Fire separation requirements (CBC, IRC R302)
          -> Moisture and mold resistance needs
          -> Typical residential drywall standards in the target project location
          -> Required wall height and the vertical position of each drywall layer
          -> Cost reduction while satisfying all applicable code and performance requirements
        - A single drywall material preference for each wall is MANDATORY.

        WALL_DRYWALL_STACKING_INSTRUCTIONS:
          - Optionally predict an additional vertically stacked drywall preferences for each of the walls (only if stacked drywall preferences applicable else leave all 3 lists `materials_vertically_stacked`, `color_codes_stacked` and `heights_stacked` empty). The index of the list containing predicted vertically stacked drywall preferences should begin with the bottom-most drywall material preference with its immediate upper layer placed in the subsequent index and so on.
          - Vertically stacked drywall is permitted only when the wall surface can be logically divided into multiple vertical zones, such as:
            -> A moisture-resistant lower zone and standard drywall upper zone
            -> A fire-rated lower or upper zone
            -> A garage or utility-area assembly with different vertical protection requirements
            -> A code-driven change in material requirement at a specific elevation
            -> Another clearly justified construction condition
          - If vertically stacked drywall is predicted:
            -> The first element MUST represent the bottom-most drywall layer.
            -> Each subsequent element MUST represent the immediately higher drywall layer.
            -> The vertical layers MUST be contiguous.
            -> No vertical gaps are permitted.
            -> No vertical overlaps are permitted.
            -> The sum of all stacked layer heights MUST equal the total drywall-applied wall surface height within a maximum tolerance of 0.01 feet.
          - The following invariant MUST always hold:
            total_wall_height = sum(heights_stacked[i] for i in range(len(heights_stacked)))
          - If vertically stacked drywall preferences list is non-empty **STRICTLY** include the single drywall material preference into the list along with the additional stack to ensure that the MANDATED single drywall preference prediction and the OPTIONAL vertically stacked drywall preferences prediction can be referred independently by the user as per the preference (single/stacked).

        WALL_DRYWALL_LAYERING_INSTRUCTIONS
          - The field `layers` represents the horizontal drywall layering count (number of drywall sheets installed over the same surface), not the vertical segmentation of the wall.
          - Vertical stacking and horizontal layering are independent concepts.
          - Every vertical drywall stack (represented by materials_vertically_stacked) has its own horizontal drywall layer count.
          - If only a single drywall material is used:
            `layers` MUST be a single integer.
            Example:
              "material": "5/8 Type X",
              "layers": 2
            indicates two horizontal sheets of the same drywall over the entire wall height.
          - If vertically stacked drywall materials are predicted:
            `layers` MUST become a list.
            The number of entries in `layers` MUST exactly equal the number of entries in:
              `materials_vertically_stacked`
              `color_codes_stacked`
              `heights_stacked`
            Each element specifies the horizontal sheet count for the corresponding vertical drywall zone.

        CEILING_DRYWALL_STACKING_INSTRUCTIONS:
          - A ceiling drywall assembly may contain either:
            1. A single drywall material, or
            2. Multiple vertically stacked drywall layers.
          - Each ceiling drywall layer is applied over the SAME ceiling surface area.
            Therefore, all vertically stacked ceiling drywall layers MUST have the same predicted ceiling area.
          - The order of `materials_vertically_stacked` MUST start with the lowest/bottom-most drywall layer and proceed upward toward the ceiling assembly.
          - The primary `material` field MUST always contain exactly one drywall material preference for the ceiling.
          - If `materials_vertically_stacked` is non-empty, it MUST include the primary `material` as one of the layers.
          - The number of entries in:
            `materials_vertically_stacked`
            `color_codes_stacked`
            MUST be identical.
          - The stacked ceiling layers are NOT vertically segmented by ceiling height. They are separate material layers installed over the same ceiling plane.
          - If no additional ceiling drywall layer is required, all stacked arrays MUST be empty.
          - If stacked layers are predicted, the primary single-material prediction MUST still remain independently available through the primary `material`, `color_code`.

        CEILING_DRYWALL_LAYERING_INSTRUCTIONS
          - Since all stacked ceiling drywall materials occupy the same ceiling area:
          - If only one drywall material is used:
            `layers` MUST be a single integer.
            Example:
              "material": "5/8 Type X",
              "layers": 2
          - If multiple ceiling drywall layers are predicted:
            `layers` MUST be a list.
          - The number of entries MUST exactly equal the number of entries in:
            `materials_vertically_stacked`
            `color_codes_stacked`
            Each value specifies the horizontal sheet count for that ceiling assembly layer.

      JURISDICTION_AND_CODE_SELECTION:
        Project Location:
          - State / Province / Region / Pincode: {project_location}

        Regulatory Requirements:
          - Determine the applicable residential building code, fire code, drywall requirements, and construction practices for the specified project location.
          - Use the commonly adopted residential drywall standards, fire separation requirements, moisture protection requirements, and local code amendments for that jurisdiction.
          - Apply the building regulations that are typically enforced by the local Authority Having Jurisdiction (AHJ).
          - Follow local residential construction practices unless the architectural drawings explicitly specify otherwise.

        Fallback Behavior:
          - If the provided project location is missing, ambiguous, invalid, or cannot be confidently mapped to a real jurisdiction, default to:
            • California Residential Code (CRC)
            • California Building Code (CBC)
            • California Fire Code (CFC)
            • California Residential Code adoption of the IRC where applicable
          - Treat California as the default jurisdiction only in these fallback situations.

        General Assumptions:
          - Residential construction.
          - Standard wood or steel stud framing unless drawings indicate otherwise.
          - Do not assume commercial construction requirements unless explicitly shown.

        You must only support the drywall types from the provided templates,
        DRYWALL TEMPLATES: {drywall_templates}

        **STRICTLY** use the field `sku_variant` which contains both `sku_id` and `sku_description` as the target drywall material and the field `color_code` to map to it's target color code accompanied by the fields `fire_rating` aand `thickness` to derive it's fire rating and thickness respectively.
        Do not invent other drywall materials or color codes which are not included into the template list. All of the provided drywall types are associated with a definite color code presented in BGR (blue, green, red) format.
        If an appropriate/optimal drywall material for a given wall or polygon is not provided with the `DRYWALL_TEMPLATES` mention the target drywall material as `DISABLED` with [0, 0, 255] in BGR tuple as its target color code.

  OUTPUT:
    Your output must be precise, code-aligned, and structured. Do not hallucinate dimensions or materials. If information is ambiguous, state assumptions explicitly.
    **STRICTLY**
      - `wall_parameters` field should contain predicted wall parameters and drywall assembly for all the perimeter walls provided in the input that also corresponds with the perimeter lines highlighted with blue bounding boxes of the highlighted polygon.
      - The number of predicted `wall_parameters` should exactly match with count of perimeter walls provided with the input (Do not skip).
      - The order of the walls provided in the `wall_parameters` list should follow the order in which the perimeter walls are provided in the input.
      - Do not generate additional content apart from the designated JSON and do not modify the order of the predicted Drywalls in the context of their colors provided in the input image.
    Please refer the following as a reference and ensure to replace every consecutive pair of open/closed curly braces with a single one during the generation of the output.
    {{
      "ceiling": {{
        "room_name": "<Detected Room Name the ceiling belongs to / NULL>",
        "area": <Area of the ceiling in SQFT (Square Feet)>,
        "confidence_area": <confidence score in predicting the area of the ceiling between 0 and 1 in float rounded upto 2 decimal places (e.g., 0.87)>,
        "ceiling_type": "<Type code of the ceiling>",
        "height": <height of the ceiling (centroid of the ceiling axis, if sloped)>,
        "confidence_height": <confidence score in predicting the height of the ceiling between 0 and 1 in float rounded upto 2 decimal places (e.g., 0.87)>,
        "pitch": {{
          "rise": <rise of the slope in float>,
          "run": <run of the slope in float>
        }},
        "slope_enabled": <is sloping supported given the type of ceiling used (True/False)>,
        "tilt_axis": "<axial direction of the tilted slope / NULL>",
        "drywall_assembly": {{
          "material": "<drywall material for the ceiling>",
          "color_code": <color code for the predicted ceiling drywall type in a BGR tuple (`Blue`, `Green`, `Red`)>,
          "materials_vertically_stacked": ["<vertically stacked drywall material preference 1 for the ceiling (optional)>", "<vertically stacked drywall material preference 2 for the ceiling (optional)>"],
          "color_codes_stacked": [<color code for the vertically stacked drywall type 1 in a BGR tuple (`Blue`, `Green`, `Red`) for the ceiling>, <color code for the vertically stacked drywall type 2 in a BGR tuple (`Blue`, `Green`, `Red`) for the ceiling>],
          "thickness": <thickness of the predicted ceiling drywall type in feet>,
          "layers": <number of required drywall layers>,
          "fire_rating": <fire-rating of the predicted drywall type in hours>,
          "waste_factor": "<waste factor of the predicted drywall in percentage>"
        }},
        "code_references": ["<applied Dywall code reference 1>", "<applied Dywall code reference 2>", "<applied Dywall code reference 3>"],
        "recommendation": "<recommendation on special requirements including cost reduction (if any)>"
      }},
      "wall_parameters": [
        {{
          "room_name": "<Detected Room Name the perimeter wall 1 belongs to / NULL>",
          "length": <length of perimeter wall 1 in feet>,
          "confidence_length": <confidence score in predicting the length of the perimeter wall 1 between 0 and 1 in float rounded upto 2 decimal places (e.g., 0.87)>,
          "width": <width of the perimeter wall 1 in feet / None>,
          "wall_type": "<type of the perimeter wall 1>",
          "openings": [
            {{"opening_type": "<Type of the perimeter wall 1 opening 1>", "count": <count of the opening type 1>, "length": <length of the opening type 1 in feet>, "height": <height of the opening type 1 in feet>}},
            {{"opening_type": "<Type of the perimeter wall 1 opening 2>", "count": <count of the opening type 2>, "length": <length of the opening type 2 in feet>, "height": <height of the opening type 2 in feet>}}
          ]
          "drywall_assembly": {{
            "material": "<drywall material for the perimeter wall 1>",
            "height": <height of the perimeter wall 1 surface the drywall is applied upon in feet>,
            "confidence_height": <confidence score in predicting the height of the perimeter wall 1 surface the drywall is applied upon between 0 and 1 in float rounded upto 2 decimal places (e.g., 0.87)>,
            "color_code": <color code for the predicted perimeter wall 1 drywall type in a BGR tuple (`Blue`, `Green`, `Red`)>,
            "materials_vertically_stacked": ["<vertically stacked drywall material preference 1 for perimeter wall 1 (optional)>", "<vertically stacked drywall material preference 2 for perimeter wall 1 (optional)>"],
            "color_codes_stacked": [<color code for the vertically stacked drywall type 1 in a BGR tuple (`Blue`, `Green`, `Red`) for perimeter wall 1>, <color code for the vertically stacked drywall type 2 in a BGR tuple (`Blue`, `Green`, `Red`) for perimeter wall 1>]
            "heights_stacked": [<height of the vertically stacked drywall type 1 for perimeter wall 1 in feet>, <height of the vertically stacked drywall type 2 of perimeter wall 1 in feet>]
            "thickness": <thickness of the predicted wall drywall type in feet>,
            "layers": <number of required drywall layers>,
            "fire_rating": <average fire-rating of the predicted drywall types in hours>,
            "waste_factor": "<waste factor of the predicted drywall in percentage>"
          }},
          "code_references": ["<applied Dywall code reference 1>", "<applied Dywall code reference 2>", "<applied Dywall code reference 3>"],
          "recommendation": "<recommendation on special requirements for perimeter wall 1 including cost reduction (if any). Generate separate recommendations for single drywall material and the vetically stacked drywall materials (If predicted)>"
        }},
        {{
          "room_name": "<Detected Room Name the perimeter wall 2 belongs to / NULL>",
          "length": <length of perimeter wall 2 in feet>,
          "confidence_length": <confidence score in predicting the length of the perimeter wall 2 between 0 and 1 in float rounded upto 2 decimal places (e.g., 0.87)>,
          "width": <width of the perimeter wall 2 in feet / None>,
          "wall_type": "<type of the perimeter wall 2>",
          "openings": [
            {{"opening_type": "<Type of the perimeter wall 2 opening 1>", "count": <count of the opening type 1>, "length": <length of the opening type 1 in feet>, "height": <height of the opening type 1 in feet>}}
          ]
          "drywall_assembly": {{
            "material": "<drywall material for the perimeter wall 2>",
            "height": <height of the perimeter wall 2 surface the drywall is applied upon in feet>,
            "confidence_height": <confidence score in predicting the height of the perimeter wall 2 surface the drywall is applied upon between 0 and 1 in float rounded upto 2 decimal places (e.g., 0.87)>,
            "color_code": <color code for the predicted perimeter wall 2 drywall type in a BGR tuple (`Blue`, `Green`, `Red`)>,
            "materials_vertically_stacked": ["<vertically stacked drywall material preference 1 for perimeter wall 2 (optional)>", "<vertically stacked drywall material preference 2 for perimeter wall 2 (optional)>"],
            "color_codes_stacked": [<color code for the vertically stacked drywall type 1 in a BGR tuple (`Blue`, `Green`, `Red`) for perimeter wall 2>, <color code for the vertically stacked drywall type 2 in a BGR tuple (`Blue`, `Green`, `Red`) for perimeter wall 2>]
            "heights_stacked": [<height of the vertically stacked drywall type 1 for perimeter wall 2 in feet>, <height of the vertically stacked drywall type 2 of perimeter wall 2 in feet>]
            "thickness": <thickness of the predicted wall drywall type in feet>,
            "layers": <number of required drywall layers>,
            "fire_rating": <average fire-rating of the predicted drywall types in hours>,
            "waste_factor": "<waste factor of the predicted drywall in percentage>"
          }},
          "code_references": ["<applied Dywall code reference 1>", "<applied Dywall code reference 2>", "<applied Dywall code reference 3>"],
          "recommendation": "<recommendation on special requirements for perimeter wall 2 including cost reduction (if any). Generate separate recommendations for single drywall material and the vetically stacked drywall materials (If predicted)>"
        }}
      ]
    }}
"""

polygon_and_drywall_predictor_image_samples, polygon_and_drywall_predictor_json_samples, polygon_and_drywall_predicted_json_samples = glob("prompts/polygon_detector_and_drywall_predictor_california_few_shot/polygon_and_drywall_predictor_*.png"), glob("prompts/polygon_detector_and_drywall_predictor_california_few_shot/polygon_and_drywall_predictor_*.json"), glob("prompts/polygon_detector_and_drywall_predictor_california_few_shot/polygon_and_drywall_predicted_*.json")
POLYGON_DETECTOR_AND_DRYWALL_PREDICTOR_CALIFORNIA_FEW_SHOT = list()
for polygon_and_drywall_predictor_image_sample, polygon_and_drywall_predictor_json_sample, polygon_and_drywall_predicted_json_sample in zip(polygon_and_drywall_predictor_image_samples, polygon_and_drywall_predictor_json_samples, polygon_and_drywall_predicted_json_samples):
    canvas = cv2.imread(polygon_and_drywall_predictor_image_sample)
    _, canvas_buffer_array = cv2.imencode(".png", canvas)
    bytes_canvas = canvas_buffer_array.tobytes()
    with open(polygon_and_drywall_predictor_json_sample, 'r') as f:
        polygon_and_drywall_predictor = json.load(f)
    with open(polygon_and_drywall_predicted_json_sample, 'r') as f:
        polygon_and_drywall_predicted = json.load(f)
    sample_one_shot = [
      Content(
        role="user",
        parts=[
          Part.from_text("Quantify the highlighted polygon in red with the dimension numbers mapped with the OCR predicted data and indentify the drywall types of the ceiling and the perimeter walls."),
          Part.from_data(data=bytes_canvas, mime_type="image/png")
        ]
      ),
      Content(
        role="user",
        parts=[
            Part.from_text(json.dumps(polygon_and_drywall_predictor))
        ]
      ),
      Content(
        role="model",
        parts=[
            Part.from_text(json.dumps(polygon_and_drywall_predicted))
        ]
      )
    ]
    POLYGON_DETECTOR_AND_DRYWALL_PREDICTOR_CALIFORNIA_FEW_SHOT.extend(sample_one_shot)

def ensure_not_nan(v: float) -> float:
    if v is None:
        return v
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        raise ValueError("NaN or Inf not allowed")
    return v

class ThicknessValidatorHelper(BaseModel):

    @field_validator("thickness", check_fields=False)
    @classmethod
    def validate_float(cls, v):
      if "thickness" not in cls.active_validators:
        return v
      return ensure_not_nan(v)

class HeightValidatorHelper(BaseModel):

    @field_validator("height", check_fields=False)
    @classmethod
    def validate_float(cls, v):
      if "height" not in cls.active_validators:
        return v
      return ensure_not_nan(v)

class AreaValidatorHelper(BaseModel):

    @field_validator("area", check_fields=False)
    @classmethod
    def validate_float(cls, v):
      if "area" not in cls.active_validators:
        return v
      return ensure_not_nan(v)

class LengthValidatorHelper(BaseModel):

    @field_validator("length", check_fields=False)
    @classmethod
    def validate_float(cls, v):
      if "length" not in cls.active_validators:
        return v
      return ensure_not_nan(v)

class WidthValidatorHelper(BaseModel):

    @field_validator("width", check_fields=False)
    @classmethod
    def validate_optional_float(cls, v):
      if "width" not in cls.active_validators:
        return v
      if v is None:
          return v
      return ensure_not_nan(v)

class LayersValidatorHelper(BaseModel):

    @field_validator("layers", check_fields=False)
    @classmethod
    def validate_positive(cls, layers):
      if "layers" not in cls.active_validators:
        return layers
      if isinstance(layers, int):
        if layers < 1:
          raise ValueError("layers must be >= 1")
      else:
        if any(l < 1 for l in layers):
          raise ValueError("Every stacked layer count must be >= 1")
      return layers

class ColorCodeValidatorHelper(BaseModel):

    @field_validator("color_code", check_fields=False)
    @classmethod
    def validate_bgr(cls, v):
      if "color_code" not in cls.active_validators:
        return v
      if len(v) != 3:
          raise ValueError("color_code must be BGR tuple")
      if not all(0 <= c <= 255 for c in v):
          raise ValueError("Invalid BGR value")
      return v

class StackedLayersValidatorHelper(BaseModel):

    @model_validator(mode="after")
    def validate_stacked_layers(self):
      if "stacked_layers" not in self.active_validators:
          return self
      if not len(self.materials_vertically_stacked) == len(self.color_codes_stacked):
          raise ValueError("Vertically stacked material count does not equate with stacked color codes count for the ceiling")
      if self.materials_vertically_stacked:
          if self.material not in self.materials_vertically_stacked:
              raise ValueError(
                "Primary material must also appear in materials_vertically_stacked."
              )
          if not isinstance(self.layers, list):
              self.layers = [1] * len(self.materials_vertically_stacked)

          elif len(self.layers) != len(self.materials_vertically_stacked):
              raise ValueError(
                  "`layers` must contain one entry per stacked ceiling layer."
              )

      else:
          if isinstance(self.layers, list):
              if not self.layers:
                  self.layers = 1
              elif len(self.layers) == 1:
                  self.layers = self.layers[0]
              else:
                  raise ValueError(
                      "`layers` must be a single integer for non-stacked ceilings."
                  )

      return self

class StackCountValidatorHelper(BaseModel):

    @model_validator(mode="after")
    def check_stack_count(self):
      if "stack_count" not in self.active_validators:
          return self
      if not len(self.materials_vertically_stacked) == len(self.color_codes_stacked):
          raise ValueError("Vertically stacked material count does not equate with stacked color codes count and stacked heights count for the walls")
      return self

class WallCountValidatorHelper(BaseModel):

    @model_validator(mode="after")
    def check_wall_count(self):
      if "wall_count" not in self.active_validators:
          return self
      if len(self.wall_parameters) < 1:
          raise ValueError("At least one wall required")
      return self

class DrywallAssemblyCeiling(
  ThicknessValidatorHelper,
  LayersValidatorHelper,
  ColorCodeValidatorHelper,
  StackedLayersValidatorHelper,
  BaseModel,
):
    active_validators: ClassVar[set[str]] = {"thickness", "layers", "color_code", "stacked_layers"}
    model_config = ConfigDict(extra="forbid")

    material: str = Field(description="'<drywall material for the ceiling>'")
    color_code: Tuple[int, int, int] = Field(description="<color code for the predicted ceiling drywall type in a BGR tuple (`Blue`, `Green`, `Red`)>")
    materials_vertically_stacked: List[str] = Field(description="['<vertically stacked drywall material preference 1 for the ceiling (optional)>', '<vertically stacked drywall material preference 2 for the ceiling (optional)>']")
    color_codes_stacked: List[Tuple[int, int, int]] = Field(description="[<color code for the vertically stacked drywall type 1 in a BGR tuple (`Blue`, `Green`, `Red`) for the ceiling>, <color code for the vertically stacked drywall type 2 in a BGR tuple (`Blue`, `Green`, `Red`) for the ceiling>]")
    thickness: float = Field(description="<thickness of the predicted ceiling drywall type in feet>")
    layers: Union[int, List[int]] = Field(description="<number of required drywall layers>")
    fire_rating: Optional[Union[str, float]] = Field(description="<fire-rating of the predicted drywall type in hours>")
    waste_factor: Union[str, int, float] = Field(description="'<waste factor of the predicted drywall in percentage>'")

class DrywallAssemblyWall(
  ThicknessValidatorHelper,
  HeightValidatorHelper,
  ColorCodeValidatorHelper,
  StackCountValidatorHelper,
  BaseModel,
):
    active_validators: ClassVar[set[str]] = {"thickness", "height", "color_code", "stack_count"}
    model_config = ConfigDict(extra="allow")

    material: str = Field(description="'<drywall material for the target perimeter wall>'")
    height: float = Field(description="<height of the target perimeter wall surface the drywall is applied upon in feet>")
    confidence_height: float = Field(ge=0, le=1)
    color_code: Tuple[int, int, int] = Field(description="<color code for the predicted target perimeter wall drywall type in a BGR tuple (`Blue`, `Green`, `Red`)>")
    materials_vertically_stacked: List = Field(description="['<vertically stacked drywall material preference 1 for target perimeter wall (optional)>', '<vertically stacked drywall material preference 2 for target perimeter wall (optional)>']")
    color_codes_stacked: List = Field(description="[<color code for the vertically stacked drywall type 1 in a BGR tuple (`Blue`, `Green`, `Red`) for target perimeter wall>, <color code for the vertically stacked drywall type 2 in a BGR tuple (`Blue`, `Green`, `Red`) for target perimeter wall>]")
    thickness: float = Field(description="<thickness of the predicted wall drywall type in feet>")
    layers: int = Field(description="<number of required drywall layers>")
    fire_rating: Optional[Union[str, float]] = Field(description="<average fire-rating of the predicted drywall types in hours>")
    waste_factor: Union[str, int, float] = Field(description="'<waste factor of the predicted drywall in percentage>'")

class Pitch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rise: float = Field(description="<rise of the slope in float>")
    run: float = Field(description="<run of the slope in float>")

class CeilingModelAndPredict(AreaValidatorHelper, HeightValidatorHelper, BaseModel):
    active_validators: ClassVar[set[str]] = {"area", "height"}
    model_config = ConfigDict(extra="forbid")

    room_name: Optional[str] = Field(description="'<Detected Room Name the ceiling belongs to / NULL>'")
    area: float = Field(description="<Area of the ceiling in SQFT (Square Feet)>")
    confidence_area: float = Field(ge=0, le=1)
    ceiling_type: str = Field(description="'<Type code of the ceiling>'")
    height: float = Field(description="<height of the ceiling (centroid of the ceiling axis, if sloped)>")
    confidence_height: float = Field(ge=0, le=1)
    pitch: Pitch
    slope_enabled: bool = Field(description="<is sloping supported given the type of ceiling used (True/False)>")
    tilt_axis: Optional[Literal["horizontal", "vertical", "NULL"]] = Field(description="'<axial direction of the tilted slope / NULL>'")
    drywall_assembly: DrywallAssemblyCeiling
    code_references: List[str] = Field(description="['<applied Dywall code reference 1>', '<applied Dywall code reference 2>', '<applied Dywall code reference 3>']")
    recommendation: Optional[str] = Field(description="'<recommendation on special requirements including cost reduction (if any)>'")

class WallParameterModelAndPredict(LengthValidatorHelper, WidthValidatorHelper, BaseModel):
    active_validators: ClassVar[set[str]] = {"length", "width"}
    model_config = ConfigDict(extra="forbid")

    room_name: Optional[str] = Field(description="'<Detected Room Name the target perimeter wall belongs to / NULL>'")
    length: float = Field(description="<length of target perimeter wall in feet>")
    confidence_length: float = Field(ge=0, le=1)
    width: Optional[float] = Field(description="<width of the target perimeter wall in feet / None>")
    wall_type: str = Field(description="'<type of the perimeter wall 2>'")
    openings: List[Dict] = Field(description="[{{'opening_type': '<Type of the target perimeter wall opening 1>', 'count': <count of the opening type 1>, 'length': <length of the opening type 1 in feet>, 'height': <height of the opening type 1 in feet>}}]")
    drywall_assembly: DrywallAssemblyWall
    code_references: List[str] = Field(description="['<applied Dywall code reference 1>', '<applied Dywall code reference 2>', '<applied Dywall code reference 3>']")
    recommendation: Optional[str] = Field(description="'<recommendation on special requirements for perimeter wall 2 including cost reduction (if any). Generate separate recommendations for single drywall material and the vetically stacked drywall materials (If predicted)>'")

class PolygonDetectorAndDrywallPredictorResponse(WallCountValidatorHelper, BaseModel):
    active_validators: ClassVar[set[str]] = {"wall_count"}
    model_config = ConfigDict(extra="forbid")

    ceiling: CeilingModelAndPredict
    wall_parameters: List[WallParameterModelAndPredict]

POLYGON_DETECTOR = """
  You are a licensed building-code-aware construction expert with Senior Architectural Drawing Interpretation Engine capabilities. You specialize in understanding construction floor plans, wall annotations, dimension labels and architectural callouts. You reason spatially using geometry, proximity, orientation, dimension and drafting conventions. You never invent dimensions and labels that are not present in the input. You return structured, deterministic outputs.

  PROVIDED:
    1. A polygon represented by a list of vertices and the polygon perimeter lines/edges joining the vertices with origin set to LEFT, TOP of the original floorplan and offset set to (0, 0):
      Vertices: [(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)]
      Perimeter wall endpoints: [
        wall: (X1, Y1) → (X2, Y2),
        wall: (X2, Y2) → (X3, Y3),
        wall: (X4, Y4) → (X3, Y3),
        wall: (X1, Y1) → (X4, Y4)
      ]

    2. A cropped snapshot of the room or the polygon from Architectural Drawing in png format inscribed with textual annotations containing the name of the room it belongs to with the wall line dimensions along with the following highlights,
      - The target polygon/room highlighted with transparent red color that corresponds with the provided polygon vertices computed from the whole floor plan using original offset with same resolution and the area is inscribed with the room name information.
      - The target polygon/room's perimeter lines highlighted with blue bounding boxes that corresponds with provided polygon perimeter wall endpoints computed from the whole floor plan using original offset with same resolution and the nearby regions are inscribed with textual annotations containing dimension marker and the dimension, width and height (optional) of the wall in `(feet) and ``(inches) describing the layout and shape of the target and adjacent rooms.

    3. A list of transcription entries extracted from a construction floorplan nearest to the given wall.
       Each entry contains:
         - text: the recognized text string
         - centroid: (cx, cy) representing the visual center of the text's bounding box

    4. A set of elevation plan snapshots (multi-page or grouped), where each elevation corresponds to one or more sides of the building.
       Each elevation includes:
         - Vertical height annotations (top plate, ridge, ceiling, slab, etc.)
         - Roof slopes / pitch annotations (e.g., 4:12, 6:12, angles)
         - Window / door vertical alignment
         - Wall-to-roof junction geometry
       Elevation pages are NOT labeled with explicit mapping to floorplan walls.
       You MUST infer correspondence using geometry, openings, and relative positioning.

    Analyze the snapshot provided from the floor plan image.

    Your task is to,
        - Predict the relevant wall dimensions (ONLY width and height) for each highlighted walls as per the instructions provided.
        - Predict the relevant ceiling dimensions (height, area, pitch_of_slope, axis_of_slope and type_of_slope) for the highlighted room/polygon as per the instructions provided.

  TASK:
    Analyze the architectural floor plan and highlighted wall segments accompanied by polygon vertices, it's perimeter wall endpoints and OCR extracted transcription entries from the floor plan to determine the following features,
      - The `width` and `height` in feet accompanied by `type` of each perimeter wall based upon the provided `WALL_EXTRACTION_INSTRUCTIONS`.
      - Identify The `ceiling_type`, `height` and `slope` of the ceiling of the highlighted room / polygon based upon the provided `CEILING_EXTRACTION_INSTRUCTIONS`.
      - Identify the `Room Name` the highlighted polygon belongs to. Follow `WALL_IDENTITY_PREDICTOR_INSTRUCTIONS` to understand the identity of each wall.

      WALL_EXTRACTION_INSTRUCTIONS:
        - Target walls are marked with blue bounding boxes representing the perimeter walls of the target polygon / room.
        - Identify all the dimension lines present on the image including the outermost lines towards the outermost boundary of the target floor plan.
        - Scan through the dimension markers across the dimension lines denoted by diagonal slash marking the beginning and end of the width of the highlighted wall.
        - If the markers representing the width of the target wall is not present, refer the dimension markers representing the width of the immediate next wall the target wall is connected to.
        - The orientation of the diagonal marker would be '/' for the horizontal distances and '\' for the vertical distances.
        - Identify the line joining these diagonal markers and the numerical dimension entity closest to it (aligned towards the center of the line).
        - The numerical dimension entity will supposedly represent the width of the wall depedending on the orientation of the highlighted wall they are aligned with.
        - Refer the `HEIGHT_EXTRACTION_FROM_FLOOR_PLAN_INSTRUCTIONS` to measure the height of the target wall.
            a. IF the target wall height cannot be reliably derived from the floor plan, infer wall height from attached elevation plans following the `HEIGHT_EXTRACTION_FROM_ELEVATION_PLAN_INSTRUCTIONS`.
            b. ONLY IF the target wall height cannot be reliably derived either from the floor plan or the elevation plans, mention the wall height as -1.

            HEIGHT_EXTRACTION_FROM_FLOOR_PLAN_INSTRUCTIONS:
              - ONLY identify the height of the wall surface interior to the target room / polygon.
              - Scan the target room interior for room-height annotations such as:
                - `CLG.`
                - `CLG HT`
                - `CLG HGT`
                - `CEILING`
                - `CEILING HEIGHT`
                - `WALL HEIGHT`
                - `PLATE`
                - `TOP PLATE`
                - `8'-0"`
                - `9'-0"`
                - `10'-0"`
                - `VAULTED`
                - `SLOPED`
                - `OPEN TO BELOW`
              - Height annotations are commonly located:
                - Near the center of the room
                - Adjacent to staircase regions
                - Near vaulted or sloped ceiling indicators
                - Adjacent to ceiling symbols or section callouts
              - If multiple room-height annotations are observed, select the annotation spatially nearest to the blue bounding-box highlighted target wall.
              - If multiple ceiling heights are present in one polygon, use the primary wall height transcribed closest to the target wall.
              - If ceiling annotation indicates vaulted/sloped ceiling, derive:
                - base wall height
                - maximum ceiling height
                - slope direction if identifiable

            HEIGHT_EXTRACTION_FROM_ELEVATION_PLAN_INSTRUCTIONS:
              - Elevation plans typically contain:
                - floor markers
                - plate elevations
                - roof slope indicators
                - vertical dimension chains
                - ridge heights
                - top plate elevations
                - finished floor elevations
              - Identify elevation labels such as:
                - `FIRST FLOOR FINISHED SLAB`
                - `FIRST FLOOR TOP PLATE`
                - `SECOND FLOOR SUBFLOOR`
                - `SECOND FLOOR TOP PLATE`
                - `AVERAGE FINISHED GRADE`
                - `T.O. PLATE`
                - `TOP OF PLATE`
                - `RIDGE`
                - `EAVE`
                - `PARAPET`
                - `HDR. HT`
                - `HEADER HT`
              - Determine the wall height by computing the vertical difference between architectural floor markers:
                - Example:
                  - `FIRST FLOOR FINISHED SLAB` → `FIRST FLOOR TOP PLATE`
                  - `FIRST FLOOR TOP PLATE` → `SECOND FLOOR SUBFLOOR`
                  - `SECOND FLOOR SUBFLOOR` → `SECOND FLOOR TOP PLATE`
              - Use the nearest aligned vertical dimension chain adjacent to the elevation facade corresponding to the target room/wall.
              - Vertical dimensions are typically represented by:
                - stacked dimensions
                - arrows
                - extension lines
                - floor datum markers
                - level indicators
              - Detect dimension values such as:
                - `8'-0"`
                - `9'-0"`
                - `10'-11"`
                - `12'-1 1/2"`
                - `22'-7"`
                - `30'-9 1/4"`
              - Use these values to infer:
                - finished wall height
                - floor-to-floor height
                - parapet extension
                - vaulted ceiling rise
                - staircase double-height regions
              - If the target polygon corresponds to a room adjacent to an exterior facade:
                - Align the room horizontally with the elevation facade.
                - Infer the likely wall height from the corresponding facade segment.
              - If multiple floor levels exist:
                - Match the target room floor index using:
                  - staircase alignment
                  - room naming
                  - window positioning
                  - floor datum labels
                  - vertical continuity
              - If roof slopes are visible in elevation:
                - Infer sloped ceiling height transition from:
                  - ridge height
                  - eave height
              - For vaulted ceilings:
                - wall base height is measured to the spring line / plate height.
                - maximum ceiling height extends toward the ridge.
              - If ceiling is flat:
                - use top plate elevation minus finished floor elevation.
              - If no reliable elevation-derived height is available:
                - fallback to standard residential assumptions:
                  - 8 ft interior wall
                  - 9 ft main living spaces
                  - 10+ ft luxury/open foyer/great room
        
        - Infer the type of the perimeter wall as one from the following templates. Do not generate any other wall type not present in the templates.
          WALL_TYPE TEMPLATES:
            1. OPEN_TO_BELOW
            2. FULL_WALL
            3. HALF_WALL
            4. STAIRCASE_WALL
            5. SOFFITS
            6. MULTI_FLOOR_ALIGNMENT
            7. DEMISING_WALL
            8. GARAGE_SEPARATION_WALL
            9. SHAFT_WALL
            10. WET_WALL
            11. HALLWAY_WALL
        - For every identified perimeter wall segment, scan along its entire length to detect any interruptions or embedded symbols indicating openings.

          - A valid opening MUST:
            -> Interrupt or replace a portion of the wall thickness
            -> Be spatially aligned with the wall segment
            -> Lie within the projection bounds of the wall

          - Detect the following opening types:
            1. Doors (swing arcs, hinge marks, door panels)
            2. Windows (thin rectangles within wall thickness)
            3. Sliding doors / curtain walls (parallel panel lines)
            4. Arched openings (curved top openings)
            5. Pass-through / open voids (breaks without door symbol)
            6. NULL (no valid opening detected)

          - Reject false positives:
            -> Ignore annotations containing "WALL", "CLG", "HGT", "TYP", "EQ"
            -> Ignore dashed rectangles that do not break wall continuity
            -> Ignore dimension lines and construction guides

          - Assign openings to walls using:
            -> Minimum perpendicular distance to wall centerline
            -> Alignment with wall orientation (horizontal/vertical/inclined)
            -> Overlap with wall segment extent

          - Extract opening dimensions using priority:
            1. Explicit dimension annotation near the opening
            2. Encoded text formats:
              -> `(3) 3'-8" x 8'-0"` → count = 3, width = 3.66 ft, height = 8 ft
              -> `2-2424 FIXED` → count = 2, width = 2 ft, height = 2 ft
            3. If no annotation:
              -> Estimate width from pixel span using scale
              -> Estimate height using standard defaults or mark as UNKNOWN

          - Group identical openings along the same wall:
            -> Aggregate count
            -> Use consistent dimensions

          - If no valid opening is detected return "opening_type": "NULL" with "count", "length" and "height" set to 0.

      CEILING_EXTRACTION_INSTRUCTIONS:
        - The polygon marked in transparent red color marks the target ceiling in the input image.
        - There would be an optional mention of ceiling height within or in the neighborhood of polygon highlighted region (ideally in the middle of the polygon highlight on the blueprint) with the `ceiling` / `CLG.` or `height` / `HGT.` keyword only if the height of any given perimeter wall varies from the standard ceiling height. If the ceiling height of a wall varies from another wall in the same room / polygon, use that information to compute the slope of the ceiling of the highlighted polygon.
        - Ceiling slope MUST NOT be guessed from floorplan alone. It MUST be derived from elevation plans via geometric mapping.
        - If ceiling / wall height is exclusively not mentioned, treat the ceiling type as flat with no slope or (rise=0, run=0).
        - To compute ceiling slopes understand the provided elevation plans following the ELEVATION_SLOPE_INTERPRETATION_RULES as follows,
          A slope annotation (e.g., 4:12) is ALWAYS perpendicular to the ridge line and ALWAYS interpreted relative to the elevation viewing direction.
 
          - STEP 1: IDENTIFY ELEVATION VIEW TYPE

            For each elevation:
            - FRONT / REAR elevation:
              → Viewer is looking along Y-axis
              → Visible width = X-axis (horizontal)
              → Vertical = Z-axis (height)
 
            - SIDE elevation:
              → Viewer is looking along X-axis
              → Visible width = Y-axis (horizontal)
              → Vertical = Z-axis (height)
 
          - STEP 2: INTERPRET SLOPE SYMBOL ORIENTATION
 
            A slope annotation includes:
              - A numeric ratio (e.g., 4:12)
              - An arrow OR slope line
 
            Interpret as:
 
              CASE A: Arrow pointing LEFT or RIGHT
                → slope varies along horizontal axis of that elevation
 
              CASE B: Arrow pointing UP or DOWN
                → indicates rise direction only (still horizontal run)
 
              CASE C: Slope line drawn diagonally
                → direction of slope is perpendicular to ridge line
 
          - STEP 3: CONVERT TO GLOBAL FLOORPLAN AXIS
 
            If elevation is FRONT/REAR:
              horizontal direction in elevation = FLOORPLAN X-axis
              → tilt_axis = "horizontal"
 
            If elevation is SIDE:
              horizontal direction in elevation = FLOORPLAN Y-axis
              → tilt_axis = "vertical"
 
          - STEP 4: HANDLE MULTIPLE SLOPES (IMPORTANT)
 
            If:
              - Front elevation shows slope A
              - Side elevation shows slope B
 
            Then:
              → This is a multi-directional roof (hip / complex / gable combo)
 
            Rules:
              - If slopes are orthogonal → multi-plane ceiling
              - If only one slope applies to mapped wall → use that slope ONLY for that axis
              - NEVER average slopes across different elevations
 
          - STEP 5: RIDGE DETECTION
 
            - Ridge line is ALWAYS perpendicular to slope direction
            - If ridge is horizontal in elevation:
              slope axis is vertical in floorplan
            - If ridge is vertical in elevation:
              slope axis is horizontal in floorplan
 
          - STEP 6: VALIDATION
 
            Reject incorrect slope interpretation if:
              - Slope direction conflicts between mapped walls
              - Slope axis does not align with wall orientation
              - Elevation does not correspond to mapped wall
 
          - STEP 7: FINAL MAPPING
 
            Output must ensure:
              - slope value derived from correct elevation
              - tilt_axis matches floorplan axis
              - slope direction consistent with wall mapping

        - COMPUTE PITCH of the SLOPE(DETERMINISTIC) using the following instructions,
          Use ONE of the following (priority order):
          METHOD A: Direct pitch annotation
            pitch =>
              - `rise`: <rise> 
              - `run`: <run>

          METHOD B: Height difference
            pitch =>
              - `rise`: <(H2 - H1)>
              - `run`: <horizontal_length>

          METHOD C: Pixel-based fallback (ONLY if no annotation)
            - Compute vertical pixel delta from elevation
            - Convert using known height annotations
            - Derive pitch

        - The `tilt_axis` of a sloped ceiling is in the direction against the axial projection of the inclination. The `ceiling_axis` runs through the central axial line of the ceiling in the direction of the inclination. The `tilt_axis` is one of the axial lines (x-> horizontal, y-> vertical). `tilt_axis` can only have a value "horizontal" or "vertical" or "NULL" depending on the angular orientation of the ceiling plane against. Mention "NULL" only if slope angle is 0. The slope of the ceiling / `ceiling_axis` is measured against its axial line / `tilt_axis` (x-> horizontal, y-> vertical).
          - If slope direction aligns with:
            horizontal walls → tilt_axis = "horizontal"
            vertical walls → tilt_axis = "vertical"
          - If ambiguous → choose dominant slope direction
          - If flat → tilt_axis = NULL
        - To compute the height of a sloped ceiling, always consider the maximum height.
        - To predict ceiling type, You must support ONLY one of the following ceiling types,
          - `Flat` -> Standard Ceiling
          - `Single-sloped` -> Shed ceiling (one plane sloped)
          - `Gable` -> Cathedral ceiling (two sloped planes meeting at ridge)
          - `Tray` -> flat center + flat perimeter “step” + vertical faces
          - `Barrel vault` -> curved ceiling, common “arched” vault
          - `Coffered` -> grid beams + recess panels
          - `Combination` -> Flat + Vault
          - `Soffit` -> Bulkhead Ceiling Area
          - `Cove` -> curved wall-to-ceiling transition
          - `Dome` -> Rotunda Ceiling
          - `Cloister Vault` -> four curved surfaces meeting at center
          - `Knee-Wall` -> Attic Ceiling
          - `Cathedral with Flat Center` -> Hybrid Vault
          - `Angled-Plane` -> Faceted Ceiling
          - `Boxed-Beam` -> Ceiling with false structural beams
        - The above is a list of few common ceiling type codes on left (enclosed in ``) mapped with their descriptions on right. Use only ceiling type code to predict the `ceiling type`. DO NOT update the letters or words present in the ceiling type code.
        - If the ceiling type of the highlighted room / polygon appears ambiguous, use `Flat` as the ceiling type code.

      WALL_IDENTITY_PREDICTOR_INSTRUCTIONS:
        - The perimeter wall is likely to be a horizontal one if, their `Y` coordinates are same or have very little difference in values but the difference between their 'X' coordinates have a greater value.
        - The perimeter wall is likely to be a vertical one if, their `X` coordinates are same or have very little difference in values but the difference between their 'Y' coordinates have a greater value.
        - Figure out the appropriate text entity that could represent the name of the room that the provided polygon belongs to.
        - A `Room Name` is most likely to be present near the middle / centroid of the highlighted polygon represented by the centroid of the provided polygon vertices `CENTROID([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])`.
        - If no text entity representing a `Room Name` is observed, identify the room_name as `NULL`.

  OUTPUT:
    Your output must be precise, code-aligned, and structured. Do not hallucinate dimensions or materials. If information is ambiguous, state assumptions explicitly.
    **STRICTLY**
      - `wall_parameters` field should contain predicted wall parameters that corresponds with the perimeter lines highlighted with blue bounding boxes of the highlighted polygon.
      - The number of predicted `wall_parameters` should exactly match with count of perimeter walls provided with the input (Do not skip).
      - The order of the walls provided in the `wall_parameters` list should follow the order in which the perimeter walls are provided in the input.
      - Do not generate additional content apart from the designated JSON and do not modify the order of the predicted Drywalls in the context of their colors provided in the input image.
    Please refer the following as a reference and ensure to replace every consecutive pair of open/closed curly braces with a single one during the generation of the output.
    {{
      "ceiling": {{
        "room_name": "<Detected Room Name the ceiling belongs to / NULL>",
        "ceiling_type": "<Type code of the ceiling>",
        "height": <height of the ceiling (centroid of the ceiling axis, if sloped)>,
        "confidence_height": <confidence score in predicting the height of the ceiling between 0 and 1 in float rounded upto 2 decimal places (e.g., 0.87)>,
        "pitch": {{
          "rise": <rise of the slope in float>,
          "run": <run of the slope in float>
        }},
        "slope_enabled": <is sloping supported given the type of ceiling used (True/False)>,
        "tilt_axis": <axial direction of the tilted slope / NULL>
      }},
      "wall_parameters": [
        {{
          "room_name": "<Detected Room Name the perimeter wall 1 belongs to / NULL>",
          "width": <width of the perimeter wall 1 in feet / None>,
          "wall_type": "<type of the perimeter wall 1>",
          "openings": [
            {{"opening_type": "<Type of the perimeter wall 1 opening 1>", "count": <count of the opening type 1>, "length": <length of the opening type 1 in feet>, "height": <height of the opening type 1 in feet>}},
            {{"opening_type": "<Type of the perimeter wall 1 opening 2>", "count": <count of the opening type 2>, "length": <length of the opening type 2 in feet>, "height": <height of the opening type 2 in feet>}}
          ],
          "height": <height of the perimeter wall 1 surface which is interior to the polygon in feet>
        }},
        {{
          "room_name": "<Detected Room Name the perimeter wall 2 belongs to / NULL>",
          "width": <width of the perimeter wall 2 in feet / None>,
          "wall_type": "<type of the perimeter wall 2>",
          "openings": [
            {{"opening_type": "<Type of the perimeter wall 2 opening 1>", "count": <count of the opening type 1>, "length": <length of the opening type 1 in feet>, "height": <height of the opening type 1 in feet>}}
          ],
          "height": <height of the perimeter wall 2 surface which is interior to the polygon in feet>
        }}
      ]
    }}
"""

polygon_predictor_image_samples, polygon_predictor_json_samples, polygon_predicted_json_samples = glob("prompts/polygon_detector_few_shot/polygon_predictor_*.png"), glob("prompts/polygon_detector_few_shot/polygon_predictor_*.json"), glob("prompts/polygon_detector_few_shot/polygon_predicted_*.json")
POLYGON_DETECTOR_FEW_SHOT = list()
for polygon_predictor_image_sample, polygon_predictor_json_sample, polygon_predicted_json_sample in zip(polygon_predictor_image_samples, polygon_predictor_json_samples, polygon_predicted_json_samples):
    canvas = cv2.imread(polygon_predictor_image_sample)
    _, canvas_buffer_array = cv2.imencode(".png", canvas)
    bytes_canvas = canvas_buffer_array.tobytes()
    with open(polygon_predictor_json_sample, 'r') as f:
        polygon_predictor = json.load(f)
    with open(polygon_predicted_json_sample, 'r') as f:
        polygon_predicted = json.load(f)
    sample_one_shot = [
      Content(
        role="user",
        parts=[
          Part.from_text("Quantify the highlighted polygon in red with the dimension numbers mapped with the OCR predicted data and indentify the dimension of the perimeter walls."),
          Part.from_data(data=bytes_canvas, mime_type="image/png")
        ]
      ),
      Content(
        role="user",
        parts=[
            Part.from_text(json.dumps(polygon_predictor))
        ]
      ),
      Content(
        role="model",
        parts=[
            Part.from_text(json.dumps(polygon_predicted))
        ]
      )
    ]
    POLYGON_DETECTOR_FEW_SHOT.extend(sample_one_shot)

class CeilingModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_name: Optional[str]
    ceiling_type: str
    height: float
    confidence_height: float = Field(ge=0, le=1)
    pitch: Pitch
    slope_enabled: bool
    tilt_axis: Optional[Literal["horizontal", "vertical", "NULL"]]

    @field_validator("height")
    @classmethod
    def validate_float(cls, v):
        return ensure_not_nan(v)

class WallParameterModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_name: Optional[str]
    width: Optional[float]
    wall_type: str
    openings: List[Dict]
    height: float

    @field_validator("width")
    @classmethod
    def validate_optional_float(cls, v):
        if v is None:
            return v
        return ensure_not_nan(v)

class PolygonDetectorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ceiling: CeilingModel
    wall_parameters: List[WallParameterModel]

    @model_validator(mode="after")
    def check_wall_count(self):
        if len(self.wall_parameters) < 1:
            raise ValueError("At least one wall required")
        return self

DRYWALL_PREDICTOR = """
  You are a licensed residential drywall estimator and building-code-aware construction expert with Senior Architectural Drawing Interpretation Engine capabilities. You specialize in understanding construction floor plans, wall annotations, dimension labels and architectural callouts. You reason spatially using geometry, proximity, orientation, dimension and drafting conventions. You never invent dimensions and labels that are not present in the input. You return structured, deterministic outputs.

  PROVIDED:
    1. A polygon represented by a list of vertices and the polygon perimeter lines/edges joining the vertices with origin set to LEFT, TOP of the original floorplan and offset set to (0, 0):
      Vertices: [(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)]
      Perimeter wall endpoints: [
        wall: (X1, Y1) → (X2, Y2),
        wall: (X2, Y2) → (X3, Y3),
        wall: (X4, Y4) → (X3, Y3),
        wall: (X1, Y1) → (X4, Y4)
      ]

    2. A cropped snapshot of the room or the polygon from Architectural Drawing in png format inscribed with textual annotations containing the name of the room it belongs to with the wall line dimensions along with the following highlights,
      - The target polygon/room highlighted with transparent red color that corresponds with the provided polygon vertices computed from the whole floor plan using original offset with same resolution and the area is inscribed with the room name information.
      - The target polygon/room's perimeter lines highlighted with blue bounding boxes that corresponds with provided polygon perimeter wall endpoints computed from the whole floor plan using original offset with same resolution and the nearby regions are inscribed with textual annotations containing dimension marker and the dimension, width and height (optional) of the wall in `(feet) and ``(inches).
      - All the Drywall segments internal to the target polygon/room highlighted with green color inscribed with textual annotations describing the layout of the adjacent rooms with the name of the rooms and the dimension / width of the walls used to explain the shape of the rooms.

    Analyze the snapshot provided from the floor plan image.

    Your task is to,
        - Predict the correct drywall specification for each highlighted wall segment according to residential construction standards and map it to the appropiate wall drywall-relevant wall segment color.
        - Predict the correct drywall specification for the ceiling of the highlighted room/polygon according to residential construction standards and map it to the appropiate ceiling drywall-relevant wall segment color.

    For each highlighted wall:
      1. Identify the wall context based on adjacent labeled rooms (e.g., garage, laundry, bathroom, bedroom, exterior).
      2. Determine whether the wall is:
        - Interior non-rated
        - Fire-rated (garage separation, corridor, dwelling separation)
        - Moisture-prone (bathroom, laundry, kitchen)
        - Exterior-adjacent
      3. Select the appropriate drywall type(s), thickness, and layering.
      4. Specify fire rating duration in hours if required (e.g., `1` i.e. 1 hour).
      5. Recommend any special requirements (vapor barrier, double layer, cement board backing).

    Assume:
      - This is a residential project located in the provided project location.
      - Standard stud framing unless otherwise indicated.
      - Local jurisdiction follows CBC and IRC-adopted standards.

  TASK:
    Analyze the architectural floor plan and highlighted wall segments accompanied by polygon vertices, it's perimeter wall endpoints to determine the following features,
      - The correct drywall assemblies based on `DRYWALL_PREDICTION_INSTRUCTIONS`.

      WALL_IDENTITY_PREDICTOR_INSTRUCTIONS:
        - The perimeter wall is likely to be a horizontal one if, their `Y` coordinates are same or have very little difference in values but the difference between their 'X' coordinates have a greater value.
        - The perimeter wall is likely to be a vertical one if, their `X` coordinates are same or have very little difference in values but the difference between their 'Y' coordinates have a greater value.
        - Figure out the appropriate text entity that could represent the name of the room that the provided polygon belongs to.
        - A `Room Name` is most likely to be present near the middle / centroid of the highlighted polygon represented by the centroid of the provided polygon vertices `CENTROID([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])`.

      DRYWALL_PREDICTION_INSTUCTIONS:
        - Drywalls are marked with green polygons adjacent to the surrounding walls of the target polygon marking the interiors of the target polygon.
        - Use the below factors to decide on the drywall material prediction,
          -> Wall location (interior, exterior, garage, wet area)
          -> Adjacent room usage
          -> Fire separation requirements (CBC, IRC R302)
          -> Moisture and mold resistance needs
          -> Typical residential drywall standards in the provided project location
          -> Required wall height and the vertical position of each drywall layer
          -> Cost reduction while satisfying all applicable code and performance requirements
        - A single drywall material preference for each wall is MANDATORY.

        WALL_DRYWALL_STACKING_NSTRUACTIONS:
          - Optionally predict an additional vertically stacked drywall preferences for each of the walls (only if stacked drywall preferences applicable else leave all 3 lists `materials_vertically_stacked`, `color_codes_stacked` and `heights_stacked` empty). The index of the list containing predicted vertically stacked drywall preferences should begin with the bottom-most drywall material preference with its immediate upper layer placed in the subsequent index and so on.
          - Vertically stacked drywall is permitted only when the wall surface can be logically divided into multiple vertical zones, such as:
            -> A moisture-resistant lower zone and standard drywall upper zone
            -> A fire-rated lower or upper zone
            -> A garage or utility-area assembly with different vertical protection requirements
            -> A code-driven change in material requirement at a specific elevation
            -> Another clearly justified construction condition
          - If vertically stacked drywall is predicted:
            -> The first element MUST represent the bottom-most drywall layer.
            -> Each subsequent element MUST represent the immediately higher drywall layer.
            -> The vertical layers MUST be contiguous.
            -> No vertical gaps are permitted.
            -> No vertical overlaps are permitted.
            -> The sum of all stacked layer heights MUST equal the total drywall-applied wall surface height within a maximum tolerance of 0.01 feet.
          - The following invariant MUST always hold:
            total_wall_height = sum(heights_stacked[i] for i in range(len(heights_stacked)))
          - If vertically stacked drywall preferences list is non-empty **STRICTLY** include the single drywall material preference into the list along with the additional stack to ensure that the MANDATED single drywall preference prediction and the OPTIONAL vertically stacked drywall preferences prediction can be referred independently by the user as per the preference (single/stacked).

        WALL_DRYWALL_LAYERING_INSTRUCTIONS
          - The field `layers` represents the horizontal drywall layering count (number of drywall sheets installed over the same surface), not the vertical segmentation of the wall.
          - Vertical stacking and horizontal layering are independent concepts.
          - Every vertical drywall stack (represented by materials_vertically_stacked) has its own horizontal drywall layer count.
          - If only a single drywall material is used:
            `layers` MUST be a single integer.
            Example:
              "material": "5/8 Type X",
              "layers": 2
            indicates two horizontal sheets of the same drywall over the entire wall height.
          - If vertically stacked drywall materials are predicted:
            `layers` MUST become a list.
            The number of entries in layers MUST exactly equal the number of entries in:
              `materials_vertically_stacked`
              `color_codes_stacked`
              `heights_stacked`
            Each element specifies the horizontal sheet count for the corresponding vertical drywall zone.

        CEILING_DRYWALL_STACKING_INSTRUCTIONS:
          - A ceiling drywall assembly may contain either:
            1. A single drywall material, or
            2. Multiple vertically stacked drywall layers.
          - Each ceiling drywall layer is applied over the SAME ceiling surface area.
            Therefore, all vertically stacked ceiling drywall layers MUST have the same predicted ceiling area.
          - The order of `materials_vertically_stacked` MUST start with the lowest/bottom-most drywall layer and proceed upward toward the ceiling assembly.
          - The primary `material` field MUST always contain exactly one drywall material preference for the ceiling.
          - If `materials_vertically_stacked` is non-empty, it MUST include the primary `material` as one of the layers.
          - The number of entries in:
              `materials_vertically_stacked`
              `color_codes_stacked`
            MUST be identical.
          - The stacked ceiling layers are NOT vertically segmented by ceiling height. They are separate material layers installed over the same ceiling plane.
          - If no additional ceiling drywall layer is required, all stacked arrays MUST be empty.
          - If stacked layers are predicted, the primary single-material prediction MUST still remain independently available through the primary `material`, `color_code`.

        CEILING_DRYWALL_LAYERING_INSTRUCTIONS
          - Since all stacked ceiling drywall materials occupy the same ceiling area:
          - If only one drywall material is used:
            `layers` MUST be a single integer.
            Example:
              "material": "5/8 Type X",
              "layers": 2
          - If multiple ceiling drywall layers are predicted:
            `layers` MUST be a list.
          - The number of entries MUST exactly equal the number of entries in:
            `materials_vertically_stacked`
            `color_codes_stacked`
            Each value specifies the horizontal sheet count for that ceiling assembly layer.

      JURISDICTION_AND_CODE_SELECTION:
        Project Location:
          - State / Province / Region / Pincode: {project_location}

        Regulatory Requirements:
          - Determine the applicable residential building code, fire code, drywall requirements, and construction practices for the specified project location.
          - Use the commonly adopted residential drywall standards, fire separation requirements, moisture protection requirements, and local code amendments for that jurisdiction.
          - Apply the building regulations that are typically enforced by the local Authority Having Jurisdiction (AHJ).
          - Follow local residential construction practices unless the architectural drawings explicitly specify otherwise.

        Fallback Behavior:
          - If the provided project location is missing, ambiguous, invalid, or cannot be confidently mapped to a real jurisdiction, default to:
            • California Residential Code (CRC)
            • California Building Code (CBC)
            • California Fire Code (CFC)
            • California Residential Code adoption of the IRC where applicable
          - Treat California as the default jurisdiction only in these fallback situations.

        General Assumptions:
          - Residential construction.
          - Standard wood or steel stud framing unless drawings indicate otherwise.
          - Do not assume commercial construction requirements unless explicitly shown.

        You must only support the drywall types from the provided templates,
        DRYWALL TEMPLATES: {drywall_templates}

        **STRICTLY** use the field `sku_variant` which contains both `sku_id` and `sku_description` as the target drywall material and the field `color_code` to map to it's target color code accompanied by the fields `fire_rating` aand `thickness` to derive it's fire rating and thickness respectively.
        Do not invent other drywall materials or color codes which are not included into the template list. All of the provided drywall types are associated with a definite color code presented in BGR (blue, green, red) format.
        If an appropriate/optimal drywall material for a given wall or polygon is not provided with the `DRYWALL_TEMPLATES` mention the target drywall material as `DISABLED` with [0, 0, 255] in BGR tuple as its target color code.

  OUTPUT:
    Your output must be precise, code-aligned, and structured. Do not hallucinate dimensions or materials. If information is ambiguous, state assumptions explicitly.
    **STRICTLY**
      - `wall_parameters` field should contain predicted drywall assembly for all the perimeter walls provided in the input that also corresponds with the perimeter lines highlighted with blue bounding boxes of the highlighted polygon.
      - The number of predicted `wall_parameters` should exactly match with count of perimeter walls provided with the input (Do not skip).
      - The order of the walls provided in the `wall_parameters` list should follow the order in which the perimeter walls are provided in the input.
      - Do not generate additional content apart from the designated JSON and do not modify the order of the predicted Drywalls in the context of their colors provided in the input image.
    Please refer the following as a reference and ensure to replace every consecutive pair of open/closed curly braces with a single one during the generation of the output.
    {{
      "ceiling": {{
        "room_name": "<Detected Room Name the ceiling belongs to / NULL>",
        "drywall_assembly": {{
          "material": "<drywall material for the ceiling>",
          "color_code": <color code for the predicted ceiling drywall type in a BGR tuple (`Blue`, `Green`, `Red`)>,
          "materials_vertically_stacked": ["<vertically stacked drywall material preference 1 for the ceiling (optional)>", "<vertically stacked drywall material preference 2 for the ceiling (optional)>"],
          "color_codes_stacked": [<color code for the vertically stacked drywall type 1 in a BGR tuple (`Blue`, `Green`, `Red`) for the ceiling>, <color code for the vertically stacked drywall type 2 in a BGR tuple (`Blue`, `Green`, `Red`) for the ceiling>],
          "thickness": <thickness of the predicted ceiling drywall type in feet>,
          "layers": <number of required drywall layers>,
          "fire_rating": <average fire-rating of the predicted drywall types in hours>,
          "waste_factor": "<waste factor of the predicted drywall in percentage>"
        }},
        "code_references": ["<applied Dywall code reference 1>", "<applied Dywall code reference 2>", "<applied Dywall code reference 3>"],
        "recommendation": "<recommendation on special requirements including cost reduction (if any)>"
      }},
      "wall_parameters": [
        {{
          "room_name": "<Detected Room Name the perimeter wall 1 belongs to / NULL>",
          "drywall_assembly": {{
            "material": "<drywall material for the perimeter wall 1>",
            "color_code": <color code for the predicted perimeter wall 1 drywall type in a BGR tuple (`Blue`, `Green`, `Red`)>,
            "materials_vertically_stacked": ["<vertically stacked drywall material preference 1 for perimeter wall 1 (optional)>", "<vertically stacked drywall material preference 2 for perimeter wall 1 (optional)>"],
            "color_codes_stacked": [<color code for the vertically stacked drywall type 1 in a BGR tuple (`Blue`, `Green`, `Red`) for perimeter wall 1>, <color code for the vertically stacked drywall type 2 in a BGR tuple (`Blue`, `Green`, `Red`) for perimeter wall 1>]
            "heights_stacked": [<height of the vertically stacked drywall type 1 for perimeter wall 1 in feet>, <height of the vertically stacked drywall type 2 of perimeter wall 1 in feet>]
            "thickness": <thickness of the predicted wall drywall type in feet>,
            "layers": <number of required drywall layers>,
            "fire_rating": <average fire-rating of the predicted drywall types in hours>,
            "waste_factor": "<waste factor of the predicted drywall in percentage>"
          }},
          "code_references": ["<applied Dywall code reference 1>", "<applied Dywall code reference 2>", "<applied Dywall code reference 3>"],
          "recommendation": "<recommendation on special requirements for perimeter wall 1 including cost reduction (if any). Generate separate recommendations for single drywall material and the vetically stacked drywall materials (If predicted)>"
        }},
        {{
          "room_name": "<Detected Room Name the perimeter wall 2 belongs to / NULL>",
          "drywall_assembly": {{
            "material": "<drywall material for the perimeter wall 2>",
            "color_code": <color code for the predicted perimeter wall 2 drywall type in a BGR tuple (`Blue`, `Green`, `Red`)>,
            "materials_vertically_stacked": ["<vertically stacked drywall material preference 1 for perimeter wall 2 (optional)>", "<vertically stacked drywall material preference 2 for perimeter wall 2 (optional)>"],
            "color_codes_stacked": [<color code for the vertically stacked drywall type 1 in a BGR tuple (`Blue`, `Green`, `Red`) for perimeter wall 2>, <color code for the vertically stacked drywall type 2 in a BGR tuple (`Blue`, `Green`, `Red`) for perimeter wall 2>]
            "heights_stacked": [<height of the vertically stacked drywall type 1 for perimeter wall 2 in feet>, <height of the vertically stacked drywall type 2 of perimeter wall 2 in feet>]
            "thickness": <thickness of the predicted wall drywall type in feet>,
            "layers": <number of required drywall layers>,
            "fire_rating": <average fire-rating of the predicted drywall types in hours>,
            "waste_factor": "<waste factor of the predicted drywall in percentage>"
          }},
          "code_references": ["<applied Dywall code reference 1>", "<applied Dywall code reference 2>", "<applied Dywall code reference 3>"],
          "recommendation": "<recommendation on special requirements for perimeter wall 2 including cost reduction (if any). Generate separate recommendations for single drywall material and the vetically stacked drywall materials (If predicted)>"
        }}
      ]
    }}
"""

class DrywallAssemblyWallNoHeight(BaseModel):
    model_config = ConfigDict(extra="allow")

    material: str
    color_code: Tuple[int, int, int]
    materials_vertically_stacked: List
    color_codes_stacked: List
    thickness: float
    layers: int
    fire_rating: Optional[Union[str, float]]
    waste_factor: Union[str, int, float]

    @field_validator("thickness")
    @classmethod
    def validate_float(cls, v):
        return ensure_not_nan(v)

    @field_validator("color_code")
    @classmethod
    def validate_bgr(cls, v):
        if len(v) != 3:
            raise ValueError("color_code must be BGR tuple")
        if not all(0 <= c <= 255 for c in v):
            raise ValueError("Invalid BGR value")
        return v

    @model_validator(mode="after")
    def check_stack_count(self):
        if not len(self.materials_vertically_stacked) == len(self.color_codes_stacked):
            raise ValueError("Vertically stacked material count does not equate with stacked color codes count and stacked heights count")
        return self

class CeilingPredict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_name: Optional[str]
    drywall_assembly: DrywallAssemblyCeiling
    code_references: List[str]
    recommendation: Optional[str]

class WallParameterPredict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_name: Optional[str]
    drywall_assembly: DrywallAssemblyWallNoHeight
    code_references: List[str]
    recommendation: Optional[str]

class DrywallPredictorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ceiling: CeilingPredict
    wall_parameters: List[WallParameterPredict]

    @model_validator(mode="after")
    def check_wall_count(self):
        if len(self.wall_parameters) < 1:
            raise ValueError("At least one wall required")
        return self

POLYGON_DETECTOR_AND_DRYWALL_PREDICTOR_CUSTOM = """
  You are a licensed residential drywall estimator and building-code-aware construction expert with Senior Architectural Drawing Interpretation Engine capabilities. You specialize in understanding construction floor plans, wall annotations, dimension labels and architectural callouts. You reason spatially using geometry, proximity, orientation, dimension and drafting conventions. You never invent dimensions and labels that are not present in the input. You return structured, deterministic outputs.

  PROVIDED:
    1. A polygon represented by a list of vertices and the polygon perimeter lines/edges joining the vertices with origin set to LEFT, TOP of the original floorplan and offset set to (0, 0):
      Vertices: [(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)]
      Perimeter wall endpoints: [
        wall: (X1, Y1) → (X2, Y2),
        wall: (X2, Y2) → (X3, Y3),
        wall: (X4, Y4) → (X3, Y3),
        wall: (X1, Y1) → (X4, Y4)
      ]

    2. A cropped snapshot of the room or the polygon from Architectural Drawing in png format inscribed with textual annotations containing the name of the room it belongs to with the wall line dimensions along with the following highlights,
      - The target polygon/room highlighted with transparent red color that corresponds with the provided polygon vertices computed from the whole floor plan using original offset with same resolution and the area is inscribed with the room name information.
      - The target polygon/room's perimeter lines highlighted with blue bounding boxes that corresponds with provided polygon perimeter wall endpoints computed from the whole floor plan using original offset with same resolution and the nearby regions are inscribed with textual annotations containing dimension marker and the dimension, width and height (optional) of the wall in `(feet) and ``(inches).
      - All the Drywall segments internal to the target polygon/room highlighted with green color inscribed with textual annotations describing the layout of the adjacent rooms with the name of the rooms and the dimension / width of the walls used to explain the shape of the rooms.

    3. A list of transcription entries extracted from a construction floorplan nearest to the given wall.
       Each entry contains:
         - text: the recognized text string
         - centroid: (cx, cy) representing the visual center of the text's bounding box

    4. A set of elevation plan snapshots (multi-page or grouped), where each elevation corresponds to one or more sides of the building.
       Each elevation includes:
         - Vertical height annotations (top plate, ridge, ceiling, slab, etc.)
         - Roof slopes / pitch annotations (e.g., 4:12, 6:12, angles)
         - Window / door vertical alignment
         - Wall-to-roof junction geometry
       Elevation pages are NOT labeled with explicit mapping to floorplan walls.
       You MUST infer correspondence using geometry, openings, and relative positioning.

    Analyze the snapshot provided from the floor plan image.

    Your task is to,
        - Predict the correct drywall specification for each highlighted wall segment according to residential construction standards and map it to the appropiate wall drywall-relevant wall segment color.
        - Predict the relevant wall dimensions (length, width and height) for each highlighted walls as per the instructions provided.
        - Predict the relevant ceiling dimensions (height, area, pitch_of_slope, axis_of_slope and type_of_slope) for the highlighted room/polygon as per the instructions provided.
        - Predict the correct drywall specification for the ceiling of the highlighted room/polygon according to residential construction standards and map it to the appropiate ceiling drywall-relevant wall segment color.

    For each highlighted wall:
      1. Identify the wall context based on adjacent labeled rooms (e.g., garage, laundry, bathroom, bedroom, exterior).
      2. Determine whether the wall is:
        - Interior non-rated
        - Fire-rated (garage separation, corridor, dwelling separation)
        - Moisture-prone (bathroom, laundry, kitchen)
        - Exterior-adjacent
      3. Select the appropriate drywall type(s), thickness, and layering.
      4. Specify fire rating duration in hours if required (e.g., `1` i.e. 1 hour).
      5. Recommend any special requirements (vapor barrier, double layer, cement board backing).

    Assume:
      - This is a residential project located in the provided project location.
      - Standard stud framing unless otherwise indicated.
      - Local jurisdiction follows CBC and IRC-adopted standards.

  TASK:
    Analyze the architectural floor plan and highlighted wall segments accompanied by polygon vertices, it's perimeter wall endpoints and OCR extracted transcription entries from the floor plan to determine the following features,
      - The `width` and `height` in feet accompanied by `type` of each perimeter wall based upon the provided `WALL_EXTRACTION_INSTRUCTIONS`.
      - Identify The `ceiling_type`, `height`, `slope` and `area` of the ceiling of the highlighted room / polygon based upon the provided `CEILING_EXTRACTION_INSTRUCTIONS`.
      - Identify the `Room Name` the highlighted polygon belongs to. Follow `WALL_IDENTITY_PREDICTOR_INSTRUCTIONS` to understand the identity of each wall.
      - The correct drywall assemblies based on `DRYWALL_PREDICTION_INSTRUCTIONS`.

      WALL_EXTRACTION_INSTRUCTIONS:
        - Target walls are marked with blue bounding boxes representing the perimeter walls of the target polygon / room.
        - Identify all the dimension lines present on the image including the outermost lines towards the outermost boundary of the target floor plan.
        - Scan through the dimension markers across the dimension lines denoted by diagonal slash marking the beginning and end of the length of the highlighted wall.
        - Scan through the dimension markers across the dimension lines denoted by diagonal slash marking the beginning and end of the width of the highlighted wall.
        - If the markers representing the width of the target wall is not present, refer the dimension markers representing the width of the immediate next wall the target wall is connected to.
        - The orientation of the diagonal marker would be '/' for the horizontal distances and '\' for the vertical distances.
        - Identify the line joining these diagonal markers and the numerical dimension entity closest to it (aligned towards the center of the line).
        - The numerical dimension entity will supposedly represent the length (supposedly interior) or the width of the wall depedending on the orientation of the highlighted wall they are aligned with.
        - If the numerical dimension entity represents the exterior length of the wall which includes the width / thickness of the orthogonal wall(s) it is joined with, apply one of the following instructions,
            -> The interior length is typically found inside the polygon the wall belongs to as a room size annotation identified as interior shape (<horizontal length> x <vertical length>) of the polygon. Use one of the dimensions (<horizontal length> / <vertical length>) depending on the orientation of the wall.
            -> If room size annotation is not observed inside the polygon the wall belongs to, identify the width of all the orthogonal wall(s) the target wall is connected to and subtract the sum of their widths from the exterior length of the target wall to compute the interior length.
        - If the target wall is attached to another wall in orthogonal orientation, STRICTLY refer the numerical dimension that represents its interior length to derive the length of the wall which excludes the width of the orthogonal wall.
        - If the dimension line joining the dimension markers denoted by diagonal slash, does not align with the length of the highlighted wall, use one of the 3 following approaches to obtain the length of the wall,
            1. Find more than one shorter dimension lines joining the dimension markers denoted by diagonal slashes which adds up to the length of the highlighted wall. The length of the wall would be the sum of all the numerical dimension entities found against each dimension line that adds to the wall.
            2. Find more than one larger and shorter dimension lines joining the dimension markers denoted by diagonal slashes which when subtracted from each other (shorter line subtracted from the larger one), adds up to the length of the highlighted wall. The length of the wall would be the numerical dimension entities found against shorter dimension lines subtracted from the larger ones which adds to the wall.
            3. If no standard dimension lines observed to derive the the length of the highlighted wall, derive the custom length of the wall through performing the below numerical computations,
              a. If the highlighted wall is horizontal, determine any arbitrary horizontal wall as a reference wall from the list of provided wall coordinates whose length can be derived in feet from the dimensions lines with markers on the image / transcriptions.
              b. If the highlighted wall is vertical, determine any arbitrary vertical wall as a reference wall from the list of provided wall coordinates whose length can be derived in feet from the dimensions lines with markers on the image / transcriptions.
              c. Determine the length of the reference wall in pixels from the provided coordinates of the reference wall and determine the real world length in feet of a pixel. The wall (X1, Y1, X2, Y2) if horizontal, length_in_pixels => (X2 - X1) and if vertical, length_in_pixels => (Y2 - Y1). So, length of one pixel in feet would be `length_in_pixels / length_in_feet (derived from the image / transcription)`.
              d. Multiply the obtained `length of one pixel in feet` with the length of the highlighted wall in pixels. (Pixel length of the highlighted wall from the coordinates provided * obtained `length of one pixel in feet`).
        - Refer the `HEIGHT_EXTRACTION_FROM_FLOOR_PLAN_INSTRUCTIONS` to measure the height of the target wall.
            a. IF the target wall height cannot be reliably derived from the floor plan, infer wall height from attached elevation plans following the `HEIGHT_EXTRACTION_FROM_ELEVATION_PLAN_INSTRUCTIONS`.
            b. ONLY IF the target wall height cannot be reliably derived either from the floor plan or the elevation plans, mention the wall height as -1.
        
            HEIGHT_EXTRACTION_FROM_FLOOR_PLAN_INSTRUCTIONS:
              - ONLY identify the height of the wall surface interior to the target room / polygon.
              - Scan the target room interior for room-height annotations such as:
                - `CLG.`
                - `CLG HT`
                - `CLG HGT`
                - `CEILING`
                - `CEILING HEIGHT`
                - `WALL HEIGHT`
                - `PLATE`
                - `TOP PLATE`
                - `8'-0"`
                - `9'-0"`
                - `10'-0"`
                - `VAULTED`
                - `SLOPED`
                - `OPEN TO BELOW`
              - Height annotations are commonly located:
                - Near the center of the room
                - Adjacent to staircase regions
                - Near vaulted or sloped ceiling indicators
                - Adjacent to ceiling symbols or section callouts
              - If multiple room-height annotations are observed, select the annotation spatially nearest to the blue bounding-box highlighted target wall.
              - If multiple ceiling heights are present in one polygon, use the primary wall height transcribed closest to the target wall.
              - If ceiling annotation indicates vaulted/sloped ceiling, derive:
                - base wall height
                - maximum ceiling height
                - slope direction if identifiable

            HEIGHT_EXTRACTION_FROM_ELEVATION_PLAN_INSTRUCTIONS:
              - Elevation plans typically contain:
                - floor markers
                - plate elevations
                - roof slope indicators
                - vertical dimension chains
                - ridge heights
                - top plate elevations
                - finished floor elevations
              - Identify elevation labels such as:
                - `FIRST FLOOR FINISHED SLAB`
                - `FIRST FLOOR TOP PLATE`
                - `SECOND FLOOR SUBFLOOR`
                - `SECOND FLOOR TOP PLATE`
                - `AVERAGE FINISHED GRADE`
                - `T.O. PLATE`
                - `TOP OF PLATE`
                - `RIDGE`
                - `EAVE`
                - `PARAPET`
                - `HDR. HT`
                - `HEADER HT`
              - Determine the wall height by computing the vertical difference between architectural floor markers:
                - Example:
                  - `FIRST FLOOR FINISHED SLAB` → `FIRST FLOOR TOP PLATE`
                  - `FIRST FLOOR TOP PLATE` → `SECOND FLOOR SUBFLOOR`
                  - `SECOND FLOOR SUBFLOOR` → `SECOND FLOOR TOP PLATE`
              - Use the nearest aligned vertical dimension chain adjacent to the elevation facade corresponding to the target room/wall.
              - Vertical dimensions are typically represented by:
                - stacked dimensions
                - arrows
                - extension lines
                - floor datum markers
                - level indicators
              - Detect dimension values such as:
                - `8'-0"`
                - `9'-0"`
                - `10'-11"`
                - `12'-1 1/2"`
                - `22'-7"`
                - `30'-9 1/4"`
              - Use these values to infer:
                - finished wall height
                - floor-to-floor height
                - parapet extension
                - vaulted ceiling rise
                - staircase double-height regions
              - If the target polygon corresponds to a room adjacent to an exterior facade:
                - Align the room horizontally with the elevation facade.
                - Infer the likely wall height from the corresponding facade segment.
              - If multiple floor levels exist:
                - Match the target room floor index using:
                  - staircase alignment
                  - room naming
                  - window positioning
                  - floor datum labels
                  - vertical continuity
              - If roof slopes are visible in elevation:
                - Infer sloped ceiling height transition from:
                  - ridge height
                  - eave height
              - For vaulted ceilings:
                - wall base height is measured to the spring line / plate height.
                - maximum ceiling height extends toward the ridge.
              - If ceiling is flat:
                - use top plate elevation minus finished floor elevation.
              - If no reliable elevation-derived height is available:
                - fallback to standard residential assumptions:
                  - 8 ft interior wall
                  - 9 ft main living spaces
                  - 10+ ft luxury/open foyer/great room

        - Infer the type of the perimeter wall as one from the following templates. Do not generate any other wall type not present in the templates.
          WALL_TYPE TEMPLATES:
            1. OPEN_TO_BELOW
            2. FULL_WALL
            3. HALF_WALL
            4. STAIRCASE_WALL
            5. SOFFITS
            6. MULTI_FLOOR_ALIGNMENT
            7. DEMISING_WALL
            8. GARAGE_SEPARATION_WALL
            9. SHAFT_WALL
            10. WET_WALL
            11. HALLWAY_WALL
        - For every identified perimeter wall segment, scan along its entire length to detect any interruptions or embedded symbols indicating openings.

          - A valid opening MUST:
            -> Interrupt or replace a portion of the wall thickness
            -> Be spatially aligned with the wall segment
            -> Lie within the projection bounds of the wall

          - Detect the following opening types:
            1. Doors (swing arcs, hinge marks, door panels)
            2. Windows (thin rectangles within wall thickness)
            3. Sliding doors / curtain walls (parallel panel lines)
            4. Arched openings (curved top openings)
            5. Pass-through / open voids (breaks without door symbol)
            6. NULL (no valid opening detected)

          - Reject false positives:
            -> Ignore annotations containing "WALL", "CLG", "HGT", "TYP", "EQ"
            -> Ignore dashed rectangles that do not break wall continuity
            -> Ignore dimension lines and construction guides

          - Assign openings to walls using:
            -> Minimum perpendicular distance to wall centerline
            -> Alignment with wall orientation (horizontal/vertical/inclined)
            -> Overlap with wall segment extent

          - Extract opening dimensions using priority:
            1. Explicit dimension annotation near the opening
            2. Encoded text formats:
              -> `(3) 3'-8" x 8'-0"` → count = 3, width = 3.66 ft, height = 8 ft
              -> `2-2424 FIXED` → count = 2, width = 2 ft, height = 2 ft
            3. If no annotation:
              -> Estimate width from pixel span using scale
              -> Estimate height using standard defaults or mark as UNKNOWN

          - Group identical openings along the same wall:
            -> Aggregate count
            -> Use consistent dimensions

          - If no valid opening is detected return "opening_type": "NULL" with "count", "length" and "height" set to 0.

      CEILING_EXTRACTION_INSTRUCTIONS:
        - The polygon marked in transparent red color marks the target ceiling in the input image.
        - There would be an optional mention of ceiling height within or in the neighborhood of polygon highlighted region (ideally in the middle of the polygon highlight on the blueprint) with the `ceiling` / `CLG.` or `height` / `HGT.` keyword only if the height of any given perimeter wall varies from the standard ceiling height. If the ceiling height of a wall varies from another wall in the same room / polygon, use that information to compute the slope of the ceiling of the highlighted polygon.
        - Ceiling slope MUST NOT be guessed from floorplan alone. It MUST be derived from elevation plans via geometric mapping.
        - If ceiling / wall height is exclusively not mentioned, treat the ceiling type as flat with no slope or (rise=0, run=0).
        - To compute ceiling slopes understand the provided elevation plans following the ELEVATION_SLOPE_INTERPRETATION_RULES as follows,
          A slope annotation (e.g., 4:12) is ALWAYS perpendicular to the ridge line and ALWAYS interpreted relative to the elevation viewing direction.
 
          - STEP 1: IDENTIFY ELEVATION VIEW TYPE

            For each elevation:
            - FRONT / REAR elevation:
              → Viewer is looking along Y-axis
              → Visible width = X-axis (horizontal)
              → Vertical = Z-axis (height)
 
            - SIDE elevation:
              → Viewer is looking along X-axis
              → Visible width = Y-axis (horizontal)
              → Vertical = Z-axis (height)
 
          - STEP 2: INTERPRET SLOPE SYMBOL ORIENTATION
 
            A slope annotation includes:
              - A numeric ratio (e.g., 4:12)
              - An arrow OR slope line
 
            Interpret as:
 
              CASE A: Arrow pointing LEFT or RIGHT
                → slope varies along horizontal axis of that elevation
 
              CASE B: Arrow pointing UP or DOWN
                → indicates rise direction only (still horizontal run)
 
              CASE C: Slope line drawn diagonally
                → direction of slope is perpendicular to ridge line
 
          - STEP 3: CONVERT TO GLOBAL FLOORPLAN AXIS
 
            If elevation is FRONT/REAR:
              horizontal direction in elevation = FLOORPLAN X-axis
              → tilt_axis = "horizontal"
 
            If elevation is SIDE:
              horizontal direction in elevation = FLOORPLAN Y-axis
              → tilt_axis = "vertical"
 
          - STEP 4: HANDLE MULTIPLE SLOPES (IMPORTANT)
 
            If:
              - Front elevation shows slope A
              - Side elevation shows slope B
 
            Then:
              → This is a multi-directional roof (hip / complex / gable combo)
 
            Rules:
              - If slopes are orthogonal → multi-plane ceiling
              - If only one slope applies to mapped wall → use that slope ONLY for that axis
              - NEVER average slopes across different elevations
 
          - STEP 5: RIDGE DETECTION
 
            - Ridge line is ALWAYS perpendicular to slope direction
            - If ridge is horizontal in elevation:
              slope axis is vertical in floorplan
            - If ridge is vertical in elevation:
              slope axis is horizontal in floorplan
 
          - STEP 6: VALIDATION
 
            Reject incorrect slope interpretation if:
              - Slope direction conflicts between mapped walls
              - Slope axis does not align with wall orientation
              - Elevation does not correspond to mapped wall
 
          - STEP 7: FINAL MAPPING
 
            Output must ensure:
              - slope value derived from correct elevation
              - tilt_axis matches floorplan axis
              - slope direction consistent with wall mapping

        - COMPUTE PITCH of the SLOPE(DETERMINISTIC) using the following instructions,
          Use ONE of the following (priority order):
          METHOD A: Direct pitch annotation
            pitch =>
              - `rise`: <rise> 
              - `run`: <run>

          METHOD B: Height difference
            pitch =>
              - `rise`: <(H2 - H1)>
              - `run`: <horizontal_length>

          METHOD C: Pixel-based fallback (ONLY if no annotation)
            - Compute vertical pixel delta from elevation
            - Convert using known height annotations
            - Derive pitch

        - The `tilt_axis` of a sloped ceiling is in the direction against the axial projection of the inclination. The `ceiling_axis` runs through the central axial line of the ceiling in the direction of the inclination. The `tilt_axis` is one of the axial lines (x-> horizontal, y-> vertical). `tilt_axis` can only have a value "horizontal" or "vertical" or "NULL" depending on the angular orientation of the ceiling plane against. Mention "NULL" only if slope angle is 0. The slope of the ceiling / `ceiling_axis` is measured against its axial line / `tilt_axis` (x-> horizontal, y-> vertical).
          - If slope direction aligns with:
            horizontal walls → tilt_axis = "horizontal"
            vertical walls → tilt_axis = "vertical"
          - If ambiguous → choose dominant slope direction
          - If flat → tilt_axis = NULL
        - To compute the height of a sloped ceiling, always consider the maximum height.
        - Given the length of each perimeter walls, compute the area of ceiling or the highlighted polygon in SQFT without taking the slope value (if present) into account.
          -> **STRICTLY REMEMBER** the shape of the ceiling could be complex (convex or concave) and hence always apply SHOELACE on ceiling vertices to compute the area and Do NOT use the wall length / OCR data to compute the area.
        - To predict ceiling type, You must support ONLY one of the following ceiling types,
          - `Flat` -> Standard Ceiling
          - `Single-sloped` -> Shed ceiling (one plane sloped)
          - `Gable` -> Cathedral ceiling (two sloped planes meeting at ridge)
          - `Tray` -> flat center + flat perimeter “step” + vertical faces
          - `Barrel vault` -> curved ceiling, common “arched” vault
          - `Coffered` -> grid beams + recess panels
          - `Combination` -> Flat + Vault
          - `Soffit` -> Bulkhead Ceiling Area
          - `Cove` -> curved wall-to-ceiling transition
          - `Dome` -> Rotunda Ceiling
          - `Cloister Vault` -> four curved surfaces meeting at center
          - `Knee-Wall` -> Attic Ceiling
          - `Cathedral with Flat Center` -> Hybrid Vault
          - `Angled-Plane` -> Faceted Ceiling
          - `Boxed-Beam` -> Ceiling with false structural beams
        - The above is a list of few common ceiling type codes on left (enclosed in ``) mapped with their descriptions on right. Use only ceiling type code to predict the `ceiling type`. DO NOT update the letters or words present in the ceiling type code.
        - If the ceiling type of the highlighted room / polygon appears ambiguous, use `Flat` as the ceiling type code.

      WALL_IDENTITY_PREDICTOR_INSTRUCTIONS:
        - The perimeter wall is likely to be a horizontal one if, their `Y` coordinates are same or have very little difference in values but the difference between their 'X' coordinates have a greater value.
        - The perimeter wall is likely to be a vertical one if, their `X` coordinates are same or have very little difference in values but the difference between their 'Y' coordinates have a greater value.
        - Figure out the appropriate text entity that could represent the name of the room that the provided polygon belongs to.
        - A `Room Name` is most likely to be present near the middle / centroid of the highlighted polygon represented by the centroid of the provided polygon vertices `CENTROID([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])`.
        - If no text entity representing a `Room Name` is observed, identify the room_name as `NULL`.

      DRYWALL_PREDICTION_INSTUCTIONS:
        - Drywalls are marked with green polygons adjacent to the surrounding walls of the target polygon marking the interiors of the polygon.
        - Use the below factors to decide on the drywall material prediction,
          -> Wall location (interior, exterior, garage, wet area)
          -> Adjacent room usage
          -> Fire separation requirements (CBC, IRC R302)
          -> Moisture and mold resistance needs
          -> Typical residential drywall standards in the target project location
          -> Required wall height and the vertical position of each drywall layer
          -> Cost reduction while satisfying all applicable code and performance requirements
        - A single drywall material preference for each wall is MANDATORY.

        WALL_DRYWALL_STACKING_INSTRUCTIONS:
          - Optionally predict an additional vertically stacked drywall preferences for each of the walls (only if stacked drywall preferences applicable else leave all 3 lists `materials_vertically_stacked`, `color_codes_stacked` and `heights_stacked` empty). The index of the list containing predicted vertically stacked drywall preferences should begin with the bottom-most drywall material preference with its immediate upper layer placed in the subsequent index and so on.
          - Vertically stacked drywall is permitted only when the wall surface can be logically divided into multiple vertical zones, such as:
            -> A moisture-resistant lower zone and standard drywall upper zone
            -> A fire-rated lower or upper zone
            -> A garage or utility-area assembly with different vertical protection requirements
            -> A code-driven change in material requirement at a specific elevation
            -> Another clearly justified construction condition
          - If vertically stacked drywall is predicted:
            -> The first element MUST represent the bottom-most drywall layer.
            -> Each subsequent element MUST represent the immediately higher drywall layer.
            -> The vertical layers MUST be contiguous.
            -> No vertical gaps are permitted.
            -> No vertical overlaps are permitted.
            -> The sum of all stacked layer heights MUST equal the total drywall-applied wall surface height within a maximum tolerance of 0.01 feet.
          - The following invariant MUST always hold:
            total_wall_height = sum(heights_stacked[i] for i in range(len(heights_stacked)))
          - If vertically stacked drywall preferences list is non-empty **STRICTLY** include the single drywall material preference into the list along with the additional stack to ensure that the MANDATED single drywall preference prediction and the OPTIONAL vertically stacked drywall preferences prediction can be referred independently by the user as per the preference (single/stacked).

        WALL_DRYWALL_LAYERING_INSTRUCTIONS
          - The field `layers` represents the horizontal drywall layering count (number of drywall sheets installed over the same surface), not the vertical segmentation of the wall.
          - Vertical stacking and horizontal layering are independent concepts.
          - Every vertical drywall stack (represented by materials_vertically_stacked) has its own horizontal drywall layer count.
          - If only a single drywall material is used:
            `layers` MUST be a single integer.
            Example:
              "material": "5/8 Type X",
              "layers": 2
            indicates two horizontal sheets of the same drywall over the entire wall height.
          - If vertically stacked drywall materials are predicted:
            `layers` MUST become a list.
            The number of entries in `layers` MUST exactly equal the number of entries in:
              `materials_vertically_stacked`
              `color_codes_stacked`
              `heights_stacked`
            Each element specifies the horizontal sheet count for the corresponding vertical drywall zone.

        CEILING_DRYWALL_STACKING_INSTRUCTIONS:
          - A ceiling drywall assembly may contain either:
            1. A single drywall material, or
            2. Multiple vertically stacked drywall layers.
          - Each ceiling drywall layer is applied over the SAME ceiling surface area.
            Therefore, all vertically stacked ceiling drywall layers MUST have the same predicted ceiling area.
          - The order of `materials_vertically_stacked` MUST start with the lowest/bottom-most drywall layer and proceed upward toward the ceiling assembly.
          - The primary `material` field MUST always contain exactly one drywall material preference for the ceiling.
          - If `materials_vertically_stacked` is non-empty, it MUST include the primary `material` as one of the layers.
          - The number of entries in:
            `materials_vertically_stacked`
            `color_codes_stacked`
            MUST be identical.
          - The stacked ceiling layers are NOT vertically segmented by ceiling height. They are separate material layers installed over the same ceiling plane.
          - If no additional ceiling drywall layer is required, all stacked arrays MUST be empty.
          - If stacked layers are predicted, the primary single-material prediction MUST still remain independently available through the primary `material`, `color_code`.

        CEILING_DRYWALL_LAYERING_INSTRUCTIONS
          - Since all stacked ceiling drywall materials occupy the same ceiling area:
          - If only one drywall material is used:
            `layers` MUST be a single integer.
            Example:
              "material": "5/8 Type X",
              "layers": 2
          - If multiple ceiling drywall layers are predicted:
            `layers` MUST be a list.
          - The number of entries MUST exactly equal the number of entries in:
            `materials_vertically_stacked`
            `color_codes_stacked`
            Each value specifies the horizontal sheet count for that ceiling assembly layer.

      JURISDICTION_AND_CODE_SELECTION:
        Project Location:
          - State / Province / Region / Pincode: {project_location}

        Regulatory Requirements:
          - Determine the applicable residential building code, fire code, drywall requirements, and construction practices for the specified project location.
          - Use the commonly adopted residential drywall standards, fire separation requirements, moisture protection requirements, and local code amendments for that jurisdiction.
          - Apply the building regulations that are typically enforced by the local Authority Having Jurisdiction (AHJ).
          - Follow local residential construction practices unless the architectural drawings explicitly specify otherwise.

        Fallback Behavior:
          - If the provided project location is missing, ambiguous, invalid, or cannot be confidently mapped to a real jurisdiction, default to:
            • California Residential Code (CRC)
            • California Building Code (CBC)
            • California Fire Code (CFC)
            • California Residential Code adoption of the IRC where applicable
          - Treat California as the default jurisdiction only in these fallback situations.

        General Assumptions:
          - Residential construction.
          - Standard wood or steel stud framing unless drawings indicate otherwise.
          - Do not assume commercial construction requirements unless explicitly shown.

        You must only support the drywall types from the provided templates,
        DRYWALL TEMPLATES: {drywall_templates}

        **STRICTLY** use the field `sku_variant` which contains both `sku_id` and `sku_description` as the target drywall material and the field `color_code` to map to it's target color code accompanied by the fields `fire_rating` aand `thickness` to derive it's fire rating and thickness respectively.
        Do not invent other drywall materials or color codes which are not included into the template list. All of the provided drywall types are associated with a definite color code presented in BGR (blue, green, red) format.
        If an appropriate/optimal drywall material for a given wall or polygon is not provided with the `DRYWALL_TEMPLATES` mention the target drywall material as `DISABLED` with [0, 0, 255] in BGR tuple as its target color code.

  PREDICTION_MODE
    The prediction request only operates in one mode.

    CUSTOM_SCHEMA
      The user has already selected one or more parameters to predict.
      A custom JSON schema will be provided separately as OUTPUT_SCHEMA.

    OUTPUT_SCHEMA contains the complete output contract.
    Predict ONLY the fields present in OUTPUT_SCHEMA.

    PREDICTION_INSTRUCTIONS:

      - Do NOT predict fields that are absent from OUTPUT_SCHEMA.
      - Do NOT infer additional fields.
      - Do NOT return the default output payload.
      - Preserve the exact hierarchy, field names, nesting and ordering of OUTPUT_SCHEMA.
      - Populate every field in OUTPUT_SCHEMA.
      - If a field cannot be determined, populate it according to the schema requirements (NULL, -1, empty list, etc.) rather than inventing additional fields.

  USER_SELECTED_PARAMETERS
    The application may allow the user to preselect only a subset of walls, ceilings or drywall parameters.

    Examples include:
      - Wall height only
      - Wall drywall assemblies only
      - Ceiling height
      - Wall height and Wall drywall assemblies and ceiling height
      - Ceiling Drywall assembly only
      - Ceiling Drywall assembly and wall drywall assemblies

    When USER_SELECTED_PARAMETERS are provided:
      - Ignore every prediction task outside the requested parameters.
      - Perform only the reasoning necessary to populate OUTPUT_SCHEMA.
      - Do not estimate auxiliary values unless they are required to derive a requested field.
      - Internal reasoning may use intermediate values, but they must not appear in the output unless present in OUTPUT_SCHEMA.

  OUTPUT_SCHEMA
    The output schema is supplied dynamically by the application.

    The schema defines:
    {output_schema}
"""

FEEDBACK_GENERATOR = """
  INTERNAL SELF-REVIEW (Do not skip):
    You are given {max_retry} attempts to retry the generation process and the following are the list of errors encountered during your previous attempts.
    {exceptions}
    STRICTY confirm no previous error remains before producing the final output.
"""

SCALE_AND_CEILING_HEIGHT_DETECTOR = """
  You are an expert architectural drawing text parser

  PROVIDED:
    1. A snapshot of the full Architectural Drawing in png format with the following highlight,
      - The target drawing of interest is enclosed with a green bounding box that encloses architectural plan(s) with a very tight aproximation.

  TASK:
    Identify the standard `ceiling_height` and `scale` applied on ONLY the target architectural plan enclosed with a green bounding box mentioned in the relevant section containing the textual metadata of the enclosed floorplan.
    INSTRUCTIONS:
      - Identify the `scale` ONLY for the highlighted target drawing, representing the ratio between paper length and real-world length.
      - Normalize architectural scales into:
          "<paper_length_in_inches>``:<real_world_length_in_feet>`<real_world_length_in_inches>``"
          Example:
            1/4" = 1'-0"  →  0.25``:1`0``
            1/8" = 1'-0"  →  0.125``:1`0``
      - SUPPORTED `Architectural Scales` are:
        {supported_scales_architectural}
      - STRICT DRAWING ASSOCIATION RULES:
        Only extract a scale if it is explicitly associated with the highlighted target drawing by one or more of the following:
          • Located directly adjacent to the highlighted drawing title
          • Inside the title block for the highlighted drawing
          • Explicitly labeled as the scale of the highlighted drawing
          • Unique and unambiguous on the page
      - MULTI-SCALE / REPRODUCTION RULE:
        If multiple scales appear for different sheet sizes, print layouts, or reproduction formats
        (e.g., "1/4\" = 1'-0\" AT 22\"x34\" LAYOUT" and "1/8\" = 1'-0\" AT 11\"x17\" LAYOUT"),
        DO NOT infer the target drawing scale.
        These are print/reproduction scales and are ambiguous unless the target drawing explicitly specifies which applies.
      - AMBIGUITY RULE:
        Return `NULL` for `scale` when:
          • Multiple competing scales exist
          • The scale belongs to page layout, viewport, or print size
          • The scale cannot be confidently tied to the highlighted drawing
          • The page contains only sheet-level scale references
          • The text contains phrases such as:
            "AT 22x34 LAYOUT", "AT 11x17 LAYOUT",
            "NOT TO SCALE", "NTS", "FOR REFERENCE ONLY"
      - NEVER infer or guess a scale from geometry, dimensions, room sizes, wall lengths, known object sizes, or typical architectural conventions.
      - If scale is written in a non-standard format, preserve the exact textual format.
      - If no unambiguous target-drawing scale exists, STRICTLY return:
        `scale = NULL`
      - Look for a keyword matching `ceiling height` in the highlighted drawing title section and extract the nearest numerical value.
      - If multiple ceiling heights exist, prefer the standard/typical ceiling height.
      - If ceiling height is not present, return `NULL`.

  OUTPUT:
    Your output should be in the JSON format containing the standard `ceiling_height` and `scale` of the floorplan.
    **STRICTLY** Do not generate additional content apart from the designated JSON.
    Please refer the following as a reference and ensure to replace every consecutive pair of open/closed curly braces with a single one during the generation of the output.
    {{
        "ceiling_height": <Standard ceiling height mentioned in the transcriptions converted to feet in float>,
        "scale": "<Scale of the drawing mentioned in the transcriptions i.e. number_in_inches``: number_in_feet`number_in_inches`` / NULL>",
        "scale_confidence": <confidence score in detecting the scale from the annotated text between 0 and 1 in float rounded upto 2 decimal places (e.g., 0.87)>,
    }}
"""

class ScaleAndCeilingHeightDetectorResponse(BaseModel):
    ceiling_height: Union[float, int]
    scale: Optional[str]
    scale_confidence: float = Field(ge=0, le=1)

SCALE_DETECTOR = """
  You are an expert architectural drawing text parser

  PROVIDED:
    1. A snapshot of the full Architectural Drawing in png format with the following highlight,
      - The target drawing of interest is enclosed with a green bounding box that encloses architectural plan(s) with a very tight aproximation.

  TASK:
    Identify the standard `scale` applied on ONLY the target architectural plan enclosed with a green bounding box mentioned in the relevant section containing the textual metadata of the enclosed floorplan.
    INSTRUCTIONS:
      - Identify the `scale` ONLY for the highlighted target drawing, representing the ratio between paper length and real-world length.
      - Normalize architectural scales into:
          "<paper_length_in_inches>``:<real_world_length_in_feet>`<real_world_length_in_inches>``"
          Example:
            1/4" = 1'-0"  →  0.25``:1`0``
            1/8" = 1'-0"  →  0.125``:1`0``
      - SUPPORTED `Architectural Scales` are:
        {supported_scales_architectural}
      - STRICT DRAWING ASSOCIATION RULES:
        Only extract a scale if it is explicitly associated with the highlighted target drawing by one or more of the following:
          • Located directly adjacent to the highlighted drawing title
          • Inside the title block for the highlighted drawing
          • Explicitly labeled as the scale of the highlighted drawing
          • Unique and unambiguous on the page
      - MULTI-SCALE / REPRODUCTION RULE:
        If multiple scales appear for different sheet sizes, print layouts, or reproduction formats
        (e.g., "1/4\" = 1'-0\" AT 22\"x34\" LAYOUT" and "1/8\" = 1'-0\" AT 11\"x17\" LAYOUT"),
        DO NOT infer the target drawing scale.
        These are print/reproduction scales and are ambiguous unless the target drawing explicitly specifies which applies.
      - AMBIGUITY RULE:
        Return `NULL` for `scale` when:
          • Multiple competing scales exist
          • The scale belongs to page layout, viewport, or print size
          • The scale cannot be confidently tied to the highlighted drawing
          • The page contains only sheet-level scale references
          • The text contains phrases such as:
            "AT 22x34 LAYOUT", "AT 11x17 LAYOUT",
            "NOT TO SCALE", "NTS", "FOR REFERENCE ONLY"
      - NEVER infer or guess a scale from geometry, dimensions, room sizes, wall lengths, known object sizes, or typical architectural conventions.
      - If scale is written in a non-standard format, preserve the exact textual format.
      - If no unambiguous target-drawing scale exists, STRICTLY return:
        `scale = NULL`

  OUTPUT:
    Your output should be in the JSON format containing the `scale` of the floorplan.
    **STRICTLY** Do not generate additional content apart from the designated JSON.
    Please refer the following as a reference and ensure to replace every consecutive pair of open/closed curly braces with a single one during the generation of the output.
    {{
        "scale": "<Scale of the drawing mentioned in the transcriptions i.e. number_in_inches``: number_in_feet`number_in_inches`` / NULL>",
        "scale_confidence": <confidence score in detecting the scale from the annotated text between 0 and 1 in float rounded upto 2 decimal places (e.g., 0.87)>,
    }}
"""

class ScaleDetectorResponse(BaseModel):
    scale: Optional[str]
    scale_confidence: float = Field(ge=0, le=1)

CEILING_HEIGHT_DETECTOR = """
  You are an expert architectural drawing text parser

  PROVIDED:
    1. A snapshot of the full Architectural Drawing in png format with the following highlight,
      - The target drawing of interest is enclosed with a green bounding box that encloses architectural plan(s) with a very tight aproximation.

  TASK:
    Identify the standard `ceiling_height` applied on ONLY the target architectural plan enclosed with a green bounding box mentioned in the relevant section containing the textual metadata of the enclosed floorplan.
    INSTRUCTIONS:
      - Look for a keyword matching `ceiling height` in the highlighted drawing title section and extract the nearest numerical value.
      - If multiple ceiling heights exist, prefer the standard/typical ceiling height.
      - If ceiling height is not present, return `NULL`.

  OUTPUT:
    Your output should be in the JSON format containing the standard `ceiling_height` of the floorplan.
    **STRICTLY** Do not generate additional content apart from the designated JSON.
    {
        "ceiling_height": <Standard ceiling height mentioned in the transcriptions converted to feet in float, or NULL>
    }
"""

class CeilingHeightDetectorResponse(BaseModel):
    ceiling_height: Optional[Union[float, int]]

CEILING_CHOICES = [
    "Flat",
    "Single-sloped",
    "Gable",
    "Tray",
    "Barrel vault",
    "Coffered",
    "Combination",
    "Soffit",
    "Cove",
    "Dome",
    "Cloister Vault",
    "Knee-Wall",
    "Cathedral with Flat Center",
    "Angled-Plane",
    "Boxed-Beam"
]

WALL_CHOICES = [
  {"wall_type": "OPEN_TO_BELOW", "is_height_static": False},
  {"wall_type": "FULL_WALL", "is_height_static": True},
  {"wall_type": "HALF_WALL", "is_height_static": False},
  {"wall_type": "STAIRCASE_WALL", "is_height_static": False},
  {"wall_type": "SOFFITS", "is_height_static": False},
  {"wall_type": "MULTI_FLOOR_ALIGNMENT", "is_height_static": False},
  {"wall_type": "DEMISING_WALL", "is_height_static": True},
  {"wall_type": "GARAGE_SEPARATION_WALL", "is_height_static": True},
  {"wall_type": "SHAFT_WALL", "is_height_static": False},
  {"wall_type": "WET_WALL", "is_height_static": False},
  {"wall_type": "HALLWAY_WALL", "is_height_static": True} 
]

OPENING_TYPE_CHOICES = {
    "Doors": [90, 32, 201],
    "Windows": [32, 153, 201],
    "Sliding doors": [201, 32, 193],
    "Arched openings": [7, 102, 77],
    "Pass-through": [214, 98, 26]
}

validator_helpers = {
  "thickness": ThicknessValidatorHelper,
  "layers": LayersValidatorHelper,
  "color_code": ColorCodeValidatorHelper,
  "stacked_layers": StackedLayersValidatorHelper,
  "height": HeightValidatorHelper,
  "stack_count": StackCountValidatorHelper,
  "area": AreaValidatorHelper,
  "length": LengthValidatorHelper,
  "width": WidthValidatorHelper,
  "wall_count": WallCountValidatorHelper
}

def load_schema_pydantic(model: type[BaseModel]):
    def build_annotation(annotation):
        origin = get_origin(annotation)

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return build_model(annotation)

        if origin is list:
            return [build_annotation(get_args(annotation)[0])]

        if origin is tuple:
            return [
                build_annotation(arg)
                for arg in get_args(annotation)
            ]

        if origin is dict:
            return {}

        if origin is None:
            if annotation is str:
                return "<string>"
            if annotation is int:
                return "<integer>"
            if annotation is float:
                return "<float>"
            if annotation is bool:
                return "<boolean>"

        if origin is not None:
            args = [a for a in get_args(annotation) if a is not type(None)]
            if args:
                return build_annotation(args[0])

        return "<value>"

    def build_model(model_cls):

        output = {}

        for name, field in model_cls.model_fields.items():

            value = build_annotation(field.annotation)

            if isinstance(value, dict):
                output[name] = value
            else:
                output[name] = (
                    f"{value}"
                    + (f" // {field.description}" if field.description else "")
                )

        return output

    return json.dumps(build_model(model), indent=2)

def prune_model(
  model: type[BaseModel],
  remove_fields: set[str] | None=None,
  remove_validators: set[str] | None=None
) -> type[BaseModel]:
    remove_fields = remove_fields or set()
    remove_validators = remove_validators or set()
    fields = {
        name: (
            field.annotation,
            field.default if not field.is_required() else ...
        )
        for name, field in model.model_fields.items()
        if name not in remove_fields
    }

    active_validators = (
        set(model.active_validators) - remove_validators
    )
    base_classes = list()
    for validator in model.active_validators:
      base_classes.append(validator_helpers[validator])
    MultiBase = type(
      f"{model.__name__}Base",
      tuple(base_classes),
      {
        "active_validators": active_validators,
      },
    )
    return create_model(
        f"{model.__name__}Pruned",
        __base__ = MultiBase,
        **fields,
    )
