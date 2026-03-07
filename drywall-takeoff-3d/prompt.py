from typing import Dict
from pydantic import BaseModel


ARCHITECTURAL_DRAWING_CLASSIFIER = """
    You are an expert architectural drawing classifier.

    PROVIDED:
        A single page extracted from an architectural construction plan project document entitled to a planned residence in `PNG` format.

    TASK:
        Classify the construction drawing into exactly ONE of the following categories:

        - FLOOR_PLAN
        - ROOF_PLAN
        - ELECTRICAL_PLAN
        - FOUNDATION_PLAN
        - ELEVATION_PLAN
        - NOT_ARCHITECTURAL_PLAN

        INSTRUCTIONS:
        - Choose exactly one category from the allowed list.
        - Use architectural conventions (symbols, annotations, layout, views).
        - Consider labels, dimensions, symbols, and drawing orientation.
        - A page containing architecture plan will contain the architecture metadata information in text at the stray sections of the image, usually at the right and bottom section of the image. Generate a mask factor containing the information on the stray section of the image following the below instrution,
            - Generate the horizontal `mask_factor` (boundary -> [0, 1]) of the image by determining the fraction of the total width of the image that contains text information and isolated from the architecture drawing on the page on the right-most section of the page.
            - Generate the vertical `mask_factor` (boundary -> [0, 1]) of the image by determining the fraction of the total height of the image that contains text information and isolated from the architecture drawing on the page on the bottom-most section of the page.

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
        {{
            "plan_type": "<FLOOR_PLAN>/<ROOF_PLAN>/<ELECTRICAL_PLAN>/<FOUNDATION_PLAN>/<ELEVATION_PLAN>/<NOT_ARCHITECTURAL_PLAN>",
            "mask_factor":
                {{
                    "horizontal": <mask factor for the width of the image in float rounded upto 2 decimal places>,
                    "vertical": <mask factor for the height of the image in float rounded upto 2 decimal places>
                }}
        }}
"""

class ArchitecturalDrawingClassifierResponse(BaseModel):
    plan_type: str
    mask_factor: Dict
