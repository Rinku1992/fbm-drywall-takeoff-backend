"""
PDF → PNG page rendering for the classifier service.

Forked from drywall-takeoff-3d/preprocessing.py with one deliberate
difference: DPI is 150, matching the resolution the classifier was
trained on. Do NOT unify with the drywall-takeoff-3d copy without first
confirming both services agree on DPI.
"""
from pathlib import Path
from pdf2image import convert_from_path

# Classifier model was trained on DPI 150 PNGs. Keep this constant
# until/unless the model is retrained at a different DPI.
CLASSIFIER_DPI = 150


def process_page(pdf_path, page_index, image_path_page):
    pdf_page = convert_from_path(
        pdf_path,
        dpi=CLASSIFIER_DPI,
        first_page=page_index + 1,
        last_page=page_index + 1,
    )[0]
    save(pdf_page, image_path_page)
    del pdf_page


def save(pdf_page, image_path_page):
    pdf_page.save(image_path_page, "PNG")


def preprocess(pdf_path, page_index, image_path="/tmp/floor_plan.png"):
    image_path = Path(image_path)
    image_path_page = image_path.parent.joinpath(
        image_path.stem
    ).with_suffix(f".{str(page_index).zfill(2)}{image_path.suffix}")
    process_page(pdf_path, page_index, image_path_page)
    return image_path_page
