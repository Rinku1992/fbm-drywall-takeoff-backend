from pathlib import Path
from pdf2image import convert_from_path
import cv2


def process_page(pdf_path, page_index, image_path_page):
    pdf_page = convert_from_path(
        pdf_path,
        dpi=300,
        first_page=page_index+1,
        last_page=page_index+1
    )[0]
    save(pdf_page, image_path_page)
    del pdf_page

def save(pdf_page, image_path_page):
    pdf_page.save(image_path_page, "PNG")

def preprocess(pdf_path, page_index, image_path="/tmp/floor_plan.png"):
    image_path = Path(image_path)
    image_path_page = image_path.parent.joinpath(image_path.stem).with_suffix(f".{str(page_index).zfill(2)}{image_path.suffix}")
    process_page(pdf_path, page_index, image_path_page)

    return image_path_page
