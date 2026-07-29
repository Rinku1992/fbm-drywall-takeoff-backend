import logging
from pathlib import Path
from pdf2image import convert_from_path
import cv2


def process_page(pdf_path, page_index, image_path_page, maximum_dpi, minimum_dpi, dpi_reduction_factor: float=0.75):
    current_dpi = maximum_dpi

    while True:
        try:
            pdf_page = convert_from_path(
                pdf_path,
                dpi=current_dpi,
                first_page=page_index+1,
                last_page=page_index+1,
            )[0]
            break

        except Image.DecompressionBombError:
            if current_dpi <= minimum_dpi:
                logging.error(
                    "DecompressionBombError persists at minimum DPI=%s",
                    minimum_dpi,
                )
                raise

            new_dpi = max(
                minimum_dpi,
                int(current_dpi * dpi_reduction_factor)
            )
            current_dpi = new_dpi
    save(pdf_page, image_path_page)
    del pdf_page
    return current_dpi

def save(pdf_page, image_path_page):
    pdf_page.save(image_path_page, "PNG")

def preprocess(pdf_path, page_index, maximum_dpi=400, minimum_dpi=250, image_path="/tmp/floor_plan.png"):
    image_path = Path(image_path)
    image_path_page = image_path.parent.joinpath(image_path.stem).with_suffix(f".{str(page_index).zfill(2)}{image_path.suffix}")
    _ = process_page(pdf_path, page_index, image_path_page, maximum_dpi, minimum_dpi)

    return image_path_page
