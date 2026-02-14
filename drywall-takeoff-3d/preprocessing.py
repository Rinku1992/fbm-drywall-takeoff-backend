from pathlib import Path
from pypdf import PdfReader, PdfWriter
from pdf2image import convert_from_path
import cv2


def pdf2image(pdf_path, image_path):
    pages = convert_from_path(
        pdf_path,
        dpi=400,
    )
    reader = PdfReader(pdf_path)

    image_path = Path(image_path)
    image_path_pages = list()
    vector_pdf_pages = list()
    for index, (page, vector_page) in enumerate(zip(pages, reader.pages)):
        image_path_page = image_path.parent.joinpath(image_path.stem).with_suffix(f".{str(index).zfill(2)}{image_path.suffix}")
        page.save(image_path_page, "PNG")
        image_path_pages.append(image_path_page)

        writer = PdfWriter()
        writer.add_page(vector_page)
        vector_pdf_page = image_path.parent.joinpath(str(index).zfill(2)).joinpath(f"scaled_{image_path.stem}").with_suffix(".pdf")
        vector_pdf_page.parent.mkdir(parents=True, exist_ok=True)
        with open(vector_pdf_page, "wb") as f:
            writer.write(f)
        vector_pdf_pages.append(vector_pdf_page)
    return vector_pdf_pages, image_path_pages

def to_sharp(image_path_pages, output_path=None):
    sharpened_images = list()
    for index, image_path in enumerate(image_path_pages):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        clean = cv2.fastNlMeansDenoising(binary, h=30)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        sharpened = cv2.dilate(clean, kernel, iterations=1)
        sharpened = cv2.erode(sharpened, kernel, iterations=1)
        sharpened_images.append(sharpened)

        if output_path:
            output_path = Path(output_path)
            output_path = output_path.parent.joinpath(output_path.stem).with_suffix(f".{str(index).zfill(2)}{output_path.suffix}")
            cv2.imwrite(output_path, sharpened)
    return sharpened_images

def preprocess(pdf_path, image_path="/tmp/floor_plan.png"):
    vector_pdf_pages, image_path_pages = pdf2image(pdf_path, image_path)
    to_sharp(image_path_pages, output_path=image_path)

    return vector_pdf_pages, image_path_pages
