from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from google.cloud import vision
from google.oauth2 import service_account
import cv2

__all__ = ["Transcriber"]


class Transcriber:

    def __init__(self, credentials, hyperparameters):
        self._credentials = credentials
        self._hyperparameters = hyperparameters
        self._transcription_block_centroids = dict()

    def _image_to_string(
        self,
        image_to_string_client,
        v_stride_index,
        h_stride_index,
        kernel_parameters,
        n_horizontal_strides,
        image_array,
        output_path
    ):
        X1 = h_stride_index * kernel_parameters["stride"]
        X2 = X1 + kernel_parameters["width"]
        Y1 = v_stride_index * kernel_parameters["stride"]
        Y2 = Y1 + kernel_parameters["height"]

        cropped = image_array[Y1:Y2, X1:X2]
        cv2.imwrite(f"/tmp/{output_path}_{str((v_stride_index*n_horizontal_strides)+h_stride_index).zfill(3)}.png", cropped)

        with open(f"/tmp/{output_path}_{str((v_stride_index*n_horizontal_strides)+h_stride_index).zfill(3)}.png", "rb") as f:
            content = f.read()
        image = vision.Image(content=content)

        response = image_to_string_client.document_text_detection(image=image)
        response_json = json.loads(response.__class__.to_json(response))
        if response_json["textAnnotations"]:
            text = response_json["textAnnotations"][0]["description"]
            bounding_box_A, bounding_box_B, bounding_box_C, bounding_box_D  = response_json["textAnnotations"][0]["boundingPoly"]["vertices"]
            centroid_x = (bounding_box_A['x'] + bounding_box_B['x'] + bounding_box_C['x'] + bounding_box_D['x']) / 4
            centroid_y = (bounding_box_A['y'] + bounding_box_B['y'] + bounding_box_C['y'] + bounding_box_D['y']) / 4
            self._transcription_block_centroids[text] = [centroid_x + X1, centroid_y + Y1]

        with open(f"/tmp/{output_path}_{str((v_stride_index*n_horizontal_strides)+h_stride_index).zfill(3)}.json", "w", encoding="utf-8") as f:
            json.dump(response_json, f, ensure_ascii=False, indent=2)

    def transcribe(self, image_path: Path):
        credentials = service_account.Credentials.from_service_account_file(self._credentials["service_drywall_account_key"])
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)

        kernel_parameters = self._hyperparameters["modelling"]["kernel"]
        image = cv2.imread(image_path)
        n_horizontal_strides = (image.shape[1] // kernel_parameters["stride"]) + 1
        n_vertical_strides = (image.shape[0] // kernel_parameters["stride"]) + 1

        futures = list()
        with ThreadPoolExecutor(max_workers=50) as executor:
            for v_stride_index in range(n_vertical_strides):
                for h_stride_index in range(n_horizontal_strides):
                    futures.append(executor.submit(
                        self._image_to_string,
                        vision_client,
                        v_stride_index,
                        h_stride_index,
                        kernel_parameters,
                        n_horizontal_strides,
                        image,
                        "ocr_clip",
                    ))

        [future.result() for future in futures]
        return self._transcription_block_centroids
