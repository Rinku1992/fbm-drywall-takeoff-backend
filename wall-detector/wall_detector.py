import logging
import os
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from peft import PeftModel
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np

__all__ = ["WallDetector"]


class WallDetector:
    def __init__(
        self,
        ckpt_path=None,
        lora_path=None,
        stable_diffusion_ckpt=None,
    ):
        self.ckpt_path = ckpt_path or os.getenv("CONTROLNET_PATH", "/models/Finetuned_Iter_17/controlnet")
        self.lora_path = lora_path or os.getenv("LORA_PATH", "/models/Finetuned_Iter_17/unet")
        self.stable_diffusion_ckpt = stable_diffusion_ckpt or os.getenv("SD_MODEL_PATH", "/models/stable-diffusion-v1-4")

        controlnet = ControlNetModel.from_pretrained(self.ckpt_path, torch_dtype=torch.float16)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.stable_diffusion_ckpt,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        pipe.unet = PeftModel.from_pretrained(pipe.unet, self.lora_path)
        self.pipe = pipe.to(self.device)

        logging.info(
            "SYSTEM: Model paths loaded SD=%s CONTROLNET=%s LORA=%s",
            self.stable_diffusion_ckpt,
            self.ckpt_path,
            self.lora_path,
        )

    def detect(self, image_path, hyperparameters, mask_offset=None):
        image = Image.open(image_path).convert("RGB")
        width_original, height_original = image.size
        if hyperparameters["RESOLUTION"]["KEEP_ORIGINAL"]:
            width, height = image.size
        else:
            width, height = hyperparameters["RESOLUTION"]["WIDTH"], hyperparameters["RESOLUTION"]["HEIGHT"]
            image = image.resize((width, height))

        out = self.pipe(
            "A floor plan",
            num_inference_steps=hyperparameters["N_INFERENCE_STEPS"],
            image=image,
            height=height,
            width=width,
            controlnet_conditioning_scale=hyperparameters["CONTROLNET_CONDITIONING_SCALE"],
            guidance_scale=hyperparameters["GUIDANCE_SCALE"],
            generator=[torch.manual_seed(s) for s in range(hyperparameters["N_IMAGES"])],
            num_images_per_prompt=hyperparameters["N_IMAGES"],
        )
        votes = np.stack([(np.asarray(img).mean(axis=-1) > 127) for img in out.images])
        image_detected = Image.fromarray((votes.mean(axis=0) >= 0.5).astype(np.uint8) * 255)

        if mask_offset:
            logging.info("SYSTEM: Masking segmented image with Offset: %s", mask_offset)
            image_detected = image_detected.resize((width_original, height_original))
            image_arr = np.array(image_detected)
            mask_height_factor = mask_offset["vertical"]
            mask_width_factor = mask_offset["horizontal"]
            mask_height_factor = max(0, mask_height_factor - 0.1)
            mask_width_factor = max(0, mask_width_factor - 0.05)
            if mask_width_factor > 0:
                image_arr[:, -round(width_original * mask_width_factor):] = 255
            if mask_height_factor > 0:
                image_arr[-round(height_original * mask_height_factor):, :] = 255
            image_detected = Image.fromarray(image_arr)
            image_detected = image_detected.resize((width, height))

        save_w = hyperparameters["SAVE_RESOLUTION"]["WIDTH"]
        save_h = hyperparameters["SAVE_RESOLUTION"]["HEIGHT"]
        return image_detected.resize((save_w, save_h), Image.NEAREST)
