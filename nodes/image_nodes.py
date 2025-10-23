"""
PowerVision 图像相关节点

包含图像和视频加载、处理相关的节点
"""

import os
import hashlib
import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence
from typing import Tuple, Union, Any
from pathlib import Path

import folder_paths
import node_helpers
from comfy.comfy_types import IO, ComfyNodeABC
from comfy_api.latest import InputImpl


class PowerVisionImageLoader:
    """PowerVision 图像加载器节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
            and f.split(".")[-1].lower() in ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]
        ]
        return {
            "required": {"image": (sorted(files), {"image_upload": True})},
        }

    CATEGORY = "PowerVision/Image Caption"
    RETURN_TYPES = ("IMAGE", "MASK", "PATH")
    RETURN_NAMES = ("image", "mask", "path")
    FUNCTION = "load_image"

    def load_image(self, image: str) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """加载图像文件"""
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ["MPO"]

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask, image_path)

    @classmethod
    def IS_CHANGED(cls, image: str) -> str:
        """检查图像是否已更改"""
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image: str) -> Union[bool, str]:
        """验证输入"""
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


class PowerVisionVideoLoader(ComfyNodeABC):
    """PowerVision 视频加载器节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return {
            "required": {"file": (sorted(files), {"video_upload": True})},
        }

    CATEGORY = "PowerVision/Image Caption"
    RETURN_TYPES = (IO.VIDEO, "PATH")
    RETURN_NAMES = ("video", "path")
    FUNCTION = "load_video"

    def load_video(self, file: str) -> Tuple[Any, str]:
        """加载视频文件"""
        video_path = folder_paths.get_annotated_filepath(file)
        return (InputImpl.VideoFromFile(video_path), video_path)

    @classmethod
    def IS_CHANGED(cls, file: str) -> float:
        """检查视频是否已更改"""
        video_path = folder_paths.get_annotated_filepath(file)
        return os.path.getmtime(video_path)

    @classmethod
    def VALIDATE_INPUTS(cls, file: str) -> Union[bool, str]:
        """验证输入"""
        if not folder_paths.exists_annotated_filepath(file):
            return f"Invalid video file: {file}"
        return True


class PowerVisionImageProcessor:
    """PowerVision 图像处理器节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "normalize": ("BOOLEAN", {"default": True}),
            }
        }

    CATEGORY = "PowerVision/Image Caption"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_image",)
    FUNCTION = "process_image"

    def process_image(self, image: torch.Tensor, max_size: int, normalize: bool) -> Tuple[torch.Tensor]:
        """处理图像"""
        from utils import image_processor
        
        # 转换为PIL图像
        pil_image = image_processor.tensor_to_pil(image)
        
        # 调整大小
        pil_image = image_processor.resize_image(pil_image, max_size)
        
        # 转换回张量
        processed_tensor = image_processor.pil_to_tensor(pil_image)
        
        # 标准化
        if normalize:
            processed_tensor = image_processor.normalize_image(processed_tensor)
        
        return (processed_tensor,)

