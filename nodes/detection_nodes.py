"""
PowerVision 检测相关节点

包含目标检测、边界框处理相关的节点
"""

import os
import ast
import json
import torch
from PIL import Image
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from .model_nodes import QwenModel
import folder_paths
import comfy.model_management


@dataclass
class DetectedBox:
    """检测到的边界框数据类"""
    bbox: List[int]
    score: float
    label: str = ""


def parse_json(json_output: str) -> str:
    """从模型响应字符串中提取JSON载荷"""
    if "```json" in json_output:
        json_output = json_output.split("```json", 1)[1]
        json_output = json_output.split("```", 1)[0]

    try:
        parsed = json.loads(json_output)
        if isinstance(parsed, dict) and "content" in parsed:
            inner = parsed["content"]
            if isinstance(inner, str):
                json_output = inner
    except Exception:
        pass
    return json_output


def parse_boxes(
    text: str,
    img_width: int,
    img_height: int,
    input_w: int,
    input_h: int,
    score_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """从模型的原始JSON输出中解析边界框"""
    text = parse_json(text)
    try:
        data = json.loads(text)
    except Exception:
        try:
            data = ast.literal_eval(text)
        except Exception:
            end_idx = text.rfind('"}') + len('"}')
            truncated = text[:end_idx] + "]"
            data = ast.literal_eval(truncated)
    
    if isinstance(data, dict):
        inner = data.get("content")
        if isinstance(inner, str):
            try:
                data = ast.literal_eval(inner)
            except Exception:
                data = []
        else:
            data = []
    
    items: List[DetectedBox] = []
    x_scale = img_width / input_w
    y_scale = img_height / input_h

    for item in data:
        box = item.get("bbox_2d") or item.get("bbox") or item
        label = item.get("label", "")
        score = float(item.get("score", 1.0))
        y1, x1, y2, x2 = box[1], box[0], box[3], box[2]
        abs_y1 = int(y1 * y_scale)
        abs_x1 = int(x1 * x_scale)
        abs_y2 = int(y2 * y_scale)
        abs_x2 = int(x2 * x_scale)
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        if score >= score_threshold:
            items.append(DetectedBox([abs_x1, abs_y1, abs_x2, abs_y2], score, label))
    
    items.sort(key=lambda x: x.score, reverse=True)
    return [
        {"score": b.score, "bbox": b.bbox, "label": b.label}
        for b in items
    ]


class PowerVisionObjectDetection:
    """PowerVision 目标检测节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_model": ("QWEN_MODEL",),
                "image": ("IMAGE",),
                "target": ("STRING", {"default": "object"}),
                "bbox_selection": ("STRING", {"default": "all"}),
                "score_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "merge_boxes": ("BOOLEAN", {"default": False}),
                "unload_model": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("JSON", "BBOX")
    RETURN_NAMES = ("detection_result", "bboxes")
    FUNCTION = "detect"
    CATEGORY = "PowerVision/Target Detection"

    def detect(
        self,
        qwen_model: QwenModel,
        image: torch.Tensor,
        target: str,
        bbox_selection: str = "all",
        score_threshold: float = 0.0,
        merge_boxes: bool = False,
        unload_model: bool = False,
    ) -> Tuple[str, List[List[int]]]:
        """执行目标检测"""
        model = qwen_model.model
        processor = qwen_model.processor
        device = qwen_model.device
        
        # 设备管理
        current_device = str(next(model.parameters()).device)
        expected_device = device
        
        if device == "auto":
            if torch.cuda.is_available():
                expected_device = "cuda:0"
            else:
                expected_device = "cpu"
        elif device.startswith("cuda") and not torch.cuda.is_available():
            expected_device = "cpu"
        
        if not current_device.startswith(expected_device):
            if expected_device.startswith("cuda"):
                device_index = 0
                if ":" in expected_device:
                    device_index = int(expected_device.split(":")[1])
                if device_index < torch.cuda.device_count():
                    model.to(expected_device)
                    if torch.cuda.is_available():
                        torch.cuda.set_device(device_index)
                else:
                    model.to("cuda:0")
                    if torch.cuda.is_available():
                        torch.cuda.set_device(0)
            else:
                model.to("cpu")
        
        current_device = str(next(model.parameters()).device)
        if current_device.startswith("cuda") and torch.cuda.is_available():
            try:
                device_index = int(current_device.split(":")[1])
                torch.cuda.set_device(device_index)
            except Exception:
                pass

        prompt = f"Locate the {target} and output bbox in JSON"

        if isinstance(image, torch.Tensor):
            image = (image.squeeze().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Unsupported image type")

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": [{"type": "text", "text": prompt}, {"image": image}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_device = next(model.parameters()).device
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model_device)
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=1024)
        gen_ids = [output_ids[len(inp):] for inp, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        
        input_h = inputs['image_grid_thw'][0][1] * 14
        input_w = inputs['image_grid_thw'][0][2] * 14
        items = parse_boxes(
            output_text,
            image.width,
            image.height,
            input_w,
            input_h,
            score_threshold,
        )

        selection = bbox_selection.strip().lower()
        boxes = items
        if selection != "all" and selection:
            idxs = []
            for part in selection.replace(",", " ").split():
                try:
                    idxs.append(int(part))
                except Exception:
                    continue
            boxes = [boxes[i] for i in idxs if 0 <= i < len(boxes)]

        if merge_boxes and boxes:
            x1 = min(b["bbox"][0] for b in boxes)
            y1 = min(b["bbox"][1] for b in boxes)
            x2 = max(b["bbox"][2] for b in boxes)
            y2 = max(b["bbox"][3] for b in boxes)
            score = max(b["score"] for b in boxes)
            label = boxes[0].get("label", target)
            boxes = [{"bbox": [x1, y1, x2, y2], "score": score, "label": label}]

        json_boxes = [
            {"bbox_2d": b["bbox"], "label": b.get("label", target)} for b in boxes
        ]
        json_output = json.dumps(json_boxes, ensure_ascii=False)
        bboxes_only = [b["bbox"] for b in boxes]

        if unload_model:
            model.to('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return (json_output, bboxes_only)


class PowerVisionBBoxProcessor:
    """PowerVision 边界框处理器节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"bboxes": ("BBOX",)}}

    RETURN_TYPES = ("BBOXES",)
    RETURN_NAMES = ("processed_bboxes",)
    FUNCTION = "convert"
    CATEGORY = "PowerVision/Target Detection"

    def convert(self, bboxes: List[List[int]]) -> Tuple[List[List[List[int]]]]:
        """转换边界框格式"""
        if not isinstance(bboxes, list):
            raise ValueError("bboxes must be a list")

        if bboxes and isinstance(bboxes[0], (list, tuple)) and bboxes[0] and isinstance(bboxes[0][0], (list, tuple)):
            return (bboxes,)

        return ([bboxes],)


class PowerVisionDetectionFilter:
    """PowerVision 检测结果过滤器节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "detection_result": ("JSON",),
                "min_score": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_boxes": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "filter_labels": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("JSON", "BBOX")
    RETURN_NAMES = ("filtered_result", "filtered_bboxes")
    FUNCTION = "filter"
    CATEGORY = "PowerVision/Target Detection"

    def filter(
        self, 
        detection_result: str, 
        min_score: float, 
        max_boxes: int, 
        filter_labels: str
    ) -> Tuple[str, List[List[int]]]:
        """过滤检测结果"""
        try:
            data = json.loads(detection_result)
            if not isinstance(data, list):
                return (detection_result, [])
            
            # 按分数过滤
            filtered = [item for item in data if item.get("score", 0) >= min_score]
            
            # 按标签过滤
            if filter_labels.strip():
                labels = [label.strip() for label in filter_labels.split(",")]
                filtered = [item for item in filtered if item.get("label", "") in labels]
            
            # 按分数排序并限制数量
            filtered.sort(key=lambda x: x.get("score", 0), reverse=True)
            filtered = filtered[:max_boxes]
            
            # 提取边界框
            bboxes = [item.get("bbox_2d", item.get("bbox", [])) for item in filtered]
            
            return (json.dumps(filtered, ensure_ascii=False), bboxes)
            
        except Exception as e:
            print(f"PowerVision: 过滤检测结果时出错: {e}")
            return (detection_result, [])

