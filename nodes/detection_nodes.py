"""
PowerVision 检测相关节点

包含目标检测、边界框处理相关的节点
"""

import os
import ast
import json
import re
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
    """从模型响应字符串中提取JSON载荷，过滤 Thinking 模型的思考内容"""
    
    # 过滤 Thinking 模型的思考内容（支持多种标签格式）
    thinking_patterns = [
        (r'<think>.*?</think>', '<think>'),
        (r'<thinking>.*?</thinking>', '<thinking>'),
    ]
    
    for pattern, marker in thinking_patterns:
        if marker in json_output:
            json_output = re.sub(pattern, '', json_output, flags=re.DOTALL)
    
    # 提取 JSON 代码块
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
    target: str = "object",
    model_type: str = "qwen2.5",
) -> List[Dict[str, Any]]:
    """Return bounding boxes parsed from the model's raw JSON output."""
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
    
    # 如果 data 本身就是坐标数组（不是列表中的项），需要特殊处理
    if isinstance(data, list) and len(data) == 4 and all(isinstance(x, (int, float)) for x in data):
        # data 就是 [x1, y1, x2, y2]，需要包装成 [{"bbox_2d": [...]}]
        data = [{"bbox_2d": data, "label": target}]
    
    items: List[DetectedBox] = []
    x_scale = img_width / input_w
    y_scale = img_height / input_h

    for item in data:
        # 处理非字典项（如直接的坐标数组）
        if not isinstance(item, dict):
            # 如果 item 本身就是坐标数组 [x1, y1, x2, y2]
            if isinstance(item, (list, tuple)) and len(item) == 4:
                box = item
                label = target
                score = 1.0
            else:
                continue
        else:
            box = item.get("bbox_2d") or item.get("bbox")
            label = item.get("label", target)
            score = float(item.get("score", 1.0))
            
            # 如果 box 为空，跳过此项
            if box is None:
                continue
        
        # 确保 box 是列表
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        
        if model_type == "qwen3":
            # Qwen3-VL 使用归一化到 1000 的坐标系
            print(f"PowerVision: Qwen3-VL 原始坐标 box={box}, img_size={img_width}x{img_height}, input_size={input_w}x{input_h}")
            
            # Qwen3-VL 返回的坐标是 [x1, y1, x2, y2] 格式，归一化到 1000
            norm_x1, norm_y1, norm_x2, norm_y2 = box[0], box[1], box[2], box[3]
            
            # 计算从归一化坐标到输入尺寸的比例
            norm_to_input_w = input_w / 1000.0
            norm_to_input_h = input_h / 1000.0
            
            # 将归一化坐标转换为输入尺寸坐标
            input_x1 = norm_x1 * norm_to_input_w
            input_y1 = norm_y1 * norm_to_input_h
            input_x2 = norm_x2 * norm_to_input_w
            input_y2 = norm_y2 * norm_to_input_h
            
            # 将输入尺寸坐标缩放到原始图像坐标
            abs_x1 = int(input_x1 * x_scale)
            abs_y1 = int(input_y1 * y_scale)
            abs_x2 = int(input_x2 * x_scale)
            abs_y2 = int(input_y2 * y_scale)
            
            print(f"PowerVision: Qwen3-VL 转换后坐标 [{abs_x1}, {abs_y1}, {abs_x2}, {abs_y2}]")
        else:
            # Qwen2.5-VL 的坐标系：直接使用输入尺寸坐标
            print(f"PowerVision: Qwen2.5-VL 原始坐标 box={box}, img_size={img_width}x{img_height}, input_size={input_w}x{input_h}, scale={x_scale:.2f}x{y_scale:.2f}")
            
            y1, x1, y2, x2 = box[1], box[0], box[3], box[2]
            
            # 检查坐标是否超出输入尺寸范围
            if x1 > input_w or x2 > input_w or y1 > input_h or y2 > input_h:
                print(f"PowerVision: 警告！坐标超出输入尺寸: x1={x1}, y1={y1}, x2={x2}, y2={y2} vs input_w={input_w}, input_h={input_h}")
            
            abs_y1 = int(y1 * y_scale)
            abs_x1 = int(x1 * x_scale)
            abs_y2 = int(y2 * y_scale)
            abs_x2 = int(x2 * x_scale)
            
            print(f"PowerVision: Qwen2.5-VL 缩放后坐标 [{abs_x1}, {abs_y1}, {abs_x2}, {abs_y2}]")
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        
        # 限制坐标在图像范围内
        abs_x1 = max(0, min(abs_x1, img_width - 1))
        abs_y1 = max(0, min(abs_y1, img_height - 1))
        abs_x2 = max(0, min(abs_x2, img_width))
        abs_y2 = max(0, min(abs_y2, img_height))
        
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

        # 使用优化的 prompt，明确指定坐标格式和图像坐标系
        prompt = f"""Find the {target} in the image and output bounding box coordinates in JSON format.
Output format: [{{"bbox_2d": [x1, y1, x2, y2], "label": "{target}"}}]
Coordinates: (x1,y1) is top-left corner, (x2,y2) is bottom-right corner.
Only return the exact bounding box coordinates of the {target}, nothing else."""

        if isinstance(image, torch.Tensor):
            image = (image.squeeze().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Unsupported image type")

        # 构建消息，按照 Qwen VL 标准格式
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image}]},
        ]
        
        # 应用 chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 获取模型设备
        model_device = next(model.parameters()).device
        
        # 使用 processor 处理输入
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model_device)
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=1024)
        gen_ids = [output_ids[len(inp):] for inp, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        
        print(f"PowerVision: 模型输出原始文本: {output_text}")
        
        input_h = inputs['image_grid_thw'][0][1] * 14
        input_w = inputs['image_grid_thw'][0][2] * 14
        
        print(f"PowerVision: 图像预处理后尺寸 input_w={input_w}, input_h={input_h}")
        
        items = parse_boxes(
            output_text,
            image.width,
            image.height,
            input_w,
            input_h,
            score_threshold,
            target,
            qwen_model.model_type,
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

