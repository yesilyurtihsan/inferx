"""YOLO object detection inferencer using OpenVINO Runtime"""

import openvino as ov
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple
import logging

from .openvino_inferencer import OpenVINOInferencer
from ..utils import ImageProcessor

logger = logging.getLogger(__name__)


class YOLOOpenVINOInferencer(OpenVINOInferencer):
    """YOLO object detection inferencer optimized for OpenVINO Runtime"""
    
    def __init__(self, model_path: Union[str, Path], config: Optional[Dict] = None):
        """Initialize YOLO OpenVINO inferencer
        
        Args:
            model_path: Path to YOLO OpenVINO model file (.xml)
            config: Optional configuration dictionary
        """
        # Default YOLO config optimized for OpenVINO
        default_yolo_config = {
            "device": "AUTO",  # Let OpenVINO choose best device
            "performance_hint": "THROUGHPUT",  # Optimize for throughput
            "precision": "FP16",  # Use FP16 for better performance
            "input_size": 640,
            "confidence_threshold": 0.25,
            "nms_threshold": 0.45,
            "max_detections": 100,
            "preprocessing": {
                "target_size": [640, 640],
                "normalize": False,  # YOLO models usually expect [0, 1] range
                "color_format": "RGB"
            },
            "class_names": [
                "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                "train", "truck", "boat", "traffic light", "fire hydrant",
                "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket",
                "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                "mouse", "remote", "keyboard", "cell phone", "microwave",
                "oven", "toaster", "sink", "refrigerator", "book", "clock",
                "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
            ]
        }
        
        # Merge user config with YOLO-specific defaults
        if config:
            self._merge_config(default_yolo_config, config)
        
        # Update input size in preprocessing config
        input_size = default_yolo_config.get("input_size", 640)
        default_yolo_config["preprocessing"]["target_size"] = [input_size, input_size]
        
        super().__init__(model_path, default_yolo_config)
    
    def _setup_device_config(self) -> Dict[str, Any]:
        """Setup OpenVINO device configuration optimized for YOLO"""
        config = super()._setup_device_config()
        
        # YOLO-specific optimizations
        device = self.config.get("device", "AUTO")
        
        if device == "CPU" or device == "AUTO":
            # CPU optimizations for YOLO
            config["CPU_BIND_THREAD"] = "YES"
            config["CPU_THREADS_NUM"] = str(self.config.get("num_threads", 0))
            
            # Use more streams for better throughput
            if not self.config.get("num_streams"):
                config["CPU_THROUGHPUT_STREAMS"] = "CPU_THROUGHPUT_AUTO"
        
        elif device == "GPU":
            # GPU optimizations for YOLO
            if not self.config.get("num_streams"):
                config["GPU_THROUGHPUT_STREAMS"] = "GPU_THROUGHPUT_AUTO"
        
        return config
    
    def _letterbox(self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """Letterbox resize image while maintaining aspect ratio (OpenVINO optimized)
        
        Args:
            img: Input image
            new_shape: Target shape (height, width)
            
        Returns:
            Tuple of (resized_image, scale_ratio, padding)
        """
        shape = img.shape[:2]  # current shape [height, width]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # width, height
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        return img, (r, r), (dw, dh)
    
    def preprocess(self, input_data: Any) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """YOLO preprocessing optimized for OpenVINO: letterbox resize + normalize + transpose
        
        Args:
            input_data: Input data (image path or numpy array)
            
        Returns:
            Tuple of (preprocessed_image, scale_ratio, padding)
        """
        if isinstance(input_data, (str, Path)):
            # Load image from file
            img = ImageProcessor.load_image(input_data)
        elif isinstance(input_data, np.ndarray):
            img = input_data
        else:
            raise ValueError(f"Unsupported input data type: {type(input_data)}")
        
        # Store original image dimensions
        self.img_height, self.img_width = img.shape[:2]
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get input size from model shape or config
        if hasattr(self, 'input_shape') and len(self.input_shape) >= 4:
            input_height = self.input_shape[2]
            input_width = self.input_shape[3]
        else:
            input_size = self.config.get("input_size", 640)
            input_height = input_width = input_size
        
        # Letterbox resize
        img, ratio, pad = self._letterbox(img, (input_height, input_width))
        
        # Normalize to [0, 1]
        img = np.array(img) / 255.0
        
        # Transpose HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension and convert to float32
        img = np.expand_dims(img, axis=0).astype(np.float32)
        
        return img, ratio, pad
    
    def _run_inference(self, preprocessed_data: np.ndarray) -> List[np.ndarray]:
        """Run YOLO model inference on OpenVINO
        
        Args:
            preprocessed_data: Preprocessed input data
            
        Returns:
            Raw model outputs
        """
        try:
            # Create inference request (reusable for better performance)
            if not hasattr(self, '_infer_request'):
                self._infer_request = self.compiled_model.create_infer_request()
            
            # Set input tensor
            self._infer_request.set_input_tensor(preprocessed_data)
            
            # Run inference
            self._infer_request.infer()
            
            # Get outputs
            outputs = []
            for output_layer in self.output_layers.values():
                output_tensor = self._infer_request.get_output_tensor(output_layer.index)
                outputs.append(output_tensor.data.copy())
            
            logger.debug(f"YOLO OpenVINO inference completed, output shapes: {[out.shape for out in outputs]}")
            return outputs
            
        except Exception as e:
            logger.error(f"YOLO OpenVINO inference failed: {e}")
            raise RuntimeError(f"YOLO OpenVINO inference failed: {e}")
    
    def postprocess(self, model_outputs: List[np.ndarray], ratio: Tuple[float, float], pad: Tuple[int, int]) -> Dict[str, Any]:
        """YOLO postprocessing: NMS + format results (OpenVINO optimized)
        
        Args:
            model_outputs: Raw outputs from model inference
            ratio: Scale ratio from letterbox resize
            pad: Padding from letterbox resize
            
        Returns:
            Dictionary containing processed results with bounding boxes
        """
        # Get output - handle different YOLO output formats
        if len(model_outputs[0].shape) == 3:
            # YOLOv5/v8 format: [1, num_boxes, 85]
            outputs = np.transpose(np.squeeze(model_outputs[0]), (1, 0))
        else:
            # Other formats
            outputs = np.squeeze(model_outputs[0])
            if len(outputs.shape) == 2:
                outputs = np.transpose(outputs)
        
        boxes, scores, class_ids = [], [], []
        
        # Calculate gain factor for coordinate scaling
        gain = min(ratio[0], ratio[1])
        
        # Process detections
        num_detections = outputs.shape[0] if len(outputs.shape) > 1 else 0
        
        for i in range(num_detections):
            detection = outputs[i]
            
            # Extract class scores (skip first 4 coordinates + confidence)
            if len(detection) > 5:
                confidence = detection[4]
                class_scores = detection[5:]
            else:
                # Handle models without separate confidence score
                class_scores = detection[4:]
                confidence = np.max(class_scores)
            
            max_score = np.max(class_scores)
            
            if max_score >= self.config["confidence_threshold"]:
                class_id = np.argmax(class_scores)
                
                # Get bounding box coordinates
                x_center, y_center, width, height = detection[0], detection[1], detection[2], detection[3]
                
                # Adjust for padding and scale back to original image size
                x_center = (x_center - pad[0]) / gain
                y_center = (y_center - pad[1]) / gain
                width = width / gain
                height = height / gain
                
                # Convert center format to corner format
                left = int(x_center - width / 2)
                top = int(y_center - height / 2)
                right = int(x_center + width / 2)
                bottom = int(y_center + height / 2)
                
                # Clip to image boundaries
                left = max(0, left)
                top = max(0, top)
                right = min(self.img_width, right)
                bottom = min(self.img_height, bottom)
                
                boxes.append([left, top, right - left, bottom - top])  # [x, y, w, h]
                scores.append(float(max_score))
                class_ids.append(int(class_id))
        
        # Apply Non-Maximum Suppression using OpenCV
        detections = []
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(
                boxes, 
                scores, 
                self.config["confidence_threshold"], 
                self.config["nms_threshold"]
            )
            
            # Format final results
            if len(indices) > 0:
                # Handle different OpenCV NMS return formats
                if isinstance(indices, np.ndarray):
                    indices = indices.flatten()
                
                for i in indices:
                    if isinstance(i, (list, np.ndarray)):
                        i = i[0]
                    
                    box = boxes[i]
                    score = scores[i]
                    class_id = class_ids[i]
                    
                    detection = {
                        "bbox": [float(x) for x in box],  # [x, y, width, height]
                        "confidence": float(score),
                        "class_id": int(class_id),
                        "class_name": self.config["class_names"][class_id] if class_id < len(self.config["class_names"]) else f"class_{class_id}"
                    }
                    detections.append(detection)
        
        return {
            "detections": detections,
            "num_detections": len(detections),
            "model_type": "yolo_openvino"
        }
    
    def predict(self, input_data: Any) -> Dict[str, Any]:
        """Run complete YOLO prediction pipeline using OpenVINO
        
        Args:
            input_data: Input data (image path or numpy array)
            
        Returns:
            Dictionary containing detection results
        """
        if self.compiled_model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Preprocess input
        preprocessed_data, ratio, pad = self.preprocess(input_data)
        
        # Run inference
        model_outputs = self._run_inference(preprocessed_data)
        
        # Postprocess outputs
        results = self.postprocess(model_outputs, ratio, pad)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the YOLO OpenVINO model
        
        Returns:
            Dictionary containing model metadata
        """
        base_info = super().get_model_info()
        
        # Add YOLO-specific information
        yolo_info = {
            "model_type": "yolo_openvino",
            "input_size": self.config.get("input_size", 640),
            "confidence_threshold": self.config["confidence_threshold"],
            "nms_threshold": self.config["nms_threshold"],
            "num_classes": len(self.config["class_names"])
        }
        
        base_info.update(yolo_info)
        return base_info