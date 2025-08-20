"""YOLO object detection inferencer based on Ultralytics implementation"""

import onnxruntime as ort
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple
import logging

from .base import BaseInferencer
from ..utils import ImageProcessor
from ..exceptions import (
    ModelLoadError,
    InferenceFailedError,
    InputInvalidFormatError,
    create_input_error,
    ErrorCode
)

logger = logging.getLogger(__name__)


class YOLOInferencer(BaseInferencer):
    """YOLO object detection inferencer"""
    
    def __init__(self, model_path: Union[str, Path], config: Optional[Dict] = None):
        """Initialize YOLO inferencer
        
        Args:
            model_path: Path to YOLO ONNX model file
            config: Optional configuration dictionary
        """
        # Default YOLO config
        default_config = {
            "input_size": 640,
            "confidence_threshold": 0.25,
            "nms_threshold": 0.45,
            "max_detections": 100,
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
        
        if config:
            default_config.update(config)
            
        super().__init__(model_path, default_config)
    
    def _load_model(self) -> None:
        """Load ONNX model with YOLO-optimized settings"""
        try:
            # Try GPU first, fallback to CPU
            available_providers = ort.get_available_providers()
            providers = []
            
            if "CUDAExecutionProvider" in available_providers:
                providers.append("CUDAExecutionProvider")
            elif "OpenVINOExecutionProvider" in available_providers:
                providers.append("OpenVINOExecutionProvider")
            
            providers.append("CPUExecutionProvider")
            
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=session_options,
                providers=providers
            )
            
            # Get model metadata
            model_inputs = self.session.get_inputs()
            self.input_name = model_inputs[0].name
            self.input_shape = model_inputs[0].shape
            
            # Get input dimensions
            self.input_width = self.input_shape[2] if len(self.input_shape) > 2 else 640
            self.input_height = self.input_shape[3] if len(self.input_shape) > 3 else 640
            
            logger.info(f"Loaded YOLO model: {self.model_path}")
            logger.info(f"Input shape: {self.input_shape}")
            logger.info(f"Providers: {self.session.get_providers()}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model {self.model_path}: {e}")
            raise ModelLoadError(
                model_path=str(self.model_path),
                runtime="onnx",
                original_error=e,
                context={
                    "available_providers": ort.get_available_providers(),
                    "model_extension": self.model_path.suffix
                }
            )
    
    def _letterbox(self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """Letterbox resize image while maintaining aspect ratio
        
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
        """YOLO preprocessing: letterbox resize + normalize + transpose
        
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
            raise InputInvalidFormatError(
                input_path=str(input_data) if hasattr(input_data, '__str__') else "unknown",
                expected_formats=["image_path", "numpy_array"],
                context={"actual_type": type(input_data).__name__}
            )
        
        # Store original image dimensions
        self.img_height, self.img_width = img.shape[:2]
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Letterbox resize
        img, ratio, pad = self._letterbox(img, (self.input_height, self.input_width))
        
        # Normalize to [0, 1]
        img = np.array(img) / 255.0
        
        # Transpose HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0).astype(np.float32)
        
        return img, ratio, pad
    
    def _run_inference(self, preprocessed_data: np.ndarray) -> List[np.ndarray]:
        """Run YOLO model inference
        
        Args:
            preprocessed_data: Preprocessed input data
            
        Returns:
            Raw model outputs
        """
        try:
            # Run inference
            outputs = self.session.run(None, {self.input_name: preprocessed_data})
            
            logger.debug(f"YOLO inference completed, output shapes: {[out.shape for out in outputs]}")
            return outputs
            
        except Exception as e:
            logger.error(f"YOLO inference failed: {e}")
            raise InferenceFailedError(
                model_type="yolo",
                original_error=e,
                context={
                    "input_shape": preprocessed_data.shape if hasattr(preprocessed_data, 'shape') else "unknown",
                    "model_providers": self.session.get_providers() if self.session else "unknown"
                }
            )
    
    def postprocess(self, model_outputs: List[np.ndarray], ratio: Tuple[float, float], pad: Tuple[int, int]) -> Dict[str, Any]:
        """YOLO postprocessing: NMS + format results
        
        Args:
            model_outputs: Raw outputs from model inference
            ratio: Scale ratio from letterbox resize
            pad: Padding from letterbox resize
            
        Returns:
            Dictionary containing processed results with bounding boxes
        """
        # Get output
        outputs = np.transpose(np.squeeze(model_outputs[0]))
        rows = outputs.shape[0]
        
        boxes, scores, class_ids = [], [], []
        
        # Calculate gain factor
        gain = min(self.input_height / self.img_height, self.input_width / self.img_width)
        
        # Adjust for padding
        outputs[:, 0] -= pad[1]  # x
        outputs[:, 1] -= pad[0]  # y
        
        # Process detections
        for i in range(rows):
            classes_scores = outputs[i][4:]  # class probabilities
            max_score = np.amax(classes_scores)
            
            if max_score >= self.config["confidence_threshold"]:
                class_id = np.argmax(classes_scores)
                
                # Get bounding box coordinates
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                
                # Scale back to original image size
                left = int((x - w / 2) / gain)
                top = int((y - h / 2) / gain)
                width = int(w / gain)
                height = int(h / gain)
                
                boxes.append([left, top, width, height])
                scores.append(max_score)
                class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression using OpenCV
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(
                boxes, 
                scores, 
                self.config["confidence_threshold"], 
                self.config["nms_threshold"]
            )
            
            # Format final results
            detections = []
            if len(indices) > 0:
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
        else:
            detections = []
        
        return {
            "detections": detections,
            "num_detections": len(detections),
            "model_type": "yolo"
        }
    
    def predict(self, input_data: Any) -> Dict[str, Any]:
        """Run complete YOLO prediction pipeline
        
        Args:
            input_data: Input data (image path or numpy array)
            
        Returns:
            Dictionary containing detection results
        """
        # Model check is handled by parent class predict() method
        
        # Preprocess input
        preprocessed_data, ratio, pad = self.preprocess(input_data)
        
        # Run inference
        model_outputs = self._run_inference(preprocessed_data)
        
        # Postprocess outputs
        results = self.postprocess(model_outputs, ratio, pad)
        
        return results