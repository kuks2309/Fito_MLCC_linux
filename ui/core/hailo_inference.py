#!/usr/bin/env python3
"""
Hailo-8 YOLOv8-seg 추론 엔진
MLCC Index 검출용
"""
import numpy as np
import cv2
from PyQt5.QtCore import QObject, pyqtSignal

try:
    from hailo_platform import HEF, VDevice, FormatType
    from hailo_platform.pyhailort.pyhailort import InferVStreams, InputVStreamParams, OutputVStreamParams
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("Warning: hailo_platform not available")


class HailoInference(QObject):
    """Hailo-8 YOLOv8-seg 추론 엔진"""

    inference_done = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    # 클래스 정보
    CLASS_NAMES = ['index0']
    COLORS = [(0, 255, 0)]

    def __init__(self, hef_path: str, input_shape: tuple = (640, 544)):
        super().__init__()
        self.hef_path = hef_path
        self.input_shape = input_shape  # (height, width)
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45

        self.hef = None
        self.target = None
        self.network_group = None
        self.infer_pipeline = None
        self.input_vstream_infos = None
        self.initialized = False

    def initialize(self) -> bool:
        """HEF 로드 및 디바이스 초기화"""
        if not HAILO_AVAILABLE:
            self.error_occurred.emit("Hailo platform not available")
            return False

        try:
            self.hef = HEF(self.hef_path)
            self.input_vstream_infos = self.hef.get_input_vstream_infos()
            self.target = VDevice()
            self.network_group = self.target.configure(self.hef)[0]

            input_params = InputVStreamParams.make(self.network_group, format_type=FormatType.UINT8)
            output_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)

            self.infer_pipeline = InferVStreams(self.network_group, input_params, output_params)
            self.infer_pipeline.__enter__()
            self.network_group.activate().__enter__()

            self.initialized = True
            return True

        except Exception as e:
            self.error_occurred.emit(f"Initialization failed: {str(e)}")
            return False

    def cleanup(self):
        """리소스 정리"""
        self.initialized = False
        if self.infer_pipeline:
            try:
                self.infer_pipeline.__exit__(None, None, None)
            except:
                pass
        if self.target:
            try:
                self.target.__exit__(None, None, None)
            except:
                pass

    def set_conf_threshold(self, value: float):
        """Confidence threshold 설정"""
        self.conf_threshold = value

    def set_iou_threshold(self, value: float):
        """IOU threshold 설정"""
        self.iou_threshold = value

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리"""
        img_resized = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        return np.expand_dims(img_rgb, axis=0).astype(np.uint8)

    def infer(self, image: np.ndarray) -> dict:
        """
        단일 이미지 추론

        Args:
            image: BGR 이미지 (numpy array)

        Returns:
            dict: {
                'boxes': list of [x1, y1, x2, y2],
                'scores': list of float,
                'class_ids': list of int,
                'masks': list of binary masks,
                'original_size': (height, width)
            }
        """
        if not self.initialized:
            self.error_occurred.emit("Engine not initialized")
            return None

        try:
            original_size = image.shape[:2]
            input_data = self.preprocess(image)

            input_dict = {self.input_vstream_infos[0].name: input_data}
            output_dict = self.infer_pipeline.infer(input_dict)

            boxes, scores, class_ids, mask_coeffs, proto = self._process_output(output_dict)

            # Generate masks
            masks = []
            for box, mask_coeff in zip(boxes, mask_coeffs):
                mask = self._generate_mask(proto, mask_coeff, box, original_size)
                masks.append(mask)

            result = {
                'boxes': boxes,
                'scores': scores,
                'class_ids': class_ids,
                'masks': masks,
                'original_size': original_size
            }

            self.inference_done.emit(result)
            return result

        except Exception as e:
            self.error_occurred.emit(f"Inference failed: {str(e)}")
            return None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _process_output(self, outputs):
        """YOLOv8-seg 출력 디코딩"""
        # Find model prefix
        model_prefix = None
        for key in outputs.keys():
            if 'conv44' in key:
                model_prefix = key.rsplit('/conv44', 1)[0]
                break

        if model_prefix is None:
            return [], [], [], [], None

        scales = [
            {'bbox': outputs[f'{model_prefix}/conv44'], 'cls': outputs[f'{model_prefix}/conv45'],
             'mask': outputs[f'{model_prefix}/conv46'], 'stride': 8},
            {'bbox': outputs[f'{model_prefix}/conv60'], 'cls': outputs[f'{model_prefix}/conv61'],
             'mask': outputs[f'{model_prefix}/conv62'], 'stride': 16},
            {'bbox': outputs[f'{model_prefix}/conv73'], 'cls': outputs[f'{model_prefix}/conv74'],
             'mask': outputs[f'{model_prefix}/conv75'], 'stride': 32}
        ]

        proto = outputs[f'{model_prefix}/conv48'][0]

        all_boxes, all_scores, all_class_ids, all_mask_coeffs = [], [], [], []

        for scale in scales:
            bbox_out = scale['bbox'][0]
            cls_out = scale['cls'][0]
            mask_out = scale['mask'][0]
            stride = scale['stride']
            h, w = bbox_out.shape[:2]

            cls_probs = cls_out
            max_cls_probs = np.max(cls_probs, axis=-1)
            class_ids = np.argmax(cls_probs, axis=-1)
            mask = max_cls_probs > self.conf_threshold

            if not np.any(mask):
                continue

            y_indices, x_indices = np.where(mask)

            for y, x in zip(y_indices, x_indices):
                score = max_cls_probs[y, x]
                class_id = class_ids[y, x]
                bbox_raw = bbox_out[y, x]

                # DFL decoding
                dfl_vals = []
                for i in range(4):
                    vals = bbox_raw[i*16:(i+1)*16]
                    vals = np.exp(vals - np.max(vals))
                    vals = vals / vals.sum()
                    dfl_vals.append(np.sum(vals * np.arange(16)))

                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride
                x1 = cx - dfl_vals[0] * stride
                y1 = cy - dfl_vals[1] * stride
                x2 = cx + dfl_vals[2] * stride
                y2 = cy + dfl_vals[3] * stride

                all_boxes.append([x1, y1, x2 - x1, y2 - y1])
                all_scores.append(float(score))
                all_class_ids.append(int(class_id))
                all_mask_coeffs.append(mask_out[y, x])

        if len(all_boxes) == 0:
            return [], [], [], [], proto

        # NMS
        boxes = np.array(all_boxes)
        scores = np.array(all_scores)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(),
                                    self.conf_threshold, self.iou_threshold)
        indices = indices.flatten() if len(indices) > 0 else []

        final_boxes, final_scores, final_class_ids, final_mask_coeffs = [], [], [], []
        for i in indices:
            x, y, w, h = boxes[i]
            final_boxes.append([x, y, x + w, y + h])
            final_scores.append(scores[i])
            final_class_ids.append(all_class_ids[i])
            final_mask_coeffs.append(all_mask_coeffs[i])

        return final_boxes, final_scores, final_class_ids, final_mask_coeffs, proto

    def _generate_mask(self, proto, mask_coeff, box, img_shape):
        """세그멘테이션 마스크 생성"""
        mask = np.tensordot(proto, mask_coeff, axes=([2], [0]))
        mask = self._sigmoid(mask)
        mask = cv2.resize(mask, (self.input_shape[1], self.input_shape[0]))

        x1, y1, x2, y2 = [int(c) for c in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.input_shape[1], x2), min(self.input_shape[0], y2)

        box_mask = np.zeros((self.input_shape[0], self.input_shape[1]), dtype=np.float32)
        box_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        box_mask = cv2.resize(box_mask, (img_shape[1], img_shape[0]))

        return box_mask > 0.5

    def draw_results(self, image: np.ndarray, result: dict,
                     show_boxes: bool = True, show_masks: bool = True) -> np.ndarray:
        """결과 시각화"""
        if result is None:
            return image

        img = image.copy()
        boxes = result['boxes']
        scores = result['scores']
        class_ids = result['class_ids']
        masks = result['masks']
        original_size = result['original_size']

        scale_x = original_size[1] / self.input_shape[1]
        scale_y = original_size[0] / self.input_shape[0]

        for box, score, class_id, mask in zip(boxes, scores, class_ids, masks):
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)

            color = self.COLORS[class_id % len(self.COLORS)]

            if show_masks and mask is not None:
                colored_mask = np.zeros_like(img)
                colored_mask[mask] = color
                img = cv2.addWeighted(img, 1, colored_mask, 0.4, 0)

            if show_boxes:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                class_name = self.CLASS_NAMES[class_id] if class_id < len(self.CLASS_NAMES) else f"class_{class_id}"
                label = f"{class_name}: {score:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return img
