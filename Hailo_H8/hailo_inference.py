#!/usr/bin/env python3
"""
Hailo-8 YOLOv8-seg 추론 모듈 (C++ 백엔드 사용)
MLCC Index 검출용 독립 inference 클래스
"""
import numpy as np
import cv2
import os
import sys

# C++ 모듈 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import hailo_yolov8_cpp
    USE_CPP = True
except ImportError:
    print("Warning: C++ module not found, using Python fallback")
    USE_CPP = False

# 기본 설정
DEFAULT_HEF_PATH = "/home/amap/Project/FITO2026/MLCC_Index/Hailo_H8/fito_best.hef"
INPUT_SHAPE = (640, 544)  # (height, width)

# 클래스 정보
CLASS_NAMES = ['index0']
COLORS = [(0, 255, 0)]


class HailoInference:
    """Hailo-8 YOLOv8-seg 추론 클래스 (C++ 백엔드)"""

    def __init__(self, hef_path: str = DEFAULT_HEF_PATH,
                 input_shape: tuple = INPUT_SHAPE,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        """
        Args:
            hef_path: HEF 모델 파일 경로
            input_shape: 입력 이미지 크기 (height, width)
            conf_threshold: Confidence threshold
            iou_threshold: IOU threshold for NMS
        """
        self.hef_path = hef_path
        self.input_shape = input_shape
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.initialized = False

        if USE_CPP:
            self.engine = hailo_yolov8_cpp.HailoYolov8Seg(
                hef_path, input_shape[0], input_shape[1]
            )
            self.engine.set_conf_threshold(conf_threshold)
            self.engine.set_iou_threshold(iou_threshold)
        else:
            self.engine = None

    def initialize(self) -> bool:
        """HEF 로드 및 디바이스 초기화"""
        if not USE_CPP:
            print("Error: C++ module not available")
            return False

        if not os.path.exists(self.hef_path):
            print(f"Error: HEF file not found: {self.hef_path}")
            return False

        result = self.engine.initialize()
        self.initialized = result
        return result

    def cleanup(self):
        """리소스 정리"""
        self.initialized = False
        if USE_CPP and self.engine:
            self.engine.cleanup()

    def set_conf_threshold(self, value: float):
        """Confidence threshold 설정"""
        self.conf_threshold = value
        if USE_CPP and self.engine:
            self.engine.set_conf_threshold(value)

    def set_iou_threshold(self, value: float):
        """IOU threshold 설정"""
        self.iou_threshold = value
        if USE_CPP and self.engine:
            self.engine.set_iou_threshold(value)

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
                'original_size': (height, width),
                'count': int
            }
        """
        if not self.initialized:
            print("Error: Engine not initialized")
            return None

        if not USE_CPP:
            return None

        try:
            result = self.engine.infer_dict(image)

            # boxes를 list로 변환 (tuple에서)
            boxes = []
            for box in result['boxes']:
                boxes.append([box[0], box[1], box[2], box[3]])
            result['boxes'] = boxes

            return result

        except Exception as e:
            print(f"Inference failed: {str(e)}")
            return None

    def draw_results(self, image: np.ndarray, result: dict,
                     show_boxes: bool = True,
                     show_masks: bool = True,
                     show_labels: bool = True,
                     show_cutting_line: bool = False,
                     cutting_y: int = None) -> np.ndarray:
        """
        결과 시각화

        Args:
            image: 원본 이미지
            result: infer() 결과
            show_boxes: 바운딩 박스 표시 여부
            show_masks: 세그멘테이션 마스크 표시 여부
            show_labels: 라벨 표시 여부
            show_cutting_line: 커팅 라인 표시 여부
            cutting_y: 커팅 라인 Y 좌표

        Returns:
            결과 이미지
        """
        if result is None:
            return image

        img = image.copy()
        boxes = result['boxes']
        scores = result['scores']
        class_ids = result['class_ids']
        masks = result['masks']

        # C++ 버전은 이미 원본 좌표로 반환됨 (스케일링 불필요)
        for box, score, class_id, mask in zip(boxes, scores, class_ids, masks):
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            color = COLORS[class_id % len(COLORS)]

            # 마스크 표시
            if show_masks and mask is not None:
                colored_mask = np.zeros_like(img)
                colored_mask[mask] = color
                img = cv2.addWeighted(img, 1, colored_mask, 0.4, 0)

            # 바운딩 박스 표시
            if show_boxes:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 8)

            # 라벨 표시
            if show_labels:
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
                label = f"{class_name}: {score:.2f}"
                cv2.putText(img, label, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 3.2, color, 8)

        # 커팅 라인 표시 (각 검출마다 개별 라인)
        if show_cutting_line and cutting_y is not None:
            h, w = img.shape[:2]
            if isinstance(cutting_y, list):
                # 여러 개의 커팅 라인
                for i, cy in enumerate(cutting_y):
                    cv2.line(img, (0, cy), (w, cy), (0, 0, 255), 4)
                    cv2.putText(img, f"Cut[{i}] Y={cy}", (20, cy - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
            else:
                # 단일 커팅 라인 (하위 호환)
                cv2.line(img, (0, cutting_y), (w, cutting_y), (0, 0, 255), 4)
                cv2.putText(img, f"Cutting Y={cutting_y}", (20, cutting_y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

        return img


# 테스트용
if __name__ == "__main__":
    import glob

    # 테스트 이미지 경로
    TEST_IMAGE_DIR = "/home/amap/Project/FITO2026/MLCC_Index/images/Side_Left"

    print(f"Using C++ backend: {USE_CPP}")

    # Inference 객체 생성 및 초기화
    inference = HailoInference()
    if not inference.initialize():
        print("Failed to initialize")
        exit(1)

    # 이미지 찾기
    image_files = glob.glob(os.path.join(TEST_IMAGE_DIR, "*.bmp"))
    if not image_files:
        print(f"No images found in {TEST_IMAGE_DIR}")
        inference.cleanup()
        exit(1)

    print(f"Found {len(image_files)} images")

    # 첫 번째 이미지로 테스트
    test_image = cv2.imread(image_files[0])
    if test_image is not None:
        result = inference.infer(test_image)
        if result:
            print(f"Detected: {result['count']} objects")
            print(f"Inference time: {result.get('inference_time_ms', 'N/A')} ms")
            print(f"Total time: {result.get('total_time_ms', 'N/A')} ms")
            for i, (box, score) in enumerate(zip(result['boxes'], result['scores'])):
                print(f"  [{i}] box={box}, score={score:.3f}")

            # 결과 이미지 표시
            result_img = inference.draw_results(test_image, result)
            cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
            cv2.imshow("Result", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    inference.cleanup()
