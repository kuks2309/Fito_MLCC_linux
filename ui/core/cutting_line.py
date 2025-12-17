#!/usr/bin/env python3
"""
절단선 추출 알고리즘
MLCC Index 검출 결과 기반 절단선 계산
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional


class CuttingLineExtractor:
    """절단선 추출기"""

    def __init__(self, offset: int = 0):
        """
        Args:
            offset: 절단선 Y 오프셋 (픽셀)
        """
        self.offset = offset

    def extract_from_detections(self, boxes: List[List[float]],
                                 masks: List[np.ndarray] = None,
                                 image_shape: Tuple[int, int] = None) -> Optional[int]:
        """
        검출 결과에서 절단선 Y 좌표 추출

        Args:
            boxes: [[x1, y1, x2, y2], ...] 형태의 바운딩 박스 리스트
            masks: 세그멘테이션 마스크 리스트 (옵션)
            image_shape: (height, width)

        Returns:
            절단선 Y 좌표 또는 None
        """
        if not boxes:
            return None

        if masks and len(masks) > 0:
            # 마스크 기반 절단선 계산
            return self._extract_from_masks(masks, boxes)
        else:
            # 바운딩 박스 기반 절단선 계산
            return self._extract_from_boxes(boxes)

    def _extract_from_boxes(self, boxes: List[List[float]]) -> int:
        """바운딩 박스 중심점 기반 절단선 계산"""
        center_ys = []
        for box in boxes:
            x1, y1, x2, y2 = box
            center_y = (y1 + y2) / 2
            center_ys.append(center_y)

        # 중앙값 사용 (outlier에 강건함)
        cutting_y = int(np.median(center_ys)) + self.offset
        return cutting_y

    def _extract_from_masks(self, masks: List[np.ndarray],
                            boxes: List[List[float]]) -> int:
        """마스크 기반 절단선 계산 (더 정밀)"""
        center_ys = []

        for mask, box in zip(masks, boxes):
            if mask is None:
                # 마스크 없으면 박스 사용
                x1, y1, x2, y2 = box
                center_ys.append((y1 + y2) / 2)
            else:
                # 마스크 중심 계산
                y_coords, x_coords = np.where(mask)
                if len(y_coords) > 0:
                    center_y = np.mean(y_coords)
                    center_ys.append(center_y)
                else:
                    x1, y1, x2, y2 = box
                    center_ys.append((y1 + y2) / 2)

        if not center_ys:
            return None

        cutting_y = int(np.median(center_ys)) + self.offset
        return cutting_y

    def extract_with_ransac(self, masks: List[np.ndarray],
                            boxes: List[List[float]],
                            image_shape: Tuple[int, int]) -> Tuple[Optional[int], Optional[float]]:
        """
        RANSAC 기반 정밀 절단선 추출

        Args:
            masks: 마스크 리스트
            boxes: 바운딩 박스 리스트
            image_shape: (height, width)

        Returns:
            (절단선 Y 좌표, 기울기) 또는 (None, None)
        """
        if not boxes:
            return None, None

        # 각 검출의 중심점 수집
        points = []
        for mask, box in zip(masks, boxes):
            if mask is not None:
                y_coords, x_coords = np.where(mask)
                if len(y_coords) > 0:
                    cx = np.mean(x_coords)
                    cy = np.mean(y_coords)
                    points.append([cx, cy])
            else:
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                points.append([cx, cy])

        if len(points) < 2:
            if points:
                return int(points[0][1]) + self.offset, 0.0
            return None, None

        points = np.array(points, dtype=np.float32)

        # RANSAC으로 라인 피팅
        line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = line.flatten()

        # 이미지 중앙에서의 Y 좌표 계산
        img_center_x = image_shape[1] / 2
        if abs(vx) > 1e-6:
            t = (img_center_x - x0) / vx
            cutting_y = int(y0 + t * vy) + self.offset
        else:
            cutting_y = int(y0) + self.offset

        # 기울기 계산 (degree)
        slope = np.arctan2(vy, vx) * 180 / np.pi

        return cutting_y, slope

    def draw_cutting_line(self, image: np.ndarray, cutting_y: int,
                          color: Tuple[int, int, int] = (0, 0, 255),
                          thickness: int = 2,
                          label: bool = True) -> np.ndarray:
        """절단선 그리기"""
        img = image.copy()
        h, w = img.shape[:2]

        # 수평선 그리기
        cv2.line(img, (0, cutting_y), (w, cutting_y), color, thickness)

        # 라벨 표시
        if label:
            text = f"Cutting Line: Y={cutting_y}"
            cv2.putText(img, text, (10, cutting_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return img

    def draw_cutting_line_with_slope(self, image: np.ndarray,
                                     cutting_y: int, slope: float,
                                     color: Tuple[int, int, int] = (0, 0, 255),
                                     thickness: int = 2) -> np.ndarray:
        """기울기 있는 절단선 그리기"""
        img = image.copy()
        h, w = img.shape[:2]

        # 라디안 변환
        angle_rad = slope * np.pi / 180
        center_x = w / 2

        # 양 끝점 계산
        x1 = 0
        y1 = int(cutting_y + (center_x - x1) * np.tan(angle_rad))
        x2 = w
        y2 = int(cutting_y + (center_x - x2) * np.tan(angle_rad))

        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

        # 라벨
        text = f"Y={cutting_y}, Slope={slope:.2f}deg"
        cv2.putText(img, text, (10, min(y1, y2) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return img
