#!/usr/bin/env python3
"""
OpenCV-based Blob Analyzer for MLCC Index Detection
Halcon Blob 분석 알고리즘을 OpenCV로 구현
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import time


class BlobAnalyzer:
    """
    Halcon 스타일의 Blob 분석기 (OpenCV 구현)

    주요 기능:
    1. 이진화 및 형태학적 처리
    2. 연결 성분 분석 (Connection)
    3. 크기 기반 필터링 (SelectShape)
    4. Projection 기반 분리
    5. Index Pitch 검증
    """

    def __init__(self):
        # 기본 파라미터 (Halcon 코드 기반)
        self.threshold_value = 40  # 이진화 임계값 (Halcon: nBinaryLevel)

        # Gain/Offset 전처리 (Halcon: ScaleImage)
        self.use_gain_offset = False
        self.gain = 1.0           # Halcon: dGainValue
        self.offset = 0           # Halcon: nOffsetValue

        # 형태학 파라미터 (Halcon: OpeningRectangle1, ClosingRectangle1)
        self.use_opening = False
        self.use_closing = False
        self.opening_x = 3        # Halcon: nOpenValueX
        self.opening_y = 3        # Halcon: nOpenValueY
        self.closing_x = 3        # Halcon: nCloseValueX
        self.closing_y = 3        # Halcon: nCloseValueY

        # 크기 필터 (예상 Index 크기 기준) - MLCC Index 실제 크기
        self.min_area = 3000      # 최소 면적
        self.max_area = 100000    # 최대 면적
        self.expected_width = 80   # 예상 Index 폭
        self.expected_height = 120  # 예상 Index 높이
        self.width_tolerance = 0.3  # 폭 허용 범위 (30% ~ 300%)
        self.height_tolerance = 0.3  # 높이 허용 범위 (30% ~ 300%)

        # Index Pitch 파라미터
        self.index_pitch = 500.0  # Index 간격 (픽셀)
        self.pitch_tolerance = 0.33  # Pitch 허용 오차 (33%)

        # 커버 제거 옵션
        self.use_remove_cover = True
        self.cover_filter_size = 400  # 커버 제거용 필터 크기

    def set_expected_size(self, width: int, height: int):
        """예상 Index 크기 설정"""
        self.expected_width = width
        self.expected_height = height

    def set_index_pitch(self, pitch: float):
        """Index Pitch 설정"""
        self.index_pitch = pitch

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        이미지 전처리 (Halcon의 HV_ImgProcess_Prepare와 동일)

        Halcon 처리 순서:
        1. ScaleImage (Gain/Offset 적용) - 선택적
        2. Threshold (고정 임계값 이진화)
        3. OpeningRectangle1 (노이즈 제거) - 선택적
        4. ClosingRectangle1 (구멍 메우기) - 선택적

        Args:
            image: 입력 BGR 이미지

        Returns:
            이진화된 마스크
        """
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. Gain/Offset 전처리 (Halcon: ScaleImage)
        # output = input * gain + offset
        if self.use_gain_offset:
            gray = cv2.convertScaleAbs(gray, alpha=self.gain, beta=self.offset)

        # 2. 고정 임계값 이진화 (Halcon: Threshold)
        # Halcon: Threshold(image, region, nThreshold, 256)
        # 밝은 영역 선택: threshold_value ~ 255
        _, binary = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)

        # 3. Opening (노이즈 제거) - Halcon: OpeningRectangle1
        if self.use_opening:
            kernel_open = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (max(self.opening_x, 1), max(self.opening_y, 1))
            )
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

        # 4. Closing (구멍 메우기) - Halcon: ClosingRectangle1
        if self.use_closing:
            kernel_close = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (max(self.closing_x, 1), max(self.closing_y, 1))
            )
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

        return binary

    def remove_cover(self, binary: np.ndarray) -> np.ndarray:
        """
        커버 영역 제거 (Halcon의 RemoveCover 로직과 동일)

        Halcon 원본 코드:
        ClosingCircle(ho_OrgRegion, &LongRegion, 20.5);
        OpeningRectangle1(LongRegion, &LongRegion, 10, iFilterSize);
        DilationCircle(LongRegion, &LongRegion, 20.5);
        Difference(ho_OrgRegion, LongRegion, &diffRegion);  // 원본에서 빼기
        OpeningCircle(diffRegion, &ho_OrgRegion, 6.6);

        Args:
            binary: 이진화된 이미지 (ho_OrgRegion)

        Returns:
            커버가 제거된 이진 이미지
        """
        if not self.use_remove_cover:
            return binary

        # 원본 보존 (Difference에서 사용)
        ho_OrgRegion = binary.copy()

        # 2-1. ClosingCircle(ho_OrgRegion, &LongRegion, 20.5)
        # 원본에 Closing 적용 → LongRegion
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))  # 20.5 * 2 = 41
        long_region = cv2.morphologyEx(ho_OrgRegion, cv2.MORPH_CLOSE, kernel_close)

        # 2-2. OpeningRectangle1(LongRegion, &LongRegion, 10, iFilterSize)
        # 세로로 긴 구조(커버) 추출
        # Halcon OpeningRectangle1(width, height) → OpenCV (width, height)
        kernel_rect = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (10, self.cover_filter_size)
        )
        long_region = cv2.morphologyEx(long_region, cv2.MORPH_OPEN, kernel_rect)

        # 2-3. DilationCircle(LongRegion, &LongRegion, 20.5)
        # 커버 영역 확장
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))  # 20.5 * 2 = 41
        long_region = cv2.dilate(long_region, kernel_dilate)

        # 2-4. Difference(ho_OrgRegion, LongRegion, &diffRegion)
        # 원본(ho_OrgRegion)에서 커버(LongRegion) 빼기
        diff_region = cv2.subtract(ho_OrgRegion, long_region)

        # 2-5. OpeningCircle(diffRegion, &ho_OrgRegion, 6.6)
        # 노이즈 제거
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))  # 6.6 * 2 = 13
        result = cv2.morphologyEx(diff_region, cv2.MORPH_OPEN, kernel_open)

        return result

    def horizontal_projection_split(self, binary: np.ndarray) -> List[Tuple[int, int]]:
        """
        수평 Projection을 이용한 영역 분리 (Halcon의 GrayProjections)

        Args:
            binary: 이진화된 이미지

        Returns:
            (start_y, end_y) 튜플의 리스트
        """
        # 수평 방향 프로젝션 (각 행의 백색 픽셀 합계)
        h_proj = np.sum(binary, axis=1).astype(np.float32)

        if np.max(h_proj) == 0:
            return []

        # 임계값 적용 (최대값의 50%)
        threshold = np.max(h_proj) * 0.5

        # 연속 영역 찾기
        regions = []
        in_region = False
        start_y = 0

        for y in range(len(h_proj)):
            if h_proj[y] > threshold:
                if not in_region:
                    start_y = y
                    in_region = True
            else:
                if in_region:
                    regions.append((start_y, y))
                    in_region = False

        # 마지막 영역 처리
        if in_region:
            regions.append((start_y, len(h_proj)))

        return regions

    def find_connected_components(self, binary: np.ndarray) -> List[Dict]:
        """
        연결 성분 분석 (Halcon의 Connection)

        Args:
            binary: 이진화된 이미지

        Returns:
            검출된 Blob 정보 리스트
        """
        # 연결 성분 분석
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        blobs = []
        for i in range(1, num_labels):  # 0은 배경
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]

            blobs.append({
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'area': area,
                'center_x': cx,
                'center_y': cy,
                'label': i
            })

        return blobs

    def filter_by_size(self, blobs: List[Dict]) -> List[Dict]:
        """
        크기 기반 필터링 (Halcon의 SelectShape)

        Args:
            blobs: Blob 리스트

        Returns:
            필터링된 Blob 리스트
        """
        # 면적 기반 필터링
        filtered = []
        for blob in blobs:
            area = blob['area']
            w = blob['width']
            h = blob['height']

            # 면적 필터
            if area < self.min_area or area > self.max_area:
                continue

            # 종횡비 필터 (너무 길쭉한 것 제외)
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            if aspect_ratio > 5.0:  # 종횡비 5 이상은 제외
                continue

            filtered.append(blob)

        return filtered

    def validate_pitch(self, blobs: List[Dict], find_first: bool = True) -> Tuple[int, bool]:
        """
        Index Pitch 검증 (Halcon 코드의 핵심 로직)
        연속된 두 Index 간의 거리가 설정된 Pitch와 일치하는지 확인

        Args:
            blobs: Y좌표로 정렬된 Blob 리스트
            find_first: True면 첫 번째 유효 Index 찾기, False면 마지막

        Returns:
            (유효한 Index의 인덱스, 검증 성공 여부)
        """
        if len(blobs) < 2:
            return 0, False

        pitch_tol = self.index_pitch * self.pitch_tolerance

        for k in range(len(blobs) - 1):
            actual_pitch = abs(blobs[k]['center_y'] - blobs[k + 1]['center_y'])

            if (self.index_pitch - pitch_tol < actual_pitch < self.index_pitch + pitch_tol):
                return k, True

        return 0, False

    def analyze(self, image: np.ndarray, search_roi: Tuple[int, int, int, int] = None) -> Dict:
        """
        메인 분석 함수

        Args:
            image: 입력 BGR 이미지
            search_roi: 검색 영역 (x, y, w, h), None이면 전체 이미지

        Returns:
            분석 결과 딕셔너리
        """
        timing = {}  # 각 단계별 시간 저장

        h, w = image.shape[:2]

        # ROI 적용
        t0 = time.time()
        if search_roi is not None:
            rx, ry, rw, rh = search_roi
            roi_image = image[ry:ry+rh, rx:rx+rw]
            offset_x, offset_y = rx, ry
        else:
            roi_image = image
            offset_x, offset_y = 0, 0
        timing['roi'] = (time.time() - t0) * 1000

        # 1. 전처리 (이진화)
        t0 = time.time()
        binary = self.preprocess(roi_image)
        timing['preprocess'] = (time.time() - t0) * 1000

        # 2. 커버 제거 (옵션)
        t0 = time.time()
        if self.use_remove_cover:
            binary = self.remove_cover(binary)
        timing['remove_cover'] = (time.time() - t0) * 1000

        # 3. 연결 성분 분석
        t0 = time.time()
        blobs = self.find_connected_components(binary)
        timing['connected_comp'] = (time.time() - t0) * 1000

        # 4. 크기 기반 필터링
        t0 = time.time()
        filtered_blobs = self.filter_by_size(blobs)
        timing['filter'] = (time.time() - t0) * 1000

        # 5. Y 좌표로 정렬 (상단부터)
        filtered_blobs.sort(key=lambda b: b['center_y'])

        # 6. 결과 좌표 보정 (ROI offset 적용)
        for blob in filtered_blobs:
            blob['x'] += offset_x
            blob['y'] += offset_y
            blob['center_x'] += offset_x
            blob['center_y'] += offset_y

        # 7. Index Pitch 검증
        t0 = time.time()
        first_valid_idx, pitch_valid = self.validate_pitch(filtered_blobs)
        timing['validate_pitch'] = (time.time() - t0) * 1000

        # 전체 처리 시간
        total_time = sum(timing.values())

        # 결과 반환 (AI 결과와 호환되는 형식)
        boxes = []
        centers = []
        for blob in filtered_blobs:
            x1 = blob['x']
            y1 = blob['y']
            x2 = blob['x'] + blob['width']
            y2 = blob['y'] + blob['height']
            boxes.append([x1, y1, x2, y2])
            centers.append((blob['center_x'], blob['center_y']))

        return {
            'boxes': boxes,
            'scores': [1.0] * len(boxes),  # Blob은 score 없음, 1.0으로 설정
            'class_ids': [0] * len(boxes),
            'masks': [None] * len(boxes),  # Blob은 마스크 없음
            'centers': centers,
            'count': len(filtered_blobs),
            'blobs': filtered_blobs,
            'pitch_valid': pitch_valid,
            'process_time_ms': total_time,
            'timing': timing,  # 단계별 시간
            'binary_image': binary  # 디버그용
        }

    def draw_results(self, image: np.ndarray, result: Dict,
                     show_boxes: bool = True,
                     show_centers: bool = True,
                     show_cutting_line: bool = True,
                     cutting_y: List[int] = None) -> np.ndarray:
        """
        결과 시각화

        Args:
            image: 원본 이미지
            result: analyze() 결과
            show_boxes: 바운딩 박스 표시
            show_centers: 중심점 표시
            show_cutting_line: 커팅 라인 표시
            cutting_y: 커팅 라인 Y 좌표 리스트

        Returns:
            시각화된 이미지
        """
        if result is None:
            return image

        img = image.copy()
        boxes = result['boxes']
        blobs = result.get('blobs', [])

        # 모든 검출 표시
        for i, (box, blob) in enumerate(zip(boxes, blobs)):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = int(blob['center_x']), int(blob['center_y'])

            color = (255, 165, 0)  # 주황색
            thickness = 2

            # 바운딩 박스
            if show_boxes:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

                # 라벨
                label = f"Blob[{i}]"
                cv2.putText(img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            # 중심점
            if show_centers:
                cv2.drawMarker(img, (cx, cy), color,
                              cv2.MARKER_CROSS, 30, thickness)

        # 커팅 라인 표시
        if show_cutting_line and cutting_y is not None:
            h, w = img.shape[:2]
            if isinstance(cutting_y, list):
                for i, cy in enumerate(cutting_y):
                    cv2.line(img, (0, cy), (w, cy), (0, 0, 255), 4)
                    cv2.putText(img, f"Cut[{i}] Y={cy}", (20, cy - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
            else:
                cv2.line(img, (0, cutting_y), (w, cutting_y), (0, 0, 255), 4)
                cv2.putText(img, f"Cutting Y={cutting_y}", (20, cutting_y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

        # 정보 텍스트
        info_text = f"Blob Analysis: {result['count']} detected"
        if result.get('pitch_valid'):
            info_text += " (Pitch OK)"
        cv2.putText(img, info_text, (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 165, 0), 3)
        cv2.putText(img, f"Time: {result['process_time_ms']:.1f}ms", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 165, 0), 2)

        return img


# 테스트용
if __name__ == "__main__":
    import glob

    TEST_IMAGE_DIR = "/home/amap/Project/FITO2026/MLCC_Index/images/Side_Left"

    analyzer = BlobAnalyzer()

    # 파라미터 설정 (실제 MLCC Index 크기에 맞게 조정 필요)
    analyzer.set_expected_size(width=150, height=200)
    analyzer.set_index_pitch(500)  # 픽셀 단위

    # 이미지 찾기
    image_files = glob.glob(f"{TEST_IMAGE_DIR}/*.bmp")
    if not image_files:
        print(f"No images found in {TEST_IMAGE_DIR}")
        exit(1)

    print(f"Found {len(image_files)} images")

    # 첫 번째 이미지 테스트
    test_image = cv2.imread(image_files[0])
    if test_image is not None:
        print(f"Image shape: {test_image.shape}")

        result = analyzer.analyze(test_image)

        print(f"\n=== Results ===")
        print(f"Detected: {result['count']} blobs")
        print(f"Process time: {result['process_time_ms']:.1f} ms")
        print(f"Pitch valid: {result['pitch_valid']}")


        for i, blob in enumerate(result['blobs']):
            print(f"  [{i}] center=({blob['center_x']:.1f}, {blob['center_y']:.1f}), "
                  f"size={blob['width']}x{blob['height']}")

        # 결과 이미지 표시
        cutting_y = [int(b['center_y']) for b in result['blobs']]
        result_img = analyzer.draw_results(test_image, result, cutting_y=cutting_y)

        cv2.namedWindow("Blob Analysis Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Blob Analysis Result", result_img)

        # 이진화 이미지도 표시
        cv2.namedWindow("Binary", cv2.WINDOW_NORMAL)
        cv2.imshow("Binary", result['binary_image'])

        cv2.waitKey(0)
        cv2.destroyAllWindows()
