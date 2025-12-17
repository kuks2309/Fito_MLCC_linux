#!/usr/bin/env python3
"""
MLCC Index Viewer - Hailo-8 기반 MLCC Index 검출 뷰어
UI 파일 기반 버전
"""
import sys
import os
import json
import time
import signal
import random

# Qt 플랫폼 플러그인 경로 설정
if "QT_QPA_PLATFORM_PLUGIN_PATH" not in os.environ:
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins"

# PyQt5를 먼저 import
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QButtonGroup
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic

# 그 후 OpenCV import
import glob
import cv2
import numpy as np

# 프로젝트 경로 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_dir, "Hailo_H8"))

from hailo_inference import HailoInference
from blob_analyzer import BlobAnalyzer
from projection_detector import ProjectionIndexDetector
import index_hough_line_cpp


class MLCCViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        # UI 파일 로드
        ui_path = os.path.join(project_dir, "ui", "mlcc_viewer.ui")
        uic.loadUi(ui_path, self)

        # 멤버 변수 초기화
        self.image_files = []
        self.current_index = -1
        self.current_image = None
        self.current_path = None
        self.result_image = None
        self.last_result = None
        self.cutting_y = None
        self.hailo_initialized = False

        # 설정 로드
        self.settings_path = os.path.join(project_dir, "config", "settings.json")
        self.settings = self.load_settings()

        # Hailo-8 초기화
        self.init_hailo()

        # Blob Analyzer 초기화 (Halcon 방식)
        self.blob_analyzer = BlobAnalyzer()

        # Projection Index Detector 초기화 (C++ 미분 방식)
        self.projection_detector = ProjectionIndexDetector(use_cpp=True)

        # Hough Line Detector 초기화 (C++)
        self.hough_detector = index_hough_line_cpp.IndexHoughLineDetector()

        # Index 파라미터 초기화
        self.min_index_height = 50
        self.max_index_height = 300
        self.index_pitch = 0
        self.h_diff_threshold = 50  # 수평 프로젝션 diff 임계값

        # 학습된 통계 초기화
        self.learned_stats = None
        self.load_learned_stats()

        # Auto Detect 라디오 버튼 그룹 설정
        self.auto_detect_group = QButtonGroup(self)
        self.auto_detect_group.addButton(self.radio_no_auto, 0)
        self.auto_detect_group.addButton(self.radio_auto_ai, 1)
        self.auto_detect_group.addButton(self.radio_auto_blob, 2)
        self.auto_detect_group.addButton(self.radio_auto_halcon, 3)

        # 시그널 연결
        self.connect_signals()

        # 이벤트 필터 설치
        self.listWidget_images.installEventFilter(self)

        # 설정 적용
        self.apply_settings()

        # 마지막으로 열었던 폴더 자동 로드
        self.load_last_folder()

    def connect_signals(self):
        """UI 위젯들의 시그널 연결"""
        # 버튼
        self.btn_open_folder.clicked.connect(self.open_folder)
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_AI_analyze.clicked.connect(self.analyze_image)
        self.btn_blob_analyze.clicked.connect(self.analyze_blob)
        self.btn_show_binary.clicked.connect(self.show_binary_image)
        self.btn_show_cover_removed.clicked.connect(self.show_cover_removed_image)
        self.btn_show_projection.clicked.connect(self.show_projection_image)
        self.btn_save_settings.clicked.connect(self.on_save_settings)
        self.btn_save.clicked.connect(self.save_result)

        # 리스트 위젯
        self.listWidget_images.currentRowChanged.connect(self.on_image_selected)

        # AI 파라미터 슬라이더
        self.slider_conf.valueChanged.connect(self.on_conf_changed)
        self.slider_iou.valueChanged.connect(self.on_iou_changed)

        # Halcon Blob 파라미터
        self.slider_halcon_threshold.valueChanged.connect(self.on_halcon_threshold_changed)
        self.spinBox_halcon_min_area.valueChanged.connect(self.on_halcon_params_changed)
        self.spinBox_halcon_max_area.valueChanged.connect(self.on_halcon_params_changed)
        self.checkBox_halcon_remove_cover.stateChanged.connect(self.on_halcon_params_changed)
        self.spinBox_halcon_cover_size.valueChanged.connect(self.on_halcon_params_changed)

        # OpenCV Blob 버튼
        self.btn_show_vertical_edge.clicked.connect(self.show_vertical_edge)
        self.btn_show_hough_lines.clicked.connect(self.show_hough_lines)
        self.btn_opencv_show_projection.clicked.connect(self.show_opencv_raw_projection)
        self.btn_opencv_detect_index.clicked.connect(self.detect_projection_index)

        # Index 파라미터
        self.spinBox_min_index_height.valueChanged.connect(self.on_index_params_changed)
        self.spinBox_max_index_height.valueChanged.connect(self.on_index_params_changed)
        self.spinBox_index_pitch.valueChanged.connect(self.on_index_params_changed)
        self.spinBox_h_diff_threshold.valueChanged.connect(self.on_index_params_changed)

        # Learn 버튼
        self.btn_learn.clicked.connect(self.learn_from_folder)

        # Display 옵션
        self.checkBox_boxes.stateChanged.connect(self.update_display)
        self.checkBox_masks.stateChanged.connect(self.update_display)
        self.checkBox_labels.stateChanged.connect(self.update_display)
        self.checkBox_cutting_line.stateChanged.connect(self.update_display)

    def load_settings(self):
        """설정 파일 로드"""
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def save_settings(self):
        """설정 파일 저장"""
        os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
        with open(self.settings_path, 'w') as f:
            json.dump(self.settings, f, indent=2)
        print(f"Settings saved to {self.settings_path}")

    def apply_settings(self):
        """저장된 설정 적용"""
        if "conf_threshold" in self.settings:
            self.slider_conf.setValue(int(self.settings["conf_threshold"] * 100))
        if "iou_threshold" in self.settings:
            self.slider_iou.setValue(int(self.settings["iou_threshold"] * 100))
        if "blob_threshold" in self.settings:
            self.slider_threshold.setValue(self.settings["blob_threshold"])
        if "blob_min_area" in self.settings:
            self.spinBox_min_area.setValue(self.settings["blob_min_area"])
        if "blob_max_area" in self.settings:
            self.spinBox_max_area.setValue(self.settings["blob_max_area"])

        # 학습 상태 UI 업데이트
        self.update_learn_status()

    def load_last_folder(self):
        """마지막으로 열었던 폴더 자동 로드"""
        last_dir = self.settings.get("last_image_dir", "")
        if last_dir and os.path.isdir(last_dir):
            self.load_images(last_dir)
            self.statusbar.showMessage(f"Loaded last folder: {last_dir}")

    def init_hailo(self):
        """Hailo-8 초기화"""
        try:
            hef_path = os.path.join(project_dir, "Hailo_H8", "deploy", "fito_best.hef")
            self.inference = HailoInference(hef_path)
            if self.inference.initialize():
                self.hailo_initialized = True
                self.statusbar.showMessage("Hailo-8 initialized")
            else:
                self.hailo_initialized = False
                self.statusbar.showMessage("Hailo-8 init failed")
        except Exception as e:
            self.inference = None
            self.hailo_initialized = False
            self.statusbar.showMessage(f"Hailo-8 init failed: {e}")

    def on_conf_changed(self, value):
        self.label_conf_value.setText(f"{value/100:.2f}")
        if self.hailo_initialized:
            self.inference.set_conf_threshold(value / 100.0)

    def on_iou_changed(self, value):
        self.label_iou_value.setText(f"{value/100:.2f}")
        if self.hailo_initialized:
            self.inference.set_iou_threshold(value / 100.0)

    def on_halcon_threshold_changed(self, value):
        """Halcon Blob Threshold 변경"""
        self.label_halcon_threshold_value.setText(str(value))
        self.blob_analyzer.threshold_value = value

    def on_halcon_params_changed(self):
        """Halcon Blob 파라미터 변경 시 호출"""
        self.blob_analyzer.min_area = self.spinBox_halcon_min_area.value()
        self.blob_analyzer.max_area = self.spinBox_halcon_max_area.value()
        self.blob_analyzer.use_remove_cover = self.checkBox_halcon_remove_cover.isChecked()
        self.blob_analyzer.cover_filter_size = self.spinBox_halcon_cover_size.value()

    def on_index_params_changed(self):
        """Index 파라미터 변경 시 호출"""
        self.min_index_height = self.spinBox_min_index_height.value()
        self.max_index_height = self.spinBox_max_index_height.value()
        self.index_pitch = self.spinBox_index_pitch.value()
        self.h_diff_threshold = self.spinBox_h_diff_threshold.value()

    def load_learned_stats(self):
        """저장된 학습 통계 로드"""
        stats_path = os.path.join(project_dir, "config", "learned_stats.json")
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as f:
                    self.learned_stats = json.load(f)
                print(f"Loaded learned stats from {stats_path}")
            except Exception as e:
                print(f"Failed to load learned stats: {e}")
                self.learned_stats = None

    def save_learned_stats(self):
        """학습 통계 저장"""
        if self.learned_stats is None:
            return
        stats_path = os.path.join(project_dir, "config", "learned_stats.json")
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(self.learned_stats, f, indent=2)
        print(f"Saved learned stats to {stats_path}")

    def learn_from_folder(self):
        """현재 폴더의 이미지들에서 프로젝션 통계 학습"""
        if not self.image_files:
            QMessageBox.warning(self, "Warning", "No images loaded")
            return

        self.statusbar.showMessage("Learning from folder...")
        QApplication.processEvents()

        # 수직 프로젝션 통계 수집
        v_mean_list = []
        v_max_list = []
        v_diff_list = []  # max - mean 차이

        # 수평 프로젝션 통계 수집 (ROI 영역 내)
        h_mean_list = []
        h_max_list = []
        h_diff_list = []

        for i, img_path in enumerate(self.image_files):
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # === 수직 프로젝션 (각 열의 합) ===
            v_proj = np.sum(gray, axis=0).astype(np.float32)
            v_mean = np.mean(v_proj)
            v_max = np.max(v_proj)
            v_diff = v_max - v_mean

            v_mean_list.append(float(v_mean))
            v_max_list.append(float(v_max))
            v_diff_list.append(float(v_diff))

            # === 수평 프로젝션 (ROI 영역 내에서 계산) ===
            # 수직 프로젝션으로 ROI 찾기
            v_threshold = (v_mean + v_max) / 2
            above_threshold = v_proj > v_threshold
            v_regions = []
            in_region = False
            start_x = 0
            for x in range(w):
                if above_threshold[x] and not in_region:
                    in_region = True
                    start_x = x
                elif not above_threshold[x] and in_region:
                    in_region = False
                    v_regions.append((start_x, x - 1))
            if in_region:
                v_regions.append((start_x, w - 1))

            # ROI 영역 내에서 수평 프로젝션 계산
            if len(v_regions) >= 2:
                roi_x1 = v_regions[0][1] + 1
                roi_x2 = v_regions[-1][0] - 1
                roi_width = roi_x2 - roi_x1 + 1
                if roi_width > 0:
                    h_proj = np.sum(gray[:, roi_x1:roi_x2+1], axis=1).astype(np.float32) / roi_width
                else:
                    h_proj = np.sum(gray, axis=1).astype(np.float32) / w
            else:
                h_proj = np.sum(gray, axis=1).astype(np.float32) / w

            h_mean = np.mean(h_proj)
            h_max = np.max(h_proj)
            h_diff = h_max - h_mean

            h_mean_list.append(float(h_mean))
            h_max_list.append(float(h_max))
            h_diff_list.append(float(h_diff))

            # 진행 상황 표시
            if (i + 1) % 10 == 0:
                self.statusbar.showMessage(f"Learning... {i + 1}/{len(self.image_files)}")
                QApplication.processEvents()

        # 통계 계산
        self.learned_stats = {
            "image_count": len(v_mean_list),
            "vertical": {
                "mean_avg": float(np.mean(v_mean_list)),
                "mean_std": float(np.std(v_mean_list)),
                "max_avg": float(np.mean(v_max_list)),
                "max_std": float(np.std(v_max_list)),
                "diff_avg": float(np.mean(v_diff_list)),
                "diff_std": float(np.std(v_diff_list)),
                "diff_min": float(np.min(v_diff_list)),
                "diff_max": float(np.max(v_diff_list)),
            },
            "horizontal": {
                "mean_avg": float(np.mean(h_mean_list)),
                "mean_std": float(np.std(h_mean_list)),
                "max_avg": float(np.mean(h_max_list)),
                "max_std": float(np.std(h_max_list)),
                "diff_avg": float(np.mean(h_diff_list)),
                "diff_std": float(np.std(h_diff_list)),
                "diff_min": float(np.min(h_diff_list)),
                "diff_max": float(np.max(h_diff_list)),
            }
        }

        # 저장
        self.save_learned_stats()

        # UI 업데이트
        self.update_learn_status()

        # 결과 메시지
        v_stats = self.learned_stats["vertical"]
        msg = f"Learned from {len(v_mean_list)} images\n\n"
        msg += f"Vertical Projection:\n"
        msg += f"  diff(max-mean): avg={v_stats['diff_avg']:.0f}, std={v_stats['diff_std']:.0f}\n"
        msg += f"  range: {v_stats['diff_min']:.0f} ~ {v_stats['diff_max']:.0f}"

        QMessageBox.information(self, "Learning Complete", msg)
        self.statusbar.showMessage(f"Learning complete: {len(v_mean_list)} images processed")

    def update_learn_status(self):
        """학습 상태 UI 업데이트"""
        if self.learned_stats:
            count = self.learned_stats.get("image_count", 0)
            v_diff = self.learned_stats["vertical"]["diff_avg"]
            self.label_learn_status.setText(f"Learned: {count} imgs, v_diff={v_diff:.0f}")
            self.label_learn_status.setStyleSheet("color: green; font-style: italic;")
        else:
            self.label_learn_status.setText("Not learned")
            self.label_learn_status.setStyleSheet("color: gray; font-style: italic;")

    def open_folder(self):
        last_dir = self.settings.get("last_image_dir", "")
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder", last_dir
        )
        if folder:
            self.settings["last_image_dir"] = folder
            self.load_images(folder)

    def load_images(self, folder):
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']:
            self.image_files.extend(glob.glob(os.path.join(folder, ext)))
        self.image_files.sort()

        self.listWidget_images.clear()
        for f in self.image_files:
            self.listWidget_images.addItem(os.path.basename(f))

        self.label_image_count.setText(f"{len(self.image_files)} images")

        # Learn 버튼 활성화
        self.btn_learn.setEnabled(len(self.image_files) > 0)

        if self.image_files:
            self.listWidget_images.setCurrentRow(0)

    def on_image_selected(self, row):
        if 0 <= row < len(self.image_files):
            self.current_index = row
            self.load_image(self.image_files[row])
            self.btn_prev.setEnabled(row > 0)
            self.btn_next.setEnabled(row < len(self.image_files) - 1)
            self.btn_AI_analyze.setEnabled(self.hailo_initialized)
            self.btn_blob_analyze.setEnabled(True)
            # Halcon Blob 버튼
            self.btn_show_binary.setEnabled(True)
            self.btn_show_cover_removed.setEnabled(True)
            self.btn_show_projection.setEnabled(True)
            # OpenCV Blob 버튼
            self.btn_show_vertical_edge.setEnabled(True)
            self.btn_show_hough_lines.setEnabled(True)
            self.btn_opencv_show_projection.setEnabled(True)
            self.btn_opencv_detect_index.setEnabled(True)

    def load_image(self, path):
        self.current_path = path
        self.current_image = cv2.imread(path)
        self.result_image = None
        self.last_result = None
        self.current_rotation_angle = 0.0  # 현재 회전 각도 저장

        if self.current_image is not None:
            # Random Rotate 옵션이 활성화되어 있으면 회전 적용
            if self.checkBox_random_rotate.isChecked():
                self.current_rotation_angle = random.uniform(-20.0, 20.0)
                self.current_image = self.rotate_image(self.current_image, self.current_rotation_angle)

            self.display_image(self.current_image)
            self.update_image_info(path)

            # 회전 각도 정보 표시
            if self.current_rotation_angle != 0.0:
                self.statusbar.showMessage(f"[{self.current_index + 1}/{len(self.image_files)}] {os.path.basename(path)} (Rotated: {self.current_rotation_angle:.1f}°)")
            else:
                self.statusbar.showMessage(f"[{self.current_index + 1}/{len(self.image_files)}] {os.path.basename(path)}")

            self.label_detections.setText("Detections: -")
            self.label_cutting_line.setText("Cutting Line: -")
            self.btn_save.setEnabled(False)

            # Auto Detect 모드에 따라 자동 분석
            auto_mode = self.auto_detect_group.checkedId()
            if auto_mode == 1 and self.hailo_initialized:
                self.analyze_image()
            elif auto_mode == 2:
                self.show_opencv_raw_projection()
            elif auto_mode == 3:
                self.analyze_blob()

    def rotate_image(self, image, angle):
        """이미지를 중심 기준으로 회전 (검은색 배경)"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # 회전 행렬 생성
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 회전 적용 (검은색 배경)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))
        return rotated

    def update_image_info(self, path):
        """이미지 정보 업데이트"""
        filename = os.path.basename(path)
        self.label_filename.setText(f"File: {filename}")

        h, w = self.current_image.shape[:2]
        self.label_resolution.setText(f"Resolution: {w} x {h}")

        if len(self.current_image.shape) == 3:
            channels = self.current_image.shape[2]
            depth = self.current_image.dtype
            self.label_depth.setText(f"Depth: {channels}ch, {depth}")
        else:
            depth = self.current_image.dtype
            self.label_depth.setText(f"Depth: 1ch (Gray), {depth}")

        size_bytes = self.current_image.nbytes
        if size_bytes > 1024 * 1024:
            size_str = f"{size_bytes / (1024*1024):.2f} MB"
        else:
            size_str = f"{size_bytes / 1024:.2f} KB"
        self.label_data_size.setText(f"Data Size: {size_str}")

        file_size = os.path.getsize(path)
        if file_size > 1024 * 1024:
            file_str = f"{file_size / (1024*1024):.2f} MB"
        else:
            file_str = f"{file_size / 1024:.2f} KB"
        self.label_file_size.setText(f"File Size: {file_str}")

    def display_image(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        scaled = pixmap.scaled(
            self.label_image.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.label_image.setPixmap(scaled)

    def update_display(self):
        """체크박스 변경 시 결과 이미지 업데이트"""
        if self.last_result is not None and self.current_image is not None:
            self.result_image = self.inference.draw_results(
                self.current_image,
                self.last_result,
                show_boxes=self.checkBox_boxes.isChecked(),
                show_masks=self.checkBox_masks.isChecked(),
                show_labels=self.checkBox_labels.isChecked(),
                show_cutting_line=self.checkBox_cutting_line.isChecked(),
                cutting_y=self.cutting_y
            )
            self.display_image(self.result_image)

    def prev_image(self):
        if self.current_index > 0:
            self.listWidget_images.setCurrentRow(self.current_index - 1)

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.listWidget_images.setCurrentRow(self.current_index + 1)

    def analyze_image(self):
        """Hailo-8로 이미지 분석"""
        if not self.hailo_initialized:
            self.statusbar.showMessage("Hailo-8 not initialized")
            return

        if self.current_image is None:
            return

        self.statusbar.showMessage("Analyzing...")
        QApplication.processEvents()

        # 추론 실행 (시간 측정)
        start_time = time.time()
        self.last_result = self.inference.infer(self.current_image)
        infer_time = (time.time() - start_time) * 1000

        if self.last_result is not None:
            self.label_inference_time.setText(f"Inference: {infer_time:.1f} ms")
            count = self.last_result['count']
            self.label_detections.setText(f"Detections: {count}")

            if count > 0:
                # 세그먼트 마스크의 중심(centroid) 계산
                masks = self.last_result.get('masks', [])
                boxes = self.last_result['boxes']
                self.cutting_y = []

                for i, mask in enumerate(masks):
                    if mask is not None and np.any(mask):
                        # 마스크의 y 좌표들의 평균 (centroid)
                        y_coords = np.where(mask)[0]
                        center_y = int(np.mean(y_coords))
                        self.cutting_y.append(center_y)
                    else:
                        # 마스크가 없으면 바운딩 박스 중심 사용 (fallback)
                        box = boxes[i]
                        center_y = int((box[1] + box[3]) / 2)
                        self.cutting_y.append(center_y)

                self.cutting_y.sort()
                self.label_cutting_line.setText(f"Cutting Lines: {count}개")
            else:
                self.cutting_y = None
                self.label_cutting_line.setText("Cutting Line: -")

            self.result_image = self.inference.draw_results(
                self.current_image,
                self.last_result,
                show_boxes=self.checkBox_boxes.isChecked(),
                show_masks=self.checkBox_masks.isChecked(),
                show_labels=self.checkBox_labels.isChecked(),
                show_cutting_line=self.checkBox_cutting_line.isChecked(),
                cutting_y=self.cutting_y
            )
            self.display_image(self.result_image)
            self.btn_save.setEnabled(True)

            self.statusbar.showMessage(f"Detected {count} objects")
        else:
            self.statusbar.showMessage("Inference failed")
            self.label_detections.setText("Detections: Error")

    def analyze_blob(self):
        """OpenCV Blob으로 이미지 분석"""
        if self.current_image is None:
            return

        self.statusbar.showMessage("Analyzing with OpenCV Blob...")
        QApplication.processEvents()

        self.last_result = self.blob_analyzer.analyze(self.current_image)

        if self.last_result is not None:
            process_time = self.last_result.get('process_time_ms', 0)
            timing = self.last_result.get('timing', {})

            # 단계별 시간 표시
            timing_str = " | ".join([f"{k}:{v:.1f}" for k, v in timing.items()])
            self.label_inference_time.setText(f"Total: {process_time:.1f} ms")

            count = self.last_result['count']
            self.label_detections.setText(f"Blobs: {count}")

            if count > 0:
                self.cutting_y = [int(b['center_y']) for b in self.last_result['blobs']]
                self.cutting_y.sort()
                self.label_cutting_line.setText(f"Cutting Lines: {count}개")
            else:
                self.cutting_y = None
                self.label_cutting_line.setText("Cutting Line: -")

            self.result_image = self.blob_analyzer.draw_results(
                self.current_image,
                self.last_result,
                show_boxes=self.checkBox_boxes.isChecked(),
                show_centers=True,
                show_cutting_line=self.checkBox_cutting_line.isChecked(),
                cutting_y=self.cutting_y
            )
            self.display_image(self.result_image)
            self.btn_save.setEnabled(True)

            # 상태바에 단계별 시간 표시
            self.statusbar.showMessage(f"Blob: {count}개 | {timing_str}")
        else:
            self.statusbar.showMessage("Blob analysis failed")
            self.label_detections.setText("Blobs: Error")

    def show_binary_image(self):
        """이진화 결과만 표시 (Halcon Threshold 프리뷰)"""
        if self.current_image is None:
            return

        self.statusbar.showMessage("Generating binary image...")
        QApplication.processEvents()

        start_time = time.time()
        binary = self.blob_analyzer.preprocess(self.current_image)
        process_time = (time.time() - start_time) * 1000

        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        self.display_image(binary_bgr)
        self.result_image = binary_bgr

        threshold_val = self.blob_analyzer.threshold_value
        self.statusbar.showMessage(f"Halcon Binary (Threshold: {threshold_val}) - {process_time:.1f} ms")
        self.label_inference_time.setText(f"Process: {process_time:.1f} ms")
        self.label_detections.setText("Mode: Halcon Binary")
        self.btn_save.setEnabled(True)

    def show_cover_removed_image(self):
        """커버 제거 결과 표시 (Step 2 검증용)"""
        if self.current_image is None:
            return

        self.statusbar.showMessage("Generating cover removed image...")
        QApplication.processEvents()

        start_time = time.time()

        # Step 1: 이진화
        binary = self.blob_analyzer.preprocess(self.current_image)

        # Step 2: 커버 제거
        # 임시로 use_remove_cover를 True로 설정
        original_setting = self.blob_analyzer.use_remove_cover
        self.blob_analyzer.use_remove_cover = True
        cover_removed = self.blob_analyzer.remove_cover(binary)
        self.blob_analyzer.use_remove_cover = original_setting

        process_time = (time.time() - start_time) * 1000

        # BGR로 변환하여 표시
        cover_removed_bgr = cv2.cvtColor(cover_removed, cv2.COLOR_GRAY2BGR)

        self.display_image(cover_removed_bgr)
        self.result_image = cover_removed_bgr

        cover_size = self.blob_analyzer.cover_filter_size
        self.statusbar.showMessage(f"Cover Removed (Filter Size: {cover_size}) - {process_time:.1f} ms")
        self.label_inference_time.setText(f"Process: {process_time:.1f} ms")
        self.label_detections.setText("Mode: Cover Removed")
        self.btn_save.setEnabled(True)

    def show_projection_image(self):
        """Projection 기반 분리 결과 표시 (Step 3 검증용)

        Halcon 원본 코드:
        1. RegionToBin() → 이진 이미지 생성
        2. GrayProjections(simple, hv_hor) → 수평 방향 프로젝션 계산
        3. Threshold(hv_Max × 0.5) → 프로젝션 임계값 적용
        4. Connection() → 프로젝션 영역 분리
        """
        if self.current_image is None:
            return

        self.statusbar.showMessage("Generating projection image...")
        QApplication.processEvents()

        start_time = time.time()

        # Step 1: 이진화
        binary = self.blob_analyzer.preprocess(self.current_image)

        # Step 2: 커버 제거 (옵션에 따라)
        if self.blob_analyzer.use_remove_cover:
            binary = self.blob_analyzer.remove_cover(binary)

        # Step 3: Projection 계산
        h, w = binary.shape
        h_proj = np.sum(binary, axis=1).astype(np.float32) / 255.0  # 픽셀 수로 변환

        # 임계값 계산 (최대값의 50%)
        max_proj = np.max(h_proj)
        threshold = max_proj * 0.5

        # 영역 분리
        regions = self.blob_analyzer.horizontal_projection_split(binary)

        process_time = (time.time() - start_time) * 1000

        # 시각화: 원본 이미지 + Projection 그래프 + 영역 표시
        result_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # Projection 그래프 그리기 (좌측에 표시)
        proj_width = 100
        proj_scale = proj_width / max_proj if max_proj > 0 else 1

        for y in range(h):
            bar_len = int(h_proj[y] * proj_scale)
            if h_proj[y] > threshold:
                color = (0, 255, 0)  # 임계값 이상: 녹색
            else:
                color = (128, 128, 128)  # 임계값 이하: 회색
            cv2.line(result_img, (0, y), (bar_len, y), color, 1)

        # 임계값 선 표시
        threshold_x = int(threshold * proj_scale)
        cv2.line(result_img, (threshold_x, 0), (threshold_x, h), (0, 0, 255), 1)

        # 분리된 영역 표시
        for i, (start_y, end_y) in enumerate(regions):
            # 영역 경계선
            cv2.line(result_img, (0, start_y), (w, start_y), (255, 0, 0), 2)
            cv2.line(result_img, (0, end_y), (w, end_y), (255, 0, 0), 2)
            # 영역 번호
            cv2.putText(result_img, f"R{i+1}", (w-50, (start_y + end_y) // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        self.display_image(result_img)
        self.result_image = result_img

        self.statusbar.showMessage(f"Projection (Regions: {len(regions)}, Threshold: {threshold:.0f}) - {process_time:.1f} ms")
        self.label_inference_time.setText(f"Process: {process_time:.1f} ms")
        self.label_detections.setText(f"Mode: Projection ({len(regions)} regions)")
        self.btn_save.setEnabled(True)

    def show_vertical_edge(self):
        """수직 엣지 이미지 표시 (C++ Sobel X 필터)"""
        if self.current_image is None:
            return

        h, w = self.current_image.shape[:2]

        # C++ 함수 호출 (1/4 축소 + Sobel X + 원본 크기 복원)
        edge_img, process_time = self.projection_detector._detector.compute_vertical_edge(
            self.current_image, 4)

        # BGR로 변환하여 표시
        result_img = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)

        self.display_image(result_img)
        self.result_image = result_img

        # 콘솔에 디버깅 정보 출력
        print(f"=== Vertical Edge Debug (C++ 1/4 + Sobel X) ===")
        print(f"Size: {w}x{h} -> {w//4}x{h//4}")
        print(f"Time: {process_time:.1f} ms")
        print()

        self.statusbar.showMessage(f"Vertical Edge (C++) - {process_time:.1f} ms")
        self.label_inference_time.setText(f"Calc: {process_time:.1f} ms")
        self.label_detections.setText("Mode: Vertical Edge (C++)")
        self.btn_save.setEnabled(True)

    def show_hough_lines(self):
        """Hough 변환으로 수직선(-15°~+15°) 검출하여 정렬 상태 확인"""
        if self.current_image is None:
            return

        h, w = self.current_image.shape[:2]

        # Hough 파라미터 설정 (긴 선만 검출)
        # 1/4 축소된 이미지에서 높이의 50% 이상 투표 필요
        scaled_h = h // 4
        min_line_votes = max(200, scaled_h // 2)  # 최소 200, 또는 축소 높이의 50%
        self.hough_detector.set_angle_range(-22.0, 22.0)  # -22° ~ +22° 검출
        self.hough_detector.set_accumulator_threshold(min_line_votes)
        self.hough_detector.set_max_lines(50)  # 많이 검출 후 그룹핑

        # C++ Hough Line 검출 (1/4 축소 + Sobel X + Hough)
        result = self.hough_detector.detect_from_image(self.current_image, 4)

        # 결과 이미지에 검출된 선 그리기
        result_img = self.current_image.copy()

        # 거리 기반 클러스터링으로 2개 그룹 찾기
        if len(result.lines) >= 2:
            # 각 라인의 중심 x 좌표 계산 및 정렬
            lines_with_x = [(line, (line.x1 + line.x2) / 2) for line in result.lines]
            lines_with_x.sort(key=lambda x: x[1])  # x 좌표로 정렬

            # 인접 라인 간 최대 거리 찾기 (가장 큰 gap이 두 그룹 사이)
            max_gap = 0
            split_idx = 0
            min_group_distance = w * 0.1  # 최소 그룹 간 거리: 이미지 폭의 10%

            for i in range(len(lines_with_x) - 1):
                gap = lines_with_x[i + 1][1] - lines_with_x[i][1]
                if gap > max_gap and gap > min_group_distance:
                    max_gap = gap
                    split_idx = i + 1

            # 그룹 분리 (gap이 충분히 크면)
            if max_gap > min_group_distance:
                group1_lines = [l for l, x in lines_with_x[:split_idx]]
                group2_lines = [l for l, x in lines_with_x[split_idx:]]
            else:
                # gap이 작으면 모두 같은 그룹으로 처리
                group1_lines = [l for l, x in lines_with_x]
                group2_lines = []

            # 각 그룹에서 가장 강한 라인 (votes 기준) 선택
            grouped_lines = []
            if group1_lines:
                best_line1 = max(group1_lines, key=lambda l: l.votes)
                grouped_lines.append(('Line1', best_line1))
            if group2_lines:
                best_line2 = max(group2_lines, key=lambda l: l.votes)
                grouped_lines.append(('Line2', best_line2))

            # 대표 라인 그리기
            colors = [(0, 255, 0), (255, 0, 0)]  # 녹색, 파란색
            mean_angle = 0.0
            for i, (group_name, line) in enumerate(grouped_lines):
                color = colors[i % 2]
                cv2.line(result_img, (line.x1, line.y1), (line.x2, line.y2), color, 3)
                mean_angle += line.angle_deg

                # 라인 정보 표시
                mid_y = (line.y1 + line.y2) // 2
                mid_x = (line.x1 + line.x2) // 2
                cv2.putText(result_img, f"{line.angle_deg:.1f}°",
                           (mid_x + 10, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if grouped_lines:
                mean_angle /= len(grouped_lines)
            num_groups = len(grouped_lines)

            # 두 그룹 간 거리 표시
            if len(grouped_lines) == 2:
                x1 = (grouped_lines[0][1].x1 + grouped_lines[0][1].x2) / 2
                x2 = (grouped_lines[1][1].x1 + grouped_lines[1][1].x2) / 2
                distance = abs(x2 - x1)
                cv2.putText(result_img, f"Distance: {distance:.0f}px", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # 라인이 2개 미만이면 그대로 표시
            for line in result.lines:
                cv2.line(result_img, (line.x1, line.y1), (line.x2, line.y2), (0, 255, 0), 3)
            mean_angle = result.mean_angle_deg
            num_groups = len(result.lines)

        # 정렬 상태 텍스트 표시
        align_text = f"Mean Angle: {mean_angle:.1f} deg"
        is_aligned = abs(mean_angle) < 5.0
        align_status = "ALIGNED" if is_aligned else "MISALIGNED"
        align_color = (0, 255, 0) if is_aligned else (0, 0, 255)

        cv2.putText(result_img, align_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_img, align_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, align_color, 2)
        cv2.putText(result_img, f"Groups: {num_groups}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        self.display_image(result_img)
        self.result_image = result_img

        # 콘솔에 디버깅 정보 출력
        print(f"=== Hough Line Debug (C++ -22~+22 deg) ===")
        print(f"Lines detected: {len(result.lines)} -> Grouped: {num_groups}")
        print(f"Mean angle: {mean_angle:.1f} deg")
        print(f"Aligned: {is_aligned}")
        print(f"Time: {result.process_time_ms:.1f} ms")
        print()

        self.statusbar.showMessage(f"Hough Lines (C++) - {result.process_time_ms:.1f} ms, {num_groups} groups")
        self.label_inference_time.setText(f"Calc: {result.process_time_ms:.1f} ms")
        self.label_detections.setText(f"Groups: {num_groups}, Angle: {mean_angle:.1f}°")
        self.btn_save.setEnabled(True)

    def show_opencv_raw_projection(self):
        """수직 프로젝션으로 ROI 영역 찾고, 그 영역 내에서 수평 프로젝션 수행"""
        if self.current_image is None:
            return

        self.statusbar.showMessage("Generating projection...")
        QApplication.processEvents()

        start_time = time.time()

        # Grayscale 변환
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # === Step 1: 수직 프로젝션으로 ROI 영역 찾기 ===
        v_proj = np.sum(gray, axis=0).astype(np.float32)

        # Threshold 계산: (mean + max) / 2
        v_mean = np.mean(v_proj)
        v_max = np.max(v_proj)
        v_threshold = (v_mean + v_max) / 2

        # 수직 프로젝션 diff 계산 (max - mean)
        v_diff = v_max - v_mean
        v_threshold_diff = 0  # 기본값

        # 학습된 통계를 사용하여 녹색 바 존재 여부 판단
        v_has_green_bar = True  # 기본값: 녹색 바 있다고 가정
        if self.learned_stats:
            # 학습된 diff_min의 50%를 임계값으로 사용
            # (학습 데이터는 녹색 바가 있는 이미지이므로, diff_min보다 작아야 녹색 바 없음)
            v_learned = self.learned_stats["vertical"]
            v_threshold_diff = v_learned["diff_min"] * 0.5
            v_has_green_bar = v_diff >= v_threshold_diff

        # Threshold 초과 영역 찾기 (수직 프로젝션) - 녹색 바가 있을 때만
        v_regions = []
        if v_has_green_bar:
            above_threshold = v_proj > v_threshold
            in_region = False
            start_x = 0
            for x in range(w):
                if above_threshold[x] and not in_region:
                    in_region = True
                    start_x = x
                elif not above_threshold[x] and in_region:
                    in_region = False
                    v_regions.append((start_x, x - 1))
            if in_region:
                v_regions.append((start_x, w - 1))

        # === Step 2: 첫 번째 녹색 영역 끝 ~ 마지막 녹색 영역 시작 사이에서 수평 프로젝션 수행 ===
        if len(v_regions) >= 2:
            # 첫 번째 영역의 끝 ~ 마지막 영역의 시작 (녹색 영역 사이의 inner 부분)
            roi_x1 = v_regions[0][1] + 1
            roi_x2 = v_regions[-1][0] - 1
            roi_width = roi_x2 - roi_x1 + 1
            if roi_width > 0:
                h_proj = np.sum(gray[:, roi_x1:roi_x2+1], axis=1).astype(np.float32) / roi_width
            else:
                h_proj = np.sum(gray, axis=1).astype(np.float32) / w
                roi_x1, roi_x2 = 0, w - 1
        elif len(v_regions) == 1:
            # 녹색 영역이 하나뿐이면 전체 이미지 사용
            h_proj = np.sum(gray, axis=1).astype(np.float32) / w
            roi_x1, roi_x2 = 0, w - 1
        else:
            h_proj = np.sum(gray, axis=1).astype(np.float32) / w
            roi_x1, roi_x2 = 0, w - 1

        # 수평 프로젝션 threshold 계산
        h_mean = np.mean(h_proj)
        h_max = np.max(h_proj)
        h_threshold = (h_mean + h_max) / 2

        # 수평 프로젝션 diff 계산 (max - mean)
        h_diff = h_max - h_mean
        h_threshold_diff = 0  # 기본값

        # UI에서 설정한 임계값을 사용하여 Index 존재 여부 판단
        h_threshold_diff = self.h_diff_threshold
        h_has_index = h_diff >= h_threshold_diff

        # 수평 프로젝션에서 threshold 초과 영역 찾기 - Index가 있을 때만
        h_regions_raw = []
        if h_has_index:
            h_above = h_proj > h_threshold
            in_region = False
            start_y = 0
            for y in range(h):
                if h_above[y] and not in_region:
                    in_region = True
                    start_y = y
                elif not h_above[y] and in_region:
                    in_region = False
                    h_regions_raw.append((start_y, y - 1))
            if in_region:
                h_regions_raw.append((start_y, h - 1))

        # Min/Max Height 필터링
        h_regions = []
        for (y1, y2) in h_regions_raw:
            region_height = y2 - y1 + 1
            if self.min_index_height <= region_height <= self.max_index_height:
                h_regions.append((y1, y2))

        # Pitch 검증 (index_pitch > 0 일 때만)
        if self.index_pitch > 0 and len(h_regions) > 1:
            pitch_tolerance = 0.33  # 33% 허용 오차
            min_pitch = self.index_pitch * (1.0 - pitch_tolerance)
            max_pitch = self.index_pitch * (1.0 + pitch_tolerance)

            validated = []
            # 첫 번째 유효한 쌍 찾기
            first_valid = -1
            for i in range(len(h_regions) - 1):
                center_i = (h_regions[i][0] + h_regions[i][1]) / 2
                center_j = (h_regions[i + 1][0] + h_regions[i + 1][1]) / 2
                pitch = center_j - center_i
                if min_pitch <= pitch <= max_pitch:
                    first_valid = i
                    break

            if first_valid >= 0:
                validated.append(h_regions[first_valid])
                for i in range(first_valid + 1, len(h_regions)):
                    center_prev = (validated[-1][0] + validated[-1][1]) / 2
                    center_curr = (h_regions[i][0] + h_regions[i][1]) / 2
                    pitch = center_curr - center_prev
                    if min_pitch <= pitch <= max_pitch:
                        validated.append(h_regions[i])
                h_regions = validated

        process_time = (time.time() - start_time) * 1000

        # === 시각화 ===
        proj_margin = 120
        result_h = h + proj_margin
        result_w = w + proj_margin
        result_img = np.zeros((result_h, result_w, 3), dtype=np.uint8)

        # 원본 이미지 배치 (우측 하단)
        result_img[proj_margin:, proj_margin:] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 수직 프로젝션 ROI 영역 표시 (녹색 반투명)
        for (x1, x2) in v_regions:
            overlay = result_img[proj_margin:, proj_margin + x1:proj_margin + x2 + 1].copy()
            cv2.rectangle(overlay, (0, 0), (x2 - x1, h - 1), (0, 255, 0), -1)
            alpha = 0.3
            result_img[proj_margin:, proj_margin + x1:proj_margin + x2 + 1] = \
                cv2.addWeighted(overlay, alpha,
                               result_img[proj_margin:, proj_margin + x1:proj_margin + x2 + 1], 1 - alpha, 0)
            # 영역 경계선 (녹색)
            cv2.line(result_img, (proj_margin + x1, proj_margin), (proj_margin + x1, proj_margin + h), (0, 255, 0), 1)
            cv2.line(result_img, (proj_margin + x2, proj_margin), (proj_margin + x2, proj_margin + h), (0, 255, 0), 1)

        # 수평 프로젝션 ROI 영역 표시 (노란색 사각형: 첫 녹색선 ~ 마지막 녹색선)
        if len(v_regions) > 0:
            cv2.rectangle(result_img, (proj_margin + roi_x1, proj_margin),
                         (proj_margin + roi_x2, proj_margin + h - 1), (0, 255, 255), 2)

        # 수평 프로젝션 threshold 초과 영역 표시 (파란색 사각형, 높이 1.5배 확장)
        h_regions_expanded = []
        for (y1, y2) in h_regions:
            region_height = y2 - y1 + 1
            expand = int(region_height * 0.25)  # 상하 각각 25% 확장 (총 1.5배)
            expand = max(expand, 20)  # 최소 20픽셀 확장
            y1_expanded = max(0, y1 - expand)
            y2_expanded = min(h - 1, y2 + expand)
            h_regions_expanded.append((y1_expanded, y2_expanded))
            cv2.rectangle(result_img, (proj_margin + roi_x1, proj_margin + y1_expanded),
                         (proj_margin + roi_x2, proj_margin + y2_expanded), (255, 0, 0), 2)

        # 수직 프로젝션 그래프 (상단, 노란색)
        if v_max > 0:
            v_proj_norm = (v_proj / v_max * (proj_margin - 10)).astype(np.int32)
            for x in range(w):
                bar_len = v_proj_norm[x]
                if bar_len > 0:
                    result_img[proj_margin - bar_len:proj_margin, proj_margin + x, 0] = 255
                    result_img[proj_margin - bar_len:proj_margin, proj_margin + x, 1] = 255

        # 수직 프로젝션 Threshold 선 (빨간색)
        v_threshold_y = int((v_threshold / v_max) * (proj_margin - 10)) if v_max > 0 else 0
        cv2.line(result_img, (proj_margin, proj_margin - v_threshold_y),
                (proj_margin + w, proj_margin - v_threshold_y), (0, 0, 255), 1)

        # 수평 프로젝션 그래프 (좌측, 녹색)
        if h_max > 0:
            h_proj_norm = (h_proj / h_max * (proj_margin - 10)).astype(np.int32)
            for y in range(h):
                bar_len = h_proj_norm[y]
                if bar_len > 0:
                    result_img[proj_margin + y, proj_margin - bar_len:proj_margin, 1] = 255

        # 수평 프로젝션 Threshold 선 (빨간색)
        h_threshold_x = int((h_threshold / h_max) * (proj_margin - 10)) if h_max > 0 else 0
        cv2.line(result_img, (proj_margin - h_threshold_x, proj_margin),
                (proj_margin - h_threshold_x, proj_margin + h), (0, 0, 255), 1)

        # 축 라벨
        cv2.putText(result_img, "V", (proj_margin + w // 2, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(result_img, "H", (5, proj_margin + h // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 경계선
        result_img[:, proj_margin, :] = 100
        result_img[proj_margin, :, :] = 100

        self.display_image(result_img)
        self.result_image = result_img

        # 콘솔에 디버깅 정보 출력
        print(f"=== Projection Debug ===")
        print(f"V: mean={v_mean:.0f}, max={v_max:.0f}, diff={v_diff:.0f} (threshold={v_threshold_diff:.0f})")
        print(f"H: mean={h_mean:.1f}, max={h_max:.1f}, diff={h_diff:.1f} (threshold={h_threshold_diff:.1f})")
        print(f"V regions: {len(v_regions)}, H regions: {len(h_regions)}")
        print(f"h_has_index: {h_has_index}")
        print()

        # 상태 표시
        self.statusbar.showMessage(f"V:{len(v_regions)} regions, H:{len(h_regions)} regions - {process_time:.1f} ms")
        self.label_inference_time.setText(f"Calc: {process_time:.1f} ms")
        self.label_detections.setText(f"V:{len(v_regions)}, H:{len(h_regions)}")
        self.btn_save.setEnabled(True)

    def detect_projection_index(self):
        """프로젝션 기반 Index 검출 (C++ 미분 방식)"""
        if self.current_image is None:
            return

        self.statusbar.showMessage("Detecting indices using projection...")
        QApplication.processEvents()

        # 검출 실행
        result = self.projection_detector.detect(self.current_image)

        # 결과 시각화
        result_img = self.projection_detector.draw_results(
            self.current_image, result,
            show_lines=True,
            show_boundaries=True
        )

        # Cutting line 설정 (첫 번째 Index 중심)
        if result['indices']:
            self.cutting_y = result['indices'][0].center_y
            self.label_cutting_line.setText(f"Cutting Line: Y={self.cutting_y}")
        else:
            self.cutting_y = None
            self.label_cutting_line.setText("Cutting Line: -")

        self.display_image(result_img)
        self.result_image = result_img

        # 상태 표시
        process_time = result['process_time_ms']
        count = result['count']
        self.statusbar.showMessage(f"Projection Index Detection: {count} indices found - {process_time:.2f} ms")
        self.label_inference_time.setText(f"Time: {process_time:.2f} ms")
        self.label_detections.setText(f"Detected: {count} indices")
        self.btn_save.setEnabled(True)

    def on_save_settings(self):
        """설정 저장 버튼 클릭"""
        self.settings["conf_threshold"] = self.slider_conf.value() / 100.0
        self.settings["iou_threshold"] = self.slider_iou.value() / 100.0
        self.settings["blob_threshold"] = self.slider_threshold.value()
        self.settings["blob_min_area"] = self.spinBox_min_area.value()
        self.settings["blob_max_area"] = self.spinBox_max_area.value()
        self.save_settings()
        self.statusbar.showMessage("Settings saved!")

    def save_result(self):
        if self.result_image is not None:
            default_name = os.path.splitext(os.path.basename(self.current_path))[0] + "_result.png"
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Result", default_name, "Images (*.png *.jpg *.bmp)"
            )
            if path:
                cv2.imwrite(path, self.result_image)
                self.statusbar.showMessage(f"Saved: {path}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.result_image is not None:
            self.display_image(self.result_image)
        elif self.current_image is not None:
            self.display_image(self.current_image)

    def eventFilter(self, obj, event):
        """이벤트 필터 - 리스트 위젯에서도 화살표 키 처리"""
        from PyQt5.QtCore import QEvent
        if obj == self.listWidget_images and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Left:
                self.prev_image()
                return True
            elif event.key() == Qt.Key_Right:
                self.next_image()
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        """키보드 단축키"""
        if event.key() == Qt.Key_Left or event.key() == Qt.Key_Backspace:
            self.prev_image()
        elif event.key() == Qt.Key_Right or event.key() == Qt.Key_Space:
            self.next_image()
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.btn_AI_analyze.isEnabled():
                self.analyze_image()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """종료 시 설정 저장 및 리소스 정리"""
        self.save_settings()
        if self.inference:
            self.inference.cleanup()
        event.accept()


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MLCCViewer()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
