#!/usr/bin/env python3
"""
Hailo-8 HEF 파일로 YOLOv8-seg 추론 및 결과 시각화 (뷰어)
스페이스바: 다음 이미지, 백스페이스: 이전 이미지, ESC/Q: 종료
"""
import numpy as np
import cv2
import os
import glob

from hailo_platform import HEF, VDevice, FormatType
from hailo_platform.pyhailort.pyhailort import InferVStreams, InputVStreamParams, OutputVStreamParams

# 설정
HEF_PATH = "/home/amap/yolov8_fito/Hailo_H8/fito_best.hef"
IMAGE_DIR = "/home/amap/yolov8_fito/kkw"
INPUT_SHAPE = (640, 544)  # (height, width)
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# 클래스 정보
CLASS_NAMES = ['index0']
COLORS = [(0, 255, 0)]


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def preprocess_image(image_path, target_size=INPUT_SHAPE):
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None
    original_img = img.copy()
    original_size = img.shape[:2]
    img_resized = cv2.resize(img, (target_size[1], target_size[0]))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_rgb, axis=0).astype(np.uint8)
    return img_array, original_img, original_size


def nms(boxes, scores, iou_threshold):
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONF_THRESHOLD, iou_threshold)
    return indices.flatten() if len(indices) > 0 else []


def process_output(outputs, conf_threshold=CONF_THRESHOLD):
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
        mask = max_cls_probs > conf_threshold

        if not np.any(mask):
            continue

        y_indices, x_indices = np.where(mask)

        for y, x in zip(y_indices, x_indices):
            score = max_cls_probs[y, x]
            class_id = class_ids[y, x]
            bbox_raw = bbox_out[y, x]

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

    boxes = np.array(all_boxes)
    scores = np.array(all_scores)
    indices = nms(boxes, scores, IOU_THRESHOLD)

    final_boxes, final_scores, final_class_ids, final_mask_coeffs = [], [], [], []
    for i in indices:
        x, y, w, h = boxes[i]
        final_boxes.append([x, y, x + w, y + h])
        final_scores.append(scores[i])
        final_class_ids.append(all_class_ids[i])
        final_mask_coeffs.append(all_mask_coeffs[i])

    return final_boxes, final_scores, final_class_ids, final_mask_coeffs, proto


def generate_mask(proto, mask_coeff, box, img_shape, input_shape):
    mask = np.tensordot(proto, mask_coeff, axes=([2], [0]))
    mask = sigmoid(mask)
    mask = cv2.resize(mask, (input_shape[1], input_shape[0]))

    x1, y1, x2, y2 = [int(c) for c in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(input_shape[1], x2), min(input_shape[0], y2)

    box_mask = np.zeros((input_shape[0], input_shape[1]), dtype=np.float32)
    box_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    box_mask = cv2.resize(box_mask, (img_shape[1], img_shape[0]))

    return box_mask > 0.5


def draw_results(image, boxes, scores, class_ids, masks, original_size, input_shape):
    img = image.copy()
    scale_x = original_size[1] / input_shape[1]
    scale_y = original_size[0] / input_shape[0]

    for box, score, class_id, mask in zip(boxes, scores, class_ids, masks):
        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)

        color = COLORS[class_id % len(COLORS)]

        if mask is not None:
            colored_mask = np.zeros_like(img)
            colored_mask[mask] = color
            img = cv2.addWeighted(img, 1, colored_mask, 0.4, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
        label = f"{class_name}: {score:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    return img


def main():
    print("=" * 60)
    print("Hailo-8 YOLOv8-seg 뷰어")
    print("스페이스바: 다음, 백스페이스: 이전, ESC/Q: 종료")
    print("=" * 60)

    print(f"\nHEF 로드: {HEF_PATH}")
    hef = HEF(HEF_PATH)
    input_vstream_infos = hef.get_input_vstream_infos()
    output_vstream_infos = hef.get_output_vstream_infos()

    image_extensions = ['*.bmp', '*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    image_files.sort()

    if not image_files:
        print(f"이미지를 찾을 수 없습니다: {IMAGE_DIR}")
        return

    print(f"총 {len(image_files)}개 이미지\n")

    window_name = "Hailo Viewer (Space: Next, Backspace: Prev, ESC: Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 900)

    current_idx = 0

    with VDevice() as target:
        network_group = target.configure(hef)[0]
        input_vstream_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        output_vstream_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

        with InferVStreams(network_group, input_vstream_params, output_vstream_params) as infer_pipeline:
            with network_group.activate():
                while True:
                    image_path = image_files[current_idx]
                    filename = os.path.basename(image_path)
                    print(f"[{current_idx + 1}/{len(image_files)}] {filename}", end=" ... ")

                    input_data, original_img, original_size = preprocess_image(image_path)

                    if input_data is None:
                        print("로드 실패")
                        result_img = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(result_img, f"Failed: {filename}", (50, 240),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        input_dict = {input_vstream_infos[0].name: input_data}
                        output_dict = infer_pipeline.infer(input_dict)

                        boxes, scores, class_ids, mask_coeffs, proto = process_output(output_dict)

                        masks = []
                        for box, mask_coeff in zip(boxes, mask_coeffs):
                            mask = generate_mask(proto, mask_coeff, box, original_size, INPUT_SHAPE)
                            masks.append(mask)

                        result_img = draw_results(original_img, boxes, scores, class_ids, masks,
                                                 original_size, INPUT_SHAPE)

                        print(f"검출: {len(boxes)}개")

                        info_text = f"[{current_idx + 1}/{len(image_files)}] {filename} - Detected: {len(boxes)}"
                        cv2.putText(result_img, info_text, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    cv2.imshow(window_name, result_img)

                    key = cv2.waitKey(0) & 0xFF
                    if key == 27 or key == ord('q') or key == ord('Q'):
                        print("\n종료")
                        break
                    elif key == 32 or key == 83:
                        current_idx = (current_idx + 1) % len(image_files)
                    elif key == 8 or key == 81:
                        current_idx = (current_idx - 1) % len(image_files)

    cv2.destroyAllWindows()
    print("완료!")


if __name__ == "__main__":
    main()
