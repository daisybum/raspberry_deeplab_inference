#!/usr/bin/env python3
"""
TFLite 세그멘테이션 모델 시각화 스크립트

- 테스트셋 일부(기본 3장)를 추론하여 원본, 추론 오버레이, GT 오버레이를 나란히 출력/저장합니다.
- evaluate_tflite_metrics.py의 유틸리티 함수들을 가져와 독립 실행 가능하도록 구성했습니다.

예시 실행:
python3 visualize_segmentation.py \
  --model_path models/tflite_models/Model_quant.tflite \
  --json_path /home/shpark/workspace/LastDeeplab/data/COCO/test_without_street.json \
  --image_dir /home/shpark/workspace/LastDeeplab/data/images/ \
  --num_images 3 --output_dir outputs
"""
import os
import json
import argparse
import random
from typing import List, Dict

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# -----------------------------
# Utility functions (원본 evaluate_tflite_metrics.py 일부 차용)
# -----------------------------

def create_mask_from_polygons(polygons, height, width):
    """COCO 폴리곤 -> 바이너리 마스크"""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        if len(polygon) >= 6:
            coords = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
            draw.polygon(coords, fill=1)
    return np.array(mask)


def load_coco_data(json_path: str, image_dir: str):
    """evaluate_tflite_metrics.py 동일"""
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    categories = {cat['id']: cat for cat in coco_data['categories']}
    target_classes = ['dry', 'humid', 'slush', 'snow', 'wet']
    allowed_cat_ids = [cid for cid, cat in categories.items() if cat['name'] in target_classes]
    category_mapping = {cid: i + 1 for i, cid in enumerate(sorted(allowed_cat_ids))}

    images = {img['id']: img for img in coco_data['images']}
    annotations_by_image: Dict[int, List[dict]] = {}
    for ann in coco_data['annotations']:
        annotations_by_image.setdefault(ann['image_id'], []).append(ann)

    images_data = []
    for img_id, img_info in images.items():
        img_path = os.path.join(image_dir, img_info['file_name'])
        if os.path.exists(img_path):
            images_data.append({
                'image_path': img_path,
                'image_info': img_info,
                'annotations': annotations_by_image.get(img_id, [])
            })
    return images_data, category_mapping


def create_ground_truth_mask(annotations, category_mapping, height, width):
    gt_mask = np.zeros((height, width), dtype=np.uint8)
    sorted_annotations = sorted(annotations, key=lambda x: x.get('area', 0), reverse=True)
    for ann in sorted_annotations:
        if ann['category_id'] not in category_mapping:
            continue
        cid = category_mapping[ann['category_id']]
        if 'segmentation' in ann and ann['segmentation']:
            if isinstance(ann['segmentation'], list):
                for polygon in ann['segmentation']:
                    if len(polygon) >= 6:
                        mask = create_mask_from_polygons([polygon], height, width)
                        gt_mask[mask == 1] = cid
    return gt_mask


def run_inference_single(interpreter, image_path, input_details, output_details):
    original_image_pil = Image.open(image_path).convert('RGB')
    original_size = original_image_pil.size
    _, height, width, _ = input_details['shape']
    input_type = input_details['dtype']

    image_tensor = tf.convert_to_tensor(original_image_pil, dtype=tf.float32)
    resized = tf.image.resize(image_tensor, [height, width], method=tf.image.ResizeMethod.BILINEAR)
    if input_type == np.uint8:
        input_data = tf.cast(resized, tf.uint8)
    else:
        input_data = resized
    input_data = tf.expand_dims(input_data, 0)

    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])

    seg_map = np.squeeze(output_data).astype(np.uint8)
    seg_pil = Image.fromarray(seg_map)
    resized_seg = seg_pil.resize(original_size, Image.NEAREST)
    return np.array(resized_seg)

# -----------------------------
# Visualization helpers
# -----------------------------

def get_palette():
    """클래스별 RGB 팔레트 (0~5)"""
    return {
        0: (0, 0, 0),       # background - black
        1: (255, 0, 0),     # dry - red
        2: (0, 255, 0),     # humid - green
        3: (0, 0, 255),     # slush - blue
        4: (0, 255, 255),   # snow - cyan
        5: (255, 255, 0),   # wet - yellow
    }


def mask_to_color(mask: np.ndarray, palette: dict):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in palette.items():
        color_mask[mask == cid] = color
    return color_mask


def overlay(image: np.ndarray, mask_color: np.ndarray, alpha: float = 0.5):
    return (image * (1 - alpha) + mask_color * alpha).astype(np.uint8)

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="세그멘테이션 시각화 스크립트")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--json_path', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--num_images', type=int, default=3, help='시각화할 이미지 수')
    parser.add_argument('--output_dir', type=str, default='viz_outputs', help='결과 저장 폴더')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Load data
    images_data, category_mapping = load_coco_data(args.json_path, args.image_dir)
    if not images_data:
        print("처리할 이미지가 없습니다.")
        return

    # 샘플 이미지 선택
    sample_data = random.sample(images_data, min(args.num_images, len(images_data)))
    palette = get_palette()

    for idx, data in enumerate(sample_data, 1):
        img_path = data['image_path']
        img_info = data['image_info']
        annots = data['annotations']
        h, w = img_info['height'], img_info['width']

        # GT & Prediction
        gt_mask = create_ground_truth_mask(annots, category_mapping, h, w)
        pred_mask = run_inference_single(interpreter, img_path, input_details, output_details)

        # Prepare visualization
        orig_img = np.array(Image.open(img_path).convert('RGB'))
        gt_color = mask_to_color(gt_mask, palette)
        pred_color = mask_to_color(pred_mask, palette)

        overlay_pred = overlay(orig_img, pred_color, alpha=0.5)
        overlay_gt = overlay(orig_img, gt_color, alpha=0.5)

        # Plot
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(orig_img)
        axs[0].set_title('Original')
        axs[0].axis('off')

        axs[1].imshow(overlay_pred)
        axs[1].set_title('Prediction Overlay')
        axs[1].axis('off')

        axs[2].imshow(overlay_gt)
        axs[2].set_title('GT Overlay')
        axs[2].axis('off')

        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f'viz_{idx}.png')
        plt.savefig(out_path)
        print(f'Saved visualization to {out_path}')
        plt.close(fig)


if __name__ == '__main__':
    main() 