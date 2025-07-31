#!/usr/bin/env python3
"""
benchmark_cpu_usage.py
----------------------
10개 임의의 COCO 이미지에 대해 TFLite 세그멘테이션 추론을 수행하고
평균 CPU 사용률(프로세스 기준)을 출력합니다.

사용 예시:
python3 benchmark_cpu_usage.py \
  --tflite_model model/tflite_models/seg_model_int8.tflite \
  --json_path /workspace/data/COCO/test_without_street.json \
  --image_dir /workspace/data/images/ \
  --num_images 10
"""
import os
import json
import random
import argparse
from typing import List, Dict

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import psutil
import time

# -------------------------------------------------------
# 보조 함수들 (evaluate_tflite_metrics.py 의 일부 간단 복사)
# -------------------------------------------------------

def create_mask_from_polygons(polygons, height, width):
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        if len(poly) >= 6:
            coords = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
            draw.polygon(coords, fill=1)
    return np.array(mask)


def load_coco(json_path: str, image_dir: str):
    with open(json_path, 'r') as f:
        coco = json.load(f)

    target_classes = ['dry', 'humid', 'slush', 'snow', 'wet']
    categories = {c['id']: c for c in coco['categories'] if c['name'] in target_classes}
    cat_map = {cid: idx+1 for idx, cid in enumerate(sorted(categories))}

    images = {img['id']: img for img in coco['images']}
    anns_by_img: Dict[int, List[dict]] = {}
    for ann in coco['annotations']:
        anns_by_img.setdefault(ann['image_id'], []).append(ann)

    data = []
    for img_id, info in images.items():
        img_path = os.path.join(image_dir, info['file_name'])
        if os.path.exists(img_path):
            data.append({
                'image_path': img_path,
                'info': info,
                'anns': anns_by_img.get(img_id, [])
            })
    return data, cat_map


def preprocess_image(img_path: str, in_h: int, in_w: int):
    img = Image.open(img_path).convert('RGB')
    original_size = img.size
    tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    tensor = tf.image.resize(tensor, (in_h, in_w), method='bilinear')
    tensor = tf.expand_dims(tensor, 0)
    return tensor, original_size


def run_inference(interpreter: tf.lite.Interpreter, inp, original_size):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    if input_details['dtype'] == np.uint8:
        inp = tf.cast(inp, tf.uint8)
    else:
        inp = tf.cast(inp, tf.float32)

    interpreter.set_tensor(input_details['index'], inp)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details['index'])
    pred = np.squeeze(pred)
    if pred.ndim == 3:
        pred = np.argmax(pred, axis=-1)
    img = Image.fromarray(pred.astype(np.uint8))
    img = img.resize(original_size, Image.NEAREST)
    return np.array(img)

# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="TFLite 추론 CPU 사용률 벤치마크 (임의 N장)")
    ap.add_argument('--tflite_model', type=str, required=True)
    ap.add_argument('--json_path', type=str, required=True)
    ap.add_argument('--image_dir', type=str, required=True)
    ap.add_argument('--input_height', type=int, default=512)
    ap.add_argument('--input_width', type=int, default=512)
    ap.add_argument('--num_images', type=int, default=10)
    args = ap.parse_args()

    if not os.path.exists(args.tflite_model):
        print(f"모델을 찾을 수 없습니다: {args.tflite_model}")
        return

    data, _ = load_coco(args.json_path, args.image_dir)
    if not data:
        print("이미지를 찾을 수 없습니다.")
        return

    sample_data = random.sample(data, min(args.num_images, len(data)))

    print(f"샘플 {len(sample_data)}장으로 추론 시작...\n")

    interpreter = tf.lite.Interpreter(model_path=args.tflite_model, num_threads=4)
    interpreter.allocate_tensors()

    proc = psutil.Process(os.getpid())
    cpu_usages = []
    elapsed_times = []

    # 첫 호출은 baseline
    proc.cpu_percent(interval=None)

    for idx, item in enumerate(sample_data, 1):
        inp, orig_size = preprocess_image(item['image_path'], args.input_height, args.input_width)

        t0 = time.perf_counter()
        _ = run_inference(interpreter, inp, orig_size)
        elapsed = time.perf_counter() - t0
        elapsed_times.append(elapsed)

        # interval=None 로 호출하면 이전 호출 이후 시간에 대한 % 반환
        cpu_percent = proc.cpu_percent(interval=None)
        cpu_usages.append(cpu_percent)
        print(f"[{idx}/{len(sample_data)}] time={elapsed*1000:.1f} ms | cpu={cpu_percent:.1f}%")

    print("\n===================== 결과 =====================")
    print(f"평균 추론 시간 : {np.mean(elapsed_times)*1000:.2f} ms (min={min(elapsed_times)*1000:.2f}, max={max(elapsed_times)*1000:.2f})")
    print(f"평균 CPU 사용률 : {np.mean(cpu_usages):.2f}% (min={min(cpu_usages):.2f}, max={max(cpu_usages):.2f})")
    print("==============================================")


if __name__ == '__main__':
    main() 