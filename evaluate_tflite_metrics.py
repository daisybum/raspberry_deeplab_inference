#!/usr/bin/env python3
"""
TFLite 모델 평가 스크립트
COCO 형식의 JSON 파일과 이미지를 사용하여 세그멘테이션 메트릭을 계산합니다.
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import io


def create_mask_from_polygons(polygons, height, width):
    """
    COCO 형식의 폴리곤을 마스크로 변환합니다.
    
    Args:
        polygons: COCO 형식의 segmentation 폴리곤 리스트
        height: 이미지 높이
        width: 이미지 너비
    
    Returns:
        numpy.ndarray: 바이너리 마스크 (height, width)
    """
    from PIL import Image, ImageDraw
    
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    for polygon in polygons:
        if len(polygon) >= 6:  # 최소 3개 점 (x1,y1,x2,y2,x3,y3)
            # 폴리곤 좌표를 (x,y) 튜플 리스트로 변환
            coords = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
            draw.polygon(coords, fill=1)
    
    return np.array(mask)


def load_coco_data(json_path, image_dir):
    """
    COCO JSON 파일을 로드하고 필요한 데이터를 추출합니다.
    클래스는 6개로 고정: background(0) + dry(1) + humid(2) + slush(3) + snow(4) + wet(5)
    
    Args:
        json_path: COCO JSON 파일 경로
        image_dir: 이미지 디렉터리 경로
    
    Returns:
        tuple: (images_data, category_mapping)
    """
    print(f"COCO 데이터 로딩 중: {json_path}")
    
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # 카테고리 매핑 생성 (배경=0, 5개 클래스=1,2,3,4,5)
    categories = {cat['id']: cat for cat in coco_data['categories']}
    # background 제외한 5개 클래스만 사용
    target_classes = ['dry', 'humid', 'slush', 'snow', 'wet']
    allowed_cat_ids = [cat_id for cat_id, cat in categories.items() 
                      if cat['name'] in target_classes]
    
    category_mapping = {}
    for i, cat_id in enumerate(sorted(allowed_cat_ids)):
        category_mapping[cat_id] = i + 1  # 배경은 0, 클래스는 1부터 시작
    
    print(f"카테고리 매핑:")
    print(f"  0: background")
    for cat_id, new_id in category_mapping.items():
        print(f"  {new_id}: {categories[cat_id]['name']}")
    
    # 이미지별 어노테이션 그룹화
    images = {img['id']: img for img in coco_data['images']}
    annotations_by_image = {}
    
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # 처리할 이미지 데이터 생성
    images_data = []
    for img_id, img_info in images.items():
        image_path = os.path.join(image_dir, img_info['file_name'])
        if os.path.exists(image_path):
            images_data.append({
                'image_path': image_path,
                'image_info': img_info,
                'annotations': annotations_by_image.get(img_id, [])
            })
    
    print(f"총 {len(images_data)}개 이미지를 처리합니다.")
    return images_data, category_mapping


def create_ground_truth_mask(annotations, category_mapping, height, width):
    """
    어노테이션으로부터 ground truth 마스크를 생성합니다.
    
    Args:
        annotations: 이미지의 어노테이션 리스트
        category_mapping: 카테고리 ID 매핑
        height: 이미지 높이
        width: 이미지 너비
    
    Returns:
        numpy.ndarray: Ground truth 마스크 (height, width)
    """
    gt_mask = np.zeros((height, width), dtype=np.uint8)
    
    # 면적이 큰 것부터 처리 (작은 객체가 큰 객체를 덮어쓰지 않도록)
    sorted_annotations = sorted(annotations, key=lambda x: x.get('area', 0), reverse=True)
    
    for ann in sorted_annotations:
        if ann['category_id'] not in category_mapping:
            continue
            
        category_id = category_mapping[ann['category_id']]
        
        if 'segmentation' in ann and ann['segmentation']:
            if isinstance(ann['segmentation'], list):
                # 폴리곤 형식
                for polygon in ann['segmentation']:
                    if len(polygon) >= 6:  # 최소 3개 점
                        mask = create_mask_from_polygons([polygon], height, width)
                        gt_mask[mask == 1] = category_id
    
    return gt_mask


def run_inference_single(interpreter, image_path, input_details, output_details):
    """
    단일 이미지에 대해 TFLite 모델 추론을 수행합니다.
    
    Args:
        interpreter: TFLite 인터프리터
        image_path: 이미지 파일 경로
        input_details: 모델 입력 세부사항
        output_details: 모델 출력 세부사항
    
    Returns:
        numpy.ndarray: 예측 마스크
    """
    # 이미지 로드
    original_image_pil = Image.open(image_path).convert('RGB')
    original_size = original_image_pil.size
    
    # 모델 입력 크기
    _, height, width, _ = input_details['shape']
    input_type = input_details['dtype']
    
    # 전처리
    image_tensor = tf.convert_to_tensor(original_image_pil, dtype=tf.float32)
    resized_image_tensor = tf.image.resize(
        image_tensor, [height, width], method=tf.image.ResizeMethod.BILINEAR
    )
    
    if input_type == np.uint8:
        input_data = tf.cast(resized_image_tensor, tf.uint8)
        input_data = tf.expand_dims(input_data, axis=0)
    else:
        input_data = tf.expand_dims(resized_image_tensor, axis=0)
    
    # 추론
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])
    
    # 후처리
    seg_map = np.squeeze(output_data).astype(np.uint8)
    seg_map_pil = Image.fromarray(seg_map)
    resized_seg_map = seg_map_pil.resize(original_size, Image.NEAREST)
    
    return np.array(resized_seg_map)


def calculate_metrics(pred_mask, gt_mask, num_classes):
    """
    예측 마스크와 ground truth 마스크로부터 메트릭을 계산합니다.
    
    Args:
        pred_mask: 예측 마스크 (H, W)
        gt_mask: Ground truth 마스크 (H, W)
        num_classes: 클래스 수 (배경 포함)
    
    Returns:
        dict: 계산된 메트릭들
    """
    # 유효한 픽셀만 고려 (배경 포함)
    valid_mask = (gt_mask < num_classes) & (pred_mask < num_classes)
    
    if not np.any(valid_mask):
        return {
            'pixel_accuracy': 0.0,
            'mean_iou': 0.0,
            'class_ious': [0.0] * num_classes,
            'class_accuracies': [0.0] * num_classes
        }
    
    pred_valid = pred_mask[valid_mask]
    gt_valid = gt_mask[valid_mask]
    
    # Pixel Accuracy
    pixel_accuracy = np.mean(pred_valid == gt_valid)
    
    # 클래스별 IoU 계산
    class_ious = []
    class_accuracies = []
    
    for class_id in range(num_classes):
        # True Positive, False Positive, False Negative
        tp = np.sum((pred_valid == class_id) & (gt_valid == class_id))
        fp = np.sum((pred_valid == class_id) & (gt_valid != class_id))
        fn = np.sum((pred_valid != class_id) & (gt_valid == class_id))
        
        # IoU 계산
        if tp + fp + fn > 0:
            iou = tp / (tp + fp + fn)
        else:
            iou = 0.0
        class_ious.append(iou)
        
        # 클래스별 정확도
        gt_class_pixels = np.sum(gt_valid == class_id)
        if gt_class_pixels > 0:
            class_acc = tp / gt_class_pixels
        else:
            class_acc = 0.0
        class_accuracies.append(class_acc)
    
    # Mean IoU (배경 제외)
    if len(class_ious) > 1:
        mean_iou = np.mean(class_ious[1:])  # 배경(0) 제외
    else:
        mean_iou = class_ious[0] if class_ious else 0.0
    
    return {
        'pixel_accuracy': pixel_accuracy,
        'mean_iou': mean_iou,
        'class_ious': class_ious,
        'class_accuracies': class_accuracies
    }


def main():
    parser = argparse.ArgumentParser(description="TFLite 모델 메트릭 평가 스크립트")
    
    parser.add_argument('--model_path', 
                       type=str, 
                       default='./exported_models/tflite_models/Model_quant.tflite',
                       help="TFLite 모델 파일 경로")
    
    parser.add_argument('--json_path',
                       type=str,
                       default='data/COCO/test_without_street.json',
                       help="COCO JSON 파일 경로")
    
    parser.add_argument('--image_dir',
                       type=str,
                       default='data/images',
                       help="이미지 디렉터리 경로")
    
    # 클래스는 6개로 고정 (배경 + 5개 클래스)
    # parser.add_argument는 제거하고 고정값 사용
    
    args = parser.parse_args()
    
    # TFLite 모델 로드
    print(f"TFLite 모델 로딩: {args.model_path}")
    try:
        interpreter = tf.lite.Interpreter(model_path=args.model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"오류: TFLite 모델을 로드할 수 없습니다: {e}")
        return
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    print(f"모델 입력 크기: {input_details['shape']}")
    print(f"모델 입력 타입: {input_details['dtype']}")
    
    # COCO 데이터 로드 (클래스 6개 고정: 배경 + dry, humid, slush, snow, wet)
    images_data, category_mapping = load_coco_data(
        args.json_path, args.image_dir
    )
    
    if not images_data:
        print("처리할 이미지가 없습니다.")
        return
    
    num_classes = len(category_mapping) + 1  # 배경 포함
    print(f"총 클래스 수: {num_classes} (배경 포함)")
    
    # 전체 메트릭 누적용 변수
    total_metrics = {
        'pixel_accuracy': [],
        'mean_iou': [],
        'class_ious': [[] for _ in range(num_classes)],
        'class_accuracies': [[] for _ in range(num_classes)]
    }
    
    # 각 이미지에 대해 추론 및 메트릭 계산
    print("\n이미지별 추론 및 메트릭 계산 시작...")
    
    for i, data in enumerate(tqdm(images_data, desc="Processing images")):
        try:
            image_path = data['image_path']
            image_info = data['image_info']
            annotations = data['annotations']
            
            # Ground truth 마스크 생성
            height, width = image_info['height'], image_info['width']
            gt_mask = create_ground_truth_mask(annotations, category_mapping, height, width)
            
            # 추론 수행
            pred_mask = run_inference_single(
                interpreter, image_path, input_details, output_details
            )
            
            # 메트릭 계산
            metrics = calculate_metrics(pred_mask, gt_mask, num_classes)
            
            # 누적
            total_metrics['pixel_accuracy'].append(metrics['pixel_accuracy'])
            total_metrics['mean_iou'].append(metrics['mean_iou'])
            
            for class_id in range(num_classes):
                total_metrics['class_ious'][class_id].append(metrics['class_ious'][class_id])
                total_metrics['class_accuracies'][class_id].append(metrics['class_accuracies'][class_id])
            
            # 진행 상황 출력 (매 50장마다)
            if (i + 1) % 50 == 0:
                current_pixel_acc = np.mean(total_metrics['pixel_accuracy'])
                current_mean_iou = np.mean(total_metrics['mean_iou'])
                print(f"\n현재까지 평균 - Pixel Acc: {current_pixel_acc:.4f}, Mean IoU: {current_mean_iou:.4f}")
        
        except Exception as e:
            print(f"\n오류 발생 (이미지: {data['image_path']}): {e}")
            continue
    
    # 최종 메트릭 계산 및 출력
    print("\n" + "="*60)
    print("최종 평가 결과")
    print("="*60)
    
    if total_metrics['pixel_accuracy']:
        final_pixel_acc = np.mean(total_metrics['pixel_accuracy'])
        final_mean_iou = np.mean(total_metrics['mean_iou'])
        
        print(f"전체 Pixel Accuracy: {final_pixel_acc:.4f}")
        print(f"전체 Mean IoU: {final_mean_iou:.4f}")
        
        print(f"\n클래스별 IoU:")
        class_names = ['background', 'dry', 'humid', 'slush', 'snow', 'wet']
        for class_id in range(num_classes):
            if total_metrics['class_ious'][class_id]:
                class_iou = np.mean(total_metrics['class_ious'][class_id])
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                print(f"  {class_name}: {class_iou:.4f}")
        
        print(f"\n클래스별 Accuracy:")
        for class_id in range(num_classes):
            if total_metrics['class_accuracies'][class_id]:
                class_acc = np.mean(total_metrics['class_accuracies'][class_id])
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                print(f"  {class_name}: {class_acc:.4f}")
        
        print(f"\n처리된 이미지 수: {len(total_metrics['pixel_accuracy'])}")
    else:
        print("메트릭을 계산할 수 있는 이미지가 없습니다.")
    
    print("="*60)


if __name__ == '__main__':
    main() 