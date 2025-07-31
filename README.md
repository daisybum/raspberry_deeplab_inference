# Raspberry DeepLab Inference

라즈베리파이4 (ARMv8) 환경에서 **도로 상태 5-클래스 세그멘테이션**(dry / humid / slush / snow / wet)을 실시간으로 추론하고, COCO 형식 GT로 성능을 평가·시각화하는 프로젝트입니다.

## 프로젝트 구조

```
raspberry_deeplab_inference/
├─ evaluate_tflite_metrics.py   # COCO 평가 (PixelAcc / IoU / Dice + 시간·메모리 통계)
├─ visualize_segmentation.py    # 원본·Pred·GT 오버레이 시각화
├─ models/
│   ├─ tflite_models/Model_quant.tflite     # 양자화 TFLite 모델 (512×512, float32)
│   └─ ... (SavedModel 백업)
├─ docker-compose.yml & Dockerfile         # 라즈베리파이4용 실행 환경
└─ README.md
```

## 요구 사항

| 항목 | 버전 |
|------|------|
| Python | ≥ 3.8 |
| TensorFlow | 2.13+ (라즈베리파이용 aarch64 wheel 권장) |
| psutil | latest |
| numpy / pillow / matplotlib / tqdm | | 

설치 예시:
```bash
pip install tensorflow-aarch64==2.13.0 numpy pillow matplotlib tqdm psutil
```

Docker 사용 시 `docker-compose.yml` 로 자동 설치됩니다.

## 모델 평가 (evaluate_tflite_metrics.py)

```bash
python3 evaluate_tflite_metrics.py \
  --tflite_model models/tflite_models/Model_quant.tflite \
  --json_path /path/to/COCO/test_without_street.json \
  --image_dir /path/to/images \
  --input_height 512 --input_width 512
```

스크립트 기능
1. COCO JSON + 이미지 로드 (background 포함 6-클래스)
2. 모든 이미지 추론 → PixelAcc / IoU / Dice 계산 (배경 포함 & 제외)
3. **추론 시간**·**메모리 사용량**(RSS) max / avg / min 통계 출력

### 출력 예시 (라즈베리파이4, 1482장)

```
============================================================
TFLite 모델 최종 평가 결과
============================================================
Pixel Accuracy (전체): 0.9825
Pixel Accuracy (배경 제외): 0.9721
Mean IoU (전체): 0.9655
Mean IoU (배경 제외): 0.9624
Mean Dice (전체): 0.9716
Mean Dice (배경 제외): 0.9703

추론 시간 (초): max=0.0684, avg=0.0431, min=0.0410
메모리 사용량 (MB): max=182.44, avg=178.87, min=176.33

클래스별 IoU / Dice:
  background | IoU: 0.9827 | Dice: 0.9912
  dry        | IoU: 0.9565 | Dice: 0.9609
  humid      | IoU: 0.9389 | Dice: 0.9447
  slush      | IoU: 0.9616 | Dice: 0.9715
  snow       | IoU: 0.9659 | Dice: 0.9709
  wet        | IoU: 0.9873 | Dice: 0.9902
============================================================
```

> **하드웨어**: Raspberry Pi 4 Model B 4GB, 64-bit OS, Thread ×4, XNNPACK delegate

## 시각화 (visualize_segmentation.py)

테스트셋 중 임의의 N장(기본 3장)을 선택해 **원본 / Prediction Overlay / GT Overlay** 이미지를 `viz_outputs/` 폴더에 저장합니다.

```bash
python3 visualize_segmentation.py \
  --model_path models/tflite_models/Model_quant.tflite \
  --json_path /path/to/COCO/test_without_street.json \
  --image_dir /path/to/images \
  --num_images 3 \
  --output_dir viz_outputs
```

<p align="center">
  <img src="viz_outputs/viz_1.png" width="720"/>
</p>

## Docker 실행 (선택)

```bash
docker compose up -d  # 라즈베리파이에서 카메라 권한 포함 실행

docker compose exec pi_inference python3 evaluate_tflite_metrics.py \
  --tflite_model models/tflite_models/Model_quant.tflite \
  --json_path /workspace/data/COCO/test_without_street.json \
  --image_dir /workspace/data/images/
```

## 라이선스

MIT License. 