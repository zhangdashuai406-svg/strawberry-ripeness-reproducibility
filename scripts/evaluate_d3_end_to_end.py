from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ULTRALYTICS_ROOT = WORKSPACE_ROOT / "vs_code" / "ultralytics-main"
if str(ULTRALYTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_ROOT))

from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)
YOLO_NAMES = {0: "fully_ripened", 1: "green", 2: "half_ripened"}
RANK_TO_YOLO = np.asarray([1, 2, 0], dtype=np.int64)
SDS_NATIVE_TO_YOLO = {0: 2, 1: 1, 2: 2, 3: 0, 4: 2, 5: 1}
# Yurdakul et al. 2026 (Kaggle mahyeks, Roboflow export, 1734 objects):
#   native id 0 = fully-ripe (520), 1 = semi-ripe (247), 2 = unripe (967),
#   verified against paper Table 3 and visually (red/orange/green). See
#   data/yurdakul_turkey_frozen/dataset_audit.json.
YURDAKUL_NATIVE_TO_YOLO = {0: 0, 1: 2, 2: 1}

# The provenance flags written into each record are derived from the directories the run was
# pointed at, against the one registry of frozen partitions in scripts/frozen_partitions.py.
# They used to be literals, which made them a description of the code rather than of the run.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from frozen_partitions import frozen_partition_evaluated


class MultiHeadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = mobilenet_v3_small(weights=None)
        self.encoder = nn.Sequential(
            base.features,
            base.avgpool,
            nn.Flatten(),
            base.classifier[0],
            base.classifier[1],
            base.classifier[2],
        )
        self.head3 = nn.Linear(1024, 3)
        self.head6 = nn.Linear(1024, 6)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(image)
        return self.head3(features), self.head6(features)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot decode image: {path}")
    return image


def read_labels(path: Path, width: int, height: int, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    classes: list[int] = []
    boxes: list[list[float]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            native = int(float(fields[0]))
            if dataset == "strawberry_ds":
                cls = SDS_NATIVE_TO_YOLO[native]
            elif dataset == "yurdakul":
                cls = YURDAKUL_NATIVE_TO_YOLO[native]
            else:
                cls = native
            xc, yc, bw, bh = map(float, fields[1:5])
            classes.append(cls)
            boxes.append(
                [
                    (xc - bw / 2) * width,
                    (yc - bh / 2) * height,
                    (xc + bw / 2) * width,
                    (yc + bh / 2) * height,
                ]
            )
    return np.asarray(classes, dtype=np.int64), np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    wh = np.maximum(rb - lt, 0.0)
    inter = wh[:, 0] * wh[:, 1]
    area_a = max(float((box[2] - box[0]) * (box[3] - box[1])), 0.0)
    area_b = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def crop_with_margin(image: np.ndarray, box: np.ndarray, margin: float = 0.10) -> Image.Image:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    bw, bh = x2 - x1, y2 - y1
    left = max(0, int(np.floor(x1 - margin * bw)))
    top = max(0, int(np.floor(y1 - margin * bh)))
    right = min(width, int(np.ceil(x2 + margin * bw)))
    bottom = min(height, int(np.ceil(y2 + margin * bh)))
    if right <= left or bottom <= top:
        raise RuntimeError(f"Invalid predicted crop: {box.tolist()}")
    crop_rgb = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


def build_classifier(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, transforms.Compose]:
    model = MultiHeadModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return model, transform


@torch.inference_mode()
def classify_crops(
    model: nn.Module,
    transform,
    crops: list[Image.Image],
    dataset: str,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not crops:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
    probabilities = []
    for start in range(0, len(crops), batch_size):
        batch = torch.stack([transform(crop) for crop in crops[start : start + batch_size]]).to(device)
        logits3, logits6 = model(batch)
        if dataset == "strawberry_ds":
            p6 = torch.softmax(logits6, dim=1)
            common_rank = torch.stack((p6[:, 0] + p6[:, 1], p6[:, 2] + p6[:, 3] + p6[:, 4], p6[:, 5]), dim=1)
        else:
            common_rank = torch.softmax(logits3, dim=1)
        probabilities.append(common_rank.cpu())
    probs = torch.cat(probabilities, dim=0).numpy()
    rank = probs.argmax(axis=1)
    return RANK_TO_YOLO[rank], probs[np.arange(len(probs)), rank].astype(np.float32)


def ap101(confidence: np.ndarray, true_positive: np.ndarray, total_gt: int) -> float:
    if total_gt == 0 or len(confidence) == 0:
        return 0.0
    order = np.argsort(-confidence, kind="stable")
    tp = np.cumsum(true_positive[order])
    fp = np.cumsum(1 - true_positive[order])
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.mean([precision[recall >= level].max() if np.any(recall >= level) else 0.0 for level in np.linspace(0, 1, 101)]))


def calculate_metrics(records: list[dict], score_threshold: float) -> dict:
    gt_counts = {cls: sum(int(np.sum(r["gt_classes"] == cls)) for r in records) for cls in YOLO_NAMES}
    ap_by_class: dict[str, dict] = {}
    threshold_keys = [f"{threshold:.2f}" for threshold in IOU_THRESHOLDS]

    for cls, name in YOLO_NAMES.items():
        cls_predictions = []
        for image_index, record in enumerate(records):
            indices = np.flatnonzero(record["pred_classes"] == cls)
            for pred_index in indices:
                cls_predictions.append((float(record["scores"][pred_index]), image_index, int(pred_index)))
        cls_predictions.sort(key=lambda item: item[0], reverse=True)
        aps = []
        ap_at_iou = {}
        for iou_threshold, key in zip(IOU_THRESHOLDS, threshold_keys):
            matched: dict[int, set[int]] = defaultdict(set)
            flags = []
            confidences = []
            for score, image_index, pred_index in cls_predictions:
                record = records[image_index]
                gt_indices = np.flatnonzero(record["gt_classes"] == cls)
                overlaps = box_iou(record["pred_boxes"][pred_index], record["gt_boxes"][gt_indices])
                order = np.argsort(-overlaps)
                chosen = None
                for local_index in order:
                    original_index = int(gt_indices[local_index])
                    if original_index not in matched[image_index] and overlaps[local_index] >= iou_threshold:
                        chosen = original_index
                        break
                flags.append(int(chosen is not None))
                confidences.append(score)
                if chosen is not None:
                    matched[image_index].add(chosen)
            ap = ap101(np.asarray(confidences), np.asarray(flags, dtype=np.int64), gt_counts[cls])
            ap_at_iou[key] = ap
            aps.append(ap)
        ap_by_class[name] = {"gt": gt_counts[cls], "ap50": aps[0], "map50_95": float(np.mean(aps)), "ap_by_iou": ap_at_iou}

    op_tp = op_fp = op_fn = 0
    op_per_class = {}
    for cls, name in YOLO_NAMES.items():
        tp = fp = 0
        gt_total = gt_counts[cls]
        for record in records:
            pred_indices = np.flatnonzero((record["pred_classes"] == cls) & (record["scores"] >= score_threshold))
            pred_indices = pred_indices[np.argsort(-record["scores"][pred_indices])]
            gt_indices = np.flatnonzero(record["gt_classes"] == cls)
            matched: set[int] = set()
            for pred_index in pred_indices:
                overlaps = box_iou(record["pred_boxes"][pred_index], record["gt_boxes"][gt_indices])
                order = np.argsort(-overlaps)
                chosen = None
                for local_index in order:
                    original_index = int(gt_indices[local_index])
                    if original_index not in matched and overlaps[local_index] >= 0.5:
                        chosen = original_index
                        break
                if chosen is None:
                    fp += 1
                else:
                    tp += 1
                    matched.add(chosen)
        fn = gt_total - tp
        op_tp += tp
        op_fp += fp
        op_fn += fn
        op_per_class[name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / gt_total if gt_total else 0.0,
        }

    return {
        "images": len(records),
        "total_gt": sum(gt_counts.values()),
        "total_predictions_at_detector_floor": sum(len(record["scores"]) for record in records),
        "map50": float(np.mean([value["ap50"] for value in ap_by_class.values()])),
        "map50_95": float(np.mean([value["map50_95"] for value in ap_by_class.values()])),
        "per_class": ap_by_class,
        "operating_point": {
            "score_threshold": score_threshold,
            "tp": op_tp,
            "fp": op_fp,
            "fn": op_fn,
            "precision": op_tp / (op_tp + op_fp) if op_tp + op_fp else 0.0,
            "recall": op_tp / (op_tp + op_fn) if op_tp + op_fn else 0.0,
            "per_class": op_per_class,
        },
        "error_decomposition_at_operating_point": error_decomposition(records, score_threshold),
    }


def error_decomposition(records: list[dict], score_threshold: float) -> dict:
    """Separate localization coverage from maturity classification on localized fruit."""
    total_gt = sum(len(record["gt_classes"]) for record in records)
    localized = 0
    correctly_classified = 0
    confusion = np.zeros((3, 3), dtype=np.int64)
    for record in records:
        pred_indices = np.flatnonzero(record["scores"] >= score_threshold)
        pred_indices = pred_indices[np.argsort(-record["scores"][pred_indices])]
        matched_gt: set[int] = set()
        for pred_index in pred_indices:
            overlaps = box_iou(record["pred_boxes"][pred_index], record["gt_boxes"])
            order = np.argsort(-overlaps)
            chosen = None
            for gt_index in order:
                gt_index = int(gt_index)
                if gt_index not in matched_gt and overlaps[gt_index] >= 0.5:
                    chosen = gt_index
                    break
            if chosen is None:
                continue
            matched_gt.add(chosen)
            localized += 1
            truth = int(record["gt_classes"][chosen])
            prediction = int(record["pred_classes"][pred_index])
            confusion[truth, prediction] += 1
            correctly_classified += int(truth == prediction)
    return {
        "localized_gt": localized,
        "localization_recall": localized / total_gt if total_gt else 0.0,
        "correctly_classified_localized_gt": correctly_classified,
        "classification_accuracy_given_localized": correctly_classified / localized if localized else 0.0,
        "end_to_end_correct_rate": correctly_classified / total_gt if total_gt else 0.0,
        "confusion_yolo_order_fully_green_half": confusion.tolist(),
    }


def predict_dataset(
    system: str,
    detector_path: Path,
    classifier_path: Path | None,
    images: Path,
    labels: Path,
    dataset: str,
    device_name: str,
    detector_conf: float,
    nms_iou: float,
    imgsz: int,
    batch: int,
    classifier_batch: int,
) -> tuple[list[dict], float, int]:
    torch_device = torch.device(f"cuda:{device_name}" if device_name.isdigit() and torch.cuda.is_available() else device_name)
    detector = YOLO(str(detector_path.resolve()))
    classifier = transform = None
    if system == "d3":
        if classifier_path is None:
            raise ValueError("D3 requires --classifier")
        classifier, transform = build_classifier(classifier_path, torch_device)

    image_paths = sorted(path for path in images.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    result_stream = detector.predict(
        source=str(images),
        imgsz=imgsz,
        conf=detector_conf,
        iou=nms_iou,
        batch=batch,
        device=device_name,
        verbose=False,
        save=False,
        stream=True,
    )
    records = []
    degenerate_predictions_dropped = 0
    started = time.time()
    for index, result in enumerate(result_stream, start=1):
        image_path = Path(result.path)
        # Use the exact decoded image associated with the detector result. This
        # keeps predicted coordinates and crops aligned for EXIF-rotated files.
        image = np.ascontiguousarray(result.orig_img)
        height, width = image.shape[:2]
        gt_classes, gt_boxes = read_labels(labels / f"{image_path.stem}.txt", width, height, dataset)
        if result.boxes is None:
            pred_boxes = np.empty((0, 4), dtype=np.float32)
            detector_scores = np.empty((0,), dtype=np.float32)
            detector_classes = np.empty((0,), dtype=np.int64)
        else:
            pred_boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            detector_scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
            detector_classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
        if len(pred_boxes):
            valid = (pred_boxes[:, 2] > pred_boxes[:, 0]) & (pred_boxes[:, 3] > pred_boxes[:, 1])
            degenerate_predictions_dropped += int(np.sum(~valid))
            pred_boxes = pred_boxes[valid]
            detector_scores = detector_scores[valid]
            detector_classes = detector_classes[valid]

        if system == "d0":
            pred_classes = detector_classes
            scores = detector_scores
        else:
            crops = [crop_with_margin(image, box) for box in pred_boxes]
            pred_classes, class_scores = classify_crops(
                classifier, transform, crops, dataset, torch_device, classifier_batch
            )
            scores = detector_scores * class_scores
        records.append(
            {
                "image": image_path.name,
                "gt_classes": gt_classes,
                "gt_boxes": gt_boxes,
                "pred_classes": pred_classes,
                "pred_boxes": pred_boxes,
                "scores": scores,
            }
        )
        if index % 25 == 0 or index == len(image_paths):
            print(f"[{dataset}/{system}] {index}/{len(image_paths)}", flush=True)
    if len(records) != len(image_paths):
        raise RuntimeError(f"Expected {len(image_paths)} images, evaluated {len(records)}")
    return records, time.time() - started, degenerate_predictions_dropped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=("d0", "d3"), required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--classifier", type=Path)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dataset", choices=("merchant", "external", "strawberry_ds", "yurdakul"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--detector-conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--classifier-batch", type=int, default=64)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    args = parser.parse_args()

    records, elapsed, degenerate_predictions_dropped = predict_dataset(
        args.system,
        args.detector,
        args.classifier,
        args.images,
        args.labels,
        args.dataset,
        args.device,
        args.detector_conf,
        args.nms_iou,
        args.imgsz,
        args.batch,
        args.classifier_batch,
    )
    metrics = calculate_metrics(records, args.score_threshold)
    partition = frozen_partition_evaluated(args.images, args.labels)
    output = {
        "system": args.system,
        "dataset": args.dataset,
        "detector": str(args.detector.resolve()),
        "classifier": str(args.classifier.resolve()) if args.classifier else None,
        "evaluated_from": {
            "images": str(Path(args.images).resolve()),
            "labels": str(Path(args.labels).resolve()),
        },
        "frozen_partition": partition,
        "parameters": {
            "detector_conf": args.detector_conf,
            "nms_iou": args.nms_iou,
            "imgsz": args.imgsz,
            "score_threshold": args.score_threshold,
            "d3_score": "detector_confidence * top1_classifier_probability" if args.system == "d3" else None,
            "crop_margin": 0.10 if args.system == "d3" else None,
        },
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "degenerate_predictions_dropped": degenerate_predictions_dropped,
        "validation_only": partition is None,
        "test_sets_touched": partition is not None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
