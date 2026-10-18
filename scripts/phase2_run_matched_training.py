"""Locked six-run phase-2 trainer with optimizer-step auditing.

Safety: without --execute this performs preflight only. Formal training is intentionally
not launched by this phase-2 preparation turn.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
ULTRALYTICS_ROOT = WORKSPACE / "vs_code" / "ultralytics-main"
MODEL = ULTRALYTICS_ROOT / "yolo11n.pt"
MODEL_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
DATA_ROOT = PROJECT / "data" / "phase2_matched_20260919"
RUN_ROOT = PROJECT / "runs" / "phase3_matched_20260919"
SMOKE_RUN_ROOT = PROJECT / "runs" / "phase3_smoke_20260919"
PROTOCOL = PROJECT / "reports" / "final_independent_audit" / "phase2" / "matched_training_protocol.md"
EPOCHS = 100
BATCH = 8
EXPECTED_IMAGES = 2380
EXPECTED_BATCHES_PER_EPOCH = math.ceil(EXPECTED_IMAGES / BATCH)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_optimizer_steps(nb: int = EXPECTED_BATCHES_PER_EPOCH, epochs: int = EPOCHS) -> tuple[int, list[int]]:
    warmup_iterations = max(round(3.0 * nb), 100)
    last_opt_step = -1
    total = 0
    per_epoch = []
    for epoch in range(epochs):
        epoch_steps = 0
        for batch_index in range(nb):
            ni = batch_index + nb * epoch
            accumulate = max(1, int(np.interp(ni, [0, warmup_iterations], [1, 64 / BATCH]).round())) if ni <= warmup_iterations else 8
            if ni - last_opt_step >= accumulate:
                total += 1
                epoch_steps += 1
                last_opt_step = ni
        per_epoch.append(epoch_steps)
    return total, per_epoch


def count_images_from_yaml(data_yaml: Path) -> int:
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    train = root / cfg["train"]
    return sum(1 for p in train.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})


def preflight(arm: str, seed: int, epochs: int = EPOCHS, run_root: Path = RUN_ROOT) -> dict:
    if arm not in {"c1", "a1_fixed"}:
        raise ValueError(arm)
    if seed not in {42, 43, 44}:
        raise ValueError(seed)
    data_yaml = DATA_ROOT / ("c1_merchant_repeat" if arm == "c1" else "a1_fixed") / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Materialize the matched datasets first: {data_yaml}")
    if sha256_file(MODEL) != MODEL_SHA256:
        raise RuntimeError("Initial YOLO11n weight hash differs from the locked protocol")
    images = count_images_from_yaml(data_yaml)
    if images != EXPECTED_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_IMAGES} train images, found {images}")
    steps, per_epoch = expected_optimizer_steps(epochs=epochs)
    return {
        "arm": arm,
        "seed": seed,
        "data": str(data_yaml.resolve()),
        "train_images": images,
        "epochs": epochs,
        "batch": BATCH,
        "batches_per_epoch": EXPECTED_BATCHES_PER_EPOCH,
        "mini_batches_total": EXPECTED_BATCHES_PER_EPOCH * epochs,
        "expected_optimizer_steps": steps,
        "optimizer_steps_first_five_epochs": per_epoch[:5],
        "model": str(MODEL.resolve()),
        "model_sha256": MODEL_SHA256,
        "protocol_sha256": sha256_file(PROTOCOL),
        "run_directory": str((run_root / f"{arm}_seed{seed}").resolve()),
        "run_type": "formal" if run_root == RUN_ROOT else "smoke_test",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("c1", "a1_fixed"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Explicitly authorize one formal 100-epoch training run")
    mode.add_argument("--smoke-test", action="store_true", help="Run isolated 2-epoch operational smoke test; never a formal result")
    args = parser.parse_args()
    run_epochs = 2 if args.smoke_test else EPOCHS
    run_root = SMOKE_RUN_ROOT if args.smoke_test else RUN_ROOT
    audit = preflight(args.arm, args.seed, epochs=run_epochs, run_root=run_root)
    if not args.execute and not args.smoke_test:
        print(json.dumps({"status": "PREFLIGHT_ONLY", **audit}, ensure_ascii=False, indent=2))
        return
    run_dir = Path(audit["run_directory"])
    if run_dir.exists():
        raise FileExistsError(f"Refusing to reuse or overwrite run directory: {run_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for formal matched training")
    if str(ULTRALYTICS_ROOT) not in sys.path:
        sys.path.insert(0, str(ULTRALYTICS_ROOT))
    from ultralytics import YOLO, __version__ as ultralytics_version
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    if ultralytics_version != "8.4.60":
        raise RuntimeError(f"Expected Ultralytics 8.4.60, got {ultralytics_version}")

    class CountingDetectionTrainer(DetectionTrainer):
        def __init__(self, *trainer_args, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            self.audited_optimizer_steps = 0
            self.audited_actual_optimizer_updates = 0
            self.audited_amp_scale_decrease_events = 0
            self.audited_train_batches_total = 0
            self.audited_train_batches_by_epoch = {}

        def _build_train_pipeline(self):
            super()._build_train_pipeline()
            original_step = self.optimizer.step

            def counted_step(*step_args, **step_kwargs):
                self.audited_actual_optimizer_updates += 1
                return original_step(*step_args, **step_kwargs)

            self.optimizer.step = counted_step

        def preprocess_batch(self, batch):
            epoch = int(self.epoch)
            self.audited_train_batches_total += 1
            self.audited_train_batches_by_epoch[epoch] = self.audited_train_batches_by_epoch.get(epoch, 0) + 1
            return super().preprocess_batch(batch)

        def optimizer_step(self):
            scale_before = float(self.scaler.get_scale())
            super().optimizer_step()
            self.audited_optimizer_steps += 1
            if bool(self.amp) and float(self.scaler.get_scale()) < scale_before:
                self.audited_amp_scale_decrease_events += 1

    model = YOLO(str(MODEL.resolve()))
    model.train(
        trainer=CountingDetectionTrainer,
        data=audit["data"], epochs=run_epochs, batch=BATCH, imgsz=640,
        patience=0, workers=0, device=0, seed=args.seed, deterministic=True,
        amp=True, cache=False, optimizer="SGD", lr0=0.01, lrf=0.01,
        momentum=0.937, weight_decay=0.0005, cos_lr=True,
        warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
        nbs=64, close_mosaic=10, mosaic=1.0, mixup=0.0, cutmix=0.0,
        copy_paste=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
        perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0,
        rect=False, multi_scale=0.0, compile=False, val=True,
        save=True, save_period=-1, project=str(run_root.resolve()),
        name=f"{args.arm}_seed{args.seed}", exist_ok=False, plots=True,
    )
    trainer = model.trainer
    results_csv = run_dir / "results.csv"
    with results_csv.open(encoding="utf-8", newline="") as handle:
        completed_epochs = sum(1 for _ in csv.DictReader(handle))
    actual = {
        **audit,
        "ultralytics_version": ultralytics_version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "completed_epochs": completed_epochs,
        "actual_batches_per_epoch": len(trainer.train_loader),
        "actual_train_batches_total": trainer.audited_train_batches_total,
        "actual_train_batches_by_epoch": trainer.audited_train_batches_by_epoch,
        "actual_optimizer_steps": trainer.audited_optimizer_steps,
        "actual_successful_optimizer_updates": trainer.audited_actual_optimizer_updates,
        "amp_enabled_actual": bool(trainer.amp),
        "amp_scale_decrease_events": trainer.audited_amp_scale_decrease_events,
        "amp_skipped_optimizer_steps_inferred": trainer.audited_optimizer_steps - trainer.audited_actual_optimizer_updates,
        "primary_checkpoint": str((run_dir / "weights" / "last.pt").resolve()),
        "best_checkpoint_secondary_only": str((run_dir / "weights" / "best.pt").resolve()),
    }
    expected_steps = int(audit["expected_optimizer_steps"])
    expected_batches = EXPECTED_BATCHES_PER_EPOCH * run_epochs
    success = (
        completed_epochs == run_epochs
        and len(trainer.train_loader) == EXPECTED_BATCHES_PER_EPOCH
        and trainer.audited_train_batches_total == expected_batches
        and all(trainer.audited_train_batches_by_epoch.get(epoch) == EXPECTED_BATCHES_PER_EPOCH for epoch in range(run_epochs))
        and trainer.audited_optimizer_steps == expected_steps
    )
    if not success:
        (run_dir / "matched_budget_audit_FAILED.json").write_text(json.dumps(actual, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError("Matched-budget assertion failed; do not evaluate this run")
    actual["budget_schedule_passed"] = True
    actual["successful_updates_equal_scheduled_calls"] = trainer.audited_actual_optimizer_updates == expected_steps
    actual["pairwise_successful_update_match_required_before_interpretation"] = True
    (run_dir / "matched_budget_audit.json").write_text(json.dumps(actual, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
