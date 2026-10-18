#!/usr/bin/env python3
"""Generate Phase 4B manuscript figures from the corrected Phase 3 CSVs only."""
from __future__ import annotations

import csv
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
REVISION = SCRIPT.parents[1]
PROJECT = SCRIPT.parents[1]
AUDIT = PROJECT / "results" / "corrected_phase3"
SEED_METRICS = AUDIT / "corrected_phase3_seed_metrics.csv"
MATCHED_DELTAS = AUDIT / "corrected_phase3_matched_deltas.csv"
ACTION_RISK = AUDIT / "corrected_phase3_action_risk.csv"
OUT = REVISION / "figures"
OUTPUT_SUFFIX = "_revised" if "--word-revision" in sys.argv else ""

VIEW_LABELS = {
    "external_development": "External\ndevelopment",
    "merchant_source_development": "Merchant\ndevelopment",
    "merchant_reused_historical_test": "Merchant\nhistorical",
    "external_reused_historical_test": "External\nhistorical",
    "sds_reused_historical_test": "Corrected SDS\nhistorical",
    "yurdakul_reused_historical_test": "Yurdakul\npublic snapshot",
}
METRICS = {
    "delta_map50_95": "Overall\nmAP50–95",
    "delta_fully_ap50": "Fully\nAP50",
    "delta_green_ap50": "Green\nAP50",
    "delta_half_ap50": "Half\nAP50",
}
METRIC_COLORS = {
    "delta_map50_95": "#0072B2",
    "delta_fully_ap50": "#009E73",
    "delta_green_ap50": "#CC79A7",
    "delta_half_ap50": "#D55E00",
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT.resolve()).as_posix()


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = pd.read_csv(SEED_METRICS)
    delta = pd.read_csv(MATCHED_DELTAS)
    risk = pd.read_csv(ACTION_RISK)
    assert len(seed) == 36, f"expected 36 seed-metric rows, found {len(seed)}"
    paired = delta.loc[delta["row_type"] == "paired_seed"].copy()
    assert len(paired) == 18, f"expected 18 paired rows, found {len(paired)}"
    assert len(risk) == 36, f"expected 36 action-risk rows, found {len(risk)}"
    assert set(paired["seed"].astype(int)) == {42, 43, 44}
    assert set(paired.loc[paired["seed"].astype(int).isin([42, 44]), "strict_budget_matched"].astype(str)) == {"True"}
    assert set(paired.loc[paired["seed"].astype(int) == 43, "analysis_set"]) == {"descriptive_sensitivity_only"}
    assert set(risk["matcher_for_semantic_counts"]) == {"class_agnostic_IoU>=0.5"}
    assert set(risk["matcher_for_fully_detection_metrics"]) == {"class_aware_IoU>=0.5"}
    return seed, paired, risk


def export_figure(fig: mpl.figure.Figure, stem: str) -> list[str]:
    outputs = []
    for suffix, kwargs in (
        (".png", {"dpi": 600}),
        (".pdf", {}),
        (".svg", {}),
    ):
        path = OUT / f"{stem}{OUTPUT_SUFFIX}{suffix}"
        fig.savefig(path, **kwargs)
        outputs.append(rel(path))
    return outputs


def figure_class_change(paired: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    paired = paired.copy()
    paired["seed"] = paired["seed"].astype(int)
    rows = []
    for _, row in paired.iterrows():
        for metric, label in METRICS.items():
            rows.append(
                {
                    "view": row["view"],
                    "view_label": VIEW_LABELS[row["view"]].replace("\n", " "),
                    "seed": int(row["seed"]),
                    "analysis_status": "strict" if int(row["seed"]) in (42, 44) else "sensitivity-only",
                    "metric": metric.replace("delta_", ""),
                    "metric_label": label.replace("\n", " "),
                    "delta_a1_fixed_minus_c1": float(row[metric]),
                }
            )
    source = pd.DataFrame(rows)
    source.to_csv(OUT / f"source_data_agriculture_class_performance_change{OUTPUT_SUFFIX}.csv", index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(7.16, 5.7), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.55])
    ax1 = fig.add_subplot(gs[0, 0])
    ext = source.loc[source["view"] == "external_development"]
    x = np.arange(len(METRICS))
    seed_style = {
        42: ("o", "-", "#0072B2", "Seed 42 (strict)"),
        44: ("s", "-", "#D55E00", "Seed 44 (strict)"),
        43: ("D", "--", "#555555", "Seed 43 (sensitivity only)"),
    }
    for seed in (42, 44, 43):
        subset = ext.loc[ext["seed"] == seed].set_index("metric")
        order = [key.replace("delta_", "") for key in METRICS]
        values = [subset.loc[key, "delta_a1_fixed_minus_c1"] for key in order]
        marker, linestyle, color, label = seed_style[seed]
        ax1.plot(
            x,
            values,
            marker=marker,
            linestyle=linestyle,
            color=color,
            label=label,
            markerfacecolor="white" if seed == 43 else color,
            markeredgewidth=1.1,
            linewidth=1.25,
            zorder=3,
        )
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xticks(x, [label for label in METRICS.values()])
    ax1.set_ylabel("Change (A1-fixed − C1)")
    ax1.set_title("A  External development: overall gain coexists with Green and Half AP50 losses", loc="left", fontweight="bold")
    ax1.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, loc="outside upper center", frameon=False)

    ax2 = fig.add_subplot(gs[1, 0])
    strict = source.loc[source["analysis_status"] == "strict"].copy()
    strict_mean = strict.groupby(["view", "metric"], as_index=False)["delta_a1_fixed_minus_c1"].mean()
    view_order = list(VIEW_LABELS)
    metric_order = [key.replace("delta_", "") for key in METRICS]
    matrix = np.array(
        [
            [
                strict_mean.loc[(strict_mean["view"] == view) & (strict_mean["metric"] == metric), "delta_a1_fixed_minus_c1"].iloc[0]
                for metric in metric_order
            ]
            for view in view_order
        ]
    )
    limit = max(abs(matrix.min()), abs(matrix.max()))
    image = ax2.imshow(matrix, cmap="RdBu", norm=mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if abs(value) > 0.55 * limit else "black"
            label = f"{value:+.4f}" if abs(value) < 0.001 else f"{value:+.3f}"
            ax2.text(j, i, label, ha="center", va="center", color=color, fontsize=7.2)
    ax2.set_xticks(np.arange(len(metric_order)), [METRICS[f"delta_{metric}"] for metric in metric_order])
    ax2.set_yticks(np.arange(len(view_order)), [VIEW_LABELS[view] for view in view_order])
    ax2.set_title("B  Strict paired-seed mean (seeds 42 and 44)", loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax2, pad=0.02, fraction=0.035)
    colorbar.set_label("Mean change")
    outputs = export_figure(fig, "agriculture_class_performance_change")
    plt.close(fig)
    return source, outputs


def figure_action_risk(risk: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    selected = {
        "external_development": "External development",
        "merchant_source_development": "Merchant development",
        "sds_reused_historical_test": "Corrected SDS historical",
        "yurdakul_reused_historical_test": "Yurdakul public snapshot",
    }
    subset = risk.loc[risk["view"].isin(selected)].copy()
    subset["seed"] = subset["seed"].astype(int)
    key_cols = ["view", "seed", "half_gt"]
    pivot = subset.pivot_table(index=key_cols, columns="arm", values=["half_to_fully", "half_to_fully_per_complete_half_gt"], aggfunc="first")
    rows = []
    for (view, seed, half_gt), values in pivot.iterrows():
        c1_count = int(values[("half_to_fully", "c1")])
        a1_count = int(values[("half_to_fully", "a1_fixed")])
        c1_rate = float(values[("half_to_fully_per_complete_half_gt", "c1")])
        a1_rate = float(values[("half_to_fully_per_complete_half_gt", "a1_fixed")])
        rows.append(
            {
                "view": view,
                "source_label": selected[view],
                "seed": seed,
                "analysis_status": "strict" if seed in (42, 44) else "sensitivity-only",
                "score_threshold": 0.25,
                "matcher": "class-agnostic IoU>=0.5",
                "half_gt_fixed_denominator": int(half_gt),
                "c1_half_to_fully_count": c1_count,
                "a1_fixed_half_to_fully_count": a1_count,
                "absolute_change_count": a1_count - c1_count,
                "c1_per_half_gt": c1_rate,
                "a1_fixed_per_half_gt": a1_rate,
                "normalized_change_per_half_gt": a1_rate - c1_rate,
            }
        )
    source = pd.DataFrame(rows).sort_values(["view", "seed"])
    source.to_csv(OUT / f"source_data_agriculture_half_to_fully_change{OUTPUT_SUFFIX}.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 1, figsize=(7.16, 5.7), sharex=True, layout="constrained")
    order = list(selected)
    x = np.arange(len(order))
    offsets = {42: -0.18, 43: 0.0, 44: 0.18}
    styles = {
        42: ("o", "#0072B2", "Seed 42 (strict)"),
        44: ("s", "#D55E00", "Seed 44 (strict)"),
        43: ("D", "#555555", "Seed 43 (sensitivity only)"),
    }
    for seed in (42, 44, 43):
        marker, color, label = styles[seed]
        seed_rows = source.loc[source["seed"] == seed].set_index("view")
        xpos = x + offsets[seed]
        delta_count = [seed_rows.loc[view, "absolute_change_count"] for view in order]
        delta_rate = [100 * seed_rows.loc[view, "normalized_change_per_half_gt"] for view in order]
        face = "white" if seed == 43 else color
        axes[0].scatter(xpos, delta_count, marker=marker, s=38, facecolors=face, edgecolors=color, linewidths=1.1, label=label, zorder=3)
        axes[1].scatter(xpos, delta_rate, marker=marker, s=38, facecolors=face, edgecolors=color, linewidths=1.1, zorder=3)
        for px, view, value in zip(xpos, order, delta_count):
            row = seed_rows.loc[view]
            axes[0].annotate(
                f"{int(row['c1_half_to_fully_count'])}→{int(row['a1_fixed_half_to_fully_count'])}",
                (px, value),
                xytext=(0, 5 if value >= 0 else -11),
                textcoords="offset points",
                ha="center",
                fontsize=6.2,
                color=color,
            )
    for axis in axes:
        axis.axhline(0, color="black", linewidth=0.8)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[0].set_ylim(-10, 20)
    axes[0].set_ylabel("Absolute change in\nhalf→fully matches")
    axes[1].set_ylabel("Change / fixed half GT\n(percentage points)")
    axes[1].set_xticks(x, [selected[view].replace(" ", "\n", 1) for view in order])
    axes[0].set_title("A  Fixed-threshold harvest-selection response at score 0.25 (C1→A1-fixed)", loc="left", fontweight="bold")
    axes[1].set_title("B  Same change normalized by the fixed source-specific half-GT count", loc="left", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, loc="outside upper center", frameon=False)
    axes[1].text(
        0.01,
        0.02,
        "Corrected SDS: half GT = 7; normalized values are highly discrete.",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
        color="#333333",
    )
    outputs = export_figure(fig, "agriculture_half_to_fully_change")
    plt.close(fig)
    return source, outputs


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    configure()
    seed, paired, risk = load_and_validate()
    class_source, class_outputs = figure_class_change(paired)
    risk_source, risk_outputs = figure_action_risk(risk)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": rel(SCRIPT),
        "python": platform.python_version(),
        "matplotlib": mpl.__version__,
        "pandas": pd.__version__,
        "inputs": [rel(SEED_METRICS), rel(MATCHED_DELTAS), rel(ACTION_RISK)],
        "prohibited_input_confirmed_absent": "no SDS legacy JSON is read by this script",
        "models": "C1 budget-matched merchant-repeat control and A1-fixed mapped-source arm; unchanged YOLO11n",
        "seeds": {"strict": [42, 44], "sensitivity_only": [43]},
        "protocol": {
            "metrics": "corrected Phase 3 paired outputs",
            "score_threshold": 0.25,
            "semantic_matcher": "class-agnostic IoU>=0.5",
            "normalization": "half→fully count divided by the fixed complete half-GT count for that source",
        },
        "dataset_identity": {
            "external_development": "External development view, 141 images",
            "merchant_source_development": "Merchant development view, 473 images",
            "merchant_reused_historical_test": "Merchant reused historical test, 472 images",
            "external_reused_historical_test": "External reused historical test, 142 images",
            "sds_reused_historical_test": "Corrected SDS historical view, 26 images; corrected valid JSON only",
            "yurdakul_reused_historical_test": "Yurdakul public Kaggle snapshot used by the experiment, 566 images and 1734 polygon-derived boxes; formal paper reports 1169 boxes",
        },
        "figures": [
            {
                "id": "agriculture_class_performance_change",
                "outputs": class_outputs,
                "source_data": rel(OUT / f"source_data_agriculture_class_performance_change{OUTPUT_SUFFIX}.csv"),
                "rows": len(class_source),
            },
            {
                "id": "agriculture_half_to_fully_change",
                "outputs": risk_outputs,
                "source_data": rel(OUT / f"source_data_agriculture_half_to_fully_change{OUTPUT_SUFFIX}.csv"),
                "rows": len(risk_source),
            },
        ],
    }
    (OUT / f"agriculture_figure_generation_manifest{OUTPUT_SUFFIX}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
