# Variables and evidence roles

All corrected CSVs have a header row. Blank summary fields mean not applicable, not zero. `c1` and `a1_fixed` denote the two YOLO11n fixed-budget arms; `seed` identifies stochastic initialization/sample-order repeats, not independent populations. Primary pairs are 42 and 44; 43 is sensitivity only. `source_json` is a relocated project-relative path to the corresponding released record.

## Performance CSV

`images` is the number of evaluated images. `total_gt`, `fully_gt`, `green_gt` and `half_gt` are full reference object counts, not numbers successfully matched by a detector. `map50` and `map50_95` are mean AP over the three classes, respectively at IoU 0.50 and averaged over 0.50:0.05:0.95. `fully_ap50`, `green_ap50`, `half_ap50` are class AP50; `fully_map50_95`, `green_map50_95`, `half_map50_95` are class AP averaged over those IoUs. All are on the 0–1 scale. `evidence_status` distinguishes corrected SDS from valid original non-SDS records.

## Matched-delta CSV

`row_type=paired_seed` gives a single within-seed comparison. `row_type=summary` gives a descriptive mean or SD, not a statistical-significance test. `analysis_set` identifies primary strict-budget or three-seed descriptive summaries. `n_pairs` is 1, 2 or 3. `strict_budget_matched` and `amp_update_qualification` document successful-update pairing. `c1_*` and `a1_fixed_*` give raw arm values; `delta_*` always means A1-fixed minus C1. `*_sample_sd` uses denominator n−1. Multiplying a performance difference by 100 gives percentage points, not relative percent improvement.

## Action-risk CSV

All endpoints use score 0.25 and matching IoU 0.5. `green_to_fully` and `half_to_fully` count green/half GT objects assigned to fully predictions under the class-agnostic matcher. `semantic_unsafe_count` is their sum. `*_per_complete_*_gt` normalizes by the indicated fixed complete GT count. `complete_nonripe_gt_denominator` equals green GT plus half GT. `matched_nonripe_confusion_row_sum_not_gt` is a matched-confusion row total and is not a valid substitute for complete GT.

`fully_actions_class_aware_matcher` is the number of fully predictions under class-aware matching. `fully_class_detection_precision` is TP/(TP+FP); `missed_ripe_count_class_aware` is fully-class FN and `missed_ripe_rate_class_aware` is FN/fully GT. The `cross_matcher_unsafe_over_fully_actions_diagnostic_only` column is retained for record fidelity; it combines different matching systems and must not be interpreted as harvesting-error probability. The `matcher_for_*` columns identify the relevant matching system. Count changes describe simulated image-based selection, not real harvesting events.

## Views and historical results

Development and reused historical views are different frozen splits, not different acquisition dates. SDS's corrected holdout contains 26 images, 104 boxes and seven half GT objects. Yurdakul is the frozen 566-image, 1,734 polygon-derived-box snapshot. Historical A1−B3 rows belong to their original three-seed experiment and must not be merged with corrected fixed-budget rows. Checkpoint/update audit columns record scheduled optimizer calls, successful updates and AMP skips separately. Hashes are SHA256 digests; opaque restricted file IDs encode no pixels or per-object labels.
