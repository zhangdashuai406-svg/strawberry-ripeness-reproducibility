# Strawberry ripeness detection under a fixed training budget

Reproducibility materials for the manuscript “Heterogeneous data integration improves cross-source strawberry ripeness detection under a fixed training budget”. Authors: Xinran Li, Jin Zhang, Yao Hao, Jingnan Shi and Shuang Han. This release does not imply that the manuscript has been accepted.

## Contents and scope

- `results/corrected_phase3`: the three final valid CSVs, plus invalid-output exclusion metadata. All numeric cells are preserved.
- `reports/`: 36 corresponding valid JSON outputs and the separate historical A1/B3 JSON records. Invalid legacy SDS JSON is excluded.
- `results/historical_a1_b3`: original historical paired comparison, not the corrected Phase 3 comparison.
- `scripts/plot_figures.py`: regenerates both manuscript figures solely from the valid CSVs.
- `scripts/`: historical evaluator and guarded training/materialization source for inspection. No training was launched in preparing this release. These sources retain their project-relative layout and require the separately obtained datasets and dependencies; they are not a one-command training distribution.
- `configs`, `manifests`, `mappings`, `provenance`: training settings, checkpoint identities, successful-update records, data identity and label rules. Source file IDs in private split manifests are opaque hashes; the correspondence is supplied with approved data access.

## Reproduce the figures

Install `environment/figure_requirements.txt` in a suitable Python environment, then run `python scripts/plot_figures.py --word-revision` from the repository root. Outputs are written to `figures/`. This command only reads released corrected CSVs and does not train or run inference.

## Evidence and units

C1 is the repeated-Merchant budget-matched control; A1-fixed replaces 172 repeated slots with mapped SDS images. Both use YOLO11n, 100 epochs and 2,380 image slots per epoch. Seeds 42 and 44 are the primary successful-update-matched pairs; seed 43 is sensitivity only. Differences are A1-fixed minus C1. CSV performance values use 0–1 units; multiplying a difference by 100 gives percentage points. Counts are objects, not observed robot picks. Class-agnostic assignment counts and class-aware detection endpoints use different matchers and must not be combined into an error probability. The corrected SDS test has only seven half-ripe GT objects.

## Data access

Merchant and External images and annotations are not included. Contact the corresponding author, Shuang Han, at 03775@qqhru.edu.cn with the requested dataset/view, research purpose and intended use. Access is subject to the applicable provider permissions. No response-time guarantee, new eligibility restriction or blanket provider authorization is asserted here. Public aggregate results do not grant rights to commercial images.

Obtain Strawberry-DS and Yurdakul from the upstream sources listed in `provenance/upstream_sources.md`, following their licenses. The evaluated Yurdakul Kaggle v1 archive has 566 images and 1,734 polygon-derived boxes; it is not equivalent to the paper’s 1,169-box annotation version. No upstream image archive is redistributed here.

## Limitations of this deposit

Checkpoint binaries, private source pixels/annotations and full original training logs are not included. Checkpoint hashes and extracted update audits document model identity and pairing. JSON and CSV path strings are relocated; original source hashes and transformations appear in `manifests/release_provenance.json`. Author-produced code is released under MIT; study-produced aggregate results and numeric figures are released under CC BY 4.0, as authorized by the depositing author. Third-party materials retain their own terms; see `RIGHTS.md`. No DOI has been assigned.

Legacy JSON fields such as `validation_only` and `test_sets_touched` were originally written as literals and must not be used to infer the evaluation split. The evaluated directory/view, source manifest and documented reused historical-test status establish the evidence role. These fields are preserved as historical records, not endorsed as independent split verification.

