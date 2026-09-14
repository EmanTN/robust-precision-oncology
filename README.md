# Practical methods for robust precision medicine in oncology

Code accompanying the manuscript *Practical methods for robust precision medicine in oncology* (Alnabati, Lu, Gusev, Ghassemi), under review.

Standard classifiers are trained on TCGA gene expression data under four ancestry-aware training strategies and evaluated with aggregate, group-level fairness, and calibration metrics, alongside published predictions from PhyloFrame (Smith et al., Nat Commun 2025).

## Pipeline

1. `finetune.py` tunes hyperparameters (`finetune_array.sbatch`, `feed_sbatch.sh`).

2. `finetune.py --best_run` writes predictions (`pred_array.sbatch`, `run_predictions.sh`).

3. `measure_metrics.py` computes performance and fairness metrics (`run_metrics.sh`).

4. `calibration_metrics.py` computes calibration metrics; `calibration_variants.py` tests recalibration.

`models.py` and `utils.py` are shared helpers. Settings are in `config/config.yaml`.

## Data

Datasets are not included. Sources are listed in the manuscript's Availability of data and materials statement. Paths in `config/config.yaml` must be updated to local paths.

## License

MIT
