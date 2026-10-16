# Adaptive Explainable Neural-XGBoost (AX-NXGB)

Code repository reproducing all experimental results and figures for the paper:
**"Adaptive Explainable Neural-XGBoost for Robust Streaming Prediction under Concept Drift"**

AX-NXGB couples a continuous neural representation with an adaptive gradient-boosted decision tree ensemble and dynamic SHAP feature attributions under streaming distribution shifts.

---

## Repository Structure

All scripts are standalone Python modules:

- `models.py`: Architecture implementation of AX-NXGB (delay-coordinate representation, neural autoencoder, streaming XGBoost ensemble, dynamic gate, and ADWIN monitor) along with baseline streaming classifiers (Static XGBoost, Sliding XGBoost, Incremental XGBoost, ADWIN-reset XGBoost, Online MLP, and River HAT/ARF).
- `data.py`: Streaming benchmark data loaders for Electricity, Weather, Airlines, and Covertype datasets.
- `stream_eval.py`: Chunk-wise prequential (test-then-train) streaming evaluation loop and evaluation metrics (accuracy, macro-F1, Cohen's kappa, ROC-AUC).
- `explain.py`: Dynamic SHAP engine, moving reference background distribution maintenance, Explanation Drift Index (EDI), and exact Shapley evaluation.
- `run_prediction.py`: Executes prequential predictive experiments across all datasets (`main` comparison, `ablation`, `dcl` generality, and `tradeoff` refit frequency analysis).
- `run_explain.py`: Executes explainability experiments evaluating attribution fidelity, EDI, and cosine agreement with exact Shapley values.
- `run_control.py`: Executes the controlled benchmark experiments on Electricity (label-definition and price-removal controls).
- `figures_v8.py`: Generates Figure 2 of the paper from experimental result JSONs.
- `analysis_v4.py`: Computes statistical significance tests (paired t-tests, Wilcoxon signed-rank tests, 95% confidence intervals) and post-alarm accuracy for Table II.

---

## Dependencies

```bash
pip install torch xgboost river shap scikit-learn pandas scipy matplotlib
```

---

## Reproducing Paper Results

### 1. Data Preparation
Place the benchmark datasets (`electricity.csv`, `weather.csv`, `airlines.csv`, `covtype.csv`) in a `data/` folder, or set the `STREAM_DATA_DIR` environment variable:
```bash
export STREAM_DATA_DIR="/path/to/data"
```

### 2. Predictive Benchmarks (Tables I, II, IV)
Run prequential evaluation across datasets and settings:
```bash
# Main comparison (Table I)
for d in electricity weather airlines covertype; do
    python run_prediction.py $d main
done

# Ablations (Table IV)
for d in electricity weather airlines covertype; do
    python run_prediction.py $d ablation
done

# DCL generality & refit trade-off
for d in electricity weather airlines covertype; do
    python run_prediction.py $d dcl
    python run_prediction.py $d tradeoff
done
```

### 3. Explainability Benchmarks (Table V)
```bash
for d in electricity weather airlines covertype; do
    python run_explain.py $d
done
```

### 4. Controlled Electricity Experiments (Table III)
```bash
for setting in orig noprice next; do
    python run_control.py $setting
done
```

### 5. Figures and Statistical Analysis
```bash
# Generate Figure 2
python figures_v8.py results .

# Compute statistical tests and summary metrics
python analysis_v4.py results data
```
