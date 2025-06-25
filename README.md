
# Impact of Cross-Border Flow Forecast Alignment on Flow-Based Market Coupling Efficiency

This repository contains the full simulation framework, data, and results associated with the paper:

**"Impact of Cross-Border Flow Forecast Alignment on Flow-Based Market Coupling Efficiency"**  
_Submitted to IEEE Transactions on Power Systems._

---

## 🔍 Overview

This study investigates how the accuracy and granularity of cross-border flow forecasts affect outcomes in day-ahead electricity markets under a Flow-Based Market Coupling (FBMC) regime. The simulation framework replicates realistic operational processes and analyzes both physical feasibility and economic efficiency under different forecast strategies.

### The framework includes:
- D-2 Common Grid Model (CGM) with DC-OPF optimization  
- Flow-based domain calculation (RAM/PTDF)  
- D-1 market coupling economic dispatch  
- D-1 CGM feasibility check via DC load flow  
- Forecast models for Net Positions (NP) and Power Transfer Corridors (PTC)

---

## 🛠️ How to Use the Framework

### 1. Generate Training Data
Run:
```bash
D-2_base_case_D-1_MC.py
model_D-1_CGM.py
```
This simulates the perfect foresight pipeline and generates D-1 outputs for training.

### 2. Train Forecast Models
Use:
```bash
Neural_Network_NP.py       # For NP forecasting
Neural_Network_PTC.py      # For PTC forecasting
```
Model outputs are stored in the `ML_results/` directory.

### 3. Run Scenario Pipelines

#### Forecast-Based Scenarios
```bash
full_pipeline_after_NN_NP.py        # With NP forecasts  
full_pipeline_after_NN_PTC.py       # With PTC forecasts
```

#### Perfect Forecast Scenarios
```bash
full_pipeline_after_NN_NP_perfect.py
full_pipeline_after_NN_PTC_perfect.py
```

#### Sensitivity Analysis (Partial Derivatives)
```bash
full_pipeline_after_NN_NP_PD.py
```

---

## 📁 Repository Structure

| Folder/File | Description |
|-------------|-------------|
| `data/` | Input data for generation, load, network, etc. |
| `ML_results/` | Trained model predictions (NPs, PTCs) |
| `Analysis/` | Scripts for analyzing results and computing metrics |
| `D-2_*/` | Output folders for storing simulation results by scenario from D-2 DC-OPF |
| `D-1_*/` | Output folders for storing simulation results by scenario from D-1 MC or D-1 DC-OPF |

---

## 📊 Key Analysis Scripts (in `Analysis/`)
- `network_plot.py` : Function to generate visualizations of the grid |
- `Calculate_metrics.py`: Computes FBME, overloads, cost, etc.
- `Fairness.py`: Analyzes zonal price convergence and RAMs
- `FBME_and_partial_derivatives.py`: Computes flow-based model error (FBME) and sensitivities
- `Shapley_analysis.py`: SHAP-based interpretability for FBME drivers
- `network_plot.py`: Custom grid plotting utility

---

## 📜 Citation

If you use this codebase or the results in your work, please cite:

```bibtex
@misc{SteensbergImpact-of-Cross-Border-Flow-Forecast-Alignment-on-Flow-Based,
  title   = {Impact of Cross Border Flow Forecast Alignment on Flow-Based},
  author  = {Julius Lindberg Steensberg},
  year    = {2025},
  url     = {https://github.com/Julius-Steensberg/Impact-of-Cross-Border-Flow-Forecast-Alignment-on-Flow-Based}
}
```

---

## 📬 Contact

For questions or collaborations, please contact [Julius Steensberg](mailto:j.l.steensberg@gmail.com).
