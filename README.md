# Shell.ai Hackathon 2026 - Sustainable Aviation Fuel Blending

<div align="center">

![Shell.ai](https://img.shields.io/badge/Shell.ai-2026-orange?style=for-the-badge)
![Score](https://img.shields.io/badge/Public_Leaderboard-83.16-success?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-SHAP-green?style=for-the-badge)

**Fuel Blend Properties Prediction Challenge**  
*Accelerating the transition to sustainable aviation fuels through AI*

</div>

---

## 🎯 Challenge Overview

This repository contains the solution for the **Shell.ai Hackathon 2026** focusing on Sustainable Aviation Fuel (SAF) blend properties prediction. The challenge addresses a critical real-world problem: predicting the final properties of complex fuel blends to accelerate the adoption of sustainable aviation fuels.

### 🏆 Achievement
- **Public Leaderboard Score:** 83.16
- **Execution Time:** ~21 minutes
- **Evaluation Metric:** Mean Absolute Percentage Error (MAPE)

### 🌍 Impact
This solution contributes to:
- Rapid evaluation of thousands of potential blend combinations
- Identification of optimal recipes maximizing sustainability while meeting specifications
- Reduction of development time for new sustainable fuel formulations
- Enabling real-time blend optimization in production facilities

---

## 📁 Project Structure

```
shell.ai/
├── 📂 data/                      # Dataset directory
│   ├── train.csv                 # Training data (blends + properties + targets)
│   ├── test.csv                  # Test data (blends + properties only)
│   └── sample_submission.csv     # Submission format template
├── 📂 models/                    # Saved model artifacts (optional)
├── 📂 outputs/                   # Predictions and submission files
│   └── submission.csv            # Final predictions for leaderboard
├── 📂 notebooks/                 # Jupyter notebooks
│   └── analysis.ipynb            # Exploratory data analysis and results
├── 📄 train.py                   # Main training and prediction script
├── 📄 requirements.txt           # Python dependencies
├── 📄 PROBLEM_DESCRIPTION.md     # Detailed challenge description
└── 📄 README.md                  # This file
```

---

## 🚀 Quick Start

### 1️⃣ Install Dependencies
```bash
# Clone or navigate to project directory
cd shell.ai

# Install required packages
pip install -r requirements.txt
```

### 2️⃣ Prepare Data
Ensure your dataset files are in the `data/` directory:
- `train.csv` - 65 columns: 5 component fractions + 50 component properties + 10 blend property targets
- `test.csv` - 55 columns: 5 component fractions + 50 component properties (targets to be predicted)
- `sample_submission.csv` - Submission format reference

### 3️⃣ Run Training & Prediction
```bash
python train.py
```

**Expected Output:**
```
🔧 Processing target: BlendProperty1
✅ Completed target 1 with clipping range: [x.xxxx, x.xxxx]
...
✅ Submission saved as outputs/submission.csv
```

### 4️⃣ Submit Predictions
Upload `outputs/submission.csv` to the hackathon platform.

---

## 🧪 Dataset Description

### Input Features (55 columns)

#### 1. Blend Composition (5 columns)
- `Component1_fraction` to `Component5_fraction`
- Volume percentage of each base component in the blend

#### 2. Component Properties (50 columns)
- Format: `Component{i}_Property{j}` where i∈[1,5], j∈[1,10]
- Simulates real-world Certificate of Analysis (COA)
- Properties include physical, chemical, safety, operational, and environmental characteristics

### Target Variables (10 columns)
- `BlendProperty1` to `BlendProperty10`
- Final properties of the blended fuel (drop-in replacement characteristics)

### Dataset Split
- **Training Set:** Full dataset with all 65 columns
- **Test Set:** 500 samples split into:
  - Public Leaderboard: 250 samples (Reference Cost: 2.72)
  - Private Leaderboard: 250 samples (Reference Cost: 2.58)

---

## 🔬 Technical Approach

### Feature Engineering Pipeline

Our solution employs **physics-inspired feature engineering** to capture complex fuel blending interactions:

#### 1. Volume-Based Features
- **Shannon Entropy:** `VolumeEntropy = -Σ(v_i × log(v_i))` - measures mixture complexity
- **Volume Standard Deviation:** Quantifies distribution uniformity

#### 2. Property Aggregations (for each of 10 properties)
- **Weighted Mean:** `Σ(volume_i × property_i)` - dominant mixing rule
- **Min/Max/Range:** Capture extrema and variability
- **Geometric Mean:** `exp(Σ(volume_i × log(property_i)))` - log-domain mixing
- **Harmonic Mean:** `1/Σ(volume_i/property_i)` - resistance-like properties

#### 3. Interaction Features (50 features)
- **Component-Property Products:** `Component{i}_fraction × Component{i}_Property{j}`
- Captures synergistic and antagonistic effects

#### 4. Domain-Specific Ratios
- `Comp1_Comp5_ratio`: Component balance indicator
- `Comp3_vol_prop8`: Specific interaction term

**Total Engineered Features:** 150+ features from 55 base features

---

### Model Architecture

#### Algorithm: LightGBM (Gradient Boosted Decision Trees)
- **Why LightGBM?**
  - Handles high-dimensional feature spaces efficiently
  - Robust to outliers via L1 regression objective
  - Fast training with leaf-wise growth strategy

#### Configuration
```python
lgb_params = {
    'objective': 'regression_l1',       # MAE loss (robust)
    'metric': 'mape',                   # Direct optimization of evaluation metric
    'boosting_type': 'gbdt',            # Gradient boosting
    'num_leaves': 31,                   # Tree complexity
    'learning_rate': 0.05,              # Conservative learning
    'min_child_samples': 20,            # Regularization
    'feature_fraction': 0.8,            # Feature sampling
    'bagging_fraction': 0.8,            # Row sampling
    'bagging_freq': 5,                  # Bagging frequency
    'n_estimators': 2000,               # Max iterations
    'random_state': 42                  # Reproducibility
}
```

#### Feature Selection: SHAP-Based Importance
- **SHAP (SHapley Additive exPlanations):** Model-agnostic feature importance
- **Selection Strategy:** Top 50 features per target based on mean absolute SHAP values
- **Benefits:**
  - Reduces overfitting
  - Improves model interpretability
  - Faster inference

#### Training Strategy
- **Cross-Validation:** 5-Fold KFold with shuffling
- **Early Stopping:** 50 rounds without improvement
- **Per-Target Models:** Independent models for each of 10 blend properties

#### Post-Processing
- **Physics-Based Clipping:** Predictions constrained to `[train_min, train_max]`
- Ensures physical feasibility of predictions

---

## 📊 Evaluation Metric

### Mean Absolute Percentage Error (MAPE)

$$
\text{MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
$$

### Leaderboard Score Calculation

$$
\text{Score} = \max\left(100 - 25 \times \frac{\text{MAPE}}{\text{Reference Cost}}, 10\right)
$$

- **Public Leaderboard:** Reference Cost = 2.72
- **Private Leaderboard:** Reference Cost = 2.58
- **Score Range:** 0-100 (scores 0-10 reserved for errors)

---

## 📦 Dependencies

```
numpy>=1.21.0        # Numerical operations
pandas>=1.3.0        # Data manipulation
scikit-learn>=1.0.0  # ML utilities & metrics
lightgbm>=3.3.0      # Gradient boosting framework
shap>=0.41.0         # Model interpretability
```

**Installation:**
```bash
pip install -r requirements.txt
```

---

## 🎛️ Key Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `RANDOM_STATE` | 42 | Reproducibility across runs |
| `N_FOLDS` | 5 | Cross-validation stability |
| `N_SHAP_FEATURES` | 50 | Feature selection threshold |
| `N_TARGETS` | 10 | Number of blend properties |
| `learning_rate` | 0.05 | Gradient descent step size |
| `num_leaves` | 31 | Tree complexity |
| `n_estimators` | 2000 | Maximum boosting rounds |

---

## 📈 Results & Performance

### Leaderboard Performance
- **Public Score:** 83.16
- **Execution Time:** ~21 minutes on standard hardware
- **Model Size:** Lightweight (no ensemble required)

### Key Success Factors
1. **Physics-Inspired Features:** Domain knowledge encoded in feature engineering
2. **SHAP-Based Selection:** Automatic feature relevance detection per target
3. **Robust Optimization:** L1 loss + MAPE metric alignment
4. **Physics Constraints:** Clipping ensures realistic predictions

---

## 🔍 Exploratory Analysis

See [notebooks/analysis.ipynb](notebooks/analysis.ipynb) for:
- Data distribution analysis
- Feature correlation heatmaps
- SHAP importance visualizations
- Prediction vs. actual scatter plots
- Cross-validation performance metrics

---

## 📝 Submission Format

Your `submission.csv` must contain:
- **11 columns:** `ID`, `BlendProperty1`, ..., `BlendProperty10`
- **500 rows:** One per test sample
- **Data Type:** Floating-point numbers for all blend properties
- **ID Order:** Must match `test.csv` row order

**Example:**
```csv
ID,BlendProperty1,BlendProperty2,...,BlendProperty10
0,45.231,12.456,...,89.123
1,43.987,11.234,...,87.654
...
```

---

## ⚠️ Common Errors to Avoid

| Error Code | Description | Solution |
|------------|-------------|----------|
| 0 | Not a CSV file | Ensure `.csv` extension |
| 1 | Missing column | Check all `BlendProperty{1-10}` exist |
| 2 | Non-float values | Convert all predictions to float |
| 3 | Wrong dimensions | Must be 500 rows × 10 columns (+ ID) |

---

## 🌟 About Shell.ai Hackathon

**Shell.ai Hackathon 2026** is the sixth edition of Shell's annual competition tackling real-world energy challenges through AI and digital solutions.

### Previous Challenges
- **2020:** Wind farm layout optimization
- **2021:** Solar irradiance forecasting
- **2022:** EV charging station placement
- **2023:** Biorefinery supply chain optimization
- **2024:** Fleet decarbonization
- **2026:** Sustainable aviation fuel blending 

### Mission
Accelerate the transition to a **net-zero future** by developing AI-powered tools that enable:
- Sustainable fuel adoption
- Environmental footprint reduction
- Economic viability of green energy solutions


## Contributing

While this is a hackathon submission, suggestions for improvement are welcome:
1. Enhanced feature engineering techniques
2. Alternative model architectures (ensemble methods, neural networks)
3. Hyperparameter optimization strategies
4. Domain-specific physical constraints


<div align="center">

**Built with ❤️ for a sustainable future**

*Accelerating the net-zero transition through AI*

</div>

<!-- Updated: June 2025 -->
