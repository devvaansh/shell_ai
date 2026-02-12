# Shell.ai Hackathon 2026 - Problem Description

## 🌍 Introduction

Welcome to the **sixth edition of the Shell.ai Hackathon** for Sustainable and Affordable Energy. Shell.ai Hackathon brings together brilliant minds passionate about digital solutions and AI, to tackle real energy challenges and help build a lower-carbon world where everyone can access and afford energy.

### Previous Editions
- **2020:** Wind farm layout optimization
- **2021:** Irradiance forecasting for solar power generation
- **2022:** Optimal placement of electric vehicle (EV) charging stations
- **2023:** Supply chain optimization for biorefineries
- **2024:** Fleet decarbonization
- **2026:** Blend properties estimation for sustainable fuel ✈️

---

## 🎯 Challenge - Sustainable Aviation Fuels (SAF)

The global call for sustainability is reshaping every industry, including **mobility, shipping, and aviation**. **Sustainable Aviation Fuels (SAFs)** are pivotal in this transformation, offering a powerful lever to significantly reduce the sector's environmental footprint.

### The Challenge
Integrating these innovative fuels into the existing ecosystem presents a sophisticated challenge. **Crafting the optimal fuel blend** – mixing various sustainable fuel types sourced from diverse pathways with each other or with conventional fuels – is an intricate science.

It demands a delicate balancing act:
- ✅ Ensuring adherence to **rigorous safety and performance specifications**
- ✅ Maximizing **environmental benefits**
- ✅ Maintaining **economic viability**

### Your Mission
In this hackathon, you will immerse yourselves in the critical field of **fuel blending**. Your challenge is to develop models capable of predicting the final properties of complex fuel blends based on their constituent components and proportions.

By exploring datasets rich with complex interactions, you will decipher the hidden relationships that dictate:
- Fuel performance
- Safety characteristics
- Regulatory compliance
- Environmental impact

### The Endgame
Engineer powerful predictive tools that can guide the industry in formulating the next generation of sustainable fuels, **accelerating the transition to a net-zero future**, without compromising on excellence.

---

## 📋 Problem Statement

In the fuel industry, blending different fuel components to achieve desired properties is both an **art and a science**. The relationships between component fractions and final blend properties are highly complex, involving:

- **Linear interactions:** Simple mixing rules
- **Non-linear interactions:** Synergistic or antagonistic effects
- **Conditional behaviors:** Properties that vary based on component combinations

This complexity makes accurate prediction a **challenging, high-dimensional problem**.

### Objectives

Your challenge is to develop models capable of accurately predicting the properties of fuel blends based on their **constituent components and their proportions**.

These predictions must be **precise enough** to guide real-world blending decisions where:
- 🛡️ **Safety** is paramount
- ⚡ **Performance** cannot be compromised
- 🌱 **Sustainability** is the goal

### Real-World Impact

By harnessing the power of data science and machine learning, you will help accelerate the adoption of sustainable aviation fuels by providing tools that can:

1. **Rapidly evaluate thousands of potential blend combinations**
   - Replace months of laboratory testing with minutes of computation
   
2. **Identify optimal recipes that maximize sustainability while meeting specifications**
   - Balance environmental impact with regulatory requirements
   
3. **Reduce the development time for new sustainable fuel formulations**
   - Accelerate time-to-market for innovative SAF blends
   
4. **Enable real-time blend optimization in production facilities**
   - Dynamic adjustment based on available feedstocks

---

## 📊 Data Description

The dataset is designed to simulate real-world fuel blending scenarios with Certificate of Analysis (COA) data.

### Files Provided

#### 1. `train.csv` - Training Dataset
**Purpose:** Train your predictive models

**Structure:** Each row represents a unique fuel blend with **65 columns** organized into three groups:

##### Group 1: Blend Composition (5 columns)
- `Component1_fraction`, `Component2_fraction`, ..., `Component5_fraction`
- **Description:** Volume percentage of each base component in the blend
- **Constraint:** Sum of fractions = 1.0 (100%)
- **Physical Meaning:** Volumetric mixing ratios

##### Group 2: Component Properties (50 columns)
- **Format:** `Component{i}_Property{j}` where:
  - `i` ∈ {1, 2, 3, 4, 5} - Component number
  - `j` ∈ {1, 2, ..., 10} - Property number

- **Description:** Properties for the specific batch of each component
- **Physical Meaning:** This section simulates a real-world Certificate of Analysis (COA)

**Example:**
- `Component1_Property1` - Property 1 of Component 1
- `Component3_Property7` - Property 7 of Component 3

**Property Categories:**
This suite of properties provides a holistic assessment of the fuel, detailing:
- Core physical and chemical nature
- Critical safety limits
- Operational characteristics
- Full lifecycle environmental impact

##### Group 3: Final Blend Properties - **TARGETS** (10 columns)
- `BlendProperty1`, `BlendProperty2`, ..., `BlendProperty10`
- **Description:** Properties of the final blended fuel
- **Purpose:** These are the **target variables** your model must predict
- **Physical Meaning:** Characteristics of the blend used as a 'drop-in' replacement fuel

---

#### 2. `test.csv` - Test Dataset
**Purpose:** Evaluate your model's performance

**Structure:** 
- **500 blends** not present in training data
- **55 columns:** Same structure as training data
  - 5 blend composition columns
  - 50 component property columns
- **Missing:** 10 target property columns (BlendProperty1-10)

**Your Task:** Predict the 10 missing blend properties for each of the 500 test samples

**Test Set Split:**
- **Public Leaderboard:** First 250 samples
- **Private Leaderboard:** Remaining 250 samples (final ranking)

---

#### 3. `sample_submission.csv` - Submission Format
**Purpose:** Demonstrates the required format for your predictions

**Critical Requirements:**
- ✅ Must contain exactly **11 columns:** `ID` + `BlendProperty1` through `BlendProperty10`
- ✅ Must contain exactly **500 rows** (one per test sample)
- ✅ **ID order must match** `test.csv` row order
- ✅ All blend properties must be **floating-point numbers**
- ✅ Column names must be exact: `BlendProperty1`, `BlendProperty2`, etc.

**Example Structure:**
```csv
ID,BlendProperty1,BlendProperty2,BlendProperty3,...,BlendProperty10
0,45.231,12.456,78.901,...,89.123
1,43.987,11.234,77.654,...,87.654
...
499,44.567,12.890,78.234,...,88.901
```

---

## 📏 Evaluation

### Two-Stage Evaluation Process

#### Round 1: Public Leaderboard
- **Data:** First 250 test samples
- **Visibility:** Updated in real-time throughout the competition
- **Purpose:** Allows participants to gauge model performance and iterate
- **Reference Cost:** 2.72

#### Round 2: Private Leaderboard (Final Ranking)
- **Data:** Remaining 250 test samples
- **Visibility:** Revealed only after competition closes
- **Purpose:** Determines final scores and winners
- **Reference Cost:** 2.58
- **Anti-Overfitting:** Prevents excessive optimization on public data

---

### Evaluation Metric: Mean Absolute Percentage Error (MAPE)

#### Formula

$$
\text{MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left| \frac{y_{\text{true}, i} - y_{\text{pred}, i}}{y_{\text{true}, i}} \right|
$$

Where:
- $n$ = Total number of predictions across all samples and properties
- $y_{\text{true}, i}$ = Actual value of blend property $i$
- $y_{\text{pred}, i}$ = Predicted value of blend property $i$

#### Why MAPE?
- **Scale-invariant:** Fair comparison across properties with different magnitudes
- **Interpretable:** Directly represents percentage error
- **Industry-standard:** Commonly used in manufacturing and quality control
- **Penalizes relative errors:** More stringent for critical small-value properties

#### Implementation
We recommend using the **scikit-learn MAPE API**:

```python
from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_true, y_pred)
```

The output of this function is directly used to calculate leaderboard scores.

---

### Leaderboard Score Calculation

#### Formula

$$
\text{Score} = \max\left(100 - 25 \times \frac{\text{MAPE}_{\text{solution}}}{\text{Reference Cost}}, 10\right)
$$

#### Parameters
- **Public Leaderboard Reference Cost:** 2.72
- **Private Leaderboard Reference Cost:** 2.58

#### Score Interpretation
- **Score = 100:** Perfect prediction (MAPE = 0)
- **Score > 10:** Valid solution with performance proportional to MAPE
- **Score ≤ 10:** Error occurred (see error codes below)

#### Example Calculation (Public Leaderboard)
If your solution achieves MAPE = 1.5:

$$
\text{Score} = 100 - 25 \times \frac{1.5}{2.72} = 100 - 13.79 = 86.21
$$

---

### Error Codes (Scores 0-10)

Scores between **0 and 10** are reserved for error codes. If you receive an integer score in this range, it indicates an evaluation error:

| Error Code | Description | Solution |
|------------|-------------|----------|
| **0** | Not a CSV file | Ensure file has `.csv` extension |
| **1** | Missing property column | Check all `BlendProperty{1-10}` columns exist with exact naming |
| **2** | Non-floating point numbers | Ensure all predictions are numeric floats, not strings or NaN |
| **3** | Incorrect dimensions | Submission must have exactly **500 rows** and **10 columns** (excluding ID) |

---

## 📤 Submission Guidelines

### What to Submit
1. **Solution File:** `submission.csv` containing your predictions
2. **Source Code:** Optional (you can ignore the "Upload source file" field)

### Validation Checklist
Before submitting, verify:
- ✅ File format is `.csv`
- ✅ Contains 11 columns: `ID`, `BlendProperty1`, ..., `BlendProperty10`
- ✅ Contains exactly 500 rows
- ✅ All predictions are floating-point numbers (no NaN, inf, or strings)
- ✅ ID column matches the order in `test.csv`
- ✅ Column names match exactly (case-sensitive)

### Submission Process
1. Navigate to the hackathon platform
2. Upload your `submission.csv` file
3. Wait for evaluation (typically 1-2 minutes)
4. Check your score on the public leaderboard

---

## 🧠 Key Insights & Considerations

### Physical Constraints
- **Non-negativity:** All properties should be non-negative
- **Bounded ranges:** Properties have physical limits (e.g., density, viscosity)
- **Monotonicity:** Some properties may exhibit monotonic behavior with composition

### Modeling Challenges
1. **High dimensionality:** 55 input features, 10 targets
2. **Non-linear interactions:** Volume fractions interact non-linearly with properties
3. **Multi-target correlation:** Blend properties may be correlated
4. **Batch variation:** Component properties vary by batch (COA data)

### Suggested Approaches
- **Feature engineering:** Create interaction terms, ratios, weighted averages
- **Domain knowledge:** Incorporate mixing rules from fuel chemistry
- **Regularization:** Prevent overfitting on limited training data
- **Ensemble methods:** Combine multiple model predictions
- **Cross-validation:** Robust performance estimation

---

## 🏁 Success Criteria

### Technical Excellence
- Low MAPE across all 10 blend properties
- Robust predictions on unseen test data
- Generalization from public to private leaderboard

### Innovation
- Novel feature engineering approaches
- Creative model architectures
- Interpretable predictions

### Impact
- Solutions that can guide real-world blending decisions
- Contribution to sustainable fuel development
- Advancement of AI for energy transition

---

## 🌟 Why This Matters

### Environmental Impact
- **Aviation emissions:** ~2.5% of global CO₂ emissions
- **SAF potential:** Up to 80% lifecycle CO₂ reduction vs. conventional jet fuel
- **Scaling challenge:** SAF currently <0.1% of global jet fuel supply

### Industry Transformation
Your work directly supports:
- Achieving net-zero aviation by 2050
- Reducing reliance on fossil fuels
- Enabling circular economy in fuel production
- Meeting international sustainability commitments

### AI for Good
This challenge exemplifies how AI can:
- Accelerate sustainable innovation
- Replace costly physical experiments
- Enable data-driven decision making
- Scale solutions for global impact

---

## 📚 Additional Resources

### Fuel Blending Basics
- Understand octane/cetane number blending
- Study Reid Vapor Pressure (RVP) mixing rules
- Learn about flash point and viscosity blending

### Machine Learning Techniques
- Gradient boosting (XGBoost, LightGBM, CatBoost)
- Neural networks for regression
- SHAP for model interpretability
- Hyperparameter optimization (Optuna, Hyperopt)

### Competition Strategy
- Start with simple baselines (linear regression, weighted averages)
- Perform thorough EDA (exploratory data analysis)
- Engineer domain-specific features
- Validate on multiple CV folds
- Avoid overfitting to public leaderboard

---

## 🤝 Community & Support

### Getting Help
- Review the hackathon discussion forum
- Check FAQs and pinned threads
- Reach out to organizers for technical issues

### Fair Play
- No external data allowed
- Code sharing prohibited until after competition
- Respect computational limits and submission quotas

---

<div align="center">

**Good luck, and may your models blend excellence with sustainability!** 🚀🌱

*Together, we're building a net-zero future.*

---

**Shell.ai Hackathon 2026**  
*Fuel Blend Properties Prediction Challenge*

</div>