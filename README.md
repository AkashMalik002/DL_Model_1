# Deep Learning Assignment - Modular Project Structure

**Student ID:** adid1212  
**Assignment:** Multi-Domain Deep Learning Implementation from Scratch  
**Date:** 2026

---

## 📋 Project Overview

This project implements a complete Deep Learning pipeline with **4 ML models from scratch** (using only NumPy) trained across **5 real-world domains**. The entire codebase is structured as modular Python files, with a main orchestrator coordinating all sections.

### Key Features:
- ✅ **4 Models:** Logistic Regression, Linear Regression, Softmax Regression, Multi-Layer Perceptron
- ✅ **5 Domains:** NLP, Agriculture, Database, Healthcare, FinTech
- ✅ **500+ Samples, 10+ Features** per domain
- ✅ **NumPy-Only Implementation:** No TensorFlow, PyTorch, or sklearn ML modules
- ✅ **Modular Architecture:** Separation of concerns with dedicated files for each function
- ✅ **Comprehensive Analysis:** Situated learning insights for each domain

---

## 🏗️ Project Structure

```
Ass_1/
├── README.md                    ← You are here (Project documentation)
├── main.py                      ← Main orchestrator (entry point)
├── datasets.py                  ← Section 1: Dataset creation
├── models.py                    ← Section 2: ML model implementations
├── metrics.py                   ← Section 3: Evaluation metrics
├── preprocessing.py             ← Section 4: Data preprocessing pipeline
├── training.py                  ← Section 5: Model training
├── visualization.py             ← Section 6-7: Loss/metrics visualization
├── analysis.py                  ← Section 8: Situated learning analysis
├── results.py                   ← Section 9: Final results summary
└── adid1212_assignment1.ipynb   ← Original notebook (backup)
```

---

## 🔄 Execution Flow

### **How to Run the Project:**

```bash
cd c:\Users\DELL\Documents\BiTs\Course\Sem3\DL\Ass_1
python main.py
```

---

## 📖 Detailed Section Breakdown

### **SECTION 0: IMPORTS & SETUP**
**File:** `main.py` (lines 23-36)

**What Happens:**
- Imports all required libraries (NumPy, Pandas, Matplotlib, Seaborn)
- Sets random seed for reproducibility
- Imports all custom modules

**Dependencies Used:**
- NumPy, Pandas, Matplotlib, Seaborn
- sklearn utilities (train_test_split, StandardScaler, MinMaxScaler)

**Output:**
```
✓ All required libraries imported successfully!
✓ All custom modules imported successfully!
```

---

### **SECTION 1: DATASET PREPARATION AND LOADING**
**File Called:** `datasets.py`
**Main File:** `main.py` (lines 44-53)

**What Happens:**
1. **`create_domain_datasets()`** function generates 5 binary classification datasets:
   - **Domain 1 (NLP):** Review Classification - 800 samples, 15 features
   - **Domain 2 (Agriculture):** Crop Health - 700 samples, 12 features
   - **Domain 3 (Database):** Infrastructure Monitoring - 900 samples, 14 features
   - **Domain 4 (Healthcare):** Patient Readmission - 800 samples, 13 features
   - **Domain 5 (FinTech):** Loan Default - 750 samples, 11 features

2. **`display_dataset_summary()`** prints dataset information and compliance checks

**Key Code in datasets.py:**
```python
def create_domain_datasets():
    # Uses sklearn.datasets.make_classification()
    # Creates 5 balanced binary classification datasets
    # Returns dictionary: {domain_key: {'data': df, 'name': domain_name}}
```

**Output:**
```
DATASET SUMMARY - All Domains
Domain 1: NLP Review Classification
  Shape: (800, 16) - Samples × Features
  ✓ Samples: 800 (Requirement: ≥500)
  ✓ Features: 15 (Requirement: ≥10)
  ... (similar for other domains)
```

---

### **SECTION 2-4: ML MODELS, METRICS & PREPROCESSING**
**Files Called:** `models.py`, `metrics.py`, `preprocessing.py`
**Main File:** `main.py` (lines 56-68)

#### **SECTION 2: Model Definitions** → `models.py`

**Classes Implemented (from scratch, NumPy only):**

1. **LogisticRegression**
   - Binary classification
   - Sigmoid activation
   - Binary cross-entropy loss
   - Gradient descent optimization

2. **LinearRegression**
   - Regression task
   - Linear forward pass
   - Mean Squared Error (MSE) loss

3. **SoftmaxRegression**
   - Multi-class classification
   - Softmax activation
   - Categorical cross-entropy loss
   - One-hot encoding

4. **MultiLayerPerceptron**
   - Deep neural network with customizable hidden layers
   - ReLU activation for hidden layers
   - Sigmoid/Softmax for output
   - Full backpropagation with chain rule
   - Xavier weight initialization

**Key Methods in Each Model:**
- `fit(X, y)` - Train the model
- `predict(X)` - Make predictions
- `predict_proba(X)` - Get probability estimates
- `forward()` / `backward()` - Forward & backpropagation

#### **SECTION 3: Evaluation Metrics** → `metrics.py`

**Classes Implemented:**

```python
class EvaluationMetrics:
    - accuracy(y_true, y_pred)      # (TP + TN) / All
    - precision(y_true, y_pred)     # TP / (TP + FP)
    - recall(y_true, y_pred)        # TP / (TP + FN)
    - f1_score(y_true, y_pred)      # 2 * (P * R) / (P + R)
    - rmse(y_true, y_pred)          # Root Mean Squared Error
    - mae(y_true, y_pred)           # Mean Absolute Error
    - confusion_matrix()             # Classification confusion matrix
```

#### **SECTION 4: Data Preprocessing** → `preprocessing.py`

**Function:** `preprocess_data(df, target_col, test_size, scaling)`

**Steps:**
1. **Handle Missing Values** - Fill with mean
2. **Separate Features & Target** - Extract X and y
3. **Feature Scaling** - StandardScaler or MinMaxScaler
4. **Train-Test Split** - 80% train, 20% test

**Returns:** X_train, X_test, y_train, y_test, scaler

**Output:**
```
DATA PREPROCESSING PIPELINE
[Step 1] Handling Missing Values
✓ No missing values found
[Step 2] Separating Features and Target
✓ Features shape: (800, 15)
✓ Target shape: (800,)
[Step 3] Feature Scaling
✓ Features scaled using StandardScaler
[Step 4] Train-Test Split
✓ Train set: 640 samples
✓ Test set: 160 samples
```

---

### **SECTION 5: MULTI-DOMAIN MODEL TRAINING**
**File Called:** `training.py`
**Main File:** `main.py` (lines 71-76)

**Main Function:** `train_all_domains(datasets, learning_rate=0.01, epochs=150)`

**What Happens:**
For each of the 5 domains:
1. **Load dataset** from Section 1
2. **Preprocess data** using preprocessing.py
3. **Train Baseline Model** (Logistic Regression)
   - Learning rate: 0.01
   - Epochs: 150
   - Track loss history
   - Calculate metrics (accuracy, precision, recall, F1)
4. **Train Advanced Model** (MLP)
   - Hidden layers: [64, 32]
   - Learning rate: 0.01
   - Epochs: 150
   - Calculate metrics
5. **Comparative Analysis**
   - Calculate accuracy improvement
   - Calculate training time overhead
   - Generate recommendations

**Key Function:**
```python
def train_domain_models(domain_key, dataset_info, learning_rate, epochs):
    # 1. Preprocess (calls preprocessing.py)
    # 2. Train Logistic Regression baseline (calls models.py)
    # 3. Calculate metrics (calls metrics.py)
    # 4. Train MLP (calls models.py)
    # 5. Calculate metrics (calls metrics.py)
    # 6. Compare performance
    return domain_results
```

**Output (for each domain):**
```
TRAINING DOMAIN: Domain 1: NLP Review Classification
================================
BASELINE MODEL: Logistic Regression
  Epoch 50/150 - Loss: 0.458292
  Epoch 100/150 - Loss: 0.342156
  Epoch 150/150 - Loss: 0.285643
  Accuracy: 0.8750, Precision: 0.8824, Recall: 0.8421, F1: 0.8619
  Training Time: 0.0234s

ADVANCED MODEL: Multi-Layer Perceptron
  Epoch 50/150 - Loss: 0.445123
  ...
  Accuracy: 0.9062, Precision: 0.9167, Recall: 0.8947, F1: 0.9055
  Training Time: 0.0892s

COMPARATIVE ANALYSIS
Accuracy Improvement: +3.12%
Training Time Overhead: +281.20%
```

---

### **SECTION 6: TRAINING LOSS VISUALIZATION**
**File Called:** `visualization.py`
**Main File:** `main.py` (lines 79-84)

**Function:** `visualize_training_loss(all_results)`

**What Happens:**
- Creates 3×2 subplot grid (one for each of 5 domains)
- Plots training loss curves for both Baseline and MLP
- Shows convergence behavior
- Displays final loss values

**Creates:**
- X-axis: Epochs (0-150)
- Y-axis: Loss value
- Blue line: Baseline (Logistic Regression)
- Red line: MLP model
- Annotation: Final loss values

**Output:** Matplotlib figure showing loss convergence

---

### **SECTION 7: PERFORMANCE METRICS COMPARISON**
**File Called:** `visualization.py`
**Main File:** `main.py` (lines 87-93)

**Functions Called:**
1. **`print_comprehensive_metrics(all_results)`**
   - Prints detailed metrics tables for each domain
   - Shows Accuracy, Precision, Recall, F1-Score, Training Time

2. **`visualize_metrics_comparison(all_results)`**
   - Creates 2×2 subplot grid for 4 metrics
   - Bar charts comparing Baseline vs MLP
   - Domains on X-axis, Metric values on Y-axis
   - Displays actual values on bars

**Output:**
- Printed tables with detailed metrics
- Matplotlib figure with 4 comparison plots (Accuracy, Precision, Recall, F1)

---

### **SECTION 8: SITUATED LEARNING ANALYSIS**
**File Called:** `analysis.py`
**Main File:** `main.py` (lines 96-100)

**Function:** `print_situated_learning_analysis()`

**What Happens:**
Prints comprehensive domain-specific analysis:

1. **Domain 1 (NLP - Review Moderation)**
   - Discussion: Precision vs Recall trade-offs
   - Why Recall > 95% is critical for legal compliance
   - Recommendation: Prioritize recall over precision

2. **Domain 2 (Agriculture - Crop Health)**
   - Discussion: Computational cost vs marginal accuracy gains
   - Energy constraints on drone systems
   - When to deploy baseline vs MLP

3. **Domain 3 (Database - Infrastructure)**
   - Discussion: Impact of preprocessing on reliability
   - Missing value handling strategies
   - Feature scaling effects on generalization

4. **Domain 4 (Healthcare - Patient Readmission)**
   - Discussion: Why Recall > Accuracy for clinical systems
   - Asymmetric error costs (False Negatives critical)
   - Threshold optimization

5. **Domain 5 (FinTech - Loan Default)**
   - Discussion: Bias amplification through preprocessing
   - Algorithmic fairness and demographic parity
   - Regulatory compliance requirements

**Output:** 2000+ line detailed analysis document

---

### **SECTION 9: FINAL RESULTS SUMMARY**
**File Called:** `results.py`
**Main File:** `main.py` (lines 103-107)

**Function:** `display_assignment_results(all_results)`

**What Happens:**
1. **Calls `get_assignment_results()`** - Aggregates all results into structured dictionary
2. **Prints Summary:**
   - Student ID: adid1212
   - Number of domains: 5
   - Models implemented: 4
   - Compliance verification
   - Dataset compliance checks

**Key Outputs:**
```
Student ID: adid1212
Domains: 5
Models Implemented: 4
  ✓ Logistic Regression (from scratch, NumPy)
  ✓ Linear Regression (from scratch, NumPy)
  ✓ Softmax Regression (from scratch, NumPy)
  ✓ Multi-Layer Perceptron with Backpropagation

Compliance Verification:
  ✓ No TensorFlow: True
  ✓ No PyTorch: True
  ✓ NumPy-only ML: True

Dataset Compliance:
  Domain 1: NLP Review Classification
    Total Samples: 800 (Requirement: ≥500 ✓)
    Features: 15 (Requirement: ≥10 ✓)
  ... (similar for other domains)
```

**Returns:** `assignment_results` dictionary containing:
```python
{
    'student_info': {...},
    'implementation_summary': {...},
    'dataset_statistics': {...},
    'model_performance': {...},
    'training_configuration': {...},
    'analysis_findings': {...}
}
```

---

### **SECTION 10: EXPORT INSTRUCTIONS**
**File Called:** `results.py`
**Main File:** `main.py` (lines 110-115)

**Function:** `print_export_guide()`

**What Happens:**
- Prints submission requirements checklist
- Shows final project structure
- Provides instructions for export and submission

**Output:**
```
SUBMISSION EXPORT INSTRUCTIONS
✓ Python modular project structure
✓ Complete implementation from scratch using NumPy
✓ All 4 models implemented
... (checklist items)

PROJECT STRUCTURE:
├── main.py
├── models.py
├── metrics.py
├── preprocessing.py
├── datasets.py
├── training.py
├── visualization.py
├── analysis.py
├── results.py
└── adid1212_assignment1.ipynb
```

---

## 📁 Module Dependency Graph

```
main.py (ORCHESTRATOR)
  │
  ├─→ datasets.py
  │    └─→ create_domain_datasets()
  │    └─→ display_dataset_summary()
  │
  ├─→ models.py
  │    ├─→ LogisticRegression
  │    ├─→ LinearRegression
  │    ├─→ SoftmaxRegression
  │    └─→ MultiLayerPerceptron
  │
  ├─→ metrics.py
  │    └─→ EvaluationMetrics
  │
  ├─→ preprocessing.py
  │    └─→ preprocess_data()
  │
  ├─→ training.py
  │    ├─→ train_domain_models()
  │    └─→ train_all_domains()
  │         (calls: models.py, preprocessing.py, metrics.py)
  │
  ├─→ visualization.py
  │    ├─→ visualize_training_loss()
  │    ├─→ visualize_metrics_comparison()
  │    └─→ print_comprehensive_metrics()
  │
  ├─→ analysis.py
  │    ├─→ print_situated_learning_analysis()
  │    └─→ get_domain_recommendations()
  │
  └─→ results.py
       ├─→ get_assignment_results()
       ├─→ display_assignment_results()
       └─→ print_export_guide()
```

---

## 🚀 Step-by-Step Execution

### **Step 1: Verify Environment**
```bash
python --version  # Should be 3.7+
```

### **Step 2: Install Dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### **Step 3: Navigate to Project**
```bash
cd c:\Users\DELL\Documents\BiTs\Course\Sem3\DL\Ass_1
```

### **Step 4: Run Main Orchestrator**
```bash
python main.py
```

### **Step 5: Expected Output Timeline**
1. **0-5 sec:** SECTION 0 - Imports complete
2. **5-10 sec:** SECTION 1 - Datasets created and summarized
3. **10-15 sec:** SECTION 2-4 - Models, metrics, preprocessing confirmed
4. **15-60 sec:** SECTION 5 - Training all domains (4 models × 5 domains = 20 trainings)
   - Each training: ~1-2 seconds
5. **60-65 sec:** SECTION 6 - Loss visualization displayed
6. **65-70 sec:** SECTION 7 - Metrics comparison displayed
7. **70-75 sec:** SECTION 8 - Situated learning analysis printed
8. **75-80 sec:** SECTION 9 - Final results summary
9. **80-85 sec:** SECTION 10 - Export guide printed
10. **85+ sec:** Completion message

**Total Execution Time:** ~90 seconds

---

## 📊 Key Metrics Collected

For each domain and each model:
- **Accuracy:** (TP + TN) / Total predictions
- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- **F1-Score:** Harmonic mean of precision & recall
- **Training Time:** Seconds to train

**Improvement Metrics:**
- Accuracy delta: MLP accuracy - Baseline accuracy
- Precision delta
- Recall delta
- F1-Score delta
- Time multiplier: MLP training time / Baseline training time

---

## 🔍 File-by-File Summary

| File | Purpose | Lines | Key Classes/Functions |
|------|---------|-------|----------------------|
| **main.py** | Orchestrator | 120 | All sections coordinated |
| **datasets.py** | Data creation | 70 | `create_domain_datasets()` |
| **models.py** | ML models | 500+ | 4 model classes |
| **metrics.py** | Evaluation | 50 | `EvaluationMetrics` class |
| **preprocessing.py** | Data prep | 50 | `preprocess_data()` |
| **training.py** | Model training | 100 | `train_all_domains()` |
| **visualization.py** | Plotting | 150+ | 3 visualization functions |
| **analysis.py** | Domain analysis | 200+ | Situated learning insights |
| **results.py** | Final output | 100+ | Results aggregation |

---

## ✅ Compliance Checklist

- ✅ All models from scratch (NumPy only)
- ✅ No TensorFlow, PyTorch, or sklearn ML modules
- ✅ 4 models implemented (Logistic, Linear, Softmax, MLP)
- ✅ 5 domains with 500+ samples, 10+ features each
- ✅ Forward propagation implemented
- ✅ Backpropagation implemented
- ✅ Gradient descent optimization
- ✅ Loss visualization
- ✅ Performance metrics comparison
- ✅ Situated learning analysis for all domains
- ✅ Modular architecture with separate files

---

## 📝 Notes

- **Reproducibility:** Random seed set to 42 for consistent results
- **Matplotlib Backend:** Requires display capability (use `plt.savefig()` if headless)
- **Scikit-learn Usage:** Only for `train_test_split`, `StandardScaler` (data utilities, not ML models)
- **Training Duration:** Varies based on system; typical ~90 seconds on modern hardware
- **Model Accuracy:** Baseline ~75-85%, MLP ~80-90% depending on domain

---

## 🎯 Quick Reference: Which File Does What?

Need to modify...

- **Datasets** → Edit `datasets.py`
- **Model architecture** → Edit `models.py`
- **Metrics calculation** → Edit `metrics.py`
- **Preprocessing steps** → Edit `preprocessing.py`
- **Training loop** → Edit `training.py`
- **Visualizations** → Edit `visualization.py`
- **Domain analysis** → Edit `analysis.py`
- **Final output format** → Edit `results.py`
- **Execution flow** → Edit `main.py`

---

## 📞 Contact & Support

For questions about the implementation:
- Check the docstrings in each Python file
- Review comments in main.py for section descriptions
- See inline comments in model classes for mathematical details

**Assignment ID:** adid1212  
**Last Updated:** May 23, 2026

---

**End of README**
