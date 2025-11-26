# Project Background (For Claude Code)

## Project Overview
I'm working on an ICU blood pressure prediction project. The goal is to **predict arterial blood pressure (ART) from PLETH (PPG) signals**. Data comes from hospital ICU monitors in .vital file format.

## Current Problem
- Using traditional machine learning models (XGBoost, Random Forest, etc.) with simple offset calibration shows poor performance
- **Performance degrades after calibration** because simple linear calibration cannot handle the complexity of individual differences
- Need to implement **transfer learning + personalized fine-tuning** approach to solve this

## Project Structure
```
PAE_NEW/
├── data_loader.py          # Load .vital files
├── signal_processing.py    # Signal filtering and peak detection
├── feature_extraction.py   # Extract features from PLETH (per heartbeat)
├── models.py              # Model training
├── evaluation.py          # Evaluation and visualization
├── config.py              # Configuration file
└── main.py                # Main pipeline
```

## Data Pipeline
1. .vital file → Extract PLETH and ART signals
2. Signal processing → Filtering, peak detection, heartbeat segmentation
3. Feature extraction → Extract 17-36 features per heartbeat (statistics, peaks, derivatives, etc.)
4. Model training → Predict systolic blood pressure (SBP) and diastolic blood pressure (DBP)

## Performance Goals
- MAE < 10 mmHg
- MSE < 70 (systolic) / 60 (diastolic)

---

# Transfer Learning + Fine-tuning Framework Design

## Core Concept
**Don't use simple calibration, use transfer learning instead**:
1. **General Model Stage**: Train a general model on multiple patients from training set
2. **Personalization Stage**: Fine-tune the model using a small amount of data from test patient (first 100-300 heartbeats)
3. **Prediction Stage**: Use fine-tuned model to predict remaining data for that patient

## Why It Works
- **Preserve general knowledge**: Basic PLETH-BP relationships (waveform morphology, physiological patterns)
- **Adapt to individual differences**: Skin thickness, vascular characteristics, baseline blood pressure, etc.
- **Avoid catastrophic forgetting**: Only fine-tune partial layers without destroying general features

---

# Detailed Technical Framework

## Module 1: General Model Trainer

### Functionality
Train a general model on **all training patient data**

### Input
- `X_train_all`: Feature matrix from all training patients
- `y_train_sys_all`: Systolic BP from all training patients
- `y_train_dia_all`: Diastolic BP from all training patients

### Output
- `model_general_sys`: General systolic BP model
- `model_general_dia`: General diastolic BP model

### Model Selection
Recommended: **XGBoost or LightGBM** (supports incremental training)

---

## Module 2: Personal Fine-tuner

### Functionality
Customize model for individual test patient

### Input
- `model_general`: General model
- `X_patient_calib`: Features from first N heartbeats of patient (N=100-300)
- `y_patient_calib`: Corresponding true BP values

### Fine-tuning Strategies

#### Strategy A: Continued Training (Recommended for Tree-based Models)
```
Parameter Settings:
- learning_rate = 0.01 (reduce learning rate to avoid overfitting)
- n_estimators_new = 50-100 (incrementally add new trees)
- Use xgb_model parameter to continue from general model
```

#### Strategy B: Partial Retraining (For Neural Networks)
```
- Freeze earlier layers (feature extraction layers)
- Only train last 1-2 layers (decision layers)
- Use small learning rate
```

### Output
- `model_personalized`: Personalized model

---

## Module 3: Data Splitter

### Functionality
Split test patient data into **calibration set** and **evaluation set**

### Splitting Scheme
```
Test Patient Data:
├── First 100-300 heartbeats → Calibration set (for fine-tuning)
└── Remaining heartbeats → Evaluation set (for testing)
```

### Calibration Set Size Determination
- **Time-based**: First 5 minutes (~300 heartbeats at 60 bpm)
- **Sample-based**: Fixed 100-300 heartbeats
- **Ratio-based**: First 20-30% of data

---

## Module 4: Pipeline Integrator

### Complete Workflow
```
Input: train.vital + test.vital

Step 1: Load and Process Training Set
  - Read all training patient data
  - Signal processing → Feature extraction
  - Combine into X_train_all, y_train_all

Step 2: Train General Model
  - model_general = XGBoost(...)
  - model_general.fit(X_train_all, y_train_all)
  - Save general model

Step 3: Load Test Patient
  - Read test.vital
  - Signal processing → Feature extraction
  - Get X_test, y_test

Step 4: Split Test Data
  - X_calib = X_test[:300]
  - y_calib = y_test[:300]
  - X_eval = X_test[300:]
  - y_eval = y_test[300:]

Step 5: Fine-tuning
  - model_personal = fine_tune(model_general, X_calib, y_calib)

Step 6: Evaluation Comparison
  - pred_general = model_general.predict(X_eval)
  - pred_personal = model_personal.predict(X_eval)
  - Compare MAE/MSE

Step 7: Visualization
  - Plot time series
  - Plot scatter plots
  - Generate report
```

---

# Key Technical Details

## XGBoost Fine-tuning Implementation

### Method 1: Using xgb_model Parameter (Recommended)
```
Approach:
1. Train general model to get booster object
2. Continue training from this booster using xgb_model parameter
3. Reduce learning_rate to avoid overfitting

Key Parameters:
- learning_rate: 0.01-0.05 (general model uses 0.1)
- n_estimators: 50-100 (incremental addition)
- max_depth: Keep consistent with general model
- subsample: 0.8-1.0 (use 1.0 for small datasets)
```

### Method 2: Warm Start
```
If XGBoost doesn't support, you can:
1. Use general model predictions as additional features
2. Train a small correction model
3. Final prediction = general prediction + correction prediction
```

---

## LightGBM Fine-tuning Implementation

### Using init_model Parameter
```
LightGBM natively supports incremental training:
- Pass general model path to init_model parameter
- Set small learning_rate
- Continue training
```

---

## Overfitting Prevention Strategies

### 1. Early Stopping
```
Use part of calibration set as validation set:
- 80% calibration data for training
- 20% for early stopping
```

### 2. Limit Model Complexity
```
- Reduce number of new trees
- Increase min_child_weight
- Use L1/L2 regularization
```

### 3. Data Augmentation
```
If calibration data is too small:
- Add slight noise
- Time window sliding
- Feature perturbation
```

---

# File Organization Recommendations

## New Files to Create

### `transfer_learning.py`
```
Functions:
- GeneralTrainer class: Train general model
- PersonalFineTuner class: Fine-tuning
- Save/load model methods
```

### `main_transfer_learning.py`
```
Functions:
- Complete transfer learning pipeline
- Command-line argument support
- Logging
```

### `config_transfer.py`
```
Configuration Items:
- General model parameters
- Fine-tuning parameters
- Calibration data amount
- Output paths
```

---

# Literature Reference

According to literature:
- **General model**: MAE ~10-15 mmHg
- **Offset calibration**: MAE ~8-12 mmHg (may be worse)
- **Transfer learning**: MAE ~3-5 mmHg (60-70% improvement)

Key advantages:
- Capture individual-specific PLETH-BP mapping relationships
- Adapt to individual baseline blood pressure
- Preserve general physiological patterns
