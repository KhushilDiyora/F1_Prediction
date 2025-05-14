# 🏁 F1 Qualifying Predictor: Monaco GP 2025

This project predicts the **Qualifying Q3 lap times** for the **2025 Monaco Grand Prix**, using a combination of **historical F1 data (2023–2024)**, **driver/team performance factors**, and a **Linear Regression model**. It also evaluates how accurately we can classify which drivers will likely reach Q3.

---

## 🚀 Features

- Predicts Q3 qualifying times for all 20 drivers using:
  - Team & driver performance factors
  - Monaco-specific difficulty modifiers
  - Historical lap time trends
- Trains a regression model on past qualifying data (FastF1 API)
- Includes a basic Q3 qualification classifier (based on top 50%)
- Provides model evaluation metrics (MAE, RMSE, R², classification report)
- Handles missing data, retries on fetch failures, and uses caching

---

## 🧠 Model Overview

- **Model Type**: Linear Regression
- **Features Used**: Q1 & Q2 lap times
- **Target**: Q3 lap time
- **Classification**: Predict whether driver will reach Q3 (top 50%)

---

## 📊 Sample Classification Report

From testing on 2023–2024 race data:

MODEL FIT STATUS:
✓ EXCELLENT FIT

==================================================

F1 QUALIFYING PREDICTION MODEL ACCURACY

             precision    recall  f1-score   support

   Missed Q3      0.400     1.000     0.571         4
     Made Q3      1.000     0.700     0.824        20

    accuracy                          0.750        24
   macro avg      0.700     0.850     0.697        24
weighted avg      0.900     0.750     0.782        24


FEATURE IMPORTANCE:

Q1_sec: +0.084

Q2_sec: +0.482

==================================================

Monaco GP 2025 Qualifying Predictions:

Position Driver Team Predicted Time

===========================================================================

Position  Driver              Team                     Predicted Time

---------------------------------------------------------------------------

1         Max Verstappen      Red Bull Racing          70.502s

2         Lewis Hamilton      Ferrari                  70.705s

3         Lando Norris        McLaren                  70.819s


## 🔧 Installation & Setup

### Requirements

- Python 3.8+
- FastF1
- pandas, numpy, matplotlib, seaborn
- scikit-learn

### Install Dependencies

```bash
pip install fastf1 pandas numpy matplotlib seaborn scikit-learn

Contact
Developed by KhushilDiyora
For suggestions or questions, feel free to open an issue or contribute.
