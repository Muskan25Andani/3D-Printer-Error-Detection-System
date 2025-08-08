# 🖨️ 3D Printer Error Detection System

**AI-powered fault detection for 3D printing operations using accelerometer sensor data.**

---
## 📌 Overview

This is a **Flask-based web application** designed to detect **faults or anomalies** in 3D printer operation by analyzing real-time **3-axis accelerometer data (X, Y, Z)**.

The system uses a **Logistic Regression** model trained on data from the `ADXL345` sensor. It predicts whether the printer is operating normally or if a fault is present — achieving a **training accuracy of 88.5%**.

---

## ✨ Features

- 🔹 **Real-time Fault Detection** from live accelerometer input  
- 🔹 **Interactive Web Interface** built with HTML/CSS  
- 🔹 **Sample Test Buttons** for quick demo  
- 🔹 **Model Summary Display** including type and accuracy  
- 🔹 **Mobile-friendly UI** for better accessibility

---

## 🛠️ Tech Stack

| Layer        | Technologies                           |
|--------------|----------------------------------------|
| **Backend**  | Flask (Python), scikit-learn, joblib   |
| **Frontend** | HTML5, CSS3, Jinja2, Font Awesome      |
| **ML Model** | Logistic Regression                    |
| **Dataset**  | ADXL345 accelerometer sensor data      |

---

## 📂 Project Structure

```
FINAL PROJECT/
├── 3printer.ipynb                 # Jupyter notebook for preprocessing & model training
├── ADXL345_SensorData.csv         # Dataset Used
├── app.py                         # Flask app for serving predictions
├── best_model.pkl                 # Trained Logistic Regression model
├── label_encoder.pkl              # Encoded labels for classification 
├── scaler.pkl                     # Scaler for input normalization
├── requirements.txt               # Packages Used
├── templates/
│   └── index.html                 # Web interface for the Flask app
```

---

## 🚀 How to Run Locally

1. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Flask App**
   ```bash
   python app.py
   ```

3. **Open your browser and visit**
   ```
   http://127.0.0.1:5000
   ```

---

## 📊 Model Performance

-  **Best Model:** Logistic Regression  
-  **Accuracy:** 88.48% on test data  
-  **Input:** X, Y, Z values from ADXL345  
-  **Output:** Binary classification – “Yes” (error), “No” (no error)

---

## ⚠️ Limitations

- The dataset is imbalanced, with more "YES" (error) labels than "NO", which may affect prediction accuracy for rare cases.
- Further generalization requires more balanced or real-time data from diverse printer scenarios.

---
