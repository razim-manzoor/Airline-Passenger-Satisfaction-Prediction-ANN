# Airline Passenger Satisfaction Prediction 🛫

![Python](https://img.shields.io/badge/Python-3.8-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-brightgreen)

Predict airline passenger satisfaction using a neural network model with **96.5% accuracy**. Deployed as an interactive Streamlit app for real-time predictions.

---

## Overview ✈️
This project predicts passenger satisfaction using a TensorFlow/Keras neural network trained on 120k+ records. Key highlights:
- **96.49% Test Accuracy**
- **0.995 ROC AUC Score**
- Real-time predictions via Streamlit interface.  ![Link](https://airline-passenger-satisfaction-prediction.streamlit.app)
- Robust handling of class imbalance

---

## Dataset 📊
Contains 120,000+ passenger records with:
- **Demographics**: Age, Gender, Customer Type
- **Flight Details**: Class, Distance, Delays
- **Service Ratings** (1-5): Cleanliness, Comfort, Entertainment, WiFi, etc.
- **Target**: Satisfaction (Binary: Satisfied/Neutral/Dissatisfied)

---

## Model Performance 🚀

### Key Metrics
| Metric                  | Value    |
|-------------------------|----------|
| Test Accuracy           | 96.49%   |
| Test Loss               | 0.0923   |
| ROC AUC Score           | 0.9955   |
| Precision-Recall AUC    | 0.9946   |

### Classification Report
          precision    recall  f1-score   support

       0       0.96      0.98      0.97     14690
       1       0.97      0.95      0.96     11286
       accuracy                    0.96     25976
       macro avg      0.97      0.96      0.96     25976
       weighted avg   0.96      0.96      0.96     25976
