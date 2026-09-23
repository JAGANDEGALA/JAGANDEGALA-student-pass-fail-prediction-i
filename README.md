# 🎓 Student Pass/Fail Prediction

## 📌 Project Overview

This machine learning project predicts whether a student will pass or fail based on:

* Study Hours
* Attendance
* Assignments Completed

The project demonstrates a complete classification workflow from dataset creation to model deployment using Streamlit.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit

## 🤖 Machine Learning Models

The following classification models were trained and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. KNN Classifier

## 📊 Evaluation Metric

* Accuracy Score
* Confusion Matrix

## 📈 Results

| Model               | Accuracy |
| ------------------- | -------: |
| Logistic Regression |      95% |
| Decision Tree       |     100% |
| KNN                 |      95% |

The Decision Tree model achieved the highest accuracy and was selected for deployment.

## 🌐 Streamlit Application

Users can enter:

* Study Hours
* Attendance Percentage
* Assignments Completed

The application predicts:

* Pass
* Fail

## ▶️ How to Run

Install required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run passFailpred.py
```

## ⚠️ Disclaimer

This project uses a synthetic dataset created for educational and learning purposes.
