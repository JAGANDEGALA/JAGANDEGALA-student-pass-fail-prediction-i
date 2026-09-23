import pandas as pd
import numpy as np
import streamlit as st
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


np.random.seed(42)

data = {
    "Study_Hours" : np.random.randint(1,10,100),
    "Attendance" : np.random.randint(50,100,100),
    "Assignments" :np.random.randint(1,6,100)
}

df = pd.DataFrame(data)

df["Results"] =(
    (
        df["Study_Hours"] >= 5
    ) &
    (
        df["Attendance"] >= 75
    )
)

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Assignments"
    ]
]

y = df["Results"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

lr = LogisticRegression()
lr.fit(X_train,y_train)
lr_pred = lr.predict(X_test)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train,y_train)
dt_pred = dt.predict(X_test)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)

lr_accuracy = accuracy_score(
    y_test,
    lr_pred
)

dt_accuracy = accuracy_score(
    y_test,
    dt_pred
)

knn_accuracy = accuracy_score(
    y_test,
    knn_pred
)

# print("Logistic pedictios :",lr_pred)
# print("Decision Tree prediction",dt_pred)
# print("KNN prediction :",knn_pred)
# print("Accuracy of lr:",lr_accuracy)
# print("Accuracy of dt :",dt_accuracy)
# print("Knn Accuracy :",knn_accuracy)

results = pd.DataFrame(
    {
        "Model" :[
            "Logistic Regression",
            "Decision Tree",
            "KNN"
        ],
        "Accuracy" :[
            lr_accuracy,
            dt_accuracy,
            knn_accuracy
        ]
    }
)

cm = confusion_matrix(
    y_test,
    dt_pred
)

# print(results)
print("confusio matrix of dt :")
print(cm)

with open("student_pass_fail_model.pkl","wb") as file:
    pickle.dump(dt,file)

print("Model saved successfully")

with open("student_pass_fail_model.pkl","rb") as file :
    dt = pickle.load(file)

print("Model loaded")


st.title(" 🎓Student Pass/Fail Predictor")

study_hours = st.number_input(
    "Study Hours",
    min_value=1,
    max_value=12
)

attendance = st.number_input(
    "Attendance(%)",
    min_value=0,
    max_value=100
)

assignments = st.number_input(
    "Assignment Completed",
    min_value=0,
    max_value=10
)

if st.button("Predict Result"):
    new_student = pd.DataFrame(
        [[
            study_hours,
            attendance,
            assignments
        ]],
        columns=[
            "Study_Hours",
            "Attendance",
            "Assignments"
        ]
    )

    prediction = dt.predict(new_student)

    if(prediction[0] == 1):
      st.success("✅ Student Will Pass")
    else :
      st.error("❌ Student Will Fail")

