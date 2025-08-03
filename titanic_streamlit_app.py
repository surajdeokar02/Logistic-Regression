# Logistic Regression

import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report

# Ensure working directory is script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# File paths
train_path = "Titanic_train.csv"
test_path = "Titanic_test.csv"

# File check
if not os.path.isfile(train_path) or not os.path.isfile(test_path):
    st.error(f"Missing required file in: {os.getcwd()}\n"
             f"- {'FOUND' if os.path.isfile(train_path) else 'MISSING'}: {train_path}\n"
             f"- {'FOUND' if os.path.isfile(test_path) else 'MISSING'}: {test_path}")
    st.stop()

# Load dataset
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
test_passenger_ids = test_df['PassengerId']
# Step 1: Data Exploration
train_df.info()
train_df.describe()
train_df.boxplot()


# Step 2: Data Preprocessing
@st.cache_data
def preprocess(df, is_train=True):
    df = df.copy()
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True, errors='ignore')
    if 'PassengerId' in df.columns:
        df.drop('PassengerId', axis=1, inplace=True)
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    if is_train:
        return df.drop("Survived", axis=1), df["Survived"]
    else:
        return df

X, y = preprocess(train_df, is_train=True)
X_test = preprocess(test_df, is_train=False)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Building
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)



# Step 4: Model Evaluation
st.subheader("📊 Model Evaluation")
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]

col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy", round(accuracy_score(y_val, y_pred), 3))
    st.metric("Precision", round(precision_score(y_val, y_pred), 3))
    st.metric("Recall", round(recall_score(y_val, y_pred), 3))
with col2:
    st.metric("F1 Score", round(f1_score(y_val, y_pred), 3))
    st.metric("ROC AUC", round(roc_auc_score(y_val, y_proba), 3))

with st.expander("📄 Classification Report"):
    st.text(classification_report(y_val, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_proba)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label='ROC Curve')
ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
st.pyplot(fig)

# Step 5: Interpretation
st.subheader("🧠 Model Interpretation")
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})
coefficients['Importance'] = np.abs(coefficients['Coefficient'])
coefficients = coefficients.sort_values(by='Importance', ascending=False)
st.dataframe(coefficients)

# Step 6: Streamlit UI
st.title("🚢 Titanic Survival Prediction App")

# Sidebar - User Input
st.sidebar.header("🎛️ Custom Prediction")
def user_input():
    Pclass = st.sidebar.selectbox('Pclass', [1, 2, 3])
    Sex = st.sidebar.selectbox('Sex', ['male', 'female'])
    Age = st.sidebar.slider('Age', 0, 80, 30)
    SibSp = st.sidebar.slider('SibSp', 0, 8, 0)
    Parch = st.sidebar.slider('Parch', 0, 6, 0)
    Fare = st.sidebar.slider('Fare', 0.0, 500.0, 32.0)
    Embarked = st.sidebar.selectbox('Embarked', ['C', 'Q', 'S'])
    return pd.DataFrame({
        'Pclass': [Pclass],
        'Sex': [0 if Sex == 'male' else 1],
        'Age': [Age],
        'SibSp': [SibSp],
        'Parch': [Parch],
        'Fare': [Fare],
        'Embarked': [{'C': 0, 'Q': 1, 'S': 2}[Embarked]]
    })

input_df = user_input()
if st.sidebar.button("Predict Survival"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    st.subheader("🔍 Prediction Result")
    st.success("Survived" if pred == 1 else "Did not Survive")
    st.write(f"Probability of Survival: {round(prob * 100, 2)}%")

# Test set prediction
st.subheader("📋 Titanic Test Set Predictions")
test_predictions = model.predict(X_test)
test_results = pd.DataFrame({
    "PassengerId": test_passenger_ids,
    "Predicted_Survived": test_predictions
})
st.dataframe(test_results.head(10))

# Optional download
csv = test_results.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Predictions as CSV",
    data=csv,
    file_name='titanic_predictions.csv',
    mime='text/csv'
)

