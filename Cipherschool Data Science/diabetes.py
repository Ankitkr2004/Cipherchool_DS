# Diabetes Predictor - Clean & Commented Full Pipeline
# Goal: Predict whether a patient is likely to have diabetes using selected health features
# Type: Binary Classification
# Models Used: Logistic Regression and Decision Tree
# Real-life Application: Healthcare Analytics


# Step 1: Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Step 2: Load the Dataset

df = pd.read_csv("diabetes.csv")  # Make sure the CSV file is in the same folder or provide full path

# Step 3: Identify Invalid Data — Some medical values like Glucose, BMI etc. have 0s
# These should not be 0 — we'll treat 0s in these features as missing values
features_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("\nCount of 0s in important features (treated as missing):")
print((df[features_with_zeros] == 0).sum())

# Step 4: Replace 0s with NaN to make them easy to handle
df[features_with_zeros] = df[features_with_zeros].replace(0, np.nan)

# Step 5: Count missing (NaN) values
print("\nMissing values after replacing 0s with NaN:")
print(df[features_with_zeros].isnull().sum())

# Step 6: Fill missing values using the median of each feature
# Median is robust and works well for healthcare data
for col in features_with_zeros:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)

# Step 7: Final check — Any NaN values left?
print("\nMissing values after cleaning (should be 0):")
print(df.isnull().sum())

# Step 8: Correlation heatmap (for analytics)
# Helps identify which features relate most to Outcome
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 9: Select Features & Target for ML Task
# Features chosen: Age, BMI, Glucose, BloodPressure, and DiabetesPedigreeFunction
# These are key indicators for diabetes risk
features = ['Age', 'BMI', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']  # Family history proxy
X = df[features]
y = df['Outcome']  # Target variable: 0 (No diabetes) / 1 (Diabetes)

# Step 10: Split data into training and testing
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 11: Train Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Step 12: Train Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Step 13: Evaluation function to avoid repetition
# This prints accuracy, confusion matrix, and report
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Evaluation:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

# Step 14: Evaluate both models
evaluate_model("Logistic Regression", y_test, y_pred_lr)
evaluate_model("Decision Tree", y_test, y_pred_dt)

print("\nStep 14 - Cleaned Data Preview:")
print(df.head())

# Step 15: Predict for a new patient (example)
new_patient = pd.DataFrame({
    'Age': [21],
    'BMI': [22],
    'Glucose': [140],
    'BloodPressure': [80],
    'DiabetesPedigreeFunction': [0.5]
})
custom_prediction = lr_model.predict(new_patient)
print("\nStep 15 - Custom patient prediction (0 = No Diabetes, 1 = Diabetes):", custom_prediction[0])


# Step 16: Goal Complete
print("\nGoal Completed: Model predicts diabetes using key features - Age, BMI, Glucose, Blood Pressure, and Family History.")

# Step 17: Save cleaned dataset to a new CSV file
df.to_csv("cleaned_diabetes.csv", index=False)
print("\nCleaned dataset saved to 'cleaned_diabetes.csv'")