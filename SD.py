#!/usr/bin/env python
# coding: utf-8

# In[1]:


streamlit_code = '''
import streamlit as st
import pandas as pd
import sqlite3
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report

# Set the page layout
st.set_page_config(page_title="Heart Disease Analysis", layout="wide")

# Connect to the SQLite database
def load_data():
    conn = sqlite3.connect("heart.db")  # Updated database name
    patients_data = pd.read_sql_query("SELECT * FROM patients;", conn)
    diagnosis_data = pd.read_sql_query("SELECT * FROM diagnosis;", conn)
    measurements_data = pd.read_sql_query("SELECT * FROM measurements;", conn)
    conn.close()
    
    # Merge data into a single DataFrame
    data = pd.merge(patients_data, diagnosis_data, on="patient_id")
    data = pd.merge(data, measurements_data, on="patient_id")
    return data

# Load data
data = load_data()

# Sidebar options
st.sidebar.header("Options")
selected_analysis = st.sidebar.selectbox(
    "Select Analysis",
    ["Overview", "Exploratory Data Analysis", "Hypothesis Testing", "Predictive Modeling"]
)

# Title and description
st.title("Heart Disease Analysis Dashboard")
st.markdown(
    """
    This dashboard provides insights into heart disease data through interactive visualizations, hypothesis testing, and predictive modeling.
    """
)

# Analysis: Overview
if selected_analysis == "Overview":
    st.header("Dataset Overview")
    st.write("Preview of the merged dataset:")
    st.dataframe(data.head())
    st.write(f"Number of patients: {data['patient_id'].nunique()}")
    st.write(f"Number of features: {data.shape[1]}")
    st.write("Summary statistics:")
    st.write(data.describe())

# Analysis: Exploratory Data Analysis
elif selected_analysis == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    
    # Age distribution
    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data['age'], kde=True, ax=ax, color='blue')
    ax.set_title("Age Distribution")
    st.pyplot(fig)

    # Cholesterol levels
    st.subheader("Cholesterol Levels")
    fig, ax = plt.subplots()
    sns.boxplot(x=data['chol'], color='green', ax=ax)
    ax.set_title("Cholesterol Levels")
    st.pyplot(fig)

    # Cholesterol vs Blood Pressure
    st.subheader("Cholesterol vs Blood Pressure")
    fig, ax = plt.subplots()
    sns.scatterplot(x=data['chol'], y=data['trestbps'], hue=data['sex'], ax=ax)
    ax.set_title("Cholesterol vs Blood Pressure")
    st.pyplot(fig)

    # Chest Pain by Gender
    st.subheader("Chest Pain Type by Gender")
    fig, ax = plt.subplots()
    sns.countplot(x='cp', hue='sex', data=data, ax=ax)
    ax.set_title("Chest Pain Type by Gender")
    st.pyplot(fig)

    # Correlation matrix
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = data[['age', 'chol', 'trestbps', 'thalch', 'cholesterol_blood_pressure', 'age_to_chol_ratio']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

# Analysis: Hypothesis Testing
elif selected_analysis == "Hypothesis Testing":
    st.header("Hypothesis Testing")
    
    # T-Test for Cholesterol Levels
    st.subheader("T-Test: Cholesterol Levels by Gender")
    male_chol = data[data['sex'] == 1]['chol']
    female_chol = data[data['sex'] == 0]['chol']
    t_stat, p_value = ttest_ind(male_chol, female_chol)
    st.write(f"T-statistic: {t_stat:.2f}, P-value: {p_value:.4f}")
    if p_value < 0.05:
        st.success("There is a significant difference in cholesterol levels between males and females.")
    else:
        st.warning("No significant difference in cholesterol levels between males and females.")

    # Chi-Square Test for Chest Pain Type and Gender
    st.subheader("Chi-Square Test: Chest Pain Type and Gender")
    contingency_table = pd.crosstab(data['cp'], data['sex'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    st.write(f"Chi-Square Statistic: {chi2:.2f}, P-value: {p:.4f}")
    if p < 0.05:
        st.success("There is a significant relationship between chest pain type and gender.")
    else:
        st.warning("No significant relationship between chest pain type and gender.")

# Analysis: Predictive Modeling
elif selected_analysis == "Predictive Modeling":
    st.header("Predictive Modeling")

    # Logistic Regression
    st.subheader("Logistic Regression")
    X = data[['age', 'chol']]
    y = (data['cp'] > 0).astype(int)  # Binary classification
    log_reg = LogisticRegression()
    log_reg.fit(X, y)
    st.write("Logistic Regression Coefficients:")
    st.write(f"Intercept: {log_reg.intercept_[0]:.2f}, Coefficients: {log_reg.coef_[0]}")

    # Decision Tree Classifier
    st.subheader("Decision Tree Classifier")
    X = data[['age', 'chol', 'trestbps', 'thalch']]
    y = data['cp']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Plot Decision Tree
    st.subheader("Decision Tree Visualization")
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(clf, feature_names=X.columns, class_names=[str(label) for label in y.unique()], filled=True, ax=ax)
    st.pyplot(fig)

# Footer
st.sidebar.info("Dashboard created successfully. Explore the options above to interact with the data.")

'''
with open("app.py", "w") as file:
    file.write(streamlit_code)
print("Streamlit code has been saved to 'app.py'")


# In[ ]:




