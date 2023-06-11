import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
import seaborn as sns


st.title("Exploratory Data Analysis")

# File uploader button:
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)


  # Relevant statistics about the dataset:
  st.write("Number of rows:", df.shape[0])
  st.write("Number of columns:", df.shape[1])

  num_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
  st.write("Number of categorical variables:", num_categorical)

  num_numerical = len(df.select_dtypes(include=['int64', 'float64']).columns)
  st.write("Number of numerical variables:", num_numerical)

  num_boolean = len(df.select_dtypes(include=['bool']).columns)
  st.write("Number of boolean variables:", num_boolean)

  # Input widget to select a single column from the dataset:
  column = st.sidebar.selectbox('Select a Column', df.columns)

  if df[column].dtype in [np.int64, np.float64]:
    numerical_column = column
    
    # Five-number summary in a table:
    summary = df[numerical_column].describe()
    st.write("Five-number summary:")
    st.table(summary)

    # Distribution plot:
    fig, ax = plt.subplots()
    sns.histplot(x = df[numerical_column], 
                 stat = "density", kde = True)
    plt.title(f"Histogram with KDE curve of {numerical_column}")
    plt.xlabel(numerical_column)
    plt.ylabel("Density")
    st.pyplot(fig)


  elif df[column].dtype in [object, pd.Categorical]:
    categorical_column = column

    # Proportions of each category level in a table:
    proportions = df[categorical_column].value_counts(normalize=True).reset_index()
    proportions.columns = [categorical_column, 'Proportion']
    st.write("Proportions of each category level:")
    st.table(proportions)

    # Customized barplot:
    fig, ax = plt.subplots()
    sns.barplot(data = proportions, x = categorical_column, 
                y = 'Proportion')
    plt.title(f"Proportions of {categorical_column}")
    plt.xlabel(categorical_column)
    plt.ylabel("Proportion")
    st.pyplot(fig)