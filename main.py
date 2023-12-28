#!/usr/bin/env python
# coding: utf-8

# # Salary Analysis
# This code analyzes salary data from a European country. The dataset consists of samples of annual salaries. We aim to explore the distribution and calculate the mean salary.
#Contents
# 1. **Data Loading:** Load and combine salary data from multiple CSV files.
# 2. **Exploratory Data Analysis (EDA):** Explore the distribution of salaries through visualizations and summary statistics.
# 3. **Mean Salary Calculation:** Calculate the mean annual salary based on the dataset.

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.pyplot as plt
from numpy.random import normal

def load_and_rename_csv(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['salary']
    return df

# Define the file paths
file_paths = ['data0-1.csv', 'data1-1.csv', 'data3-2.csv', 'data4-2.csv', 'data5-1.csv', 'data7-1.csv', 'data8-1.csv']

# Load and rename DataFrames using the function
df_list = [load_and_rename_csv(file_path) for file_path in file_paths]

# Concatenate the DataFrames
df_combined = pd.concat(df_list, ignore_index=True)


def convert_df_to_array(df):
    data = df.values.tolist()
    data = np.array(data)
    data = np.squeeze(data)
    return data

def plot_pdf(data):
    # Estimate  parameters (mean and standard deviation) of the normal distribution that best fits the data using maximum likelihood estimation (MLE).
    sample_mean, sample_std_dev = norm.fit(data)
    dist = norm(sample_mean, sample_std_dev)
    values = [value for value in range(np.min(data), np.max(data))]
    probabilities = [dist.pdf(value) for value in values]

    # Plot the PDF
    plt.figure(figsize = (14, 8))
    xmin, xmax = plt.xlim()
    plt.text(120000, 3200, f'Mean Salary: {sample_mean:.2f}\n mean probabilities: {np.mean(probabilities)}', fontsize=10)
    
    plt.plot(values, probabilities)
    plt.hist(data, bins = 30, label='PDF', alpha = 0.4, color = 'g')
    plt.title('Probability Density Function of Salary Distribution')
    plt.xlabel('Salary')
    plt.ylabel('Probability Density')
    
    
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return sample_mean, sample_std_dev, probabilities

X = convert_df_to_array(df = df_combined)

mean, std_dev, probabilities = plot_pdf(data = X)


# Display the result
print(f"The mean annual salary is: {round(mean, 2)}")
