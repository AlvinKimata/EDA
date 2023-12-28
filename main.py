#!/usr/bin/env python
# coding: utf-8

# # Salary Analysis
# This code analyzes salary data from a European country. The dataset consists of samples of annual salaries. We aim to explore the distribution and calculate the mean salary.
# Contents
# 1. **Data Loading:** Load and combine salary data from multiple CSV files.
# 2. **Exploratory Data Analysis (EDA):** Explore the distribution of salaries through visualizations and summary statistics.
# 3. **Mean Salary Calculation:** Calculate the mean annual salary based on the dataset.

import numpy as np
import pandas as pd
from scipy.stats import norm
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
    plt.figure(figsize=(14, 8))

    # Plot the histogram.
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g', edgecolor='black', label = 'PDF')

    # Fit a normal distribution to the data
    mu, std = norm.fit(data)

    # Plot the PDF
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.text(120000, 0.00002, f'Mean Salary: {mu:.2f}\nStandard deviation: {std:.2f}', fontsize=10)


    plt.title('Probability Density Function of Salary Distribution')
    plt.xlabel('Salary')
    plt.ylabel('Probability Density')


    plt.legend()
    plt.grid(True)
    plt.show()

    return mu, std

X = convert_df_to_array(df=df_combined)
mean, std_dev = plot_pdf(data=X)

# Display the result
print(f"The mean annual salary is: {round(mean, 2)}")
