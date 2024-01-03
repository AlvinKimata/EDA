#!/usr/bin/env python
# coding: utf-8

# # Salary Analysis
# This code analyzes salary data from a European country. The dataset consists of samples of annual salaries. We aim to explore the distribution and calculate the mean salary.
# Contents
# 1. **Data Loading:** Load and combine salary data from multiple CSV files.
# 2. **Exploratory Data Analysis (EDA):** Explore the distribution of salaries through visualizations and summary statistics.
# 3. **Mean and Standard Deviation Salary Calculation:** Calculate the mean and standard deviation annual salary based on the dataset.

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from numpy.random import normal

df = pd.read_csv('data8-1.csv')

def convert_df_to_array(df):
    data = df.values.tolist()
    data = np.array(data)
    data = np.squeeze(data)
    return data

def calculate_mean_std(data):
    # Calculate mean and standard deviation
    mu, std = norm.fit(data)
    return mu, std

def plot_pdf(data):
    plt.figure(figsize=(14, 8))

    # Plot the histogram.
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g', edgecolor='black', label='PDF')

    # Fit a normal distribution to the data
    mu, std = calculate_mean_std(data)

    # Plot the PDF
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.text(120000, 0.00002, f'Mean Salary: €{mu:.2f}\nStandard deviation: €{std:.2f}', fontsize=10)

    plt.title('Probability Density Function of Salary Distribution')
    plt.xlabel('Salary')
    plt.ylabel('Probability Density')

    plt.legend()
    plt.grid(True)
    plt.show()



data = convert_df_to_array(df=df)
mean, std_dev =  calculate_mean_std(data)
plot_pdf(data)

# Display the result
print(f"The mean annual salary is: {round(mean, 2)}")
