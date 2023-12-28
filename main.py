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
