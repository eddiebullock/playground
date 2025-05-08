
def main():
    # Generate messy data
    print("Generating messy dataset...")
    data = generate_messy_dataset(200)
    
    # Examine missing values
    columns = ["age", "income", "category"]
    missing = detect_missing_values(data, columns)
    print("\nMissing values:")
    for col, count in missing.items():
        print(f"  {col}: {count} ({count/len(data)*100:.1f}%)")
    
    # Clean data
    print("\nCleaning data...")
    cleaned_data = impute_missing_values(data, "mean")
    
    # Check for outliers
    outliers, bounds = detect_outliers_iqr(cleaned_data, "age")
    print(f"\nFound {len(outliers)} outliers in 'age' column")
    print(f"Outlier bounds: {bounds[0]:.2f} to {bounds[1]:.2f}")
    
    # Normalize data
    print("\nNormalizing data...")
    normalized_data = normalize_min_max(cleaned_data, "income")
    
    # Statistical tests
    print("\nPerforming statistical tests...")
    ages = [row["age"] for row in cleaned_data if row["age"] is not None]
    
    # Split ages into two groups for demonstration
    middle = len(ages) // 2
    group1 = ages[:middle]
    group2 = ages[middle:]
    
    # Run t-test
    t_result = t_test_two_sample(group1, group2)
    print("\nTwo-sample t-test results:")
    print(f"  t-statistic: {t_result['t_statistic']:.4f}")
    print(f"  Mean difference: {t_result['mean_difference']:.4f}")
    print(f"  Degrees of freedom: {t_result['degrees_of_freedom']:.1f}")
    print(f"  Reject null hypothesis: {t_result['reject_null']}")
    
    # Visualizations
    incomes = [row["income"] for row in cleaned_data if row["income"] is not None]
    
    print("\nIncome Distribution (ASCII Histogram):")
    print(ascii_histogram(incomes, bins=8))
    
    print("\nAge vs. Income (ASCII Scatter Plot):")
    x_data = [row["age"] for row in cleaned_data if row["age"] is not None and row["income"] is not None]
    y_data = [row["income"] for row in cleaned_data if row["age"] is not None and row["income"] is not None]
    print(ascii_scatter_plot(x_data, y_data))

if __name__ == "__main__":
    main()