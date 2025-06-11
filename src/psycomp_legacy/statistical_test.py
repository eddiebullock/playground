import math

def calculate_mean(values):
    """Calculate the mean of a list of values"""
    return sum(values) / len(values)

def calculate_variance(values, sample=True):
    """Calculate variance of a list of values"""
    mean = calculate_mean(values)
    squared_diff_sum = sum((x - mean) ** 2 for x in values)
    if sample:
        # Sample variance (divide by n-1)
        return squared_diff_sum / (len(values) - 1)
    else:
        # Population variance (divide by n)
        return squared_diff_sum / len(values)

def calculate_std_dev(values, sample=True):
    """Calculate standard deviation"""
    return math.sqrt(calculate_variance(values, sample))

def t_test_one_sample(sample, population_mean, alpha=0.05):
    """Perform one-sample t-test"""
    sample_mean = calculate_mean(sample)
    sample_std = calculate_std_dev(sample)
    n = len(sample)
    
    # Calculate t-statistic
    t_stat = (sample_mean - population_mean) / (sample_std / math.sqrt(n))
    
    # For p-value calculation, we'd need to implement the t-distribution
    # Since that's complex, here we use a simplified approach
    # In practice, you'd use a lookup table or scipy.stats
    
    # For a simplified result, we'll just compare against critical values
    # Approximate critical values for common alphas and df > 30:
    # alpha=0.05: t_crit ≈ 1.96
    # alpha=0.01: t_crit ≈ 2.58
    
    t_crit = 1.96 if alpha == 0.05 else 2.58
    
    # Check if we reject the null hypothesis
    reject_null = abs(t_stat) > t_crit
    
    return {
        "t_statistic": t_stat,
        "sample_mean": sample_mean,
        "sample_std": sample_std,
        "degrees_of_freedom": n - 1,
        "reject_null": reject_null
    }

def t_test_two_sample(sample1, sample2, equal_var=True, alpha=0.05):
    """Perform two-sample t-test"""
    mean1 = calculate_mean(sample1)
    mean2 = calculate_mean(sample2)
    var1 = calculate_variance(sample1)
    var2 = calculate_variance(sample2)
    n1 = len(sample1)
    n2 = len(sample2)
    
    if equal_var:
        # Pooled variance
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        # T-statistic
        t_stat = (mean1 - mean2) / math.sqrt(pooled_var * (1/n1 + 1/n2))
        df = n1 + n2 - 2
    else:
        # Welch's t-test
        t_stat = (mean1 - mean2) / math.sqrt(var1/n1 + var2/n2)
        # Welch-Satterthwaite equation for degrees of freedom
        numerator = (var1/n1 + var2/n2)**2
        denominator = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
        df = numerator / denominator
    
    # Similar to one-sample test, we use approximate critical values
    t_crit = 1.96 if alpha == 0.05 else 2.58
    reject_null = abs(t_stat) > t_crit
    
    return {
        "t_statistic": t_stat,
        "mean_difference": mean1 - mean2,
        "degrees_of_freedom": df,
        "reject_null": reject_null
    }
