def ascii_histogram(data, bins=10, width=50):
    """Create an ASCII histogram"""
    # Calculate bin ranges
    min_val = min(data)
    max_val = max(data)
    bin_width = (max_val - min_val) / bins
    
    # Count values in each bin
    counts = [0] * bins
    for val in data:
        bin_idx = min(int((val - min_val) / bin_width), bins - 1)
        counts[bin_idx] += 1
    
    # Find the maximum count for scaling
    max_count = max(counts)
    scale_factor = width / max_count if max_count > 0 else 1
    
    # Generate the histogram
    histogram = []
    for i in range(bins):
        bin_start = min_val + i * bin_width
        bin_end = bin_start + bin_width
        bar_length = int(counts[i] * scale_factor)
        bar = '#' * bar_length
        histogram.append(f"{bin_start:.2f}-{bin_end:.2f} | {bar} ({counts[i]})")
    
    return '\n'.join(histogram)

def ascii_scatter_plot(x_data, y_data, width=50, height=20):
    """Create a simple ASCII scatter plot"""
    if len(x_data) != len(y_data):
        return "Error: x and y data must have the same length"
    
    # Normalize data to fit in the plot area
    x_min, x_max = min(x_data), max(x_data)
    y_min, y_max = min(y_data), max(y_data)
    
    # Create empty plot
    plot = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Place points
    for x, y in zip(x_data, y_data):
        # Scale the point to fit in the plot
        scaled_x = int((x - x_min) / (x_max - x_min) * (width - 1)) if x_max > x_min else 0
        # Invert y since ASCII plots have (0,0) at the top-left
        scaled_y = height - 1 - int((y - y_min) / (y_max - y_min) * (height - 1)) if y_max > y_min else 0
        
        # Ensure within bounds
        scaled_x = max(0, min(width - 1, scaled_x))
        scaled_y = max(0, min(height - 1, scaled_y))
        
        plot[scaled_y][scaled_x] = '*'
    
    # Convert to string
    plot_str = []
    for row in plot:
        plot_str.append('|' + ''.join(row) + '|')
    
    # Add x-axis
    plot_str.append('-' * (width + 2))
    
    # Add labels
    x_label = f"x: {x_min:.2f}" + ' ' * (width - len(f"x: {x_min:.2f}") - len(f"{x_max:.2f}")) + f"{x_max:.2f}"
    plot_str.append(x_label)
    
    return '\n'.join(plot_str)