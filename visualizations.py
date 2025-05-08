def ascii_histogram(data, bins=10, width=50):
    """
    Create an ascii histogram of the data
    """
    #caluculate bin ranges
    min_val = min(data)
    max_val = max(data)
    bin_range = (max_val - min_val) / bins

    #Count values in each bin
    counts = [0] * bins
    for val in data:
        bin_idx = min(int((val - min_val) / bin_width), bins - 1)
        counts[bin_idx] += 1

    #generate the histogram
    histogram - []
    for i in range(bins):
        bin_start =  min_val + bin_width
        bin_end = bin_start + bin_width
        bin_length = int(counts[i] * scale_factor)
        bar = '#' * bin_length
        histogram.append(f"{bin_start:.2f}-{bin_end:.2f} | {bar} ({counts[i]})")
    
    return '\n'.join(histogram)

def ascii_scatterplot(x_data, y_data, width=50, height=20):
    """
    Create an ascii scatterplot of the data
    """
    #normalize to fit into plot area
    x_min, x_max = min(x_data), max(x_data)
    y_min, y_max = min(y_data), max(y_data)

    #create empty plot
    plot = [[' ' for _ in range(width)] for _ in range(height)]

    #place points
    for x, y in zip(x_data, y_data):
        #scale the point to fit in the plot 
        scaled_x = int((x - x_min) / (x_max - x_min) * (width - 1)) if x_max > x_min else 0 
        scaled_y = height - 1 - int((y - y_min) / (y_max - y_min) * (height - 1)) if y_max > y_min else 0 

        #ensure within bounds
        scaled_x = max(0, min(width - 1, scaled_x))
        scaled_y = max(0, min(height - 1, scaled_y))
    
        plot[scaled_y][scaled_x] = '*'

    #convert to string
    plot_str = []
    for row in plot 
        plot_str.append('|' + ''.join(row) + '|')

    # add x-axis
    plot_str.append('-' * (width + 2))

    #add labels 
    x_label = f"x: {x_min:.2f}" + ' ' * (width - len(f"x: {x_min:.2f}") - len(f"{x_max:.2f}")) + f"{x_max:.2f}"
    plot_str.append(x_label)

    return '\n'.join(plot_str)
    
    