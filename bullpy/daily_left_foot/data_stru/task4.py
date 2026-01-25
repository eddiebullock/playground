def squares_list():
    squares = [x**2 for x in range(10)]
    return squares 

print(squares_list())

# Read a CSV file with pandas or built-in csv and print the first 5 rows.
import pandas as pd

def read_csv():
    df = pd.read_csv('/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/data_c4_clean.csv')
    print(df.head()) # print first 5 rows

read_csv()