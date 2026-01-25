import pandas as pd

try:
    df = pd.read_csv('mental/file/path')
    print(df.head())
except:
    print("not sure about that file path lad")

    
    