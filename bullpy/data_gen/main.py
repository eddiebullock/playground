from data_loader import load_data, calculate_questionnaire_totals, create_autism_target

def main():
    """
    main function to load and prepare asc data
    """
    #file path
    file_path = "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/data_c4_raw.csv"

    #load data
    print("loading data...")
    df = load_data(file_path)

    #calculate totals
    print("calculating totals...")
    df = calculate_questionnaire_totals(df)

    #create target
    print("creating autism diagnosis target...")
    df = create_autism_target(df)

    #save prepared data 
    output_path = "prepared_data.csv"
    print(f"Type of df before saving: {type(df)}")
    df.to_csv(output_path, index=False)
    print(f"saved prepared data to {output_path}")

    #print summary stats
    print(f"final dataset shape: {df.shape}")
    print(f"autism prevalence: {df['autism_diagnosis'].mean():.3f}")

if __name__ == "__main__":
    main()