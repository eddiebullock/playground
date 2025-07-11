import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/eb2007/predict_asc_c4/data/data_c4_raw.csv', help='Path to the data file')
    parser.add_argument('--sample_size', type=int, default=None, help='Number of samples to load')
    parser.add_argument('--output_dir', type=str, default='test_output', help='Output directory')
    args = parser.parse_args()

    df = load_data(args.data_path)
    if args.sample_size:
        df = df.sample(n=args.sample_size)
    # Save or process df as needed
    # Example: save to output_dir
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'loaded_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}") 