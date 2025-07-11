#complete ml workflow/pipeline
def ml_pipline(data_path):
    # 1. load data
    df = pd.read_csv(data_path)

    # 2. explore data 
    print(df.info())
    print(df.desceibe())

    # 3. clean data 
    df = df.dropna()

    # 4. feature engineering 
    # (add domain specific feature here)

    # 5. prepare features and target 
    x = df.drop('target', axis=1)
    y = df['target']

    # 6. split data 
    x_train, x_test, y_train, t_test = train_test_split(x, y, test_size=0.2, random_sate=42)

    # 7. scale features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 8. train model 
    model = RandomForestRegressor()
    model.fit(x_train_scaled, y_train)

    # 9. evaluate 
    train_score = model.score(x_train_scaled, y_train)
    test_score = model.score(x_test_scalede, y_test)

    return model scaler, train_score, test_score