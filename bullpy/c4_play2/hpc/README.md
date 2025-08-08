# HPC Optimization System for Autism Classification

This directory contains the complete HPC optimization system for autism classification using machine learning models.

## Overview

The system performs comprehensive hyperparameter optimization and ensemble methods to achieve the best possible performance for autism classification using questionnaire data (AQ, EQ, SQR, SPQ).

## Best Performing Models (from notebook analysis)

1. **Feature Selection + Ensemble**: F1=0.7188, AUC=0.7592 (100 features)
2. **MLP with Threshold Optimization**: F1=0.7124 
3. **Random Forest with Threshold Optimization**: F1=0.7166
4. **Voting Ensemble**: F1=0.6887, AUC=0.7650

## Files Structure

```
hpc/
├── requirements.txt                    # Python dependencies
├── hpc_config.yaml                    # Configuration file
├── hpc_hyperparameter_tuning.py      # Main hyperparameter tuning script
├── hpc_ensemble_optimization.py       # Ensemble optimization script
├── setup_hpc.sh                      # Setup script
├── README.md                         # This file
├── slurm_scripts/                    # SLURM job scripts
│   ├── run_hyperparameter_tuning.slurm
│   └── run_ensemble_optimization.slurm
└── .gitignore                        # Git ignore file for data
```

## Setup Instructions

### 1. Transfer Files to HPC

```bash
# Copy the entire hpc directory to your HPC system
scp -r hpc/ username@hpc-cluster:/path/to/your/project/
```

### 2. Copy Data Files

Copy your enhanced dataset to the HPC:
```bash
# Copy the enhanced dataset
scp data/processed/data_c4_enhanced_fe_v2.csv username@hpc-cluster:/path/to/your/project/hpc/
```

### 3. Setup Environment

```bash
# SSH to HPC
ssh username@hpc-cluster

# Navigate to hpc directory
cd /path/to/your/project/hpc

# Run setup script
chmod +x setup_hpc.sh
./setup_hpc.sh
```

### 4. Submit Jobs

```bash
# Submit hyperparameter tuning job
sbatch slurm_scripts/run_hyperparameter_tuning.slurm

# Submit ensemble optimization job
sbatch slurm_scripts/run_ensemble_optimization.slurm
```

## Configuration

The `hpc_config.yaml` file contains all optimization parameters:

### Models Optimized

1. **Random Forest**
   - n_estimators: [100, 200, 500, 1000, 2000]
   - max_depth: [None, 10, 15, 20, 25, 30]
   - min_samples_split: [2, 5, 10, 20]
   - min_samples_leaf: [1, 2, 5, 10]
   - max_features: ['sqrt', 'log2', 0.3, 0.5, 0.7]

2. **XGBoost**
   - n_estimators: [100, 200, 500, 1000]
   - max_depth: [3, 5, 7, 9, 11]
   - learning_rate: [0.01, 0.05, 0.1, 0.2]
   - subsample: [0.8, 0.9, 1.0]

3. **LightGBM**
   - n_estimators: [100, 200, 500, 1000]
   - max_depth: [3, 5, 7, 9, 11]
   - learning_rate: [0.01, 0.05, 0.1, 0.2]
   - num_leaves: [31, 63, 127, 255]

4. **Logistic Regression**
   - C: [0.001, 0.01, 0.1, 1, 10, 100]
   - penalty: ['l1', 'l2', 'elasticnet']
   - solver: ['liblinear', 'saga']

5. **Neural Network (MLP)**
   - hidden_layer_sizes: Multiple architectures
   - activation: ['relu', 'tanh']
   - solver: ['adam', 'sgd']
   - alpha: [0.0001, 0.001, 0.01, 0.1]

### Ensemble Methods

1. **Voting Ensemble**
   - Methods: 'soft', 'hard'
   - Base models: RF, LR, XGB, LGB, MLP

2. **Stacking Ensemble**
   - Meta-learners: Logistic, Random Forest, MLP
   - Cross-validation: 5-fold

3. **Weighted Ensemble**
   - Custom weight optimization
   - Threshold optimization

## Resource Requirements

- **CPU**: 32 cores per job
- **Memory**: 128 GB RAM
- **Wall time**: 24-48 hours
- **Storage**: 1-2 TB for results and models

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Check job details
scontrol show job <job_id>

# Monitor logs
tail -f logs/hyperparameter_tuning_<job_id>.out
tail -f logs/hyperparameter_tuning_<job_id>.err
```

## Results

Results will be saved in:
- `results/`: JSON files with optimization results
- `models/`: Trained model files (.joblib)
- `logs/`: Detailed logs
- `plots/`: Visualization plots

## Expected Performance Improvements

Based on the notebook analysis, we expect:

1. **Feature Selection**: F1 improvement from ~0.68 to ~0.72
2. **Hyperparameter Tuning**: Additional 2-5% improvement
3. **Ensemble Methods**: Additional 1-3% improvement
4. **Threshold Optimization**: 3-5% improvement

## Troubleshooting

### Common Issues

1. **Module not found**: Install missing packages in requirements.txt
2. **Memory issues**: Reduce batch size or number of parallel jobs
3. **Time limit exceeded**: Increase wall time in SLURM script
4. **Data file not found**: Check data file path in config

### Debugging

```bash
# Test locally first
python hpc_hyperparameter_tuning.py --config hpc_config.yaml

# Check SLURM logs
cat logs/hyperparameter_tuning_<job_id>.err
```

## Data Privacy

The `.gitignore` file ensures that:
- Data files are not uploaded to version control
- Results and models are kept local
- Sensitive information is protected

## Next Steps

After running the optimization:

1. **Analyze Results**: Review the JSON result files
2. **Select Best Model**: Choose the best performing model
3. **Deploy Model**: Use the saved model for predictions
4. **Further Optimization**: Run additional experiments if needed

## Contact

For issues or questions about the HPC optimization system, refer to the main project documentation. 