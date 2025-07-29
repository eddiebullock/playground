# Deep Learning Templates for Autism Classification

This directory contains templates for implementing deep learning approaches to autism classification using questionnaire data.

## 📋 Template Overview

### 1. **Baseline Neural Network** (`deep_learning_baseline.py`)
- **Purpose**: Simple feedforward neural network baseline
- **Expected Performance**: F1 ~0.65-0.75
- **Use Case**: Establish baseline vs traditional ML methods

### 2. **Domain Adaptation** (`domain_adaptation_advanced.py`)
- **Purpose**: Domain-Adversarial Neural Network (DANN) for cross-dataset generalization
- **Expected Performance**: F1 ~0.75-0.85
- **Use Case**: Improve generalization from C4 to YBT datasets

### 3. **Hyperparameter Optimization** (`hyperparameter_optimization.py`)
- **Purpose**: Automated hyperparameter tuning using Optuna
- **Expected Performance**: F1 ~0.80-0.90
- **Use Case**: Find optimal model architecture and training parameters

## 🚀 How to Use These Templates

### **Step 1: Create a New Notebook**
1. Open Cursor/Jupyter Lab
2. Create a new notebook: `notebooks/deep_learning_experiment.ipynb`
3. Select the **"TensorFlow Environment"** kernel

### **Step 2: Copy Template Code**
1. Open the template file you want to use
2. Copy each cell section (marked with `# CELL X:`)
3. Paste into your notebook
4. Run cells sequentially

### **Step 3: Customize for Your Data**
- **Update file paths** to match your data locations
- **Adjust hyperparameters** based on your needs
- **Modify model architecture** if needed

## 📊 Expected Results

### **Performance Targets:**
| Template | F1 Score | ROC-AUC | Use Case |
|----------|----------|---------|----------|
| Baseline | 0.65-0.75 | 0.70-0.80 | Initial comparison |
| Domain Adaptation | 0.75-0.85 | 0.80-0.90 | Cross-dataset generalization |
| Optimized | 0.80-0.90 | 0.85-0.95 | Best performance |

## 🔧 Technical Requirements

### **Environment Setup:**
```bash
# Activate your deep learning environment
source tf_venv/bin/activate

# Verify packages are installed
python -c "import torch, lightning, wandb, optuna; print('All packages ready!')"
```

### **Data Requirements:**
- C4 balanced standardized dataset
- YBT balanced standardized dataset
- Common features between datasets

## 📈 Experiment Tracking

### **WandB Integration:**
- All templates include WandB logging
- Track experiments, hyperparameters, and metrics
- Compare different approaches easily

### **Model Saving:**
- Best models are automatically saved
- Checkpoint files for resuming training
- Optimization results saved separately

## 🎯 Recommended Workflow

### **Phase 1: Baseline (Week 1)**
1. Run `deep_learning_baseline.py` template
2. Compare with your current Random Forest results
3. Document baseline performance

### **Phase 2: Domain Adaptation (Week 2)**
1. Run `domain_adaptation_advanced.py` template
2. Compare with baseline results
3. Analyze domain adaptation effectiveness

### **Phase 3: Optimization (Week 3)**
1. Run `hyperparameter_optimization.py` template
2. Let Optuna find optimal parameters
3. Train final optimized model

## 🔍 Key Features

### **Automatic Features:**
- ✅ Early stopping to prevent overfitting
- ✅ Learning rate scheduling
- ✅ Model checkpointing
- ✅ Cross-dataset evaluation
- ✅ Threshold optimization
- ✅ Performance comparison

### **Advanced Features:**
- ✅ Domain adversarial training
- ✅ Gradient reversal layers
- ✅ Hyperparameter optimization
- ✅ Experiment tracking
- ✅ Reproducible results

## 📝 Usage Example

```python
# In your notebook, copy the template cells:

# CELL 1: SETUP AND DATA LOADING
# (Copy from template)

# CELL 2: DATA PREPARATION  
# (Copy from template)

# CELL 3: NEURAL NETWORK MODEL
# (Copy from template)

# ... continue with all cells
```

## 🎯 Next Steps

1. **Start with Baseline**: Use `deep_learning_baseline.py` to establish performance baseline
2. **Compare Results**: Compare with your current Random Forest performance
3. **Scale Up**: Move to domain adaptation if baseline shows promise
4. **Optimize**: Use hyperparameter optimization for best performance
5. **HPC Deployment**: Scale to HPC for large-scale experiments

## 📊 Monitoring Progress

### **Key Metrics to Track:**
- **F1 Score**: Primary performance metric
- **ROC-AUC**: Overall model discrimination
- **Training Loss**: Convergence monitoring
- **Validation Loss**: Overfitting detection

### **Success Criteria:**
- **Baseline**: F1 > 0.70 (vs current 0.619)
- **Domain Adaptation**: F1 > 0.80
- **Optimized**: F1 > 0.85

## 🚨 Troubleshooting

### **Common Issues:**
1. **CUDA/GPU errors**: Set `accelerator='cpu'` in trainer
2. **Memory issues**: Reduce batch size
3. **Convergence problems**: Adjust learning rate
4. **Overfitting**: Increase dropout or reduce model complexity

### **Getting Help:**
- Check WandB logs for detailed metrics
- Monitor training progress in real-time
- Use early stopping to prevent overfitting

---

**Ready to start your deep learning journey! 🚀** 