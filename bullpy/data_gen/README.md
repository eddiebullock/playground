# Synthetic Autism Questionnaire Data Generator

This project generates synthetic autism questionnaire data to address class imbalance in autism prediction research. The goal is to create realistic synthetic autism cases that maintain the statistical properties and response patterns of real AQ, SQ, EQ, and SPQ questionnaire data.

## Problem Statement

- **Dataset**: 750k participants with questionnaire data (AQ, SQ, EQ, SPQ)
- **Class Imbalance**: Only 44k autism cases (5.9% of dataset)
- **Challenge**: Low precision/recall for autism class due to imbalance
- **Solution**: Generate synthetic autism cases to balance the dataset

## Features

- Generates synthetic autism questionnaire responses
- Maintains realistic correlations between questionnaire items
- Preserves demographic distributions
- Creates balanced datasets for improved ML model training
- Supports multiple generation strategies (SMOTE, GANs, VAEs)

## Getting Started

1. **Set up the virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the data generator:**
   ```bash
   python generate_autism_data.py
   ```

## Project Structure

- `generate_autism_data.py` - Main synthetic data generation script
- `questionnaire_analysis.py` - Analyze real data patterns
- `synthetic_strategies.py` - Different generation approaches
- `validation.py` - Validate synthetic data quality

## Requirements

See `requirements.txt` for dependencies. 

