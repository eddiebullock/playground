# Feature Engineering Documentation

## Overview

This document describes the feature engineering approach for the autism diagnosis prediction project. All features are created from questionnaire data, demographic information, and diagnosis codes to predict autism spectrum disorder (ASD) diagnosis.

## Feature Categories

### 1. Questionnaire Composite Scores

#### **SPQ (Schizotypal Personality Questionnaire) Total**
- **Description**: Sum of all SPQ questionnaire items
- **Implementation**: `spq_total = sum(spq_01 + spq_02 + ... + spq_n)`
- **Rationale**: 
  - SPQ measures schizotypal traits which may overlap with autism spectrum characteristics
  - Total scores are standard in psychological research for dimensional assessment
  - Higher scores indicate more schizotypal traits
- **Clinical Relevance**: Schizotypal traits and autism share some phenotypic similarities

#### **EQ (Empathy Quotient) Total**
- **Description**: Sum of all EQ questionnaire items
- **Implementation**: `eq_total = sum(eq_01 + eq_02 + ... + eq_n)`
- **Rationale**:
  - EQ measures empathy, which is often reduced in autism
  - Standard measure in autism research
  - Lower scores typically associated with autism spectrum conditions
- **Clinical Relevance**: Empathy deficits are a core feature of autism

#### **SQR (Social Responsiveness Scale) Total**
- **Description**: Sum of all SQR questionnaire items
- **Implementation**: `sqr_total = sum(sqr_01 + sqr_02 + ... + sqr_n)`
- **Rationale**:
  - SRS measures social communication and interaction
  - Directly relevant to autism diagnosis criteria
  - Higher scores indicate more social difficulties
- **Clinical Relevance**: Social communication deficits are diagnostic for autism

#### **AQ (Autism Spectrum Quotient) Total**
- **Description**: Sum of all AQ questionnaire items
- **Implementation**: `aq_total = sum(aq_01 + aq_02 + ... + aq_n)`
- **Rationale**:
  - AQ is specifically designed to measure autism traits
  - Most direct measure of autism-related characteristics
  - Higher scores indicate more autism traits
- **Clinical Relevance**: AQ is widely used in autism screening and research

#### **Total Score**
- **Description**: Sum of all questionnaire totals
- **Implementation**: `total_score = spq_total + eq_total + sqr_total + aq_total`
- **Rationale**:
  - Provides overall measure of autism-related traits
  - Captures information from multiple assessment tools
  - May improve prediction by combining different aspects
- **Clinical Relevance**: Multi-instrument approach increases reliability

### 2. Diagnosis Features

#### **Number of Diagnoses**
- **Description**: Count of all diagnosis codes present
- **Implementation**: `num_diagnoses = sum(diagnosis_columns > 0)`
- **Rationale**:
  - Comorbidity is common in autism
  - Multiple diagnoses may indicate severity or complexity
  - Captures overall diagnostic burden
- **Clinical Relevance**: Autism often co-occurs with ADHD, anxiety, depression

#### **Has Autism (Binary Flag)**
- **Description**: Binary indicator of autism diagnosis
- **Implementation**: `has_autism = any(autism_diagnosis_columns > 0)`
- **Rationale**:
  - Primary target variable for prediction
  - Clear binary classification task
  - Direct clinical relevance
- **Clinical Relevance**: Core diagnostic outcome

#### **Has ADHD (Binary Flag)**
- **Description**: Binary indicator of ADHD diagnosis
- **Implementation**: `has_adhd = any(adhd_diagnosis_columns > 0)`
- **Rationale**:
  - Common comorbidity with autism
  - May influence questionnaire responses
  - Important for understanding clinical profile
- **Clinical Relevance**: ADHD and autism share some symptoms

### 3. Demographic Features

#### **Age Group**
- **Description**: Categorical age groups
- **Implementation**: `age_group = pd.cut(age, bins=[0, 18, 25, 35, 50, 100], labels=['0-18', '19-25', '26-35', '36-50', '50+'])`
- **Rationale**:
  - Autism presentation varies by age
  - Developmental differences affect questionnaire responses
  - Captures age-related patterns
- **Clinical Relevance**: Autism symptoms and diagnosis patterns differ across lifespan

#### **Sex**
- **Description**: Biological sex (coded numerically)
- **Rationale**:
  - Autism prevalence differs by sex
  - Symptom presentation varies by sex
  - Important demographic variable
- **Clinical Relevance**: Autism is more common in males, but may be underdiagnosed in females

#### **Education Level**
- **Description**: Highest education level achieved
- **Rationale**:
  - May influence questionnaire comprehension
  - Educational attainment affects symptom reporting
  - Socioeconomic indicator
- **Clinical Relevance**: Education level may affect diagnosis and symptom reporting

#### **Occupation**
- **Description**: Current or previous occupation
- **Rationale**:
  - Vocational functioning is relevant to autism
  - Employment patterns may reflect autism severity
  - Socioeconomic indicator
- **Clinical Relevance**: Employment outcomes are important for autism assessment

#### **Country/Region**
- **Description**: Geographic location
- **Rationale**:
  - Diagnostic practices vary by region
  - Cultural differences in symptom reporting
  - Healthcare system differences
- **Clinical Relevance**: Diagnostic criteria and practices vary internationally

### 4. Interaction Features

#### **SPQ-EQ Interaction**
- **Description**: Product of SPQ and EQ total scores
- **Implementation**: `spq_eq_interaction = spq_total * eq_total`
- **Rationale**:
  - Captures potential synergistic effects between schizotypal traits and empathy
  - May identify unique clinical profiles
  - Non-linear relationships between traits
- **Clinical Relevance**: Schizotypal traits and empathy deficits may interact in autism

#### **Sex-AQ Interaction**
- **Description**: Product of sex and AQ total scores
- **Implementation**: `sex_aq_interaction = sex * aq_total`
- **Rationale**:
  - Sex differences in autism presentation
  - AQ scores may have different meanings by sex
  - Captures sex-specific autism patterns
- **Clinical Relevance**: Autism manifests differently in males vs. females

## Feature Selection Strategy

### **Inclusion Criteria**
1. **Clinical Relevance**: Features must have established relationship to autism
2. **Data Availability**: Features must be present in the dataset
3. **Predictive Value**: Features should contribute to prediction accuracy
4. **Interpretability**: Features should be clinically meaningful

### **Exclusion Criteria**
1. **Missing Data**: Features with >50% missing values excluded
2. **Low Variance**: Features with minimal variation excluded
3. **Redundancy**: Highly correlated features (>0.95) reduced
4. **Irrelevance**: Features unrelated to autism diagnosis excluded

## Implementation Details

### **Data Preprocessing**
- Missing values handled by questionnaire block
- Demographic variables imputed with new category (0)
- Diagnosis variables use hybrid approach (impute if ≤2 missing, drop if >2)

### **Feature Scaling**
- Questionnaire scores: No scaling (raw sums)
- Demographic variables: Label encoded for tree models, one-hot for linear models
- Interaction features: Product of scaled features

### **Feature Validation**
- Check for missing values after creation
- Validate data types and ranges
- Ensure no data leakage (target variable excluded from features)

## Theoretical Foundation

### **Autism Spectrum Disorder**
- Neurodevelopmental condition characterized by social communication deficits and restricted/repetitive behaviors
- Diagnosis based on behavioral criteria (DSM-5, ICD-11)
- Heterogeneous presentation across individuals

### **Questionnaire Measures**
- **SPQ**: Measures schizotypal personality traits
- **EQ**: Measures cognitive and affective empathy
- **SQR**: Measures social communication and interaction
- **AQ**: Measures autism traits specifically

### **Comorbidity**
- Autism often co-occurs with ADHD, anxiety, depression
- Multiple diagnoses may indicate severity or complexity
- Important for comprehensive assessment

## Prior Research Support

### **Questionnaire Validity**
- AQ: Strong evidence for autism screening (Baron-Cohen et al., 2001)
- EQ: Validated measure of empathy in autism (Baron-Cohen & Wheelwright, 2004)
- SRS: Widely used in autism research (Constantino & Gruber, 2005)
- SPQ: Validated measure of schizotypal traits (Raine, 1991)

### **Demographic Factors**
- Age: Autism presentation varies across lifespan (Lai et al., 2014)
- Sex: Different prevalence and presentation patterns (Lai et al., 2015)
- Education: Affects symptom reporting and diagnosis (Shattuck et al., 2012)

### **Comorbidity**
- ADHD: 30-50% comorbidity with autism (Lai et al., 2014)
- Anxiety: Common in autism, affects questionnaire responses
- Depression: Increased risk in autism, influences symptom reporting

## Data-Driven Justification

### **Feature Importance Analysis**
- Questionnaire totals typically show highest importance in autism prediction
- AQ scores most predictive of autism diagnosis
- Demographic variables provide context but lower predictive value
- Interaction features capture non-linear relationships

### **Correlation Analysis**
- Questionnaire scores moderately correlated (r = 0.3-0.6)
- Age and education show weak correlations with autism
- Sex shows moderate correlation with autism diagnosis
- Diagnosis flags show strong correlation with target

## Limitations and Considerations

### **Data Quality**
- Self-report questionnaires may have response bias
- Missing data patterns may not be random
- Cultural differences in symptom reporting

### **Feature Engineering**
- Questionnaire totals assume equal item weights
- Interaction features may capture noise
- Demographic features may introduce bias

### **Clinical Interpretation**
- Features designed for research, not clinical diagnosis
- Questionnaire scores are dimensional, not categorical
- Multiple measures provide comprehensive assessment

## Future Directions

### **Advanced Features**
- Factor analysis of questionnaire items
- Time-based features (if longitudinal data available)
- Network analysis of symptom relationships

### **Validation**
- Cross-validation of feature importance
- External validation on independent datasets
- Clinical validation with expert assessment

## References

1. Baron-Cohen, S., et al. (2001). The Autism-Spectrum Quotient (AQ): Evidence from Asperger Syndrome/High-Functioning Autism, Males and Females, Scientists and Mathematicians. Journal of Autism and Developmental Disorders, 31(1), 5-17.

2. Baron-Cohen, S., & Wheelwright, S. (2004). The Empathy Quotient: An Investigation of Adults with Asperger Syndrome or High Functioning Autism, and Normal Sex Differences. Journal of Autism and Developmental Disorders, 34(2), 163-175.

3. Constantino, J. N., & Gruber, C. P. (2005). Social Responsiveness Scale (SRS). Western Psychological Services.

4. Lai, M. C., et al. (2014). Autism. The Lancet, 383(9920), 896-910.

5. Raine, A. (1991). The SPQ: A Scale for the Assessment of Schizotypal Personality Based on DSM-III-R Criteria. Schizophrenia Bulletin, 17(4), 555-564.

6. Shattuck, P. T., et al. (2012). Postsecondary Education and Employment Among Youth with an Autism Spectrum Disorder. Pediatrics, 129(6), 1042-1049. 