# Copy and paste this code into your notebook to see all features being used by the model

print("="*60)
print("ALL FEATURES USED BY THE MODEL")
print("="*60)
print(f"Total number of features: {len(x.columns)}")
print()

# Group features by category
demographic_features = ['age', 'sex_2.0', 'sex_3.0', 'sex_4.0', 'sex_unknown', 'is_stem_occupation']
total_scores = ['spq_total', 'eq_total', 'sqr_total', 'aq_total', 'd_score']
engineered_features = ['log_aq_total', 'sqrt_age', 'aq_eq_interaction', 'sqp_aq_interaction', 
                      'age_x_eq', 'age_x_aq', 'aq_spq_ratio', 'eq_sqr_ratio', 'high_aq',
                      'age_group_19-30', 'age_group_31-45', 'age_group_46-60', 'age_group_61+']

# Individual questionnaire items
spq_items = [col for col in x.columns if col.startswith('spq_') and col != 'spq_total']
eq_items = [col for col in x.columns if col.startswith('eq_') and col != 'eq_total']
sqr_items = [col for col in x.columns if col.startswith('sqr_') and col != 'sqr_total']
aq_items = [col for col in x.columns if col.startswith('aq_') and col != 'aq_total']

print("1. DEMOGRAPHIC FEATURES:")
for feat in demographic_features:
    if feat in x.columns:
        print(f"   - {feat}")
print()

print("2. TOTAL SCORES:")
for feat in total_scores:
    if feat in x.columns:
        print(f"   - {feat}")
print()

print("3. INDIVIDUAL QUESTIONNAIRE ITEMS:")
print("   SPQ items (10):", ", ".join(spq_items))
print("   EQ items (10):", ", ".join(eq_items))
print("   SQR items (10):", ", ".join(sqr_items))
print("   AQ items (10):", ", ".join(aq_items))
print()

print("4. ENGINEERED FEATURES:")
for feat in engineered_features:
    if feat in x.columns:
        print(f"   - {feat}")
print()

print("5. COMPLETE FEATURE LIST (alphabetical):")
for i, feat in enumerate(sorted(x.columns), 1):
    print(f"   {i:2d}. {feat}")
print()

print("="*60)
print("FEATURE SUMMARY")
print("="*60)
print(f"Demographic features: {len([f for f in demographic_features if f in x.columns])}")
print(f"Total scores: {len([f for f in total_scores if f in x.columns])}")
print(f"Individual SPQ items: {len(spq_items)}")
print(f"Individual EQ items: {len(eq_items)}")
print(f"Individual SQR items: {len(sqr_items)}")
print(f"Individual AQ items: {len(aq_items)}")
print(f"Engineered features: {len([f for f in engineered_features if f in x.columns])}")
print(f"TOTAL: {len(x.columns)} features")
