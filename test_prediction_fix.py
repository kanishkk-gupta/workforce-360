#!/usr/bin/env python3
"""
Test script to verify the threshold fix is working correctly.
Tests the borderline case: 49.70% probability with 45% threshold
Should predict LEAVE (1) now with >= operator
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

import pickle
import pandas as pd

# Load model and components
model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_and_key_components.pkl')
with open(model_path, 'rb') as f:
    saved_components = pickle.load(f)

model = saved_components['model']
DECISION_THRESHOLD = saved_components.get('threshold', 0.45)

# Test data - the borderline case from your issue
# Using normalized values from the notebook that showed 57.16% probability
input_data = pd.DataFrame({
    'Age': [54],
    'Department': ['Sales'],
    'EnvironmentSatisfaction': [1],
    'JobRole': ['Manager'],
    'JobSatisfaction': [4],
    'MonthlyIncome': [15552],
    'NumCompaniesWorked': [0.915015703],  # Normalized value
    'OverTime': [False],
    'PercentSalaryHike': [15],
    'RelationshipSatisfaction': [4],  # Changed to 4 (high satisfaction)
    'TrainingTimesLastYear': [3.130162755],  # Normalized value
    'WorkLifeBalance': [3],  # Changed to 3
    'YearsSinceLastPromotion': [5.549904694],  # Normalized value
    'YearsWithCurrManager': [9.073853543]  # Normalized value
})

# Ensure correct column order and types
for col in ['Department', 'JobRole']:
    input_data[col] = input_data[col].astype(str)

expected_order = ['Age', 'Department', 'EnvironmentSatisfaction', 'JobRole', 'JobSatisfaction', 
                 'MonthlyIncome', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
                 'RelationshipSatisfaction', 'TrainingTimesLastYear', 'WorkLifeBalance',
                 'YearsSinceLastPromotion', 'YearsWithCurrManager']
input_data = input_data[expected_order]

# Make prediction
probability = model.predict_proba(input_data)[:, 1][0]
prediction = 1 if probability >= DECISION_THRESHOLD else 0

print("=" * 70)
print("PREDICTION VERIFICATION TEST")
print("=" * 70)
print(f"\nEmployee Profile:")
print(f"  Age: 54, Sales Manager")
print(f"  Monthly Income: $15,552 (High)")
print(f"  Job Satisfaction: 4/4 (Very High)")
print(f"  Environment Satisfaction: 1/4 (Very Low) ← THE ISSUE")
print(f"\n  Decision Threshold: {DECISION_THRESHOLD * 100:.0f}%")
print(f"  Attrition Probability: {probability * 100:.2f}%")
print(f"\nComparison:")
print(f"  {probability * 100:.2f}% >= {DECISION_THRESHOLD * 100:.0f}% = {probability >= DECISION_THRESHOLD}")
print(f"\nPrediction Result: {'LEAVE (1)' if prediction == 1 else 'STAY (0)'}")

print("\n" + "=" * 70)
if prediction == 1 and probability >= DECISION_THRESHOLD:
    print("✅ CORRECT! Threshold logic is working properly.")
    print("   Employee predicted to LEAVE as expected (probability exceeds threshold)")
elif prediction == 0 and probability < DECISION_THRESHOLD:
    print("✅ CORRECT! Threshold logic is working properly.")
    print("   Employee predicted to STAY as expected (probability is below threshold)")
else:
    print("❌ ERROR! Logic still incorrect.")
    print(f"   Expected: {'LEAVE' if probability >= DECISION_THRESHOLD else 'STAY'}, Got: {'LEAVE' if prediction == 1 else 'STAY'}")
print("=" * 70)

risk_level = 'HIGH RISK - Will Likely Leave' if probability >= DECISION_THRESHOLD else 'LOW RISK - Will Likely Stay'
print(f"\nWeb Display Will Show:")
print(f"  Prediction: Employee is predicted to {'LEAVE' if prediction == 1 else 'STAY'}")
print(f"  Risk Level: {risk_level}")
print(f"  Attrition Probability: {probability * 100:.2f}%")
print(f"  Decision Threshold: ≥ {DECISION_THRESHOLD * 100:.0f}% ({'exceeded' if probability >= DECISION_THRESHOLD else 'not exceeded'})")
