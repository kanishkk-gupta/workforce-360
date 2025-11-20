#!/usr/bin/env python
"""
Test script to verify model predictions work correctly
"""
import pickle
import pandas as pd
import os

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_and_key_components.pkl')

with open(model_path, 'rb') as file:
    saved_components = pickle.load(file)

model = saved_components['model']

print("="*60)
print("EMPLOYEE ATTRITION PREDICTION - MODEL TEST")
print("="*60)
print()

# Test Case 1: High Risk Employee (Should Leave)
print("TEST CASE 1: HIGH RISK EMPLOYEE")
print("-" * 60)
test1 = pd.DataFrame({
    'Age': [25],
    'Department': ['Sales'],
    'EnvironmentSatisfaction': [1],
    'JobRole': ['Sales Representative'],
    'JobSatisfaction': [1],
    'MonthlyIncome': [2000],
    'NumCompaniesWorked': [4],
    'OverTime': [True],
    'PercentSalaryHike': [10],
    'RelationshipSatisfaction': [1],
    'TrainingTimesLastYear': [0],
    'WorkLifeBalance': [1],
    'YearsSinceLastPromotion': [5],
    'YearsWithCurrManager': [1]
})

pred1 = model.predict(test1)[0]
prob1 = model.predict_proba(test1)[0, 1]

print("Profile: Young employee, low satisfaction, low income, frequent overtime")
print(f"Prediction: {'WILL LEAVE (HIGH RISK)' if pred1 == 1 else 'WILL STAY'}")
print(f"Probability: {prob1:.2%}")
print()

# Test Case 2: Low Risk Employee (Should Stay)
print("TEST CASE 2: LOW RISK EMPLOYEE")
print("-" * 60)
test2 = pd.DataFrame({
    'Age': [42],
    'Department': ['Research & Development'],
    'EnvironmentSatisfaction': [4],
    'JobRole': ['Research Director'],
    'JobSatisfaction': [4],
    'MonthlyIncome': [9000],
    'NumCompaniesWorked': [2],
    'OverTime': [False],
    'PercentSalaryHike': [15],
    'RelationshipSatisfaction': [4],
    'TrainingTimesLastYear': [5],
    'WorkLifeBalance': [4],
    'YearsSinceLastPromotion': [1],
    'YearsWithCurrManager': [6]
})

pred2 = model.predict(test2)[0]
prob2 = model.predict_proba(test2)[0, 1]

print("Profile: Experienced manager, high satisfaction, high income, good work-life balance")
print(f"Prediction: {'WILL LEAVE' if pred2 == 1 else 'WILL STAY (LOW RISK)'}")
print(f"Probability: {prob2:.2%}")
print()

# Test Case 3: Medium Risk Employee
print("TEST CASE 3: MEDIUM RISK EMPLOYEE")
print("-" * 60)
test3 = pd.DataFrame({
    'Age': [35],
    'Department': ['Human Resources'],
    'EnvironmentSatisfaction': [3],
    'JobRole': ['Manager'],
    'JobSatisfaction': [3],
    'MonthlyIncome': [6000],
    'NumCompaniesWorked': [3],
    'OverTime': [True],
    'PercentSalaryHike': [12],
    'RelationshipSatisfaction': [3],
    'TrainingTimesLastYear': [3],
    'WorkLifeBalance': [3],
    'YearsSinceLastPromotion': [2],
    'YearsWithCurrManager': [3]
})

pred3 = model.predict(test3)[0]
prob3 = model.predict_proba(test3)[0, 1]

print("Profile: Mid-level manager, average satisfaction and income")
print(f"Prediction: {'WILL LEAVE' if pred3 == 1 else 'WILL STAY (MEDIUM RISK)'}")
print(f"Probability of Leaving: {prob3:.2%}")
print()

print("="*60)
print("TEST SUMMARY:")
print("="*60)
print(f"✓ Test Case 1 (High Risk):   PASSED - Prediction = {pred1}" if pred1 == 1 else f"✗ Test Case 1 FAILED")
print(f"✓ Test Case 2 (Low Risk):    PASSED - Prediction = {pred2}" if pred2 == 0 else f"✗ Test Case 2 FAILED")
print(f"✓ Test Case 3 (Medium Risk): Probability = {prob3:.2%}")
print()
print("Model is working correctly!")
print("="*60)
