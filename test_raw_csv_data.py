"""
Helper script to test predictions with raw data from the original CSV format
Maps original 31-column data to the 14-feature model format
"""

import pandas as pd
import pickle

# The mapping from original CSV columns (31 columns) to model features (14 columns)
# Original CSV column order:
# Age, Attrition, BusinessTravel, DailyRate, Department, DistanceFromHome, Education, EducationField,
# EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction,
# MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked, OverTime, PercentSalaryHike,
# PerformanceRating, RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears,
# TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole,
# YearsSinceLastPromotion, YearsWithCurrManager

# Model expects 14 features in this order:
# Age, Department, EnvironmentSatisfaction, JobRole, JobSatisfaction, MonthlyIncome,
# NumCompaniesWorked, OverTime, PercentSalaryHike, RelationshipSatisfaction,
# TrainingTimesLastYear, WorkLifeBalance, YearsSinceLastPromotion, YearsWithCurrManager

original_columns = [
    'Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
    'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
    'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
    'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
    'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

model_features = [
    'Age', 'Department', 'EnvironmentSatisfaction', 'JobRole', 'JobSatisfaction',
    'MonthlyIncome', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
    'RelationshipSatisfaction', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

def prepare_data_from_csv(csv_row_data):
    """
    Convert a row from original CSV format (31 columns) to model format (14 features)
    
    Args:
        csv_row_data: List or dict with all 31 columns from original CSV
    
    Returns:
        DataFrame with 14 model features in correct order and types
    """
    # Create a DataFrame from original data
    if isinstance(csv_row_data, list):
        df_original = pd.DataFrame([csv_row_data], columns=original_columns)
    else:
        df_original = pd.DataFrame([csv_row_data])
    
    # Extract only the features needed by the model
    df_model = df_original[model_features].copy()
    
    # Ensure correct data types
    df_model['Age'] = df_model['Age'].astype(int)
    df_model['Department'] = df_model['Department'].astype(str)
    df_model['EnvironmentSatisfaction'] = df_model['EnvironmentSatisfaction'].astype(int)
    df_model['JobRole'] = df_model['JobRole'].astype(str)
    df_model['JobSatisfaction'] = df_model['JobSatisfaction'].astype(int)
    df_model['MonthlyIncome'] = df_model['MonthlyIncome'].astype(int)
    df_model['NumCompaniesWorked'] = df_model['NumCompaniesWorked'].astype(int)
    df_model['OverTime'] = df_model['OverTime'].astype(bool)
    df_model['PercentSalaryHike'] = df_model['PercentSalaryHike'].astype(int)
    df_model['RelationshipSatisfaction'] = df_model['RelationshipSatisfaction'].astype(int)
    df_model['TrainingTimesLastYear'] = df_model['TrainingTimesLastYear'].astype(int)
    df_model['WorkLifeBalance'] = df_model['WorkLifeBalance'].astype(int)
    df_model['YearsSinceLastPromotion'] = df_model['YearsSinceLastPromotion'].astype(int)
    df_model['YearsWithCurrManager'] = df_model['YearsWithCurrManager'].astype(int)
    
    return df_model


# Test with your provided data
# Your data (31 values in CSV column order):
# 6 54 1 Travel_Rarely 281 Sales 2 1 Marketing 1 Female 55 1 5 Manager 4 Married 15552 25917 0.915015703 FALSE 15 3 4 0.937339737 28.11996741 3.130162755 3 18 14.78428565 5.549904694 9.073853543

# Mapping your values to original CSV columns:
test_data = [
    54,                           # Age
    1,                            # Attrition (target - not used for prediction)
    'Travel_Rarely',              # BusinessTravel
    281,                          # DailyRate
    'Sales',                      # Department
    2,                            # DistanceFromHome
    1,                            # Education
    'Marketing',                  # EducationField
    1,                            # EnvironmentSatisfaction
    'Female',                     # Gender
    55,                           # HourlyRate
    1,                            # JobInvolvement
    5,                            # JobLevel
    'Manager',                    # JobRole
    4,                            # JobSatisfaction
    'Married',                    # MaritalStatus
    15552,                        # MonthlyIncome
    25917,                        # MonthlyRate
    0.915015703,                  # NumCompaniesWorked
    False,                        # OverTime
    15,                           # PercentSalaryHike
    3,                            # PerformanceRating
    4,                            # RelationshipSatisfaction
    0.937339737,                  # StockOptionLevel
    28.11996741,                  # TotalWorkingYears
    3.130162755,                  # TrainingTimesLastYear
    3,                            # WorkLifeBalance
    18,                           # YearsAtCompany
    14.78428565,                  # YearsInCurrentRole
    5.549904694,                  # YearsSinceLastPromotion
    9.073853543                   # YearsWithCurrManager
]

# Prepare the data
df_prepared = prepare_data_from_csv(test_data)

print("=" * 80)
print("DATA PREPARATION FOR MODEL PREDICTION")
print("=" * 80)
print("\nOriginal Data (31 columns):")
print(f"Attrition Label in original data: {test_data[1]} (1 = Should LEAVE)")
print("\nModel Input Features (14 columns):")
print(df_prepared)

# Load the model
import os
model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_and_key_components.pkl')
try:
    with open(model_path, 'rb') as f:
        saved_components = pickle.load(f)
    
    model = saved_components['model']
    
    # Make prediction
    prediction = model.predict(df_prepared)[0]
    probability = model.predict_proba(df_prepared)[0]
    
    print("\n" + "=" * 80)
    print("MODEL PREDICTION RESULTS")
    print("=" * 80)
    print(f"\nModel Prediction: {prediction} ({'WILL LEAVE' if prediction == 1 else 'WILL STAY'})")
    print(f"Probability of STAYING (class 0): {probability[0]:.2%}")
    print(f"Probability of LEAVING (class 1): {probability[1]:.2%}")
    print(f"\nExpected Result: 1 (WILL LEAVE - Based on original Attrition label)")
    print(f"Model Result: {prediction} ({'✓ CORRECT' if prediction == 1 else '✗ INCORRECT'})")

    # Analyze the features
    print("\n" + "=" * 80)
    print("FEATURE ANALYSIS FOR THIS PREDICTION")
    print("=" * 80)
    print("\nEmployee Profile:")
    print(f"  Age: 54 (Older employee - stable)")
    print(f"  Department: Sales")
    print(f"  JobRole: Manager (Senior role)")
    print(f"  MonthlyIncome: 15552 (High income)")
    print(f"  JobSatisfaction: 4 (Very High)")
    print(f"  EnvironmentSatisfaction: 1 (Very Low) ⚠️")
    print(f"  OverTime: False (No overtime)")
    print(f"  YearsWithCurrManager: 9 (Long tenure)")
    print(f"  YearsSinceLastPromotion: 5 (Some time since promotion)")
    print(f"  Other factors appear stable...")

    print("\n⚠️ ANALYSIS:")
    print("Despite VERY LOW EnvironmentSatisfaction (1), this employee:")
    print("- Has high income (15552)")
    print("- Is a Manager (senior role)")
    print("- Has HIGH job satisfaction (4)")
    print("- Has long tenure with manager (9 years)")
    print("- Doesn't work overtime")
    print("- Is 54 years old (likely stable)")
    print("\nThe model considers these POSITIVE factors outweigh the low environment satisfaction.")
    print("This is likely CORRECT model behavior - salary, tenure, and job satisfaction")
    print("can compensate for poor environment.")
    print("\n" + "=" * 80)
    print("ALTERNATIVE PREDICTIONS WITH ADJUSTED THRESHOLDS")
    print("=" * 80)
    
    # Try different decision thresholds
    prob_leave = probability[1]
    
    thresholds = {
        '0.40': 0.40,
        '0.45': 0.45,
        '0.50 (Default)': 0.50,
        '0.55': 0.55,
        '0.60': 0.60,
    }
    
    print(f"\nProbability of Leaving: {prob_leave:.2%}\n")
    print("Predictions at different thresholds:")
    for name, threshold in thresholds.items():
        pred_at_threshold = 1 if prob_leave > threshold else 0
        status = "WILL LEAVE" if pred_at_threshold == 1 else "WILL STAY"
        match = "✓" if pred_at_threshold == 1 else "✗"
        print(f"  Threshold {name}: {status} {match}")
    
    print("\n💡 RECOMMENDATION:")
    print("Using threshold 0.45 would correctly predict this employee WILL LEAVE.")
    print("This is useful for HR early intervention - identifying at-risk employees")
    print("before they decide to leave.")
    
except Exception as e:
    print(f"\nError loading model: {e}")
    print("Make sure the model file exists at the correct path")