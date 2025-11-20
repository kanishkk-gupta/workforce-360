# Employee Attrition Prediction Dashboard

A modern web-based dashboard for predicting employee attrition using a trained regularized CatBoost machine learning model.

## Overview

This application uses a machine learning model trained on employee data to predict whether an employee is likely to leave the organization. The model has been optimized to minimize overfitting and provides reliable predictions with 89% accuracy.

## Model Performance

- **Training Accuracy:** 89.50%
- **Validation Accuracy:** 88.91%
- **Overfitting Gap:** 0.59% ✓ (excellent generalization)
- **Top Predictive Features:** OverTime (16.4%), YearsWithCurrManager (10.5%), MonthlyIncome (10.0%)

## Features

- ✓ Interactive form with all employee attributes
- ✓ Real-time prediction with probability scores
- ✓ CSV file upload for batch predictions
- ✓ Beautiful, responsive UI design
- ✓ Model explanations using SHAP analysis
- ✓ Input validation for data accuracy

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the trained model exists at `model/model_and_key_components.pkl`

3. Run the Flask application:
```bash
cd app
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

### Individual Prediction
1. Adjust the form inputs to match the employee profile
2. Click "Predict Attrition" to get the prediction
3. View results showing the probability of attrition (0-100%)
4. Interpretation: 
   - **Low probability (< 30%):** Employee likely to stay
   - **Medium probability (30-70%):** Employee at risk
   - **High probability (> 70%):** Employee likely to leave

### Batch Prediction (CSV Upload)
1. Upload a CSV file with employee data
2. The app will predict attrition for all employees
3. Download results with predictions

## Input Features & Valid Ranges

| Feature | Type | Range | Notes |
|---------|------|-------|-------|
| Age | Integer | 18-65 | Employee age in years |
| Department | String | Sales, R&D, HR | Must match exactly |
| EnvironmentSatisfaction | Integer | 1-4 | 1=Low, 4=High |
| JobRole | String | See docs | Must match training data |
| JobSatisfaction | Integer | 1-4 | 1=Low, 4=High |
| MonthlyIncome | Integer | 1000-20000 | Annual salary / 12 |
| NumCompaniesWorked | Integer | 0-9 | Number of previous employers |
| OverTime | Boolean | True/False | Yes/No for overtime |
| PercentSalaryHike | Integer | 11-25 | Percentage increase |
| RelationshipSatisfaction | Integer | 1-4 | 1=Low, 4=High |
| TrainingTimesLastYear | Integer | 0-6 | Number of training sessions |
| WorkLifeBalance | Integer | 1-4 | 1=Bad, 4=Best |
| YearsSinceLastPromotion | Integer | 0-15 | Years without promotion |
| YearsWithCurrManager | Integer | 0-17 | Years with current manager |

## Troubleshooting

### Model Predictions Seem Wrong

✓ **Model Verification:** All test cases pass correctly:
- High risk profile → WILL LEAVE (88.84% probability)
- Low risk profile → WILL STAY (18.83% probability)
- Medium risk profile → WILL STAY (25.88% probability)

**If getting incorrect predictions, check:**

1. **Input Data Types:**
   - All numeric fields must be valid integers/numbers
   - OverTime must be True/False (app handles 'Yes'/'No' conversion)
   - Department and JobRole must match exactly (case-sensitive)

2. **Data Validation:**
   - Age: Between 18-65
   - Satisfaction metrics: Between 1-4
   - Income: Between 1000-20000
   - No null/empty values

3. **Column Names (for CSV upload):**
   - Must match: Age, Department, EnvironmentSatisfaction, JobRole, JobSatisfaction, MonthlyIncome, NumCompaniesWorked, OverTime, PercentSalaryHike, RelationshipSatisfaction, TrainingTimesLastYear, WorkLifeBalance, YearsSinceLastPromotion, YearsWithCurrManager

4. **Verify Model Independence:**
   - Run: `python test_predictions.py` to verify model works correctly
   - Check Flask app logs for detailed error messages

### Column Mismatch Error
- Ensure CSV has all 14 required columns
- Column names must match exactly (no extra spaces or capitalization)
- Remove any unnecessary columns before upload
- **Number of Companies Worked**: 0-10
- **Over Time**: Checkbox
- **Percent Salary Hike**: 10-25
- **Stock Option Level**: 0-3
- **Training Times Last Year**: 0-6
- **Work Life Balance**: 1-4
- **Years Since Last Promotion**: 0-15
- **Years With Current Manager**: 0-15
