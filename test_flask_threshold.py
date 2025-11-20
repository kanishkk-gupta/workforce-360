"""
Quick test script to verify Flask app works with new threshold
Run this AFTER starting: python app/app.py
"""

import requests
import json

# API endpoint
API_URL = "http://localhost:5000/api/predict"

# Test Case 1: Your problematic employee data (should now predict LEAVE)
test_employee = {
    "age": 54,
    "department": "Sales",
    "environment_satisfaction": 1,
    "job_role": "Manager",
    "job_satisfaction": 4,
    "monthly_income": 15552,
    "num_companies_worked": 0,
    "over_time": False,
    "percent_salary_hike": 15,
    "relationship_satisfaction": 4,
    "training_times_last_year": 3,
    "work_life_balance": 3,
    "years_since_last_promotion": 5,
    "years_with_curr_manager": 9
}

print("=" * 80)
print("FLASK APP TEST - THRESHOLD IMPLEMENTATION")
print("=" * 80)
print("\nNote: Make sure Flask app is running: python app/app.py\n")

try:
    # Make prediction
    response = requests.post(API_URL, json=test_employee)
    
    if response.status_code == 200:
        result = response.json()
        
        print("✅ API RESPONSE SUCCESSFUL")
        print("-" * 80)
        print(f"Prediction: {result['prediction']} ({'LEAVE' if result['prediction'] == 1 else 'STAY'})")
        print(f"Probability: {result['probability_percent']:.2f}%")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Threshold Used: {result['threshold_used']}")
        print("-" * 80)
        
        # Verify threshold is 0.45
        if result['threshold_used'] == 0.45:
            print("✓ Threshold correctly set to 0.45")
        else:
            print(f"⚠ Threshold is {result['threshold_used']} (expected 0.45)")
            
        # Check if prediction makes sense
        if result['probability'] > 0.45 and result['prediction'] == 1:
            print("✓ Prediction logic correct (probability > threshold → LEAVE)")
        elif result['probability'] <= 0.45 and result['prediction'] == 0:
            print("✓ Prediction logic correct (probability ≤ threshold → STAY)")
        else:
            print("⚠ Prediction logic might be incorrect")
            
        print("\nFull Response:")
        print(json.dumps(result, indent=2))
        
    else:
        print(f"❌ API Error: {response.status_code}")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("❌ Could not connect to Flask app!")
    print("\nMake sure the Flask app is running:")
    print("  python app/app.py")
    print("\nThen run this test again in another terminal window")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 80)
