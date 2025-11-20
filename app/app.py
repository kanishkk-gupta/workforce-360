from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import os
from werkzeug.utils import secure_filename
import tempfile
import uuid
import json
from datetime import datetime

# Support running this module both as a package and as a script.
try:
    from . import shap_utils
except Exception:
    # When running `python app/app.py` the package context isn't set, so fall back
    # to importing the module from the same directory.
    import sys as _sys
    _app_dir = os.path.dirname(__file__)
    if _app_dir not in _sys.path:
        _sys.path.insert(0, _app_dir)
    import shap_utils

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Store uploaded CSV data in memory (in production, use a proper cache like Redis)
uploaded_csvs = {}
active_sessions = {}
employee_feedback = []

HR_USERNAME = os.getenv('HR_USERNAME', 'hr_manager')
HR_PASSWORD = os.getenv('HR_PASSWORD', 'securepass123')
EMPLOYEE_ID_COLUMNS = ['EmployeeNumber', 'Employee ID', 'EmployeeId', 'Employee_ID', 'Employee', 'EmployeeName', 'Name']
FEEDBACK_CATEGORIES = [
    'Work Environment',
    'Management',
    'Company Culture',
    'Team Collaboration',
    'Compensation & Benefits',
    'Career Growth',
    'Other'
]
JOB_LEVEL_LABELS = {
    1: 'Entry-Level',
    2: 'Mid-Level',
    3: 'Senior-Level',
    4: 'Director',
    5: 'Executive'
}

# Load default employee dataset for employee self-service flows
_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
_default_employee_csv = os.path.join(_data_dir, 'attrition_data.csv')
try:
    base_employee_df = pd.read_csv(_default_employee_csv)
except Exception:
    base_employee_df = pd.DataFrame()
latest_employee_df = base_employee_df.copy()

# Gemini / Generative AI configuration (optional)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
gemini_model = None
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception as gemini_exc:
        print(f"Gemini configuration failed: {gemini_exc}")
        gemini_model = None


def _make_serializable(obj):
    """Recursively convert numpy / pandas objects to JSON-serializable Python types."""
    import numpy as _np
    # dict
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    # numpy arrays and pandas Index -> tolist()
    try:
        if hasattr(obj, 'tolist') and not isinstance(obj, (str, bytes)):
            return _make_serializable(obj.tolist())
    except Exception:
        pass
    # numpy scalars -> native python
    if isinstance(obj, (_np.integer, _np.floating, _np.bool_)):
        try:
            return obj.item()
        except Exception:
            return obj
    return obj


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {'true', 'yes', 'y', '1', 'attrition'}
    return False


def to_bool(value):
    """Helper to convert common inputs to boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ['true', 'yes', 'y', '1', 'on']
    return bool(value)


def _ensure_hr_dataframe(session_id):
    df = uploaded_csvs.get(session_id)
    if df is None or df.empty:
        if latest_employee_df is not None and not latest_employee_df.empty:
            df = latest_employee_df.copy()
            uploaded_csvs[session_id] = df
        else:
            return None
    if 'EmployeeNumber' not in df.columns:
        df['EmployeeNumber'] = range(1, len(df) + 1)
    return df


def _save_hr_dataframe(session_id, df):
    uploaded_csvs[session_id] = df
    global latest_employee_df
    latest_employee_df = df.copy()


def _serialize_employee(row):
    return {
        'employee_number': str(row.get('EmployeeNumber', '')),
        'name': row.get('EmployeeName') or row.get('Name') or f"Employee {row.get('EmployeeNumber', '')}",
        'department': row.get('Department'),
        'job_role': row.get('JobRole'),
        'age': row.get('Age'),
        'monthly_income': row.get('MonthlyIncome'),
        'attrition': bool(_coerce_bool(row.get('Attrition'))),
        'monthly_rate': row.get('MonthlyRate'),
        'years_at_company': row.get('YearsAtCompany'),
        'job_level': row.get('JobLevel')
    }


def _chart_payload(labels, values):
    return {
        'labels': list(labels),
        'values': [int(v) for v in values]
    }


def _get_session(required_role=None):
    """Fetch the active session from the X-Session-ID header."""
    session_id = request.headers.get('X-Session-ID')
    if not session_id or session_id not in active_sessions:
        return None, None, ('Authentication required', 401)
    session = active_sessions[session_id]
    if required_role and session.get('role') != required_role:
        return session_id, None, ('Insufficient permissions for this action', 403)
    return session_id, session, None


def _find_employee_row(identifier, source_df=None):
    """Find a single employee row by ID/name in the provided dataframe (defaults to latest)."""
    if identifier is None:
        return None
    df = source_df if source_df is not None else latest_employee_df
    if df is None or df.empty:
        return None
    target = str(identifier).strip().lower()
    for column in EMPLOYEE_ID_COLUMNS:
        if column in df.columns:
            matches = df[df[column].astype(str).str.strip().str.lower() == target]
            if not matches.empty:
                return matches.iloc[0]
    return None


def _extract_feature_values(row):
    """Extract required feature values from a dataframe row."""
    required_features = {
        'Age': 'Age',
        'Department': 'Department',
        'EnvironmentSatisfaction': 'EnvironmentSatisfaction',
        'JobRole': 'JobRole',
        'JobSatisfaction': 'JobSatisfaction',
        'MonthlyIncome': 'MonthlyIncome',
        'NumCompaniesWorked': 'NumCompaniesWorked',
        'OverTime': 'OverTime',
        'PercentSalaryHike': 'PercentSalaryHike',
        'RelationshipSatisfaction': 'RelationshipSatisfaction',
        'TrainingTimesLastYear': 'TrainingTimesLastYear',
        'WorkLifeBalance': 'WorkLifeBalance',
        'YearsSinceLastPromotion': 'YearsSinceLastPromotion',
        'YearsWithCurrManager': 'YearsWithCurrManager'
    }
    missing_features = []
    feature_values = {}
    for feature_key, column in required_features.items():
        if column not in row.index:
            missing_features.append(column)
        else:
            value = row[column]
            if column == 'OverTime':
                if isinstance(value, bool):
                    feature_values[feature_key] = value
                elif isinstance(value, str):
                    feature_values[feature_key] = value.strip().upper() in ['TRUE', 'YES', 'Y', '1']
                else:
                    feature_values[feature_key] = bool(value)
            else:
                feature_values[feature_key] = value
    return feature_values, missing_features


def _create_input_dataframe(feature_values):
    """Create a dataframe suitable for model prediction from feature values."""
    input_data = pd.DataFrame({
        'Age': [int(feature_values['Age'])],
        'Department': [str(feature_values['Department'])],
        'EnvironmentSatisfaction': [int(feature_values['EnvironmentSatisfaction'])],
        'JobRole': [str(feature_values['JobRole'])],
        'JobSatisfaction': [int(feature_values['JobSatisfaction'])],
        'MonthlyIncome': [int(feature_values['MonthlyIncome'])],
        'NumCompaniesWorked': [int(feature_values['NumCompaniesWorked'])],
        'OverTime': [bool(feature_values['OverTime'])],
        'PercentSalaryHike': [int(feature_values['PercentSalaryHike'])],
        'RelationshipSatisfaction': [int(feature_values['RelationshipSatisfaction'])],
        'TrainingTimesLastYear': [int(feature_values['TrainingTimesLastYear'])],
        'WorkLifeBalance': [int(feature_values['WorkLifeBalance'])],
        'YearsSinceLastPromotion': [int(feature_values['YearsSinceLastPromotion'])],
        'YearsWithCurrManager': [int(feature_values['YearsWithCurrManager'])]
    })
    categorical_cols = ['Department', 'JobRole']
    for col in categorical_cols:
        if col in input_data.columns:
            input_data[col] = input_data[col].astype(str)
    expected_order = ['Age', 'Department', 'EnvironmentSatisfaction', 'JobRole', 'JobSatisfaction',
                      'MonthlyIncome', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
                      'RelationshipSatisfaction', 'TrainingTimesLastYear', 'WorkLifeBalance',
                      'YearsSinceLastPromotion', 'YearsWithCurrManager']
    return input_data[expected_order]


def _classify_feedback_sentiment(feedback_text, rating=None, category=None):
    """Classify sentiment of a single feedback entry."""
    default_result = {
        'label': 'neutral',
        'confidence': 0.5,
        'reason': 'Insufficient data'
    }
    if not feedback_text:
        return default_result

    if gemini_model and GEMINI_API_KEY:
        prompt = (
            "You are an HR assistant. Classify the employee feedback as Positive, Neutral, or Negative. "
            "Respond strictly in JSON with keys 'sentiment' (one of Positive, Neutral, Negative), "
            "'confidence' (0-1 float), and 'reason' (short explanation). "
            f"Feedback category: {category or 'General'}. "
            f"Employee star rating (1-5): {rating or 'N/A'}. "
            f"Feedback text: {feedback_text}"
        )
        try:
            response = gemini_model.generate_content(prompt)
            parsed = json.loads(_clean_json_text(response.text))
            label = str(parsed.get('sentiment', 'neutral')).strip().lower()
            if label not in {'positive', 'neutral', 'negative'}:
                label = 'neutral'
            confidence = float(parsed.get('confidence', 0.5))
            reason = parsed.get('reason', '').strip() or 'Model classification'
            return {
                'label': label,
                'confidence': max(0.0, min(1.0, confidence)),
                'reason': reason
            }
        except Exception as exc:
            print(f"Gemini sentiment classification failed: {exc}")

    # Fallback using rating heuristics
    if rating is not None:
        try:
            rating = int(rating)
            if rating >= 4:
                return {'label': 'positive', 'confidence': 0.6, 'reason': 'High self rating'}
            if rating <= 2:
                return {'label': 'negative', 'confidence': 0.6, 'reason': 'Low self rating'}
        except Exception:
            pass
    lowered = feedback_text.lower()
    if any(word in lowered for word in ['great', 'love', 'happy', 'excellent']):
        return {'label': 'positive', 'confidence': 0.55, 'reason': 'Positive keywords detected'}
    if any(word in lowered for word in ['poor', 'bad', 'terrible', 'hate', 'toxic']):
        return {'label': 'negative', 'confidence': 0.55, 'reason': 'Negative keywords detected'}
    return default_result


def _prepare_department_distribution(entries):
    distribution = {}
    for entry in entries:
        dept = entry.get('department') or 'Unknown'
        sentiment = (entry.get('sentiment') or {}).get('label', 'neutral')
        dept_record = distribution.setdefault(dept, {'positive': 0, 'neutral': 0, 'negative': 0})
        if sentiment not in dept_record:
            sentiment = 'neutral'
        dept_record[sentiment] += 1
    distribution_list = []
    for dept, counts in distribution.items():
        total = max(1, counts['positive'] + counts['neutral'] + counts['negative'])
        distribution_list.append({
            'department': dept,
            'positive': counts['positive'],
            'neutral': counts['neutral'],
            'negative': counts['negative'],
            'total': total
        })
    distribution_list.sort(key=lambda item: item['department'])
    return distribution_list


def _aggregate_feedback_stats(entries):
    sentiment_totals = {'positive': 0, 'neutral': 0, 'negative': 0}
    category_breakdown = {}
    for entry in entries:
        sentiment = (entry.get('sentiment') or {}).get('label', 'neutral')
        if sentiment not in sentiment_totals:
            sentiment = 'neutral'
        sentiment_totals[sentiment] += 1
        category = entry.get('category') or 'General'
        cat_stats = category_breakdown.setdefault(category, {'positive': 0, 'neutral': 0, 'negative': 0, 'count': 0})
        cat_stats[sentiment] += 1
        cat_stats['count'] += 1
    category_list = []
    for cat, stats in category_breakdown.items():
        total = max(1, stats['count'])
        category_list.append({
            'category': cat,
            'positive': stats['positive'],
            'neutral': stats['neutral'],
            'negative': stats['negative'],
            'count': total
        })
    category_list.sort(key=lambda item: item['count'], reverse=True)
    return sentiment_totals, category_list


def _generate_sentiment_overview(entries):
    """Create a Gemini summary for overall sentiment."""
    if not entries:
        return {
            'overview': 'No feedback submitted yet.',
            'actions': [],
            'recommendations': []
        }
    if gemini_model and GEMINI_API_KEY:
        payload = [
            {
                'category': entry.get('category'),
                'rating': entry.get('rating'),
                'sentiment': (entry.get('sentiment') or {}).get('label'),
                'feedback': entry.get('feedback')
            }
            for entry in entries[-50:]
        ]
        prompt = (
            "You are an HR analyst. Given the recent anonymized employee feedback entries below, "
            "return JSON with keys 'overview' (2-3 sentences summarizing mood), 'actions' "
            "(array of up to 4 quick wins), and 'recommendations' "
            "(array of up to 4 strategic suggestions with 'title' and 'detail'). Entries: "
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        try:
            response = gemini_model.generate_content(prompt)
            parsed = json.loads(_clean_json_text(response.text))
            overview = parsed.get('overview') or parsed.get('summary') or ''
            actions = parsed.get('actions') or parsed.get('recommendations') or []
            recommendations = parsed.get('recommendations') or parsed.get('actions') or []
            if isinstance(actions, dict):
                actions = [actions]
            if isinstance(recommendations, dict):
                recommendations = [recommendations]
            formatted_actions = []
            for action in actions:
                if isinstance(action, str):
                    formatted_actions.append({'action': action})
                else:
                    formatted_actions.append(action)
            return {
                'overview': overview.strip() or 'Insights generated from recent feedback.',
                'actions': formatted_actions,
                'recommendations': recommendations
            }
        except Exception as exc:
            print(f"Gemini sentiment overview failed: {exc}")

    # Fallback summary
    sentiments = {'positive': 0, 'neutral': 0, 'negative': 0}
    for entry in entries:
        sentiments[(entry.get('sentiment') or {}).get('label', 'neutral')] += 1
    overview = (
        f"Recent feedback shows {sentiments['positive']} positive, {sentiments['neutral']} neutral, "
        f"and {sentiments['negative']} negative notes. Monitor recurring themes and respond proactively."
    )
    fallback_actions = [
        {'action': 'Discuss key concerns raised in feedback during the next leadership sync.'},
        {'action': 'Prioritize quick wins for departments with higher negative sentiment.'}
    ]
    fallback_recommendations = [
        {'title': 'Pulse Conversations', 'detail': 'Leaders should host short listening sessions in teams showing negative sentiment.'},
        {'title': 'Celebrate Wins', 'detail': 'Highlight positive shout-outs in internal channels to reinforce what works well.'}
    ]
    return {'overview': overview, 'actions': fallback_actions, 'recommendations': fallback_recommendations}


def _clean_json_text(text):
    if text is None:
        return None
    cleaned = text.strip()
    if cleaned.startswith('```'):
        cleaned = cleaned.strip('`')
        # Remove potential language identifier
        if '\n' in cleaned:
            cleaned = cleaned.split('\n', 1)[1]
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _summarize_feedback(feedback_entries):
    """Prepare sentiment overview, department split, and recent feedback list."""
    overview = _generate_sentiment_overview(feedback_entries)
    distribution = _prepare_department_distribution(feedback_entries)
    sentiment_totals, category_breakdown = _aggregate_feedback_stats(feedback_entries)
    recent = [
        {
            'category': entry.get('category'),
            'rating': entry.get('rating'),
            'sentiment': (entry.get('sentiment') or {}).get('label'),
            'department': entry.get('department'),
            'feedback': entry.get('feedback'),
            'submitted_at': entry.get('submitted_at')
        }
        for entry in feedback_entries[-10:]
    ]
    recent.reverse()
    return {
        'total_feedback': len(feedback_entries),
        'sentiment_overview': overview,
        'department_distribution': distribution,
        'recent_feedback': recent,
        'sentiment_totals': sentiment_totals,
        'category_breakdown': category_breakdown,
        'recommendations': overview.get('recommendations', [])
    }


# Load the trained model and unique values from the pickle file
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'model_and_key_components.pkl')

with open(model_path, 'rb') as file:
    saved_components = pickle.load(file)

model = saved_components['model']
unique_values = saved_components.get('unique_values', {})

# Decision threshold for attrition prediction
DECISION_THRESHOLD = float(saved_components.get('threshold', 0.45))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/login', methods=['POST'])
def login():
    data = request.json or {}
    role = data.get('role')
    if role == 'hr':
        username = str(data.get('username', '')).strip()
        password = data.get('password', '')
        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400
        if username != HR_USERNAME or password != HR_PASSWORD:
            return jsonify({'error': 'Invalid HR credentials'}), 401
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = {
            'role': 'hr',
            'username': username,
            'created_at': datetime.utcnow().isoformat() + 'Z'
        }
        return jsonify({
            'session_id': session_id,
            'role': 'hr',
            'display_name': username
        })

    if role == 'employee':
        employee_id = str(data.get('employee_id', '')).strip()
        if not employee_id:
            return jsonify({'error': 'Employee ID is required'}), 400
        row = _find_employee_row(employee_id)
        if row is None:
            return jsonify({'error': 'Employee ID not found in the latest dataset'}), 404
        session_id = str(uuid.uuid4())
        display_name = row.get('EmployeeName') or row.get('Name') or f'Employee {employee_id}'
        active_sessions[session_id] = {
            'role': 'employee',
            'employee_id': employee_id,
            'display_name': display_name,
            'created_at': datetime.utcnow().isoformat() + 'Z'
        }
        return jsonify({
            'session_id': session_id,
            'role': 'employee',
            'display_name': display_name
        })

    return jsonify({'error': 'Invalid role specified'}), 400


@app.route('/api/logout', methods=['POST'])
def logout():
    session_id, session, error = _get_session()
    if error:
        message, status_code = error
        return jsonify({'error': message}), status_code
    active_sessions.pop(session_id, None)
    uploaded_csvs.pop(session_id, None)
    return jsonify({'message': 'Logged out successfully'})


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json or {}

        # Get unique values for categorical columns
        department_values = unique_values.get('Department', ['Sales', 'Research & Development', 'Human Resources'])
        job_role_values = unique_values.get('JobRole', ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
                                                         'Manufacturing Director', 'Healthcare Representative', 'Manager',
                                                         'Sales Representative', 'Research Director', 'Human Resources'])

        input_data = pd.DataFrame({
            'Age': [int(data.get('age', 30))],
            'Department': [str(data.get('department', 'Sales'))],
            'EnvironmentSatisfaction': [int(data.get('environment_satisfaction', 2))],
            'JobRole': [str(data.get('job_role', 'Sales Executive'))],
            'JobSatisfaction': [int(data.get('job_satisfaction', 2))],
            'MonthlyIncome': [int(data.get('monthly_income', 5000))],
            'NumCompaniesWorked': [int(data.get('num_companies_worked', 2))],
            'OverTime': [to_bool(data.get('over_time', False))],
            'PercentSalaryHike': [int(data.get('percent_salary_hike', 15))],
            'RelationshipSatisfaction': [int(data.get('relationship_satisfaction', 2))],
            'TrainingTimesLastYear': [int(data.get('training_times_last_year', 2))],
            'WorkLifeBalance': [int(data.get('work_life_balance', 2))],
            'YearsSinceLastPromotion': [int(data.get('years_since_last_promotion', 3))],
            'YearsWithCurrManager': [int(data.get('years_with_curr_manager', 3))]
        })

        # Ensure categorical columns are strings (as expected by CatBoost)
        categorical_cols = ['Department', 'JobRole']
        for col in categorical_cols:
            if col in input_data.columns:
                input_data[col] = input_data[col].astype(str)

        # Ensure columns are in the exact order the model expects (matching training order)
        expected_order = ['Age', 'Department', 'EnvironmentSatisfaction', 'JobRole', 'JobSatisfaction',
                         'MonthlyIncome', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
                         'RelationshipSatisfaction', 'TrainingTimesLastYear', 'WorkLifeBalance',
                         'YearsSinceLastPromotion', 'YearsWithCurrManager']
        input_data = input_data[expected_order]

        # Make prediction with probability
        probability = model.predict_proba(input_data)[:, 1][0]

        # Apply custom threshold
        prediction = 1 if probability >= DECISION_THRESHOLD else 0

        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'probability_percent': round(probability * 100, 2),
            'risk_level': 'HIGH RISK - Will Likely Leave' if probability >= DECISION_THRESHOLD else 'LOW RISK - Will Likely Stay',
            'threshold_used': DECISION_THRESHOLD,
            'will_leave': bool(prediction == 1)
        }
        # Provide SHAP-based explanation and recommendations
        try:
            explain_result = shap_utils.explain_and_recommend(model, input_data)
            result['explanation'] = explain_result
        except Exception as e:
            result['explanation_error'] = str(e)

        return jsonify(_make_serializable(result))
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in predict: {error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 400


@app.route('/api/unique_values', methods=['GET'])
def get_unique_values():
    """Endpoint to get unique values for reference"""
    return jsonify(_make_serializable(unique_values))


@app.route('/api/explain', methods=['POST'])
def explain():
    """Separate endpoint to return SHAP explanation and recommendations for a given input payload."""
    try:
        data = request.json or {}
        input_data = pd.DataFrame({
            'Age': [int(data.get('age', 30))],
            'Department': [str(data.get('department', 'Sales'))],
            'EnvironmentSatisfaction': [int(data.get('environment_satisfaction', 2))],
            'JobRole': [str(data.get('job_role', 'Sales Executive'))],
            'JobSatisfaction': [int(data.get('job_satisfaction', 2))],
            'MonthlyIncome': [int(data.get('monthly_income', 5000))],
            'NumCompaniesWorked': [int(data.get('num_companies_worked', 2))],
            'OverTime': [to_bool(data.get('over_time', False))],
            'PercentSalaryHike': [int(data.get('percent_salary_hike', 15))],
            'RelationshipSatisfaction': [int(data.get('relationship_satisfaction', 2))],
            'TrainingTimesLastYear': [int(data.get('training_times_last_year', 2))],
            'WorkLifeBalance': [int(data.get('work_life_balance', 2))],
            'YearsSinceLastPromotion': [int(data.get('years_since_last_promotion', 3))],
            'YearsWithCurrManager': [int(data.get('years_with_curr_manager', 3))]
        })

        for col in ['Department', 'JobRole']:
            if col in input_data.columns:
                input_data[col] = input_data[col].astype(str)

        expected_order = ['Age', 'Department', 'EnvironmentSatisfaction', 'JobRole', 'JobSatisfaction',
                         'MonthlyIncome', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
                         'RelationshipSatisfaction', 'TrainingTimesLastYear', 'WorkLifeBalance',
                         'YearsSinceLastPromotion', 'YearsWithCurrManager']
        input_data = input_data[expected_order]

        explain_result = shap_utils.explain_and_recommend(model, input_data)
        return jsonify(_make_serializable(explain_result))
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in explain: {error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 400


@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    """Endpoint to upload a CSV file and store it in memory (HR only)."""
    try:
        session_id, session, error = _get_session('hr')
        if error:
            message, status_code = error
            return jsonify({'error': message}), status_code
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be a CSV'}), 400

        # Read CSV into DataFrame
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({'error': f'Error reading CSV: {str(e)}'}), 400

        # Store the DataFrame for the HR session
        uploaded_csvs[session_id] = df
        global latest_employee_df
        latest_employee_df = df.copy()

        # Return session ID and available search columns
        search_columns = []
        if 'Name' in df.columns:
            search_columns.append('Name')
        if 'EmployeeName' in df.columns:
            search_columns.append('EmployeeName')
        if 'EmployeeNumber' in df.columns:
            search_columns.append('EmployeeNumber')
        if 'Employee ID' in df.columns:
            search_columns.append('Employee ID')

        return jsonify({
            'session_id': session_id,
            'rows': len(df),
            'columns': list(df.columns),
            'search_columns': search_columns,
            'message': 'CSV uploaded successfully'
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in upload_csv: {error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 400


@app.route('/api/search_employee', methods=['POST'])
def search_employee():
    """Endpoint to search for an employee by name and make a prediction (HR only)."""
    try:
        session_id, session, error = _get_session('hr')
        if error:
            message, status_code = error
            return jsonify({'error': message}), status_code
        if session_id not in uploaded_csvs:
            return jsonify({'error': 'Please upload an employee CSV before searching.'}), 400

        data = request.json or {}
        employee_name = data.get('employee_name', '').strip()

        if not employee_name:
            return jsonify({'error': 'Employee name is required'}), 400

        df = uploaded_csvs[session_id]

        # Try to find employee by name (case-insensitive)
        employee_row = None
        search_columns = ['Name', 'EmployeeName', 'Employee', 'Employee ID', 'EmployeeNumber']
        for col in search_columns:
            if col in df.columns:
                mask = df[col].astype(str).str.lower().str.contains(employee_name.lower(), na=False)
                matches = df[mask]
                if len(matches) > 0:
                    employee_row = matches.iloc[0]
                    break

        if employee_row is None:
            return jsonify({'error': f'Employee \"{employee_name}\" not found in the CSV'}), 404

        feature_values, missing_features = _extract_feature_values(employee_row)
        if missing_features:
            return jsonify({
                'error': f'Missing required columns in CSV: {", ".join(missing_features)}',
                'missing_features': missing_features,
                'available_columns': list(df.columns)
            }), 400

        input_data = _create_input_dataframe(feature_values)

        # Make predictions with threshold
        probability = model.predict_proba(input_data)[:, 1][0]
        prediction = 1 if probability >= DECISION_THRESHOLD else 0

        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'probability_percent': round(probability * 100, 2),
            'risk_level': 'HIGH RISK - Will Likely Leave' if probability >= DECISION_THRESHOLD else 'LOW RISK - Will Likely Stay',
            'threshold_used': DECISION_THRESHOLD,
            'will_leave': bool(prediction == 1),
            'employee_name': employee_name,
            'employee_data': _make_serializable(employee_row.to_dict())
        }

        # Provide SHAP-based explanation and recommendations
        try:
            explain_result = shap_utils.explain_and_recommend(model, input_data)
            result['explanation'] = explain_result
        except Exception as e:
            result['explanation_error'] = str(e)

        return jsonify(_make_serializable(result))
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in search_employee: {error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 400


@app.route('/api/hr/employees', methods=['GET', 'POST'])
def manage_employees():
    """Return or create employees within the uploaded dataset."""
    session_id, session, error = _get_session('hr')
    if error:
        message, status_code = error
        return jsonify({'error': message}), status_code
    df = _ensure_hr_dataframe(session_id)
    if df is None:
        return jsonify({'error': 'Please upload an employee CSV before managing records.'}), 400

    if request.method == 'GET':
        employees = [_serialize_employee(row) for _, row in df.iterrows()]
        return jsonify({'employees': employees})

    data = request.json or {}
    name = data.get('name') or data.get('employee_name') or 'New Employee'
    department = data.get('department') or 'General'
    job_role = data.get('job_role') or 'Staff'
    age = int(data.get('age', 30))
    monthly_income = float(data.get('monthly_income', 5000))
    attrition = _coerce_bool(data.get('attrition', False))

    next_number = data.get('employee_number')
    if not next_number:
        if 'EmployeeNumber' in df.columns and not df.empty:
            try:
                next_number = int(pd.to_numeric(df['EmployeeNumber'], errors='coerce').max()) + 1
            except Exception:
                next_number = len(df) + 1
        else:
            next_number = len(df) + 1

    new_row = {
        'EmployeeNumber': next_number,
        'EmployeeName': name,
        'Name': name,
        'Department': department,
        'JobRole': job_role,
        'Age': age,
        'MonthlyIncome': monthly_income,
        'Attrition': attrition
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _save_hr_dataframe(session_id, df)
    return jsonify({'message': 'Employee added successfully.', 'employee': _serialize_employee(new_row)})


@app.route('/api/hr/employees/<employee_id>', methods=['PUT', 'DELETE'])
def modify_employee(employee_id):
    """Update or delete a specific employee."""
    session_id, session, error = _get_session('hr')
    if error:
        message, status_code = error
        return jsonify({'error': message}), status_code
    df = _ensure_hr_dataframe(session_id)
    if df is None:
        return jsonify({'error': 'Please upload an employee CSV before managing records.'}), 400

    mask = df['EmployeeNumber'].astype(str) == str(employee_id)
    if not mask.any():
        return jsonify({'error': f'Employee {employee_id} not found.'}), 404

    if request.method == 'DELETE':
        df = df.loc[~mask].reset_index(drop=True)
        _save_hr_dataframe(session_id, df)
        return jsonify({'message': f'Employee {employee_id} deleted.'})

    data = request.json or {}
    if 'name' in data or 'employee_name' in data:
        name = data.get('name') or data.get('employee_name')
        if 'EmployeeName' in df.columns:
            df.loc[mask, 'EmployeeName'] = name
        if 'Name' in df.columns:
            df.loc[mask, 'Name'] = name
        else:
            df.loc[mask, 'Name'] = name
    if 'department' in data:
        df.loc[mask, 'Department'] = data['department']
    if 'job_role' in data:
        df.loc[mask, 'JobRole'] = data['job_role']
    if 'age' in data:
        df.loc[mask, 'Age'] = int(data['age'])
    if 'monthly_income' in data:
        df.loc[mask, 'MonthlyIncome'] = float(data['monthly_income'])
    if 'attrition' in data:
        df.loc[mask, 'Attrition'] = _coerce_bool(data['attrition'])

    _save_hr_dataframe(session_id, df)
    updated_row = df.loc[mask].iloc[0]
    return jsonify({'message': f'Employee {employee_id} updated.', 'employee': _serialize_employee(updated_row)})


@app.route('/api/hr/attrition_dashboard', methods=['GET'])
def attrition_dashboard():
    """Return KPI metrics and chart-ready data for the uploaded dataset."""
    session_id, session, error = _get_session('hr')
    if error:
        message, status_code = error
        return jsonify({'error': message}), status_code
    df = _ensure_hr_dataframe(session_id)
    if df is None:
        return jsonify({'error': 'Please upload an employee CSV before viewing analytics.'}), 400

    total_employees = len(df)
    attrition_series = df['Attrition'].apply(_coerce_bool) if 'Attrition' in df.columns else pd.Series([False] * total_employees)
    total_attrition = int(attrition_series.sum())
    attrition_rate = round((total_attrition / total_employees) * 100, 2) if total_employees else 0
    average_age = round(float(df['Age'].dropna().mean()), 2) if 'Age' in df.columns and not df['Age'].dropna().empty else None
    average_income = round(float(df['MonthlyIncome'].dropna().mean()), 2) if 'MonthlyIncome' in df.columns and not df['MonthlyIncome'].dropna().empty else None

    def _value_counts(series, labels_order=None):
        if series is None or series.empty:
            if labels_order:
                return labels_order, [0] * len(labels_order)
            return [], []
        counts = series.value_counts()
        if labels_order:
            counts = counts.reindex(labels_order, fill_value=0)
        return list(counts.index.astype(str)), [int(v) for v in counts.values]

    # Distribution stay vs leave
    distribution_labels = ['Stay', 'Leave']
    distribution_values = [int(total_employees - total_attrition), total_attrition]

    # Job level
    if 'JobLevel' in df.columns and not df['JobLevel'].dropna().empty:
        job_series = df['JobLevel'].dropna().astype(int).map(lambda lvl: JOB_LEVEL_LABELS.get(lvl, f'Level {lvl}'))
        job_labels, job_values = _value_counts(job_series)
    else:
        job_labels, job_values = [], []

    # Age groups
    age_labels = ['18-25', '26-35', '36-45', '46-55', '56+']
    if 'Age' in df.columns and not df['Age'].dropna().empty:
        age_bins = [17, 25, 35, 45, 55, 200]
        age_series = pd.cut(df['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
        _, age_values = _value_counts(age_series, age_labels)
    else:
        age_values = [0] * len(age_labels)

    # Training times
    training_labels = ['0-2', '3-5', '6+']
    if 'TrainingTimesLastYear' in df.columns and not df['TrainingTimesLastYear'].dropna().empty:
        def training_bucket(val):
            try:
                val = int(val)
            except Exception:
                val = 0
            if val <= 2:
                return '0-2'
            if val <= 5:
                return '3-5'
            return '6+'
        training_series = df['TrainingTimesLastYear'].apply(training_bucket)
        _, training_values = _value_counts(training_series, training_labels)
    else:
        training_values = [0, 0, 0]

    # Environment satisfaction
    env_labels = ['Satisfied', 'Dissatisfied']
    if 'EnvironmentSatisfaction' in df.columns and not df['EnvironmentSatisfaction'].dropna().empty:
        env_series = df['EnvironmentSatisfaction'].apply(lambda v: 'Satisfied' if pd.notna(v) and float(v) >= 3 else 'Dissatisfied')
        _, env_values = _value_counts(env_series, env_labels)
    else:
        env_values = [0, 0]

    # Job satisfaction
    if 'JobSatisfaction' in df.columns and not df['JobSatisfaction'].dropna().empty:
        job_sat_series = df['JobSatisfaction'].apply(lambda v: 'Satisfied' if pd.notna(v) and float(v) >= 3 else 'Dissatisfied')
        job_sat_labels, job_sat_values = _value_counts(job_sat_series, env_labels)
    else:
        job_sat_labels, job_sat_values = env_labels, [0, 0]

    # Years since last promotion
    promotion_labels = ['0-3', '4-6', '7-9', '10-13', '14+']
    if 'YearsSinceLastPromotion' in df.columns and not df['YearsSinceLastPromotion'].dropna().empty:
        def promo_bucket(val):
            try:
                val = int(val)
            except Exception:
                val = 0
            if val <= 3:
                return '0-3'
            if val <= 6:
                return '4-6'
            if val <= 9:
                return '7-9'
            if val <= 13:
                return '10-13'
            return '14+'
        promo_series = df['YearsSinceLastPromotion'].apply(promo_bucket)
        _, promo_values = _value_counts(promo_series, promotion_labels)
    else:
        promo_values = [0] * len(promotion_labels)

    # Work-life balance
    wlb_labels = ['Good', 'Poor']
    if 'WorkLifeBalance' in df.columns and not df['WorkLifeBalance'].dropna().empty:
        wlb_series = df['WorkLifeBalance'].apply(lambda v: 'Good' if pd.notna(v) and float(v) >= 3 else 'Poor')
        _, wlb_values = _value_counts(wlb_series, wlb_labels)
    else:
        wlb_values = [0, 0]

    response = {
        'kpis': {
            'total_employees': total_employees,
            'total_attrition': total_attrition,
            'attrition_rate': attrition_rate,
            'average_age': average_age,
            'average_income': average_income
        },
        'distribution': _chart_payload(distribution_labels, distribution_values),
        'job_level': _chart_payload(job_labels, job_values),
        'age_group': _chart_payload(age_labels, age_values),
        'training': _chart_payload(training_labels, training_values),
        'environment': _chart_payload(env_labels, env_values),
        'job_satisfaction': _chart_payload(job_sat_labels, job_sat_values),
        'promotion': _chart_payload(promotion_labels, promo_values),
        'work_life': _chart_payload(wlb_labels, wlb_values)
    }
    return jsonify(_make_serializable(response))


@app.route('/api/employee_self', methods=['GET'])
def employee_self():
    """Allow an authenticated employee to view their own attrition prediction."""
    try:
        session_id, session, error = _get_session('employee')
        if error:
            message, status_code = error
            return jsonify({'error': message}), status_code
        employee_id = session.get('employee_id')
        row = _find_employee_row(employee_id)
        if row is None:
            return jsonify({'error': 'Employee record not found in the current dataset.'}), 404

        feature_values, missing = _extract_feature_values(row)
        if missing:
            return jsonify({'error': 'Required fields are missing for your record.', 'missing_features': missing}), 400

        input_data = _create_input_dataframe(feature_values)

        probability = model.predict_proba(input_data)[:, 1][0]
        prediction = 1 if probability >= DECISION_THRESHOLD else 0

        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'probability_percent': round(probability * 100, 2),
            'risk_level': 'HIGH RISK - Will Likely Leave' if probability >= DECISION_THRESHOLD else 'LOW RISK - Will Likely Stay',
            'threshold_used': DECISION_THRESHOLD,
            'will_leave': bool(prediction == 1),
            'employee_name': row.get('EmployeeName') or row.get('Name') or f'Employee {employee_id}',
            'employee_data': _make_serializable(row.to_dict())
        }

        try:
            explain_result = shap_utils.explain_and_recommend(model, input_data)
            result['explanation'] = explain_result
        except Exception as e:
            result['explanation_error'] = str(e)

        return jsonify(_make_serializable(result))
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in employee_self: {error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 400


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Allow an authenticated employee to submit feedback."""
    try:
        session_id, session, error = _get_session('employee')
        if error:
            message, status_code = error
            return jsonify({'error': message}), status_code
        data = request.json or {}
        feedback_message = str(data.get('feedback', '')).strip()
        category = str(data.get('category', '')).strip()
        rating = data.get('rating')

        if category not in FEEDBACK_CATEGORIES:
            return jsonify({'error': 'Please select a valid feedback category.'}), 400
        try:
            rating = int(rating)
        except Exception:
            rating = 0
        if rating < 1 or rating > 5:
            return jsonify({'error': 'Please provide a star rating between 1 and 5.'}), 400
        if not feedback_message:
            return jsonify({'error': 'Feedback message is required'}), 400

        employee_id = session.get('employee_id')
        row = _find_employee_row(employee_id)
        department = 'Unknown'
        if row is not None and 'Department' in row.index:
            department = row['Department']

        sentiment = _classify_feedback_sentiment(feedback_message, rating=rating, category=category)

        entry = {
            'category': category,
            'rating': rating,
            'feedback': feedback_message,
            'department': department,
            'sentiment': sentiment,
            'submitted_at': datetime.utcnow().isoformat() + 'Z'
        }
        employee_feedback.append(entry)
        if len(employee_feedback) > 500:
            employee_feedback.pop(0)

        return jsonify({'message': 'Feedback submitted successfully'})
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in submit_feedback: {error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 400


@app.route('/api/hr/feedback_summary', methods=['GET'])
def hr_feedback_summary():
    """Provide HR with a Gemini-generated summary of employee feedback."""
    session_id, session, error = _get_session('hr')
    if error:
        message, status_code = error
        return jsonify({'error': message}), status_code

    summary = _summarize_feedback(employee_feedback)
    return jsonify(summary)


if __name__ == "__main__":
    app.run(debug=True, port=5000)