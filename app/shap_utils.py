import os
import pandas as pd
import numpy as np

# shap is an optional dependency - imported inside functions to avoid import-time errors

# Expected feature order used by the model
EXPECTED_ORDER = ['Age', 'Department', 'EnvironmentSatisfaction', 'JobRole', 'JobSatisfaction',
                  'MonthlyIncome', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
                  'RelationshipSatisfaction', 'TrainingTimesLastYear', 'WorkLifeBalance',
                  'YearsSinceLastPromotion', 'YearsWithCurrManager']

# Simple mapping from feature -> retention recommendation
RECOMMENDATION_MAP = {
    'MonthlyIncome': "Consider a compensation review (raises, bonuses, or targeted retention pay).",
    'OverTime': "Reduce overtime, offer flexible scheduling or additional time-off to improve work-life balance.",
    'JobSatisfaction': "Provide career development, recognition, and clearer growth paths to increase job satisfaction.",
    'YearsSinceLastPromotion': "Review promotion opportunities and career progression for the employee.",
    'WorkLifeBalance': "Offer flexible or remote work options and wellness programs.",
    'EnvironmentSatisfaction': "Improve immediate work environment and manager/peer support.",
    'TrainingTimesLastYear': "Provide more training/upskilling and mentorship opportunities.",
    'NumCompaniesWorked': "Offer long-term incentives and clarify career path to improve retention.",
    'RelationshipSatisfaction': "Improve manager coaching and team-building activities.",
    'PercentSalaryHike': "Consider targeted salary adjustments or performance-based bonuses.",
    'JobRole': "Consider role enrichment, rotation, or clearer expectations for this role.",
    'Age': "Consider tailored retention programs for different career stages (mentorship, leadership paths)."
}

_explainer = None
_background = None


def _load_background_sample(max_rows=100, sample_frac=None):
    global _background
    if _background is not None:
        return _background

    project_root = os.path.dirname(os.path.dirname(__file__))
    train_path = os.path.join(project_root, 'data', 'train_data.csv')
    if not os.path.exists(train_path):
        _background = None
        return None

    bg = pd.read_csv(train_path)
    # Keep expected columns if present
    cols = [c for c in EXPECTED_ORDER if c in bg.columns]
    bg = bg[cols]

    # Ensure categorical columns are string type for CatBoost compatibility
    for c in ['Department', 'JobRole']:
        if c in bg.columns:
            bg[c] = bg[c].astype(str)

    # Sampling if very large
    if sample_frac is not None:
        bg = bg.sample(frac=sample_frac, random_state=42)
    elif bg.shape[0] > max_rows:
        bg = bg.sample(max_rows, random_state=42)

    _background = bg
    return _background


def _get_explainer(model):
    """Lazily create and cache a SHAP explainer for the fitted model."""
    global _explainer
    if _explainer is not None:
        return _explainer

    try:
        import shap
    except Exception as e:
        raise ImportError("The 'shap' package is required for explanations. Install with 'pip install shap'.")

    bg = _load_background_sample()
    try:
        if bg is not None:
            _explainer = shap.TreeExplainer(model, data=bg)
        else:
            _explainer = shap.TreeExplainer(model)
    except Exception:
        # fall back to generic explainer
        _explainer = shap.Explainer(model)

    return _explainer


def explain_and_recommend(model, input_df, top_n=5):
    """Return top feature contributions and human-readable recommendations for a single input row.

    Parameters
    - model: a fitted model (CatBoost or similar)
    - input_df: pandas.DataFrame with a single-row example in EXPECTED_ORDER
    - top_n: how many top features to return

    Returns: dict with keys: 'top_features' (list), 'recommendations' (list)
    """
    if input_df.shape[0] != 1:
        raise ValueError("explain_and_recommend currently expects a single-row DataFrame")

    explainer = _get_explainer(model)

    try:
        # For older shap versions, explainer may have shap_values method
        if hasattr(explainer, 'shap_values'):
            shap_values = explainer.shap_values(input_df)
        else:
            shap_values = explainer(input_df).values
    except Exception:
        # Last resort: use explainer call
        shap_values = explainer(input_df).values

    # shap_values may be a list (per-class) or array
    if isinstance(shap_values, list) or (hasattr(shap_values, 'ndim') and getattr(shap_values, 'ndim', 0) == 3):
        # pick the positive class (index 1) if available
        if isinstance(shap_values, list) and len(shap_values) > 1:
            sv = np.array(shap_values[1])
        else:
            sv = np.array(shap_values[1])
    else:
        sv = np.array(shap_values)

    # sv should be (n_samples, n_features); we want first row
    if sv.ndim == 2:
        sv_row = sv[0]
    else:
        sv_row = sv

    feature_names = list(input_df.columns)
    abs_vals = np.abs(sv_row)
    order = np.argsort(-abs_vals)[:top_n]

    top_features = []
    total_abs = abs_vals.sum() if abs_vals.sum() != 0 else 1.0
    recommendations = []

    for idx in order:
        fname = feature_names[idx]
        fval = input_df.iloc[0, idx] if fname in input_df.columns else None
        contrib = float(sv_row[idx])
        rel = float(abs_vals[idx] / total_abs)
        # Normalize the feature value to native Python types for JSON serialization
        if pd.isna(fval):
            py_val = None
        else:
            try:
                if hasattr(fval, 'item'):
                    py_val = fval.item()
                elif hasattr(fval, 'tolist'):
                    py_val = fval.tolist()
                else:
                    py_val = fval
            except Exception:
                py_val = fval

            # Convert numpy scalar types to native Python scalars
            if isinstance(py_val, (np.integer, np.floating, np.bool_)):
                try:
                    py_val = py_val.item()
                except Exception:
                    pass

        top_features.append({
            'feature': fname,
            'value': py_val,
            'shap_value': contrib,
            'relative_importance': rel
        })

        # If the SHAP value increases probability of leaving (positive), suggest retention actions
        if contrib > 0:
            rec_text = RECOMMENDATION_MAP.get(fname, None)
            if rec_text:
                recommendations.append({
                    'feature': fname,
                    'issue': rec_text,
                    'severity': rel
                })

    # If no positive contributors found among top features, suggest general retention actions
    if not recommendations:
        recommendations.append({
            'feature': None,
            'issue': "No major positive contributors detected among top features. Consider a general retention review: compensation, manager check-in, career progression.",
            'severity': 0.0
        })

    return {
        'top_features': top_features,
        'recommendations': recommendations
    }
