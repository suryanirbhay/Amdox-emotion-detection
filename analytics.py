import pandas as pd

def calculate_stress_score(emotion):
    if emotion == "stressed":
        return 3
    elif emotion == "angry":
        return 2
    elif emotion == "sad":
        return 1
    else:
        return 0

def employee_risk_level(df, employee_id):
    """Calculate risk level for an employee based on stress scores"""
    if isinstance(df, pd.DataFrame):
        emp_data = df[df['employee_id'] == employee_id]
    else:
        emp_data = df
    
    if emp_data.empty:
        return "No Data"
    
    avg_stress = emp_data['stress_score'].mean()
    
    if avg_stress >= 2:
        return "HIGH RISK"
    elif avg_stress >= 1:
        return "MODERATE RISK"
    else:
        return "LOW RISK"
