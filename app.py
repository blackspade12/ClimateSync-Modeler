# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load("climate_model.pkl")

# Load feature list
with open("features_list.txt", "r") as f:
    features = [line.strip() for line in f]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse input data
        input_data = request.get_json()
        
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])
        
        # One-hot encode categorical data
        df = pd.get_dummies(df, drop_first=True)
        
        # Ensure all necessary columns exist (align with training data)
        missing_cols = set(features) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        
        df = df[features]

        # Make prediction
        prediction = model.predict(df)

        return jsonify({"predicted_ecosystem_health_score": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 400



def detailed_risk_assessment(risk_factors):
    """
    Perform a detailed risk assessment based on input risk factors.

    Args:
        risk_factors (dict): A dictionary of risk factors with their values.

    Returns:
        dict: A dictionary containing risk levels for each factor.
    """
    # Define thresholds for each factor
    thresholds = {
        "Temperature": {"low": 15, "high": 30},
        "Salinity": {"low": 5, "high": 15},
        "pH": {"low": 6.5, "high": 8.5},
        "Dissolved Oxygen": {"low": 4, "high": 10}
    }

    # Initialize assessments dictionary
    assessments = {}

    for factor, values in thresholds.items():
        value = risk_factors.get(factor, None)
        if value is not None:
            if value < values["low"]:
                assessments[factor] = "Low - Below Optimal"
            elif value > values["high"]:
                assessments[factor] = "Critical - Above Optimal"
            else:
                assessments[factor] = "Optimal"
        else:
            assessments[factor] = "Data Missing"

    return assessments

def suggest_mitigation(assessments):
    """
    Suggest mitigation strategies based on risk levels.

    Args:
        assessments (dict): Risk levels for each factor.

    Returns:
        dict: A dictionary containing mitigation strategies.
    """
    strategies = {}

    if assessments.get("Temperature", "") == "Critical - Above Optimal":
        strategies["Temperature"] = "Install cooling systems or increase water circulation to regulate temperature."
    elif assessments.get("Temperature", "") == "Low - Below Optimal":
        strategies["Temperature"] = "Use heaters or other thermal control measures to raise water temperature."

    if assessments.get("Salinity", "") == "Critical - Above Optimal":
        strategies["Salinity"] = "Dilute water with fresh sources to reduce salinity."
    elif assessments.get("Salinity", "") == "Low - Below Optimal":
        strategies["Salinity"] = "Add controlled amounts of saline water to maintain optimal levels."

    if assessments.get("pH", "") == "Critical - Above Optimal":
        strategies["pH"] = "Introduce pH-lowering agents like CO2 injection or buffering agents."
    elif assessments.get("pH", "") == "Low - Below Optimal":
        strategies["pH"] = "Use lime or other pH-raising substances to stabilize levels."

    if assessments.get("Dissolved Oxygen", "") == "Critical - Above Optimal":
        strategies["Dissolved Oxygen"] = "Monitor aeration systems to avoid excessive oxygenation."
    elif assessments.get("Dissolved Oxygen", "") == "Low - Below Optimal":
        strategies["Dissolved Oxygen"] = "Increase aeration or add oxygenation devices to improve levels."

    return strategies

def predict_productivity(input_factors):
    """
    Predict potential productivity losses or gains based on input factors.

    Args:
        input_factors (dict): A dictionary containing environmental and operational factors.

    Returns:
        dict: A dictionary with predicted productivity changes and contributing factors.
    """
    # Define thresholds for each factor
    thresholds = {
        "Temperature": {"low": 15, "high": 30},
        "Salinity": {"low": 5, "high": 15},
        "pH": {"low": 6.5, "high": 8.5},
        "Dissolved Oxygen": {"low": 4, "high": 10}
    }
    # Define baseline productivity and sensitivity coefficients for each factor
    baseline_productivity = 100  # Example baseline productivity (e.g., 100 tons/year)
    sensitivity = {
        "Temperature": -2,  # % change per degree above/below optimal range
        "Salinity": -1.5,  # % change per unit above/below optimal range
        "pH": -1,          # % change per unit above/below optimal range
        "Dissolved Oxygen": 2  # % increase per unit within optimal range
    }

    # Initialize productivity change and contributing factors
    productivity_change = 0
    contributing_factors = {}

    for factor, value in input_factors.items():
        if factor in sensitivity:
            if factor in thresholds:
                optimal_range = thresholds[factor]
                if value < optimal_range["low"]:
                    change = sensitivity[factor] * (optimal_range["low"] - value)
                    productivity_change += change
                    contributing_factors[factor] = change
                elif value > optimal_range["high"]:
                    change = sensitivity[factor] * (value - optimal_range["high"])
                    productivity_change += change
                    contributing_factors[factor] = change
                else:
                    # Positive contribution within optimal range
                    change = sensitivity[factor] * (value - optimal_range["low"])
                    productivity_change += change
                    contributing_factors[factor] = change

    # Calculate final productivity
    final_productivity = baseline_productivity + productivity_change

    return {
        "baseline_productivity": baseline_productivity,
        "final_productivity": max(final_productivity, 0),  # Ensure no negative productivity
        "productivity_change": productivity_change,
        "contributing_factors": contributing_factors
    }

# Example endpoint for productivity prediction
@app.route("/predict-productivity", methods=["POST"])
def predict_productivity_endpoint():
    try:
        # Parse input data
        input_data = request.get_json()
        input_factors = input_data.get("input_factors", {})

        # Predict productivity
        prediction = predict_productivity(input_factors)

        return jsonify({"productivity_prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Example endpoint for enhanced risk assessment
@app.route("/risk-assessment", methods=["POST"])
def risk_assessment():
    try:
        # Parse input data
        input_data = request.get_json()
        risk_factors = input_data.get("risk_factors", {})

        # Perform risk assessment and suggest mitigations
        assessments = detailed_risk_assessment(risk_factors)
        strategies = suggest_mitigation(assessments)

        return jsonify({"risk_assessment": assessments, "mitigation_strategies": strategies})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Start the Flask server
if __name__ == '__main__':
    # Get the port from environment variable (Render will set this)
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port, debug=False)  # Bind to 0.0.0.0 for external access
