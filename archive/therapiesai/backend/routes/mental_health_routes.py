from flask import Blueprint, request, jsonify
from services.firebase_service import get_user_entries
import spacy
import statistics


mental_health_routes = Blueprint("mental_health_routes", __name__)

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

@mental_health_routes.route('/generate-report', methods=['POST'])
def generate_report():
    """
    Generate a mental health report based on user journal entries.
    """
    try:
        data = request.json
        user_id = data.get("userId")
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        # Fetch user journal entries from Firestore
        journal_entries = get_user_entries(user_id)
        if not journal_entries:
            return jsonify({"error": "No journal entries found"}), 404

        # Analyze the entries and calculate risks
        risks = calculate_risks(journal_entries)

        # Calculate Brain Health Index
        risks["brainHealthIndex"] = calculate_brain_health_index(
            risks.get("dementia", 0),
            risks.get("alzheimers", 0)
        )

        # Calculate Mood Stability
        happiness_scores = [entry.get("happiness", 0) for entry in journal_entries]
        risks["moodStability"] = calculate_mood_stability(happiness_scores)

        # Add Trend Analysis
        risks["trendAnalysis"] = calculate_trend_analysis(happiness_scores)

        # Add Strength-Based Feedback
        risks["strengths"] = generate_strength_based_feedback(risks)

        return jsonify(risks), 200
    except Exception as e:
        print(f"Error generating report: {e}")
        return jsonify({"error": "Internal server error"}), 500


def calculate_risks(entries):
    """
    Calculate risk scores for mental health and brain diseases.
    """
    risk_scores = {
        "anxiety": 0,
        "depression": 0,
        "psychosis": 0,
        "alzheimers": 0,
        "dementia": 0,
    }
    total_entries = len(entries)

    for entry in entries:
        doc = nlp(entry["entry"])
        
        # Analyze keywords and sentiment
        for token in doc:
            if token.text.lower() in ["anxious", "stress", "worry", "overwhelmed"]:
                risk_scores["anxiety"] += 1
            if token.text.lower() in ["sad", "depressed", "hopeless", "crying"]:
                risk_scores["depression"] += 1
            if token.text.lower() in ["delusion", "hallucination", "paranoid"]:
                risk_scores["psychosis"] += 1

        # Alzheimer's and dementia risk based on indicative keywords
        if len(doc.ents) > 5:  # Example heuristic
            risk_scores["alzheimers"] += 1
        if len([t for t in doc if t.text.lower() in ["forget", "memory", "confused"]]) > 2:
            risk_scores["dementia"] += 1

    # Normalize scores as percentages
    for key in risk_scores:
        risk_scores[key] = (risk_scores[key] / total_entries) * 100

    return risk_scores


def calculate_brain_health_index(dementia, alzheimers):
    """
    Calculate Brain Health Index as 100 - average of dementia and Alzheimer's risk.
    """
    return 100 - ((dementia + alzheimers) / 2)


def calculate_mood_stability(happiness_scores):
    """
    Calculate Mood Stability based on the standard deviation of happiness scores.
    """
    if len(happiness_scores) < 2:
        return 100  # Default for single or no entries
    std_dev = statistics.stdev(happiness_scores)
    return max(0, 100 - (std_dev * 10))


def calculate_trend_analysis(happiness_scores):
    """
    Analyze trends in happiness scores over time.
    """
    if len(happiness_scores) < 2:
        return "No sufficient data for trend analysis."

    delta = happiness_scores[-1] - happiness_scores[0]
    if delta > 0:
        return "Positive trend in happiness over time."
    elif delta < 0:
        return "Negative trend in happiness over time."
    else:
        return "No significant change in happiness over time."


def generate_strength_based_feedback(risks):
    """
    Provide strengths based on areas with lower risks.
    """
    strengths = []
    for key, value in risks.items():
        if key in ["brainHealthIndex", "moodStability", "trendAnalysis"]:
            continue  # Skip non-risk categories

        if value < 30:  # Arbitrary threshold for "low risk"
            strengths.append(f"Low risk of {key.capitalize()}.")
    return strengths
