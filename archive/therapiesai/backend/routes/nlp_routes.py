from flask import Blueprint, request, jsonify
from services.firebase_service import initialize_firebase, save_journal_entry
from services.nlp_service import analyze_entry

nlp_routes = Blueprint("nlp_routes", __name__)

# Initialize Firebase once when the server starts
initialize_firebase()

@nlp_routes.route('/add-entry', methods=['POST'])
def add_entry():
    """
    Handle POST request to analyze and save a journal entry.
    """
    try:
        data = request.json
        print("Received POST request with data:", data)  # Debug frontend data

        user_id = data.get('userId')
        entry = data.get('entry')
        happiness = data.get('happiness')

        if not user_id or not entry or happiness is None:
            print("Error: Missing required fields")  # Debug missing fields
            return jsonify({"error": "Missing required fields"}), 400

        # Analyze the journal entry using the NLP service
        print("Analyzing entry with Hugging Face pipelines...")  # Debug analysis call
        analysis_result = analyze_entry(entry)
        if not analysis_result:
            return jsonify({"error": "Error analyzing entry"}), 500

        themes = analysis_result.get("themes", [])
        sentiment = analysis_result.get("sentiment", "neutral")

        print(f"Extracted themes: {themes}, sentiment: {sentiment}")  # Debug before saving to Firestore

        # Save to Firestore
        print("Saving to Firestore...")  # Debug Firestore save
        result = save_journal_entry(user_id, entry, happiness, themes, sentiment)
        return jsonify(result), 200 if "message" in result else 500
    except Exception as e:
        print(f"Error processing entry: {e}")  # Debug errors
        return jsonify({"error": "Internal server error"}), 500
