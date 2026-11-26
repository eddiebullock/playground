from flask import Blueprint, request, jsonify
from services.anonymizer import anonymize_text_with_huggingface

# Blueprint for the upload route
upload_bp = Blueprint("upload", __name__)

@upload_bp.route("/api/anonymize", methods=["OPTIONS", "POST"])
def anonymize_text():
    if request.method == "OPTIONS":
        # Allow preflight requests
        response = jsonify({"message": "Preflight check successful"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        return response, 200

    data = request.json
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    try:
        anonymized_text = anonymize_text_with_huggingface(text)
        return jsonify({
            "message": "Text anonymized successfully",
            "anonymized_text": anonymized_text
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to anonymize text: {str(e)}"}), 500
