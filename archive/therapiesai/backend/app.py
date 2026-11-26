from flask import Flask
from flask_cors import CORS
from routes.nlp_routes import nlp_routes
from routes.mental_health_routes import mental_health_routes
from services.firebase_service import initialize_firebase
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")

# Initialize Firebase
initialize_firebase()

# Create Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)

# Register routes
app.register_blueprint(nlp_routes)
app.register_blueprint(mental_health_routes)

@app.route("/")
def home():
    return "Welcome to the TherapiesAI Backend!"

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, port=port)
