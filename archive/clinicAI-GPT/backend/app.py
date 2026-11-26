from flask import Flask
from routes.upload import upload_bp
from routes.report import report_bp  # Import the report blueprint
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS for all routes and origins
CORS(app)

# Register the blueprints
app.register_blueprint(upload_bp)
app.register_blueprint(report_bp, url_prefix="/api")  # Add a prefix for the report API

if __name__ == "__main__":
    app.run(debug=True)
