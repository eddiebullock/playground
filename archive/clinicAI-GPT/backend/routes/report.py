from flask import Blueprint, request, jsonify
from sections.assessment import generate_assessment_report
from sections.communication import generate_communication_report
from sections.reciprocal_social_interaction import generate_rsi_report
from sections.rrb import generate_rrb_report
import openai
import os
import re

openai.api_key = os.getenv("OPENAI_API_KEY")

report_bp = Blueprint("report", __name__)

@report_bp.route("/generate_report", methods=["POST"])
def generate_report():
    try:
        # Retrieve clinician's notes from the request
        data = request.json
        anonymized_text = data.get("anonymized_text", "")

        if not anonymized_text.strip():
            return jsonify({"error": "No text provided"}), 400

        # Log the received input
        print(f"Anonymized text received: {anonymized_text}")

        # Generate reports for each section
        assessment = generate_assessment_report(anonymized_text)
        communication = generate_communication_report(anonymized_text)
        rsi = generate_rsi_report(anonymized_text)
        rrb = generate_rrb_report(anonymized_text)

        # Combine all sections into a final report with improved formatting
        full_report = f"""
        <h1 style="text-align:center;">Autism Assessment Report</h1>

        <h2>Background and Key Topics</h2>
        {format_subsections(assessment)}

        <h2>Communication</h2>
        {format_subsections(communication)}

        <h2>Reciprocal Social Interaction</h2>
        {format_subsections(rsi)}

        <h2>Restricted and Repetitive Behaviors</h2>
        {format_subsections(rrb)}
        """

        print("Full report generated successfully.")  # Debug log
        return jsonify({"message": "Report generated successfully", "report": full_report}), 200

    except Exception as e:
        print("Error generating report:", str(e))
        return jsonify({"error": f"Failed to generate report: {str(e)}"}), 500

def format_subsections(section_text):
    """
    Formats subsections to ensure proper spacing and paragraph separation.
    """
    formatted_text = section_text

    # Ensure subheadings are recognized and bolded properly
    formatted_text = re.sub(r"\*\*(.*?)\*\*", r"<h3>\1</h3>\n", formatted_text)

    # Ensure numbered list items have a newline before them
    formatted_text = re.sub(r"(\d+\.)", r"\n\1 ", formatted_text)

    # Ensure each paragraph is wrapped properly
    paragraphs = formatted_text.split("\n")
    formatted_text = "".join(
        f"<p>{para.strip()}</p>\n" if not para.strip().startswith("<h3>") else para for para in paragraphs
    )

    return formatted_text


