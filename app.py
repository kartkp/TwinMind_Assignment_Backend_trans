import google.generativeai as genai
import os
import json
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def get_model():
    """Initialize and configure the Gemini AI model with API credentials."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY is missing")
        raise RuntimeError("GEMINI_API_KEY missing")

    genai.configure(api_key=api_key)
    
    logger.info("Initializing Gemini Model")
    
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash", 
        generation_config={"response_mime_type": "application/json"}
    )


model = get_model()

def analyze_transcript(transcript: str) -> dict:
    """
    Process a transcript using Gemini AI to extract structured information.
    Returns a dictionary with title, summary, action items, and key points.
    """
    prompt = f"""Extract key information from the transcript below.

Return a JSON object with these fields:
- title: A brief descriptive title
- summary: Short overview of main discussion
- action_items: List of tasks or next steps mentioned
- key_points: List of important takeaways
- make it with less than 150 words.

Content:
{transcript}"""

    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}. Response: {response.text}")
        return {
            "title": "Parsing Error",
            "summary": "Unable to parse response as JSON.",
            "raw_response": response.text,
            "action_items": [],
            "key_points": []
        }
    except Exception as e:
        logger.error(f"API Error: {e}")
        return {
            "title": "Error",
            "summary": "An error occurred during processing.",
            "action_items": [],
            "key_points": []
        }


@app.route("/", methods=["GET"])
def home():
    """API health check endpoint."""
    return jsonify({"message": "Server is running!", "status": "ok"}), 200


@app.route("/analyze", methods=["POST"])
def analyze():
    """Endpoint to analyze transcript."""
    try:
        data = request.get_json()
        transcript = data.get("transcript", "")
        
        if not transcript:
            return jsonify({"error": "Transcript is required"}), 400
        
        result = analyze_transcript(transcript)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error in /analyze endpoint: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("Starting Flask server on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)