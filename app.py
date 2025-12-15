import google.generativeai as genai
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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