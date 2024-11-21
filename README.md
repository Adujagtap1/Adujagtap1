from flask import Flask, request, jsonify
from ibm_watson import Natural LanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# Initialize Flask App
app = Flask(__name__)

# Watson NLP Configuration
API_KEY = "your-api-key-here"  # Replace with your actual IBM Watson API Key
SERVICE_URL = "your-service-url-here"  # Replace with your Watson Service URL

# Emotion Detection Function
def emotion_predictor(api_key, service_url, text):
    """
    Function to detect emotions using Watson NLP.
    Args:
        api_key: Watson API Key
        service_url: Watson Service URL
        text: Input text to analyze
    Returns:
        dict: Emotion analysis or error message
    """
    if not text.strip():
        return {"status": "error", "message": "Empty input provided"}
    try:
        authenticator = IAMAuthenticator(api_key)
        nlp_service = Natural LanguageUnderstandingV1(
            version='2023-04-01',
            authenticator=authenticator
        )
        nlp_service.set_service_url(service_url)

        response = nlp_service.analyze(
            text=text,
            features=Features(emotion=EmotionOptions())
        ).get_result()

        emotions = response['emotion']['document']['emotion']
        return {"status": "success", "data": emotions}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Flask Route for Emotion Detection
@app.route('/predict', methods=['POST'])
def predict_emotions():
    """
    Flask route to handle emotion detection requests.
    """
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({"status": "error", "message": "Input text is required"}), 400

    result = emotion_predictor(API_KEY, SERVICE_URL, text)
    if result["status"] == "error":
        return jsonify(result), 400
    return jsonify(result)

# Run Flask Server
if __name__ == '__main__':
    app.run(debug=True)

