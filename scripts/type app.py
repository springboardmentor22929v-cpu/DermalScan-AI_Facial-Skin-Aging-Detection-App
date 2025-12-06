# File: app.py (in your project root: dermalscan-project/)

from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
import base64
# IMPORTANT: Ensure scripts/inference_model.py exists and is correct
from scripts.inference_model import predict_aging_signs
import os

app = Flask(__name__)

# Set a max content length to prevent large file uploads (e.g., 16 MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
def index():
    return "DermalScan API is running! Send POST request to /predict for predictions."

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected image file."}), 400

    if image_file:
        try:
            # Read the image file into a PIL Image object
            image_bytes = image_file.read()
            input_image_pil = Image.open(io.BytesIO(image_bytes))

            # Call your prediction function
            annotated_image_pil, predictions = predict_aging_signs(input_image_pil)

            if annotated_image_pil is None:
                return jsonify({"error": predictions[0].get("error", "Model not loaded correctly.")}), 500

            # Convert the annotated PIL Image back to bytes (JPEG format) for sending to frontend
            buffered = io.BytesIO()
            annotated_image_pil.save(buffered, format="JPEG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return jsonify({
                "status": "success",
                "predictions": predictions,
                "annotated_image": encoded_image # Base64 encoded annotated image
            }), 200

        except Exception as e:
            app.logger.error(f"Prediction failed: {e}")
            return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

    return jsonify({"error": "Unknown error processing image."}), 500

if __name__ == '__main__':
    # Ensure the model is loaded during app startup
    # This prevents loading the model on every request, improving performance
    # The inference_model.py already handles initial loading at import time.
    # We just need to make sure 'model' in inference_model is not None.
    from scripts.inference_model import model as loaded_model
    if loaded_model is None:
        print("WARNING: Model failed to load during app startup. Predictions will not work.")
        # You might want to exit or handle this more gracefully in production
    
    app.run(debug=True, port=5000) # debug=True is good for development