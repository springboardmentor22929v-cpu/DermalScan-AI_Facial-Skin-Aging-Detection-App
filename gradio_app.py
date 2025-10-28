# File: gradio_app.py (in your project root: dermalscan-project/)

import gradio as gr
import requests
import base64
import io
from PIL import Image
import os # For example paths

# URL of your running Flask API (adjust if you change port or host)
FLASK_API_URL = "http://127.0.0.1:5000/predict"

def predict_image(image_pil):
    if image_pil is None:
        return "Please upload an image.", None

    # Convert PIL image to bytes
    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    # Prepare the file for the POST request
    files = {'image': ('image.jpg', image_bytes, 'image/jpeg')}

    try:
        # Send the image to your Flask API
        response = requests.post(FLASK_API_URL, files=files)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        result = response.json()

        if result.get("status") == "success":
            # Decode the base64 annotated image
            encoded_image = result["annotated_image"]
            decoded_image_bytes = base64.b64decode(encoded_image)
            annotated_image = Image.open(io.BytesIO(decoded_image_bytes))

            # Format predictions for display
            predictions_text = ""
            if result["predictions"]:
                for pred_group in result["predictions"]:
                    # Display top predictions for aging signs
                    predictions_text += "Aging Signs:\n"
                    if "top_predictions" in pred_group:
                        for p in pred_group["top_predictions"]:
                            predictions_text += f"- {p['label']}: {p['confidence']*100:.1f}%\n"
                    
                    # Display predicted age
                    predicted_age = pred_group.get('predicted_age', 'N/A')
                    predictions_text += f"Age: {predicted_age}\n"
                    predictions_text += "----------------------------------\n" # Separator for multiple faces
                
                # Remove last separator if only one face
                if len(result["predictions"]) == 1:
                    predictions_text = predictions_text.rstrip("----------------------------------\n")

            else:
                predictions_text = "No predictions (e.g., no face detected)."


            return predictions_text, annotated_image
        else:
            return f"API Error: {result.get('error', 'Unknown error from API')}", None

    except requests.exceptions.ConnectionError:
        return f"Connection Error: Is your Flask API running at {FLASK_API_URL}?", None
    except requests.exceptions.RequestException as e:
        return f"Request Error: {e}", None
    except Exception as e:
        return f"An unexpected error occurred: {e}", None

# Gradio Interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=[
        gr.Textbox(label="Prediction Results", lines=10), # Increased lines to show more predictions
        gr.Image(type="pil", label="Annotated Image")
    ],
    title="DermalScan: AI Facial Skin Aging Detection App",
    description="Upload a facial image to detect and classify aging signs (wrinkles, dark spots, puffy eyes, clear skin) and predict age.",
    examples=[
        # Add paths to some test images from your data_split_fixed/test folder here
        # Example: os.path.join("data_split_fixed", "test", "clear skin", "image_name_orig.jpg")
        # Ensure these example images exist in your project root or relative path
    ]
)

iface.launch(share=False) # Set share=True to get a public link (temporarily)