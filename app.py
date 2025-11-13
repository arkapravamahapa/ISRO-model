import streamlit as st
from PIL import Image
import requests
import io
import base64  
import json    

# ---------------------------------------------------------------------
# AI "Brain" - Using the REAL, working API URL
# ---------------------------------------------------------------------

# THIS IS THE REAL, WORKING URL that other developers found.
# We put the model name directly in the URL.
API_URL = "https://router.huggingface.co/hf-inference/models/dandelin/vilt-b32-finetuned-vqa"

def query_api(image_b64, question, api_key):
    """Sends the image (as base64 text) and question to the real API."""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # The payload is now simpler because the model name is in the URL
    payload = {
        "inputs": {
            "image": image_b64,
            "question": question
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    
    try:
        json_response = response.json()
        if "error" in json_response:
            return f"Error from API: {json_response['error']}"
        
        # Get the best answer
        best_answer = json_response[0]['label']
        return f"{best_answer} (Confidence: {json_response[0]['score']:.2f})"
        
    except Exception as e:
        return f"Error processing API response: {e}. Response text: {response.text}"

# ---------------------------------------------------------------------
# Your app's UI code (No changes needed here)
# ---------------------------------------------------------------------

st.set_page_config(page_title="Hackathon VQA")
st.title("👁️ G-V-OSS: Visual Question Answering")
st.write("Our simplified hackathon demo (now powered by the fast API router!)")

st.subheader("Enter Your Hugging Face API Key")
st.warning("Your key is needed to talk to the free AI model.")
api_key = st.text_input("Paste your 'hf_...' key here:", type="password")

st.divider()

uploaded_file = st.file_uploader("Upload a satellite image (or any image)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your Uploaded Image", width='stretch')
    
    st.subheader("Ask a question about this image:")
    question = st.text_input("e.g., 'Are there any rivers visible?'")

    if question:
        if not api_key:
            st.error("Please enter your API key above to get an answer.")
        else:
            st.write("...AI is thinking (this will be fast!)...")
            
            # --- Convert image to Base64 text ---
            with io.BytesIO() as img_buffer:
                image.save(img_buffer, format="PNG")
                img_bytes = img_buffer.getvalue()
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')

            # Call our API function
            answer = query_api(img_b64, question, api_key)
            
            st.subheader("AI Answer:")
            st.write(answer)