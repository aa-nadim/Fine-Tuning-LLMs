from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from gradio_client import Client
import os
import io
import base64
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set Hugging Face token
os.environ['HF_TOKEN'] = ''

# Create the inference client
client = Client("black-forest-labs/FLUX.1-schnell")

@app.route('/')
def home():
    return jsonify({
        'status': 'running',
        'message': 'Image generation API is running'
    })

@app.route('/api/generate-image', methods=['POST'])
def generate_image():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        width = data.get('width', 1024)
        height = data.get('height', 1024)
        
        if not prompt:
            return jsonify({
                'success': False,
                'error': 'Prompt is required'
            }), 400
        
        # Generate image
        result = client.predict(
            prompt=prompt,
            seed=0,
            randomize_seed=True,
            width=width,
            height=height,
            num_inference_steps=4
        )
        
        # Ensure result[0] is a valid URL or file path
        try:
            if isinstance(result[0], str) and (result[0].startswith('http://') or result[0].startswith('https://')):
                response = requests.get(result[0])
                response.raise_for_status()  # Raise an exception for bad status codes
                img = Image.open(io.BytesIO(response.content))
            else:
                img = Image.open(result[0])

            # Convert image to base64
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            
            return jsonify({
                'success': True,
                'image': f'data:image/png;base64,{img_base64}'
            })
        except requests.exceptions.RequestException as e:
            return jsonify({
                'success': False,
                'error': f'Failed to fetch image: {str(e)}'
            }), 500
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to process image: {str(e)}'
            }), 500
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)