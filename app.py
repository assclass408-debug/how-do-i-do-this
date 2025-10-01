from flask import Flask, request, jsonify, Response
import requests
import json
import os

app = Flask(__name__)

# Get NVIDIA NIM API key from environment variable
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY', '')
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.json
        
        # Extract OpenAI format parameters
        messages = data.get('messages', [])
        model = data.get('model', 'meta/llama-3.1-8b-instruct')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 1024)
        stream = data.get('stream', False)
        
        # Prepare NVIDIA NIM request
        nim_request = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # Forward to NVIDIA NIM
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        if stream:
            return handle_stream(nim_request, headers)
        else:
            return handle_non_stream(nim_request, headers)
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def handle_non_stream(nim_request, headers):
    response = requests.post(
        f"{NVIDIA_BASE_URL}/chat/completions",
        headers=headers,
        json=nim_request
    )
    
    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify({"error": response.text}), response.status_code

def handle_stream(nim_request, headers):
    def generate():
        response = requests.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            headers=headers,
            json=nim_request,
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                yield line + b'\n'
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models"""
    models = {
        "object": "list",
        "data": [
            {
                "id": "meta/llama-3.1-8b-instruct",
                "object": "model",
                "created": 1677610602,
                "owned_by": "nvidia"
            },
            {
                "id": "meta/llama-3.1-70b-instruct",
                "object": "model",
                "created": 1677610602,
                "owned_by": "nvidia"
            },
            {
                "id": "mistralai/mixtral-8x7b-instruct-v0.1",
                "object": "model",
                "created": 1677610602,
                "owned_by": "nvidia"
            }
        ]
    }
    return jsonify(models)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)