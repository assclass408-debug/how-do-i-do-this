import os
from flask import Flask, request, jsonify, Response
import requests
import json
import time

app = Flask(__name__)

# Configuration - Railway will provide these as environment variables
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY', '')
NVIDIA_BASE_URL = os.environ.get('NVIDIA_BASE_URL', 'https://integrate.api.nvidia.com/v1')
PORT = int(os.environ.get('PORT', 5000))

# Model mapping (OpenAI model names to NVIDIA NIM model names)
MODEL_MAPPING = {
    "gpt-3.5-turbo": "meta/llama-3.1-8b-instruct",
    "gpt-4": "meta/llama-3.1-70b-instruct",
    "gpt-4-turbo": "meta/llama-3.1-405b-instruct",
}

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        if not NVIDIA_API_KEY:
            return jsonify({"error": "NVIDIA_API_KEY not configured"}), 500
            
        data = request.json
        
        # Get the model and map it to NVIDIA model
        openai_model = data.get('model', 'gpt-3.5-turbo')
        nvidia_model = MODEL_MAPPING.get(openai_model, MODEL_MAPPING['gpt-3.5-turbo'])
        
        # Check if streaming is requested
        stream = data.get('stream', False)
        
        # Prepare NVIDIA NIM request
        nvidia_payload = {
            "model": nvidia_model,
            "messages": data.get('messages', []),
            "temperature": data.get('temperature', 0.7),
            "top_p": data.get('top_p', 1.0),
            "max_tokens": data.get('max_tokens', 1024),
            "stream": stream
        }
        
        # Make request to NVIDIA NIM
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        if stream:
            return handle_streaming_response(nvidia_payload, headers, openai_model)
        else:
            return handle_non_streaming_response(nvidia_payload, headers, openai_model)
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def handle_non_streaming_response(nvidia_payload, headers, openai_model):
    """Handle non-streaming responses"""
    response = requests.post(
        f"{NVIDIA_BASE_URL}/chat/completions",
        headers=headers,
        json=nvidia_payload,
        timeout=60
    )
    
    if response.status_code != 200:
        return jsonify({"error": response.text}), response.status_code
    
    nvidia_response = response.json()
    
    # Convert NVIDIA response to OpenAI format
    openai_response = {
        "id": nvidia_response.get("id", f"chatcmpl-{int(time.time())}"),
        "object": "chat.completion",
        "created": nvidia_response.get("created", int(time.time())),
        "model": openai_model,
        "choices": nvidia_response.get("choices", []),
        "usage": nvidia_response.get("usage", {})
    }
    
    return jsonify(openai_response)

def handle_streaming_response(nvidia_payload, headers, openai_model):
    """Handle streaming responses"""
    def generate():
        response = requests.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            headers=headers,
            json=nvidia_payload,
            stream=True,
            timeout=60
        )
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        yield f"data: [DONE]\n\n"
                        break
                    
                    try:
                        data = json.loads(data_str)
                        # Convert to OpenAI format if needed
                        if 'model' in data:
                            data['model'] = openai_model
                        yield f"data: {json.dumps(data)}\n\n"
                    except json.JSONDecodeError:
                        continue
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models in OpenAI format"""
    models = [
        {
            "id": model,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nvidia"
        }
        for model in MODEL_MAPPING.keys()
    ]
    
    return jsonify({
        "object": "list",
        "data": models
    })

@app.route('/health', methods=['GET'])
@app.route('/', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "service": "OpenAI to NVIDIA NIM Proxy",
        "api_key_configured": bool(NVIDIA_API_KEY)
    })

if __name__ == '__main__':
    print("Starting OpenAI to NVIDIA NIM Proxy Server...")
    print(f"Server will run on port {PORT}")
    print(f"API Key configured: {bool(NVIDIA_API_KEY)}")
    app.run(host='0.0.0.0', port=PORT)