"""
Simple HTTP server for QBNN Parity chatbot (no Flask required).
Run: python parity_chatbot_server.py
Then open http://localhost:5000 in browser
"""
import torch
import torch.nn as nn
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
import numpy as np
from parity_chatbot_train import ParityQBNN

# Load model
device = "cpu"
model = ParityQBNN(n_features=4, hidden_dim=32, n_layers=2).to(device)
model.load_state_dict(torch.load("parity_qbnn.pkl", map_location=device))
model.eval()

def parse_input(user_text):
    """Parse user input as 4-bit binary pattern."""
    text = user_text.strip().replace(" ", "").lower()
    bits = []
    for char in text:
        if char in ['0', '1']:
            bits.append(int(char))

    if len(bits) != 4:
        return None, f"Please provide exactly 4 bits (0 or 1). Got {len(bits)}."

    return np.array(bits, dtype=np.float32), None

def predict_parity(bits):
    """Get QBNN prediction and quantum info."""
    x = torch.from_numpy(bits.reshape(1, -1))

    with torch.no_grad():
        h = x
        for layer, act in zip(model.layers, model.activations):
            h = layer(h)
            h = act(h)
        output = model.head(h)
        pred = output.item()
        pred_class = int(pred > 0.5)
        confidence = max(pred, 1 - pred)

    first_layer = model.layers[0]
    with torch.no_grad():
        a = first_layer.P(x)
        a = torch.clamp(a, -first_layer.a_clip, first_layer.a_clip)
        r = torch.tanh(a)
        q = 1.0 / torch.cosh(a)
        r_vals = r.numpy()[0]
        q_vals = q.numpy()[0]

    return {
        'prediction': pred_class,
        'confidence': float(confidence),
        'probability': float(pred),
        'r_mean': float(r_vals.mean()),
        'q_mean': float(q_vals.mean()),
        'coherence_l1': float(q_vals.mean()),
        'r_samples': [float(x) for x in r_vals[:5]],
        'q_samples': [float(x) for x in q_vals[:5]],
    }

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🎭 APQB Parity Chatbot</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            width: 100%;
            max-width: 700px;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
            height: 90vh;
            max-height: 800px;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 16px 16px 0 0;
            text-align: center;
        }

        .header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }

        .header p {
            font-size: 13px;
            opacity: 0.9;
        }

        .chat-box {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .message {
            display: flex;
            gap: 10px;
            animation: fadeIn 0.3s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .user-message .content {
            background: #667eea;
            color: white;
            padding: 12px 16px;
            border-radius: 12px;
            margin-left: auto;
            max-width: 70%;
            word-wrap: break-word;
        }

        .bot-message .content {
            background: #f0f0f0;
            color: #333;
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 85%;
        }

        .quantum-info {
            font-size: 12px;
            margin-top: 8px;
            padding: 8px;
            background: #f9f9f9;
            border-radius: 8px;
            border-left: 3px solid #667eea;
            font-family: monospace;
        }

        .quantum-info div {
            margin: 4px 0;
        }

        .quantum-info .label {
            font-weight: bold;
            color: #667eea;
        }

        .input-area {
            padding: 20px;
            border-top: 1px solid #eee;
            display: flex;
            gap: 10px;
        }

        .input-area input {
            flex: 1;
            border: 2px solid #eee;
            border-radius: 8px;
            padding: 12px;
            font-size: 14px;
            transition: border-color 0.2s;
        }

        .input-area input:focus {
            outline: none;
            border-color: #667eea;
        }

        .input-area button {
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 14px;
            cursor: pointer;
            transition: background 0.2s;
        }

        .input-area button:hover {
            background: #5568d3;
        }

        .input-area button:active {
            transform: scale(0.98);
        }

        .loading {
            display: inline-block;
            animation: pulse 1s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }

        .info-box {
            background: #fffbea;
            border-left: 4px solid #ffa500;
            padding: 10px;
            border-radius: 4px;
            font-size: 12px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎭 APQB Parity Chatbot</h1>
            <p>Enter 4 bits (0/1) to classify XOR pattern</p>
        </div>

        <div class="chat-box" id="chatBox">
            <div class="message bot-message">
                <div class="content">
                    <div>Hi! I'm a quantum-inspired neural network trained on the <strong>parity (XOR)</strong> task.</div>
                    <div style="margin-top: 8px;">Try entering a 4-bit pattern like: <code>1 0 1 0</code> or <code>1010</code></div>
                    <div class="info-box" style="margin-top: 8px;">
                        I'll predict whether the XOR of the first 2 bits is 1 or 0, and show you my quantum state!
                    </div>
                </div>
            </div>
        </div>

        <div class="input-area">
            <input type="text" id="userInput" placeholder="Enter 4 bits (e.g., 1 0 1 0)" autocomplete="off">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        const chatBox = document.getElementById('chatBox');
        const userInput = document.getElementById('userInput');

        function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;

            const userDiv = document.createElement('div');
            userDiv.className = 'message user-message';
            userDiv.innerHTML = `<div class="content">${escapeHtml(message)}</div>`;
            chatBox.appendChild(userDiv);

            userInput.value = '';
            chatBox.scrollTop = chatBox.scrollHeight;

            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'message bot-message';
            loadingDiv.innerHTML = `<div class="content"><span class="loading">Thinking...</span></div>`;
            chatBox.appendChild(loadingDiv);
            chatBox.scrollTop = chatBox.scrollHeight;

            fetch('/api/predict?message=' + encodeURIComponent(message))
            .then(r => r.json())
            .then(data => {
                loadingDiv.remove();

                if (data.error) {
                    const errDiv = document.createElement('div');
                    errDiv.className = 'message bot-message';
                    errDiv.innerHTML = `<div class="content" style="color: #d32f2f;">❌ ${escapeHtml(data.error)}</div>`;
                    chatBox.appendChild(errDiv);
                } else {
                    const botDiv = document.createElement('div');
                    botDiv.className = 'message bot-message';

                    const prediction = data.prediction === 1 ? '✅ 1 (XOR is TRUE)' : '⭕ 0 (XOR is FALSE)';
                    const confidence = (data.confidence * 100).toFixed(1);

                    botDiv.innerHTML = `
                        <div class="content">
                            <div><strong>Prediction:</strong> ${prediction}</div>
                            <div><strong>Confidence:</strong> ${confidence}%</div>
                            <div class="quantum-info">
                                <div><span class="label">Quantum State:</span></div>
                                <div>  r (correlation): ${data.r_mean.toFixed(3)}</div>
                                <div>  q (coherence): ${data.q_mean.toFixed(3)}</div>
                                <div>  C_l1 (coherence norm): ${data.coherence_l1.toFixed(3)}</div>
                            </div>
                        </div>
                    `;
                    chatBox.appendChild(botDiv);
                }

                chatBox.scrollTop = chatBox.scrollHeight;
            })
            .catch(err => {
                loadingDiv.remove();
                const errDiv = document.createElement('div');
                errDiv.className = 'message bot-message';
                errDiv.innerHTML = `<div class="content" style="color: #d32f2f;">❌ Connection error</div>`;
                chatBox.appendChild(errDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
            });
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        userInput.addEventListener('keypress', e => {
            if (e.key === 'Enter') sendMessage();
        });

        userInput.focus();
    </script>
</body>
</html>
"""

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
        elif self.path.startswith('/api/predict'):
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            message = params.get('message', [''])[0]

            bits, error = parse_input(message)
            if error:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': error}).encode('utf-8'))
            else:
                result = predict_parity(bits)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

if __name__ == '__main__':
    print("Starting Parity Chatbot Server...")
    print("Open http://localhost:5000 in your browser")
    server = HTTPServer(('localhost', 5000), RequestHandler)
    server.serve_forever()
