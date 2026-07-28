"""
Chatbot HTTP server - general purpose Japanese conversation.
Run: python chatbot_server.py
Then open http://localhost:5001 in browser
"""
import torch
import torch.nn as nn
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
import pickle
from chatbot_train import ChatbotQBNN
from chatbot_dataset import CONVERSATIONS, RESPONSE_CATEGORIES

# Load model and vectorizer
device = "cpu"
model = ChatbotQBNN(input_dim=128, hidden_dim=64, n_classes=12).to(device)
model.load_state_dict(torch.load("chatbot_qbnn.pkl", map_location=device))
model.eval()

with open("chatbot_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Build response dict
response_dict = {}
for question, category, response_text in CONVERSATIONS:
    if category not in response_dict:
        response_dict[category] = []
    response_dict[category].append(response_text)

def get_response(user_text):
    """Get chatbot response and quantum state info."""
    try:
        # Vectorize
        x = vectorizer.transform([user_text]).toarray().astype('float32')
        x_tensor = torch.from_numpy(x).to(device)

        # Get prediction
        with torch.no_grad():
            # Get intermediate layer output for quantum info
            h = x_tensor
            for layer, act in zip(model.layers, model.activations):
                h = layer(h)
                h = act(h)

            logits = model.head(h)
            probs = torch.softmax(logits, dim=1)
            pred_cat = probs.argmax(dim=1).item()
            confidence = probs[0, pred_cat].item()

            # Extract APQB info
            first_layer = model.layers[0]
            a = first_layer.P(x_tensor)
            a = torch.clamp(a, -first_layer.a_clip, first_layer.a_clip)
            r = torch.tanh(a)
            q = 1.0 / torch.cosh(a)
            r_vals = r.numpy()[0]
            q_vals = q.numpy()[0]

        # Get response text (random from category)
        import random
        response_text = random.choice(response_dict.get(pred_cat, ["（応答生成中...）"]))

        return {
            'response': response_text,
            'category': RESPONSE_CATEGORIES[pred_cat],
            'confidence': float(confidence),
            'r_mean': float(r_vals.mean()),
            'q_mean': float(q_vals.mean()),
            'coherence': float(q_vals.mean()),
        }
    except Exception as e:
        return {
            'error': str(e),
            'response': '申し訳ありません。もう一度試していただけますか？'
        }

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>💬 APQB チャットボット</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Hiragino Sans", sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            width: 100%;
            max-width: 750px;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
            height: 90vh;
            max-height: 900px;
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
            gap: 12px;
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

        .user-msg {
            justify-content: flex-end;
        }

        .user-msg .content {
            background: #667eea;
            color: white;
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 70%;
            word-wrap: break-word;
        }

        .bot-msg .content {
            background: #f0f0f0;
            color: #333;
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 75%;
        }

        .quantum-badge {
            font-size: 11px;
            margin-top: 8px;
            padding: 6px 10px;
            background: #f5f5f5;
            border-radius: 6px;
            border-left: 3px solid #667eea;
            font-family: monospace;
        }

        .quantum-badge div {
            margin: 3px 0;
        }

        .category-tag {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            margin-right: 6px;
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
            animation: pulse 1s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }

        .welcome {
            background: #f9f9f9;
            border-left: 4px solid #667eea;
            padding: 12px;
            border-radius: 6px;
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>💬 APQB チャットボット</h1>
            <p>量子インスパイアードニューラルネットワーク</p>
        </div>

        <div class="chat-box" id="chatBox">
            <div class="message bot-msg">
                <div class="content">
                    <div class="welcome">
                        <strong>こんにちは！👋</strong><br>
                        私は QBNN という量子インスパイアード神経回路ネットワークです。
                        何でもお話しください。天気の話から、趣味の話まで何でもOK！
                    </div>
                </div>
            </div>
        </div>

        <div class="input-area">
            <input type="text" id="userInput" placeholder="何か話しかけてください..." autocomplete="off">
            <button onclick="sendMessage()">送信</button>
        </div>
    </div>

    <script>
        const chatBox = document.getElementById('chatBox');
        const userInput = document.getElementById('userInput');

        function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;

            // User message
            const userDiv = document.createElement('div');
            userDiv.className = 'message user-msg';
            userDiv.innerHTML = `<div class="content">${escapeHtml(message)}</div>`;
            chatBox.appendChild(userDiv);

            userInput.value = '';
            chatBox.scrollTop = chatBox.scrollHeight;

            // Loading
            const loadDiv = document.createElement('div');
            loadDiv.className = 'message bot-msg';
            loadDiv.innerHTML = `<div class="content"><span class="loading">考え中...</span></div>`;
            chatBox.appendChild(loadDiv);
            chatBox.scrollTop = chatBox.scrollHeight;

            // Send
            fetch('/api/chat?message=' + encodeURIComponent(message))
            .then(r => r.json())
            .then(data => {
                loadDiv.remove();

                const botDiv = document.createElement('div');
                botDiv.className = 'message bot-msg';

                let html = `<div class="content">`;
                if (data.category) {
                    html += `<span class="category-tag">${escapeHtml(data.category)}</span>`;
                }
                html += `<div>${escapeHtml(data.response)}</div>`;

                if (data.r_mean !== undefined) {
                    html += `<div class="quantum-badge">
                        <div>r: ${data.r_mean.toFixed(3)} | q: ${data.q_mean.toFixed(3)}</div>
                        <div>確信度: ${(data.confidence * 100).toFixed(0)}%</div>
                    </div>`;
                }

                html += `</div>`;
                botDiv.innerHTML = html;
                chatBox.appendChild(botDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
            })
            .catch(err => {
                loadDiv.remove();
                const errDiv = document.createElement('div');
                errDiv.className = 'message bot-msg';
                errDiv.innerHTML = `<div class="content">申し訳ありません。エラーが発生しました。</div>`;
                chatBox.appendChild(errDiv);
            });
        }

        function escapeHtml(text) {
            const map = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'};
            return text.replace(/[&<>"']/g, m => map[m]);
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
        elif self.path.startswith('/api/chat'):
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            message = params.get('message', [''])[0]

            if not message:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Empty message'}).encode('utf-8'))
            else:
                result = get_response(message)
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(result, ensure_ascii=False).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

if __name__ == '__main__':
    print("Starting Chatbot Server...")
    print("Open http://localhost:5001 in your browser")
    server = HTTPServer(('localhost', 5001), RequestHandler)
    server.serve_forever()
