from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import torch
import argparse
from train import GPT
import os
import pickle

class ModelHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)
        
        prompt = data.get('prompt', '')
        max_new_tokens = data.get('max_new_tokens', 100)
        
        # Simple character-level encoding
        context = [self.server.stoi.get(c, 0) for c in prompt]
        context = torch.tensor(context, dtype=torch.long, device=self.server.device).unsqueeze(0)
        
        if context.shape[1] == 0:
            context = torch.zeros((1, 1), dtype=torch.long, device=self.server.device)
            
        with torch.no_grad():
            generated = self.server.model.generate(context, max_new_tokens=max_new_tokens)
            
        generated_text = ''.join([self.server.itos[i] for i in generated[0].tolist()])
        
        response = {'generated_text': generated_text}
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))

def serve(ckpt_path, port, device='cpu'):
    print(f"Loading checkpoint from {ckpt_path} onto {device}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = GPT()
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    print("Model loaded.")
    
    # Load meta for vocab
    meta_path = os.path.join(os.path.dirname(__file__), 'data', 'tinyshakespeare_meta.pkl')
    if not os.path.exists(meta_path):
        meta_path = os.path.join(os.path.dirname(__file__), 'data', 'meta.pkl')
        
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi = meta['stoi']
        itos = meta['itos']
    else:
        print("Warning: meta.pkl not found, using dummy vocabulary.")
        stoi = {}
        itos = {i: chr(i) for i in range(256)}
        
    server = HTTPServer(('0.0.0.0', port), ModelHandler)
    server.model = model
    server.device = device
    server.stoi = stoi
    server.itos = itos
    
    print(f"Starting server on port {port}...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.server_close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Serve the model via a simple HTTP API.")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to checkpoint")
    parser.add_argument('--port', type=int, default=8000, help="Port to bind to")
    parser.add_argument('--device', type=str, default='cpu', help="Device (cpu or cuda)")
    args = parser.parse_args()
    
    serve(args.ckpt, args.port, args.device)
