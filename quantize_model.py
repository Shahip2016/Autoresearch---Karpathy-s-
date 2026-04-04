import torch
import argparse
import os
from train import GPT

def quantize_model(checkpoint_path, out_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    original_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"Original model size: {original_size:.2f} MB")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = GPT()
    
    # Check if 'model' key exists
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    print("Quantizing model dynamically (qint8) for nn.Linear layers...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    print(f"Saving quantized model to {out_path}")
    torch.save(quantized_model.state_dict(), out_path)
    quantized_size = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size / original_size) * 100:.1f}%")
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Quantize PyTorch model for faster CPU inference.")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to input .pt checkpoint")
    parser.add_argument('--out', type=str, default='quantized_model.pt', help="Path to output .pt quantized checkpoint")
    args = parser.parse_args()
    
    quantize_model(args.ckpt, args.out)
