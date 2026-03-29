import os
import torch
import argparse
from train import GPT, DEVICE, BLOCK_SIZE

def export_to_onnx(checkpoint_path, onnx_path):
    """
    Exports a PyTorch model checkpoint to ONNX format.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint '{checkpoint_path}' not found.")
        return

    print(f"[*] Loading checkpoint from {checkpoint_path}...")
    
    # Initialize model with current architecture from train.py
    model = GPT().to(DEVICE)
    
    # Load state dict
    try:
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        
        # Strip torch.compile prefix if present
        unwanted_prefix = '_orig_mod.'
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(unwanted_prefix):
                new_state_dict[k[len(unwanted_prefix):]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        print("[+] Model weights loaded successfully.")
    except Exception as e:
        print(f"[-] Error loading state dict: {e}")
        return

    model.eval()

    # Create dummy input for tracing (Batch Size 1, Sequence Length BLOCK_SIZE)
    dummy_input = torch.zeros((1, BLOCK_SIZE), dtype=torch.long, device=DEVICE)
    
    print(f"[*] Exporting to ONNX: {onnx_path}...")
    try:
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_path, 
            export_params=True,
            opset_version=14, 
            do_constant_folding=True,
            input_names=['input'], 
            output_names=['logits'],
            dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'}}
        )
        print(f"[+] Export successful! Saved to {onnx_path}")
    except Exception as e:
        print(f"[-] Export failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoResearch Model Export (PT -> ONNX)")
    parser.add_argument("--checkpoint", type=str, default="out/best_model.pt", 
                        help="Path to the PyTorch checkpoint (default: out/best_model.pt)")
    parser.add_argument("--output", type=str, default="out/model.onnx", 
                        help="Path for the exported ONNX file (default: out/model.onnx)")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    export_to_onnx(args.checkpoint, args.output)
