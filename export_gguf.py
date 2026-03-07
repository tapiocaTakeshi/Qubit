import torch
import struct
import numpy as np
import sys
from gguf import GGUFWriter

def pt_to_gguf(pt_file, out_file):
    print(f"Loading {pt_file}...")
    checkpoint = torch.load(pt_file, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    print(f"Writing GGUF to {out_file}...")
    # Initialize GGUFWriter with architecture name
    writer = GGUFWriter(out_file, "qbnn")
    
    # Optional metadata
    writer.add_name("QBNN Model")
    writer.add_description("Quantum Bit Neural Network Model by tapiocaTakeshi")
    
    count = 0
    for name, tensor in state_dict.items():
        # Convert torch tensor to numpy array (must be contiguous)
        data = np.ascontiguousarray(tensor.float().numpy())
        writer.add_tensor(name, data)
        count += 1
        
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    print(f"Successfully exported {count} tensors to {out_file}.")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        pt_to_gguf(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python export_gguf.py <input.pt> <output.gguf>")
        # Default execution for the specific project file
        pt_to_gguf("neuroq_checkpoint.pt", "neuroq.gguf")
