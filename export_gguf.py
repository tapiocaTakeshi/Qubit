import torch
import struct
import numpy as np
import sys
from gguf import GGUFWriter, GGUFQuantizationType

def pt_to_gguf(pt_file, out_file, quantization="Q4_K_M"):
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

    print(f"Writing GGUF to {out_file} with {quantization} quantization...")
    # Initialize GGUFWriter with architecture name
    writer = GGUFWriter(out_file, "qbnn")

    # Optional metadata
    writer.add_name("QBNN Model")
    writer.add_description("Quantum Bit Neural Network Model by tapiocaTakeshi")
    writer.add_string("model.quantization", quantization)

    # Map quantization strings to GGUFQuantizationType
    quantization_map = {
        "Q4_K_M": GGUFQuantizationType.Q4_K_M,
        "Q4_K_S": GGUFQuantizationType.Q4_K_S,
        "Q5_K_M": GGUFQuantizationType.Q5_K_M,
        "Q5_K_S": GGUFQuantizationType.Q5_K_S,
        "Q6_K": GGUFQuantizationType.Q6_K,
        "Q8_0": GGUFQuantizationType.Q8_0,
        "F32": None,
        "F16": GGUFQuantizationType.F16,
    }

    quant_type = quantization_map.get(quantization)

    count = 0
    for name, tensor in state_dict.items():
        # Convert torch tensor to numpy array (must be contiguous)
        data = np.ascontiguousarray(tensor.float().numpy())

        # Skip quantization for embeddings, layer norms, and biases
        should_quantize = quant_type is not None and not any(
            pattern in name for pattern in ["embed", "norm", "bias"]
        )

        if should_quantize:
            writer.add_tensor(name, data, quantization_type=quant_type)
        else:
            writer.add_tensor(name, data)
        count += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Successfully exported {count} tensors to {out_file}.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert PyTorch model to GGUF format with quantization")
    parser.add_argument("input_file", nargs="?", default="neuroq_checkpoint.pt",
                        help="Input PyTorch model file (.pt)")
    parser.add_argument("output_file", nargs="?", default="neuroq.gguf",
                        help="Output GGUF file")
    parser.add_argument("--quantization", "-q", default="Q4_K_M",
                        choices=["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "F32", "F16"],
                        help="Quantization type (default: Q4_K_M)")

    args = parser.parse_args()

    print(f"Converting {args.input_file} to {args.output_file}")
    print(f"Quantization: {args.quantization}")
    pt_to_gguf(args.input_file, args.output_file, quantization=args.quantization)
