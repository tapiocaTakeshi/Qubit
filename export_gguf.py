import torch
import struct
import numpy as np
import sys
import json
from datetime import datetime
from gguf import GGUFWriter, GGMLQuantizationType

# Default GGUF runtime parameters
DEFAULT_GGUF_PARAMS = {
    "n_ctx": 512,
    "n_batch": 64,
    "n_ubatch": 64,
    "n_threads": 4,
    "n_gpu_layers": 0,
    "cache_type_k": "f16",
    "cache_type_v": "f16"
}

def pt_to_gguf(pt_file, out_file, quantization="Q4_K_M", gguf_params=None):
    if gguf_params is None:
        gguf_params = DEFAULT_GGUF_PARAMS.copy()

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

    # Required metadata
    writer.add_name("QBNN Model")
    writer.add_description("Quantum Bit Neural Network Model by tapiocaTakeshi")
    writer.add_string("model.quantization", quantization)
    writer.add_string("model.architecture", "qbnn")
    writer.add_string("model.size", "unknown")
    writer.add_string("model.created", datetime.now().isoformat())
    writer.add_bool("model.is_quantum", True)

    # Add GGUF runtime parameters
    writer.add_int32("llm.context_length", gguf_params.get("n_ctx", 512))
    writer.add_int32("llm.batch_size", gguf_params.get("n_batch", 64))
    writer.add_int32("llm.ubatch_size", gguf_params.get("n_ubatch", 64))
    writer.add_int32("llm.threads", gguf_params.get("n_threads", 4))
    writer.add_int32("llm.gpu_layers", gguf_params.get("n_gpu_layers", 0))
    writer.add_string("llm.cache_type_k", gguf_params.get("cache_type_k", "f16"))
    writer.add_string("llm.cache_type_v", gguf_params.get("cache_type_v", "f16"))

    # Save all GGUF parameters as JSON for reference
    writer.add_string("model.gguf_params", json.dumps(gguf_params))

    # Map quantization strings to GGMLQuantizationType
    quantization_map = {
        "Q4_K_M": GGMLQuantizationType.Q4_K,
        "Q4_K_S": GGMLQuantizationType.Q4_K,
        "Q5_K_M": GGMLQuantizationType.Q5_K,
        "Q5_K_S": GGMLQuantizationType.Q5_K,
        "Q6_K": GGMLQuantizationType.Q6_K,
        "Q8_0": GGMLQuantizationType.Q8_0,
        "F32": None,
        "F16": None,
    }

    quant_type = quantization_map.get(quantization)

    count = 0
    for name, tensor in state_dict.items():
        # Convert torch tensor to numpy array (must be contiguous)
        data = np.ascontiguousarray(tensor.float().numpy())

        # Only quantize large weight matrices (skip embeddings, norms, biases, and small tensors)
        should_quantize = (
            quant_type is not None
            and not any(pattern in name for pattern in ["embed", "norm", "bias"])
            and len(data.shape) >= 2  # At least 2D tensor
            and data.shape[-1] >= 256  # Last dimension large enough for quantization block
        )

        if should_quantize:
            writer.add_tensor(name, data, raw_dtype=quant_type)
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
    parser.add_argument("--gguf-params",
                        type=str,
                        help="GGUF runtime parameters as JSON (e.g., '{\"n_ctx\": 512, \"n_batch\": 64}')")

    args = parser.parse_args()

    # Parse GGUF parameters if provided
    gguf_params = None
    if args.gguf_params:
        try:
            gguf_params = json.loads(args.gguf_params)
        except json.JSONDecodeError as e:
            print(f"Error parsing GGUF parameters: {e}")
            sys.exit(1)

    print(f"Converting {args.input_file} to {args.output_file}")
    print(f"Quantization: {args.quantization}")
    if gguf_params:
        print(f"GGUF Parameters: {gguf_params}")
    pt_to_gguf(args.input_file, args.output_file, quantization=args.quantization, gguf_params=gguf_params)
