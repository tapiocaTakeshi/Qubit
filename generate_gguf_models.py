#!/usr/bin/env python3
"""
Generate GGUF models for multiple sizes (small, medium, large).
Creates quantized model files in GGUF format for different architectures.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, get_model_config_by_size, NeuroQuantumTokenizer
from qbnn_layered import EQBNNGenerativeAI

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("WARNING: gguf module not available. Install with: pip install gguf")
    GGUFWriter = None
    GGMLQuantizationType = None


class GGUFModelGenerator:
    """Generate GGUF format models for multiple sizes."""

    MODEL_SIZES = ["small", "medium", "large", "xlarge"]
    VOCAB_SIZE = 32000
    OUTPUT_DIR = "gguf_models"

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

    def __init__(self, output_dir: str = OUTPUT_DIR, device: str = "cpu", gguf_params: Dict = None):
        """Initialize the GGUF generator.

        Args:
            output_dir: Directory to save GGUF models
            device: Device to use for model creation (cpu/cuda)
            gguf_params: GGUF runtime parameters (uses defaults if not provided)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = device
        self.results = {}
        self.gguf_params = gguf_params or self.DEFAULT_GGUF_PARAMS.copy()

    def create_neuroquantum_model(self, size: str) -> torch.nn.Module:
        """Create a NeuroQuantum model of specified size."""
        config_dict = get_model_config_by_size(size=size, vocab_size=self.VOCAB_SIZE)
        config = NeuroQuantumConfig(
            vocab_size=config_dict["vocab_size"],
            embed_dim=config_dict["embed_dim"],
            hidden_dim=config_dict["hidden_dim"],
            num_heads=config_dict["num_heads"],
            num_layers=config_dict["num_layers"],
            max_seq_len=config_dict["max_seq_len"],
            lambda_entangle=config_dict["entangle_strength"],
            dropout=config_dict["dropout"],
        )
        model = NeuroQuantum(config=config)
        return model.to(self.device)

    def create_qbnn_model(self, size: str) -> Optional[torch.nn.Module]:
        """Create a QBNN model of specified size."""
        try:
            from qbnn_layered import EQBNNGenerativeAI

            # Size configurations
            size_configs = {
                "small": {
                    "vocab_size": self.VOCAB_SIZE,
                    "embedding_dim": 256,
                    "hidden_dim": 512,
                    "num_layers": 4,
                    "num_heads": 8,
                    "max_seq_len": 4096,
                },
                "medium": {
                    "vocab_size": self.VOCAB_SIZE,
                    "embedding_dim": 128,
                    "hidden_dim": 256,
                    "num_layers": 2,
                    "num_heads": 4,
                    "max_seq_len": 1024,
                },
                "large": {
                    "vocab_size": self.VOCAB_SIZE,
                    "embedding_dim": 512,
                    "hidden_dim": 1024,
                    "num_layers": 6,
                    "num_heads": 8,
                    "max_seq_len": 10000,
                },
                "xlarge": {
                    "vocab_size": self.VOCAB_SIZE,
                    "embedding_dim": 768,
                    "hidden_dim": 2048,
                    "num_layers": 12,
                    "num_heads": 12,
                    "max_seq_len": 16384,
                },
            }

            config = size_configs.get(size, size_configs["small"])
            model = EQBNNGenerativeAI(**config)
            return model.to(self.device)
        except ImportError:
            print(f"WARNING: EQBNNGenerativeAI not available for {size} model")
            return None

    def extract_quantum_characteristics_if_qbnn(self, state_dict: Dict, architecture: str) -> Dict:
        """Extract quantum characteristics for QBNN models.

        Args:
            state_dict: Model state dictionary
            architecture: Model architecture

        Returns:
            Dictionary with quantum characteristics
        """
        if architecture.lower() != "qbnn":
            return {}

        quantum_info = {
            "has_quantum_correlation": False,
            "has_entanglement": False,
            "apqb_theta_count": 0,
            "entangle_layer_count": 0,
            "quantum_tensors": [],
        }

        for name in state_dict.keys():
            if "quantum_corr" in name:
                quantum_info["has_quantum_correlation"] = True
            if "entangle" in name:
                quantum_info["has_entanglement"] = True
                quantum_info["entangle_layer_count"] += 1
            if "theta" in name:
                quantum_info["apqb_theta_count"] += 1
                quantum_info["quantum_tensors"].append(name)

        return quantum_info

    def pt_to_gguf(self, pt_file: str, gguf_file: str, model_name: str = "Qubit",
                   model_size: str = "unknown", architecture: str = "llama",
                   quantization: str = "Q4_K_M") -> bool:
        """Convert PyTorch model to GGUF format.

        Args:
            pt_file: Path to input .pt file
            gguf_file: Path to output .gguf file
            model_name: Name for the model
            model_size: Size of the model (small/medium/large)
            architecture: Model architecture name
            quantization: Quantization type (e.g., "Q4_K_M", "Q5_K_M", "F32")

        Returns:
            True if successful, False otherwise
        """
        if not GGUFWriter:
            print(f"ERROR: gguf module not available, cannot convert {pt_file}")
            return False

        try:
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

            # Extract quantum characteristics if QBNN
            quantum_info = self.extract_quantum_characteristics_if_qbnn(state_dict, architecture)

            print(f"Writing GGUF to {gguf_file}...")
            writer = GGUFWriter(gguf_file, architecture)

            # Add metadata
            writer.add_name(f"{model_name} {model_size.capitalize()}")
            writer.add_description(
                f"{model_name} {model_size.capitalize()} Model ({architecture}) by tapiocaTakeshi"
            )
            writer.add_version("1.0")
            writer.add_author("tapiocaTakeshi")
            writer.add_url("https://github.com/tapiocatakeshi/qubit")
            writer.add_source_url("https://github.com/tapiocatakeshi/qubit")

            # Add custom metadata
            writer.add_string("model.size", model_size)
            writer.add_string("model.architecture", architecture)
            writer.add_string("model.created", datetime.now().isoformat())
            writer.add_string("model.quantization", quantization)

            # Add GGUF runtime parameters
            writer.add_int32("llm.context_length", self.gguf_params.get("n_ctx", 512))
            writer.add_int32("llm.batch_size", self.gguf_params.get("n_batch", 64))
            writer.add_int32("llm.ubatch_size", self.gguf_params.get("n_ubatch", 64))
            writer.add_int32("llm.threads", self.gguf_params.get("n_threads", 4))
            writer.add_int32("llm.gpu_layers", self.gguf_params.get("n_gpu_layers", 0))
            writer.add_string("llm.cache_type_k", self.gguf_params.get("cache_type_k", "f16"))
            writer.add_string("llm.cache_type_v", self.gguf_params.get("cache_type_v", "f16"))

            # Save all GGUF parameters as JSON for reference
            writer.add_string("model.gguf_params", json.dumps(self.gguf_params))

            # Add quantum metadata flag
            writer.add_bool("model.is_quantum", bool(quantum_info))

            # Add detailed quantum metadata if QBNN
            if quantum_info:
                if quantum_info["has_quantum_correlation"]:
                    writer.add_bool("model.has_quantum_correlation", True)
                if quantum_info["has_entanglement"]:
                    writer.add_bool("model.has_entanglement", True)
                    writer.add_int32("model.entangle_layer_count", quantum_info["entangle_layer_count"])
                if quantum_info["apqb_theta_count"] > 0:
                    writer.add_int32("model.apqb_theta_count", quantum_info["apqb_theta_count"])

                # Save quantum metadata as JSON
                quantum_metadata = {
                    "type": "qbnn",
                    "has_quantum_correlation": quantum_info["has_quantum_correlation"],
                    "has_entanglement": quantum_info["has_entanglement"],
                    "apqb_theta_parameters": quantum_info["apqb_theta_count"],
                    "entanglement_layers": quantum_info["entangle_layer_count"],
                }
                writer.add_string("model.quantum_metadata", json.dumps(quantum_metadata))

            count = 0
            total_params = 0
            quantum_tensor_count = 0

            # Map quantization strings to GGMLQuantizationType
            quantization_map = {
                "Q4_K_M": GGMLQuantizationType.Q4_K if GGMLQuantizationType else None,
                "Q4_K_S": GGMLQuantizationType.Q4_K if GGMLQuantizationType else None,
                "Q5_K_M": GGMLQuantizationType.Q5_K if GGMLQuantizationType else None,
                "Q5_K_S": GGMLQuantizationType.Q5_K if GGMLQuantizationType else None,
                "Q6_K": GGMLQuantizationType.Q6_K if GGMLQuantizationType else None,
                "Q8_0": GGMLQuantizationType.Q8_0 if GGMLQuantizationType else None,
                "F32": None,  # No quantization
                "F16": None,  # Not directly available
            }

            quant_type = quantization_map.get(quantization)

            for name, tensor in state_dict.items():
                data = np.ascontiguousarray(tensor.float().detach().cpu().numpy())

                # For QBNN: preserve quantum tensors without quantization
                is_quantum = any(q in name for q in ["quantum_corr", "entangle", "theta"])

                should_quantize = (
                    quant_type is not None
                    and not any(pattern in name for pattern in ["embed", "norm", "bias"])
                    and len(data.shape) >= 2  # At least 2D tensor
                    and data.shape[-1] >= 256  # Last dimension large enough for quantization block
                    and not is_quantum  # Don't quantize quantum-specific tensors
                )

                if should_quantize:
                    writer.add_tensor(name, data, raw_dtype=quant_type)
                else:
                    writer.add_tensor(name, data)
                    if is_quantum:
                        quantum_tensor_count += 1

                count += 1
                total_params += tensor.numel()

            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            file_size_mb = os.path.getsize(gguf_file) / (1024 * 1024)
            print(f"✅ Successfully exported {count} tensors ({total_params:,} params, {file_size_mb:.2f}MB) to {gguf_file}.")
            if quantum_tensor_count > 0:
                print(f"   ⚛️  Quantum tensors preserved: {quantum_tensor_count}")
            return True

        except Exception as e:
            print(f"❌ Error converting {pt_file}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_model_checkpoint(self, architecture: str, size: str) -> Optional[str]:
        """Generate and save a model checkpoint.

        Args:
            architecture: "gemma", "neuroquantum", or "qbnn"
            size: Model size (small/medium/large)

        Returns:
            Path to saved checkpoint, or None if failed or unsupported
        """
        try:
            print(f"\n🔨 Creating {architecture} {size} model...")

            model = None
            if architecture.lower() == "gemma":
                model = self.create_neuroquantum_model(size)
            elif architecture.lower() == "neuroquantum":
                model = self.create_neuroquantum_model(size)
            elif architecture.lower() == "qbnn":
                model = self.create_qbnn_model(size)
                if model is None:
                    print(f"   ⏭️  QBNN architecture not yet supported, skipping...")
                    return None
            else:
                print(f"ERROR: Unknown architecture {architecture}")
                return None

            if model is None:
                return None

            # Generate random input for initialization
            batch_size = 1
            seq_len = 128
            input_ids = torch.randint(0, self.VOCAB_SIZE, (batch_size, seq_len))

            # Forward pass to initialize weights
            print(f"   Initializing model with forward pass...")
            with torch.no_grad():
                _ = model(input_ids.to(self.device))

            # Save checkpoint
            checkpoint_file = self.output_dir / f"{architecture}_{size}_checkpoint.pt"
            torch.save(model.state_dict(), checkpoint_file)
            print(f"   ✅ Saved checkpoint to {checkpoint_file}")

            return str(checkpoint_file)

        except Exception as e:
            print(f"❌ Error creating {architecture} {size} model: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_all(self, architectures: list = None, sizes: list = None,
                     quantization: str = "Q4_K_M", gguf_params: Dict = None) -> Dict:
        """Generate GGUF models for all specified architectures and sizes.

        Args:
            architectures: List of architectures to generate (default: ["gemma"])
            sizes: List of sizes to generate (default: ["small", "medium", "large"])
            quantization: Quantization type (default: "Q4_K_M")
            gguf_params: GGUF runtime parameters (uses defaults if not provided)

        Returns:
            Dictionary with generation results
        """
        if gguf_params:
            self.gguf_params = gguf_params
        if architectures is None:
            architectures = ["llama"]
        if sizes is None:
            sizes = self.MODEL_SIZES

        results = {}

        for architecture in architectures:
            results[architecture] = {}

            for size in sizes:
                print(f"\n{'='*60}")
                print(f"Generating {architecture.upper()} - {size.upper()}")
                print(f"Quantization: {quantization}")
                print(f"{'='*60}")

                # Generate checkpoint
                checkpoint_file = self.generate_model_checkpoint(architecture, size)
                if not checkpoint_file or checkpoint_file is None:
                    results[architecture][size] = {"status": "skipped", "error": f"{architecture} model generation not yet supported"}
                    continue

                # Convert to GGUF
                gguf_file = self.output_dir / f"{architecture}_{size}_{quantization}.gguf"
                success = self.pt_to_gguf(
                    checkpoint_file,
                    str(gguf_file),
                    model_name="Qubit",
                    model_size=size,
                    architecture=architecture,
                    quantization=quantization
                )

                if success:
                    results[architecture][size] = {
                        "status": "success",
                        "checkpoint": checkpoint_file,
                        "gguf": str(gguf_file),
                        "quantization": quantization,
                        "size_mb": os.path.getsize(gguf_file) / (1024 * 1024)
                    }
                else:
                    results[architecture][size] = {"status": "failed", "error": "GGUF conversion failed"}

        self.results = results
        return results

    def print_summary(self):
        """Print summary of generation results."""
        print(f"\n{'='*60}")
        print("GGUF Generation Summary")
        print(f"{'='*60}")

        for architecture, sizes_results in self.results.items():
            print(f"\n📦 {architecture.upper()}:")
            for size, result in sizes_results.items():
                status_icon = "✅" if result["status"] == "success" else "❌"
                print(f"  {status_icon} {size.upper()}: {result['status']}", end="")
                if result["status"] == "success":
                    print(f" ({result['size_mb']:.2f}MB)")
                else:
                    print(f" ({result.get('error', 'Unknown error')})")

        print(f"\n📁 Output directory: {self.output_dir}")
        print(f"{'='*60}\n")

    def save_manifest(self):
        """Save a manifest of generated models."""
        manifest = {
            "generated_at": datetime.now().isoformat(),
            "models": self.results,
            "output_directory": str(self.output_dir),
            "vocab_size": self.VOCAB_SIZE,
        }

        manifest_file = self.output_dir / "manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"📋 Manifest saved to {manifest_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate GGUF models for Qubit in multiple sizes"
    )
    parser.add_argument(
        "--output-dir",
        default="gguf_models",
        help="Output directory for GGUF files (default: gguf_models)"
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["llama"],
        help="Architectures to generate (default: llama). Use --architectures llama qbnn for both"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["small", "medium", "large", "xlarge"],
        help="Model sizes to generate (default: small medium large xlarge)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for model creation (cpu/cuda, default: cpu)"
    )
    parser.add_argument(
        "--skip-checkpoint-cleanup",
        action="store_true",
        help="Keep checkpoint .pt files after GGUF conversion"
    )
    parser.add_argument(
        "--quantization",
        default="Q4_K_M",
        choices=["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "F32", "F16"],
        help="Quantization type (default: Q4_K_M)"
    )
    parser.add_argument(
        "--gguf-params",
        type=str,
        help="GGUF runtime parameters as JSON (e.g., '{\"n_ctx\": 512, \"n_batch\": 64}')"
    )

    args = parser.parse_args()

    # Parse GGUF parameters if provided
    gguf_params = None
    if args.gguf_params:
        try:
            gguf_params = json.loads(args.gguf_params)
            print(f"📋 Using custom GGUF parameters: {gguf_params}")
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing GGUF parameters: {e}")
            sys.exit(1)

    print("🚀 Qubit GGUF Model Generator")
    print(f"   Output: {args.output_dir}")
    print(f"   Architectures: {', '.join(args.architectures)}")
    print(f"   Sizes: {', '.join(args.sizes)}")
    print(f"   Quantization: {args.quantization}")
    print(f"   Device: {args.device}\n")

    generator = GGUFModelGenerator(
        output_dir=args.output_dir,
        device=args.device,
        gguf_params=gguf_params
    )

    # Generate all models
    generator.generate_all(
        architectures=args.architectures,
        sizes=args.sizes,
        quantization=args.quantization,
        gguf_params=gguf_params
    )

    # Print summary
    generator.print_summary()

    # Save manifest
    generator.save_manifest()

    # Clean up checkpoints if requested
    if not args.skip_checkpoint_cleanup:
        print("🧹 Cleaning up checkpoint files...")
        for f in generator.output_dir.glob("*_checkpoint.pt"):
            f.unlink()
            print(f"   Removed {f.name}")

    print("✨ Done!")


if __name__ == "__main__":
    main()
