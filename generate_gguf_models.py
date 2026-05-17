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

from neuroquantum_layered import NeuroQuantum, get_model_config_by_size, NeuroQuantumTokenizer
from qbnn_layered import QBNNLayered

try:
    from gguf import GGUFWriter
except ImportError:
    print("WARNING: gguf module not available. Install with: pip install gguf")
    GGUFWriter = None


class GGUFModelGenerator:
    """Generate GGUF format models for multiple sizes."""

    MODEL_SIZES = ["small", "medium", "large"]
    VOCAB_SIZE = 32000
    OUTPUT_DIR = "gguf_models"

    def __init__(self, output_dir: str = OUTPUT_DIR, device: str = "cpu"):
        """Initialize the GGUF generator.

        Args:
            output_dir: Directory to save GGUF models
            device: Device to use for model creation (cpu/cuda)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = device
        self.results = {}

    def create_neuroquantum_model(self, size: str) -> torch.nn.Module:
        """Create a NeuroQuantum model of specified size."""
        config = get_model_config_by_size(size=size, vocab_size=self.VOCAB_SIZE)
        model = NeuroQuantum(
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            max_seq_len=config["max_seq_len"],
            entangle_strength=config["entangle_strength"],
            dropout=config["dropout"],
        )
        return model.to(self.device)

    def create_qbnn_model(self, size: str) -> torch.nn.Module:
        """Create a QBNN model of specified size."""
        config = get_model_config_by_size(size=size, vocab_size=self.VOCAB_SIZE)
        model = QBNNLayered(
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            max_seq_len=config["max_seq_len"],
            dropout=config["dropout"],
        )
        return model.to(self.device)

    def pt_to_gguf(self, pt_file: str, gguf_file: str, model_name: str = "Qubit",
                   model_size: str = "unknown", architecture: str = "neuroquantum") -> bool:
        """Convert PyTorch model to GGUF format.

        Args:
            pt_file: Path to input .pt file
            gguf_file: Path to output .gguf file
            model_name: Name for the model
            model_size: Size of the model (small/medium/large)
            architecture: Model architecture name

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

            count = 0
            total_params = 0
            for name, tensor in state_dict.items():
                data = np.ascontiguousarray(tensor.float().detach().cpu().numpy())
                writer.add_tensor(name, data)
                count += 1
                total_params += tensor.numel()

            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            file_size_mb = os.path.getsize(gguf_file) / (1024 * 1024)
            print(f"✅ Successfully exported {count} tensors ({total_params:,} params, {file_size_mb:.2f}MB) to {gguf_file}.")
            return True

        except Exception as e:
            print(f"❌ Error converting {pt_file}: {e}")
            return False

    def generate_model_checkpoint(self, architecture: str, size: str) -> Optional[str]:
        """Generate and save a model checkpoint.

        Args:
            architecture: "neuroquantum" or "qbnn"
            size: Model size (small/medium/large)

        Returns:
            Path to saved checkpoint, or None if failed
        """
        try:
            print(f"\n🔨 Creating {architecture} {size} model...")

            if architecture.lower() == "neuroquantum":
                model = self.create_neuroquantum_model(size)
            elif architecture.lower() == "qbnn":
                model = self.create_qbnn_model(size)
            else:
                print(f"ERROR: Unknown architecture {architecture}")
                return None

            # Generate random input for initialization
            batch_size = 1
            seq_len = 128
            input_ids = torch.randint(0, self.VOCAB_SIZE, (batch_size, seq_len))

            # Forward pass to initialize weights
            print(f"   Initializing model with forward pass...")
            with torch.no_grad():
                if architecture.lower() == "neuroquantum":
                    _ = model(input_ids.to(self.device))
                elif architecture.lower() == "qbnn":
                    _ = model(input_ids.to(self.device))

            # Save checkpoint
            checkpoint_file = self.output_dir / f"{architecture}_{size}_checkpoint.pt"
            torch.save(model.state_dict(), checkpoint_file)
            print(f"   ✅ Saved checkpoint to {checkpoint_file}")

            return str(checkpoint_file)

        except Exception as e:
            print(f"❌ Error creating {architecture} {size} model: {e}")
            return None

    def generate_all(self, architectures: list = None, sizes: list = None) -> Dict:
        """Generate GGUF models for all specified architectures and sizes.

        Args:
            architectures: List of architectures to generate (default: ["neuroquantum", "qbnn"])
            sizes: List of sizes to generate (default: ["small", "medium", "large"])

        Returns:
            Dictionary with generation results
        """
        if architectures is None:
            architectures = ["neuroquantum", "qbnn"]
        if sizes is None:
            sizes = self.MODEL_SIZES

        results = {}

        for architecture in architectures:
            results[architecture] = {}

            for size in sizes:
                print(f"\n{'='*60}")
                print(f"Generating {architecture.upper()} - {size.upper()}")
                print(f"{'='*60}")

                # Generate checkpoint
                checkpoint_file = self.generate_model_checkpoint(architecture, size)
                if not checkpoint_file:
                    results[architecture][size] = {"status": "failed", "error": "Checkpoint generation failed"}
                    continue

                # Convert to GGUF
                gguf_file = self.output_dir / f"{architecture}_{size}.gguf"
                success = self.pt_to_gguf(
                    checkpoint_file,
                    str(gguf_file),
                    model_name="Qubit",
                    model_size=size,
                    architecture=architecture
                )

                if success:
                    results[architecture][size] = {
                        "status": "success",
                        "checkpoint": checkpoint_file,
                        "gguf": str(gguf_file),
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
        default=["neuroquantum", "qbnn"],
        help="Architectures to generate (default: neuroquantum qbnn)"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["small", "medium", "large"],
        help="Model sizes to generate (default: small medium large)"
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

    args = parser.parse_args()

    print("🚀 Qubit GGUF Model Generator")
    print(f"   Output: {args.output_dir}")
    print(f"   Architectures: {', '.join(args.architectures)}")
    print(f"   Sizes: {', '.join(args.sizes)}")
    print(f"   Device: {args.device}\n")

    generator = GGUFModelGenerator(
        output_dir=args.output_dir,
        device=args.device
    )

    # Generate all models
    generator.generate_all(
        architectures=args.architectures,
        sizes=args.sizes
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
