#!/usr/bin/env python3
"""
Batch Quantization Processing Script
バッチ量子化処理スクリプト

Generate multiple quantized models with different bit-widths and configurations
in a single run, with validation and reporting.

Usage:
    python batch_quantize_models.py --checkpoint checkpoint.pt
    python batch_quantize_models.py --checkpoint checkpoint.pt --bit-widths 1,2,3
    python batch_quantize_models.py --checkpoint checkpoint.pt --export-gguf
    python batch_quantize_models.py --checkpoint checkpoint.pt --all
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import subprocess

try:
    import torch
except ImportError:
    print("❌ PyTorch not installed. Install with: pip install torch")
    sys.exit(1)


class BatchQuantizer:
    """Manage batch quantization of models."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}
        self.start_time = None
        self.batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log(self, message: str, level: str = "info"):
        """Pretty logging."""
        if not self.verbose:
            return

        emojis = {
            "info": "ℹ️ ",
            "success": "✅ ",
            "warning": "⚠️ ",
            "error": "❌ ",
            "step": "📋 ",
            "process": "⚙️ ",
        }
        prefix = emojis.get(level, "• ")
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {prefix}{message}")

    def verify_checkpoint(self, checkpoint_path: str) -> bool:
        """Verify checkpoint exists and is valid."""
        self.log(f"Verifying checkpoint: {checkpoint_path}", "step")

        if not Path(checkpoint_path).exists():
            self.log(f"Checkpoint not found: {checkpoint_path}", "error")
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.log("✓ Checkpoint is valid", "success")
            return True
        except Exception as e:
            self.log(f"Invalid checkpoint: {e}", "error")
            return False

    def run_quantization(self, checkpoint_path: str, bit_width: int,
                         output_path: Optional[str] = None) -> bool:
        """Run quantization for a specific bit-width."""

        self.log(f"Starting {bit_width}-bit quantization", "process")

        try:
            if bit_width == 1:
                cmd = [
                    'python', 'quantize_neuroquantum_1bit.py',
                    checkpoint_path
                ]
                if output_path:
                    cmd.extend(['-o', output_path])
            else:
                cmd = [
                    'python', 'quantize_neuroquantum_multibit.py',
                    checkpoint_path,
                    '--bit-width', str(bit_width)
                ]
                if output_path:
                    cmd.extend(['--output', output_path])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                self.log(f"Quantization failed for {bit_width}-bit", "error")
                self.log(f"Error: {result.stderr}", "error")
                return False

            self.log(f"✓ {bit_width}-bit quantization complete", "success")
            return True

        except subprocess.TimeoutExpired:
            self.log(f"Quantization timeout for {bit_width}-bit", "error")
            return False
        except Exception as e:
            self.log(f"Quantization error: {e}", "error")
            return False

    def run_gguf_export(self, model_path: str, bit_width: int,
                        output_path: Optional[str] = None) -> bool:
        """Export quantized model to GGUF format."""

        self.log(f"Exporting {bit_width}-bit to GGUF", "process")

        try:
            if bit_width == 1:
                cmd = ['python', 'export_1bit_gguf.py', model_path]
            else:
                cmd = [
                    'python', 'export_multibit_gguf.py',
                    model_path,
                    '--bit-width', str(bit_width)
                ]

            if output_path:
                cmd.extend(['--output', output_path])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                self.log(f"GGUF export failed for {bit_width}-bit", "error")
                self.log(f"Error: {result.stderr}", "error")
                return False

            self.log(f"✓ GGUF export complete", "success")
            return True

        except subprocess.TimeoutExpired:
            self.log(f"GGUF export timeout", "error")
            return False
        except Exception as e:
            self.log(f"Export error: {e}", "error")
            return False

    def validate_outputs(self, bit_width: int, model_path: str,
                        gguf_path: Optional[str] = None) -> Dict:
        """Validate quantized outputs."""

        self.log(f"Validating {bit_width}-bit outputs", "process")

        validation = {
            "bit_width": bit_width,
            "model_valid": False,
            "gguf_valid": False,
            "details": {}
        }

        # Check model file
        if Path(model_path).exists():
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                validation["model_valid"] = True
                validation["model_size_mb"] = Path(model_path).stat().st_size / (1024 * 1024)
                self.log(f"✓ Model file valid ({validation['model_size_mb']:.1f} MB)", "success")
            except Exception as e:
                self.log(f"Model validation failed: {e}", "error")

        # Check GGUF if path provided
        if gguf_path and Path(gguf_path).exists():
            try:
                # Try to validate with check_gguf_params.py
                result = subprocess.run(
                    ['python', 'check_gguf_params.py', gguf_path],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    validation["gguf_valid"] = True
                    validation["gguf_size_mb"] = Path(gguf_path).stat().st_size / (1024 * 1024)
                    self.log(f"✓ GGUF file valid ({validation['gguf_size_mb']:.1f} MB)", "success")
            except Exception as e:
                self.log(f"GGUF validation attempt: {e}", "warning")

        return validation

    def run_batch(self, checkpoint_path: str, bit_widths: List[int] = None,
                  export_gguf: bool = False, validate: bool = True):
        """Run complete batch quantization."""

        if bit_widths is None:
            bit_widths = [2]  # Default to 2-bit

        self.start_time = time.time()

        print("\n" + "="*70)
        print("🚀 Batch Quantization Processing")
        print("="*70)
        print(f"Batch ID: {self.batch_id}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Bit-widths: {bit_widths}")
        print(f"Export GGUF: {export_gguf}")
        print("="*70 + "\n")

        # Verify checkpoint
        if not self.verify_checkpoint(checkpoint_path):
            return False

        # Process each bit-width
        for bit_width in bit_widths:
            self.log(f"\n{'='*70}", "info")
            self.log(f"Processing {bit_width}-bit quantization", "step")
            self.log(f"{'='*70}", "info")

            result_entry = {
                "bit_width": bit_width,
                "quantization": False,
                "gguf_export": False,
                "validation": None,
                "duration": 0
            }

            step_start = time.time()

            # Generate output filename
            base_name = Path(checkpoint_path).stem
            model_output = f"{base_name}_{bit_width}bit.pt"

            # Run quantization
            if self.run_quantization(checkpoint_path, bit_width, model_output):
                result_entry["quantization"] = True

                # Export to GGUF if requested
                if export_gguf:
                    gguf_output = f"{base_name}_{bit_width}bit.gguf"
                    if self.run_gguf_export(model_output, bit_width, gguf_output):
                        result_entry["gguf_export"] = True

                        # Validate
                        if validate:
                            validation = self.validate_outputs(
                                bit_width, model_output, gguf_output
                            )
                            result_entry["validation"] = validation

            result_entry["duration"] = time.time() - step_start
            self.results[f"model_{bit_width}bit"] = result_entry

        # Print summary
        self.print_summary()

        return True

    def print_summary(self):
        """Print batch processing summary."""

        elapsed = time.time() - self.start_time

        print("\n" + "="*70)
        print("📊 Batch Processing Summary")
        print("="*70)

        successful = sum(1 for r in self.results.values()
                        if r.get("quantization"))

        print(f"\nResults: {successful}/{len(self.results)} successful")
        print(f"Total time: {elapsed:.1f} seconds\n")

        # Results table
        print("Bit-Width | Quantization | GGUF Export | Validation | Duration")
        print("-" * 70)

        for model_id, result in self.results.items():
            bw = result["bit_width"]
            q = "✓" if result["quantization"] else "✗"
            g = "✓" if result["gguf_export"] else "-"
            v = "✓" if result["validation"] and result["validation"]["model_valid"] else "-"
            d = f"{result['duration']:.1f}s"
            print(f"{bw:^9} | {q:^12} | {g:^11} | {v:^10} | {d:^8}")

        # Save report
        report_path = f"batch_report_{self.batch_id}.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        self.log(f"Report saved: {report_path}", "success")

        # Recommendations
        print("\n" + "="*70)
        print("💡 Next Steps:")
        print("="*70)
        print("\n1. Verify outputs:")
        for bw in sorted(set(r["bit_width"] for r in self.results.values())):
            print(f"   python check_gguf_params.py checkpoint_{bw}bit.gguf")

        print("\n2. Upload to Hugging Face:")
        print("   python upload_to_huggingface.py checkpoint_*bit.gguf")

        print("\n3. Review documentation:")
        print("   • MULTIBIT_QUANTIZATION_COMPARISON.md")
        print("   • GGUF_LOADING_TROUBLESHOOTING.md")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch Quantization Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s --checkpoint model.pt
  python %(prog)s --checkpoint model.pt --bit-widths 1,2,3
  python %(prog)s --checkpoint model.pt --bit-widths 2 --export-gguf
  python %(prog)s --checkpoint model.pt --all
        """
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--bit-widths', type=str, default='2',
                       help='Comma-separated bit-widths (default: 2)')
    parser.add_argument('--export-gguf', action='store_true',
                       help='Export to GGUF format')
    parser.add_argument('--all', action='store_true',
                       help='Process all (1, 2, 3-bit) and export GGUF')
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip validation')

    args = parser.parse_args()

    # Parse bit-widths
    if args.all:
        bit_widths = [1, 2, 3]
        export_gguf = True
    else:
        bit_widths = [int(x.strip()) for x in args.bit_widths.split(',')]
        export_gguf = args.export_gguf

    quantizer = BatchQuantizer()
    quantizer.run_batch(
        args.checkpoint,
        bit_widths=bit_widths,
        export_gguf=export_gguf,
        validate=not args.no_validate
    )


if __name__ == '__main__':
    main()
