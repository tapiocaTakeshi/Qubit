#!/usr/bin/env python3
"""
GGUF メタデータ検証スクリプト
Qubit GGUF ファイルの必須メタデータが正しく含まれているか検証
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple


class GGUFMetadataValidator:
    """GGUF メタデータの検証"""

    # 必須フィールド
    REQUIRED_FIELDS = {
        # モデルメタデータ
        "model.architecture": (str, "Model architecture"),
        "model.size": (str, "Model size"),
        "model.quantization": (str, "Quantization type"),

        # ランタイムパラメータ
        "llm.context_length": (int, "Context length"),
        "llm.batch_size": (int, "Batch size"),
        "llm.gpu_layers": (int, "GPU layers"),
        "llm.threads": (int, "Thread count"),
    }

    # 推奨フィールド
    RECOMMENDED_FIELDS = {
        "llm.ubatch_size": (int, "Unbatched size"),
        "llm.cache_type_k": (str, "Cache type K"),
        "llm.cache_type_v": (str, "Cache type V"),
        "model.gguf_params": (str, "GGUF parameters JSON"),
        "model.created": (str, "Creation timestamp"),
        "model.is_quantum": (bool, "Quantum flag (for QBNN)"),
    }

    def __init__(self, gguf_path: str):
        """Initialize validator

        Args:
            gguf_path: Path to GGUF file
        """
        self.gguf_path = Path(gguf_path)
        self.reader = None
        self.errors = []
        self.warnings = []
        self.info = []

    def validate(self) -> bool:
        """Validate GGUF file

        Returns:
            True if all required fields are present, False otherwise
        """
        try:
            from gguf import GGUFReader
        except ImportError:
            self.errors.append("gguf module not installed. Install with: pip install gguf")
            return False

        if not self.gguf_path.exists():
            self.errors.append(f"File not found: {self.gguf_path}")
            return False

        try:
            self.reader = GGUFReader(str(self.gguf_path))
        except Exception as e:
            self.errors.append(f"Failed to read GGUF file: {e}")
            return False

        # Check required fields
        self._check_required_fields()

        # Check recommended fields
        self._check_recommended_fields()

        # Check consistency
        self._check_consistency()

        # Check data types
        self._check_data_types()

        return len(self.errors) == 0

    def _check_required_fields(self):
        """Check if all required fields are present"""
        for field_name, (field_type, description) in self.REQUIRED_FIELDS.items():
            try:
                field = self.reader.get_field(field_name)
                if field is None:
                    self.errors.append(f"❌ Missing required field: {field_name} ({description})")
                else:
                    self.info.append(f"✅ {field_name}: {self._get_field_value(field)}")
            except KeyError:
                self.errors.append(f"❌ Missing required field: {field_name} ({description})")

    def _check_recommended_fields(self):
        """Check if recommended fields are present"""
        for field_name, (field_type, description) in self.RECOMMENDED_FIELDS.items():
            try:
                field = self.reader.get_field(field_name)
                if field is None:
                    self.warnings.append(f"⚠️  Missing recommended field: {field_name} ({description})")
                else:
                    value = self._get_field_value(field)
                    # Don't print long JSON values
                    if field_name == "model.gguf_params" and isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    self.info.append(f"✅ {field_name}: {value}")
            except KeyError:
                self.warnings.append(f"⚠️  Missing recommended field: {field_name} ({description})")

    def _check_consistency(self):
        """Check consistency between fields"""
        # Check if n_gpu_layers makes sense
        try:
            n_gpu_layers = self.reader.get_field("llm.gpu_layers").ints[0]
            if n_gpu_layers < 0:
                self.errors.append(f"❌ Invalid value for llm.gpu_layers: {n_gpu_layers} (should be >= 0)")
            elif n_gpu_layers == 0:
                self.info.append("ℹ️  GPU layers set to 0 (CPU-only by default)")
            else:
                self.info.append(f"ℹ️  GPU acceleration enabled ({n_gpu_layers} layers)")
        except:
            pass

        # Check context length
        try:
            n_ctx = self.reader.get_field("llm.context_length").ints[0]
            if n_ctx < 128:
                self.warnings.append(f"⚠️  Small context length: {n_ctx} tokens")
            elif n_ctx > 32768:
                self.info.append(f"ℹ️  Large context length: {n_ctx} tokens")
        except:
            pass

        # Check batch size consistency
        try:
            n_batch = self.reader.get_field("llm.batch_size").ints[0]
            n_ubatch = self.reader.get_field("llm.ubatch_size").ints[0]
            if n_ubatch > n_batch:
                self.warnings.append(
                    f"⚠️  ubatch_size ({n_ubatch}) > batch_size ({n_batch})"
                )
        except:
            pass

    def _check_data_types(self):
        """Check if field values have correct types"""
        for field_name, (expected_type, _) in self.REQUIRED_FIELDS.items():
            try:
                field = self.reader.get_field(field_name)
                if expected_type == str and not hasattr(field, 'strings'):
                    self.errors.append(f"❌ {field_name} should be string but isn't")
                elif expected_type == int and not hasattr(field, 'ints'):
                    self.errors.append(f"❌ {field_name} should be int but isn't")
                elif expected_type == bool and not hasattr(field, 'bools'):
                    self.errors.append(f"❌ {field_name} should be bool but isn't")
            except:
                pass

    def _get_field_value(self, field):
        """Extract value from GGUF field"""
        if hasattr(field, 'strings') and field.strings:
            return field.strings[0]
        elif hasattr(field, 'ints') and field.ints:
            return field.ints[0]
        elif hasattr(field, 'bools') and field.bools:
            return field.bools[0]
        elif hasattr(field, 'floats') and field.floats:
            return field.floats[0]
        else:
            return "(unknown type)"

    def print_report(self) -> int:
        """Print validation report

        Returns:
            Exit code (0 = valid, 1 = errors, 2 = warnings only)
        """
        print(f"\n{'='*70}")
        print(f"GGUF Metadata Validation Report")
        print(f"{'='*70}")
        print(f"\nFile: {self.gguf_path}")
        print(f"Size: {self.gguf_path.stat().st_size / (1024**2):.2f} MB")

        if self.info:
            print(f"\n✅ Valid Fields ({len(self.info)}):")
            for msg in self.info:
                print(f"   {msg}")

        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for msg in self.warnings:
                print(f"   {msg}")

        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for msg in self.errors:
                print(f"   {msg}")

            print(f"\n💡 Fix missing fields:")
            print(f"   1. Regenerate GGUF using updated export_qbnn_gguf.py or generate_gguf_models.py")
            print(f"   2. Ensure gguf_params are passed correctly during conversion")

            return_code = 1
        else:
            print(f"\n✅ All required metadata fields are present!")
            return_code = 0

        if self.warnings and return_code == 0:
            return_code = 2

        print(f"\n{'='*70}\n")
        return return_code


def validate_multiple_files(gguf_paths: List[str]) -> int:
    """Validate multiple GGUF files

    Args:
        gguf_paths: List of GGUF file paths

    Returns:
        Max return code from all validations
    """
    max_return_code = 0

    for path in gguf_paths:
        validator = GGUFMetadataValidator(path)
        validator.validate()
        return_code = validator.print_report()
        max_return_code = max(max_return_code, return_code)

    return max_return_code


def main():
    if len(sys.argv) < 2:
        print("GGUF Metadata Validator for Qubit Models")
        print("\nUsage: python validate_gguf_metadata.py <model1.gguf> [model2.gguf] ...")
        print("\nValidates that GGUF files contain all required metadata fields:")
        print("  - model.architecture")
        print("  - model.size")
        print("  - model.quantization")
        print("  - llm.context_length")
        print("  - llm.batch_size")
        print("  - llm.gpu_layers")
        print("  - llm.threads")
        print("\nExit codes:")
        print("  0 = All required fields present")
        print("  1 = Missing required fields")
        print("  2 = Warnings only (recommended fields missing)")
        sys.exit(1)

    gguf_paths = sys.argv[1:]
    return_code = validate_multiple_files(gguf_paths)
    sys.exit(return_code)


if __name__ == "__main__":
    main()
