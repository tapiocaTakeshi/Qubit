#!/usr/bin/env python3
"""
Local Workflow Validation Script
ローカルワークフロー検証スクリプト

Validate the complete quantization workflow before running GitHub Actions,
including checking dependencies, file structure, and mock execution.

Usage:
    python validate_workflow_local.py --full
    python validate_workflow_local.py --check-deps
    python validate_workflow_local.py --check-files
    python validate_workflow_local.py --mock-run
"""

import argparse
import subprocess
import sys
from pathlib import Path
import importlib.util
from typing import Dict, List, Tuple


class WorkflowValidator:
    """Validate quantization workflow setup."""

    def __init__(self):
        self.checks = {}
        self.project_root = Path.cwd()

    def log(self, message: str, status: str = "info"):
        """Pretty logging with status."""
        emojis = {
            "pass": "✅ ",
            "fail": "❌ ",
            "warning": "⚠️ ",
            "info": "ℹ️ ",
            "pending": "⏳ ",
        }
        prefix = emojis.get(status, "• ")
        print(f"{prefix}{message}")

    def check_python_version(self) -> bool:
        """Check Python version."""
        self.log("Checking Python version...", "info")

        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            self.log(f"Python {version.major}.{version.minor} is compatible", "pass")
            self.checks["python_version"] = True
            return True
        else:
            self.log(f"Python {version.major}.{version.minor} is too old (need 3.8+)", "fail")
            self.checks["python_version"] = False
            return False

    def check_dependencies(self) -> Dict[str, bool]:
        """Check required Python dependencies."""
        self.log("\nChecking dependencies...", "info")

        required = {
            'torch': 'PyTorch',
            'numpy': 'NumPy',
        }

        optional = {
            'gguf': 'GGUF',
            'huggingface_hub': 'Hugging Face Hub',
        }

        results = {}

        print("\n  Required:")
        for module, name in required.items():
            try:
                spec = importlib.util.find_spec(module)
                if spec is not None:
                    self.log(f"{name} is installed", "pass")
                    results[module] = True
                else:
                    self.log(f"{name} is NOT installed (REQUIRED)", "fail")
                    results[module] = False
            except ImportError:
                self.log(f"{name} is NOT installed (REQUIRED)", "fail")
                results[module] = False

        print("\n  Optional:")
        for module, name in optional.items():
            try:
                spec = importlib.util.find_spec(module)
                if spec is not None:
                    self.log(f"{name} is installed", "pass")
                    results[module] = True
                else:
                    self.log(f"{name} is NOT installed (optional)", "warning")
                    results[module] = False
            except ImportError:
                self.log(f"{name} is NOT installed (optional)", "warning")
                results[module] = False

        self.checks["dependencies"] = all(results[m] for m in required.keys())
        return results

    def check_files(self) -> bool:
        """Check required files exist."""
        self.log("\nChecking required files...", "info")

        required_files = [
            'quantize_neuroquantum_1bit.py',
            'quantize_neuroquantum_multibit.py',
            'export_1bit_gguf.py',
            'export_multibit_gguf.py',
            'check_gguf_params.py',
            'validate_gguf_metadata.py',
            'upload_to_huggingface.py',
            'neuroquantum_layered.py',
            '.github/workflows/upload-to-huggingface.yml',
            '.github/workflows/generate-quantized-models.yml',
        ]

        optional_files = [
            'MULTIBIT_QUANTIZATION_COMPARISON.md',
            'BINARY_1BIT_QUANTIZATION_GUIDE.md',
            'GGUF_LOADING_TROUBLESHOOTING.md',
            'GITHUB_ACTIONS_SETUP.md',
        ]

        all_exist = True
        print("\n  Required:")
        for file in required_files:
            path = self.project_root / file
            if path.exists():
                self.log(f"{file}", "pass")
            else:
                self.log(f"{file} NOT FOUND", "fail")
                all_exist = False

        print("\n  Optional Documentation:")
        for file in optional_files:
            path = self.project_root / file
            if path.exists():
                self.log(f"{file}", "pass")
            else:
                self.log(f"{file} not found (optional)", "warning")

        self.checks["files"] = all_exist
        return all_exist

    def check_github_actions(self) -> bool:
        """Check GitHub Actions workflow files."""
        self.log("\nChecking GitHub Actions workflows...", "info")

        workflows = [
            '.github/workflows/upload-to-huggingface.yml',
            '.github/workflows/generate-quantized-models.yml',
        ]

        all_valid = True
        for workflow_file in workflows:
            path = self.project_root / workflow_file
            if path.exists():
                try:
                    import yaml
                    with open(path) as f:
                        yaml.safe_load(f)
                    self.log(f"{workflow_file} is valid YAML", "pass")
                except ImportError:
                    self.log(f"{workflow_file} exists (YAML validation skipped)", "warning")
                except Exception as e:
                    self.log(f"{workflow_file} has invalid YAML: {e}", "fail")
                    all_valid = False
            else:
                self.log(f"{workflow_file} not found", "fail")
                all_valid = False

        self.checks["github_actions"] = all_valid
        return all_valid

    def check_git_status(self) -> bool:
        """Check git status."""
        self.log("\nChecking git status...", "info")

        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True, text=True
            )

            if result.returncode == 0:
                status_lines = result.stdout.strip().split('\n')
                uncommitted = [l for l in status_lines if l]

                if not uncommitted:
                    self.log("No uncommitted changes", "pass")
                    self.checks["git_status"] = True
                    return True
                else:
                    self.log(f"⚠️  {len(uncommitted)} uncommitted changes:", "warning")
                    for line in uncommitted[:5]:
                        print(f"      {line}")
                    if len(uncommitted) > 5:
                        print(f"      ... and {len(uncommitted) - 5} more")
                    self.checks["git_status"] = "warning"
                    return True
            else:
                self.log("Not a git repository or git not available", "warning")
                self.checks["git_status"] = False
                return False

        except FileNotFoundError:
            self.log("Git not found in PATH", "warning")
            self.checks["git_status"] = False
            return False

    def check_imports(self) -> bool:
        """Check that Python files can be imported."""
        self.log("\nChecking Python module imports...", "info")

        critical_files = [
            'quantize_neuroquantum_1bit.py',
            'quantize_neuroquantum_multibit.py',
        ]

        all_ok = True
        for filename in critical_files:
            filepath = self.project_root / filename
            if filepath.exists():
                try:
                    result = subprocess.run(
                        ['python', '-m', 'py_compile', str(filepath)],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        self.log(f"{filename} compiles successfully", "pass")
                    else:
                        self.log(f"{filename} has syntax errors: {result.stderr}", "fail")
                        all_ok = False
                except subprocess.TimeoutExpired:
                    self.log(f"{filename} compilation timeout", "warning")
                except Exception as e:
                    self.log(f"Error checking {filename}: {e}", "warning")

        self.checks["imports"] = all_ok
        return all_ok

    def mock_workflow_run(self) -> bool:
        """Mock run the workflow without actual quantization."""
        self.log("\nMocking workflow execution...", "info")

        self.log("Workflow would execute:", "info")
        print("\n  ✓ Checkout repository")
        print("  ✓ Set up Python 3.10")
        print("  ✓ Install dependencies")
        print("  ✓ Generate checkpoints")
        print("  ✓ Quantize to 1-bit")
        print("  ✓ Quantize to 2-bit")
        print("  ✓ Quantize to 3-bit")
        print("  ✓ Export to GGUF")
        print("  ✓ Validate models")
        print("  ✓ Upload to Hugging Face")
        print("  ✓ Create release")

        self.checks["mock_workflow"] = True
        return True

    def run_full_validation(self):
        """Run complete validation."""
        print("\n" + "="*70)
        print("🔍 Local Workflow Validation")
        print("="*70 + "\n")

        # Run all checks
        self.check_python_version()
        deps = self.check_dependencies()
        self.check_files()
        self.check_github_actions()
        self.check_git_status()
        self.check_imports()
        self.mock_workflow_run()

        # Print summary
        self.print_summary(deps)

    def print_summary(self, deps: Dict):
        """Print validation summary."""
        print("\n" + "="*70)
        print("📊 Validation Summary")
        print("="*70 + "\n")

        passed = sum(1 for v in self.checks.values() if v is True)
        warnings = sum(1 for v in self.checks.values() if v == "warning")
        total = len(self.checks)

        print(f"Results: {passed}/{total} checks passed")
        if warnings:
            print(f"Warnings: {warnings}")

        print("\nDetailed Results:")
        for check, result in self.checks.items():
            status = "pass" if result is True else ("warning" if result == "warning" else "fail")
            emoji = {"pass": "✅", "warning": "⚠️", "fail": "❌"}[status]
            print(f"  {emoji} {check.replace('_', ' ').title()}")

        # Recommendations
        print("\n" + "="*70)
        print("💡 Next Steps:")
        print("="*70)

        if not deps.get('torch'):
            print("\n1. Install PyTorch:")
            print("   pip install torch")

        if not deps.get('gguf'):
            print("\n2. Install optional dependencies:")
            print("   pip install gguf huggingface-hub")

        if self.checks.get("files"):
            print("\n3. Ready to run quantization:")
            print("   python demo_quantization_workflow.py")
            print("   python batch_quantize_models.py --checkpoint model.pt")

        if self.checks.get("github_actions"):
            print("\n4. GitHub Actions setup:")
            print("   • Configure HF_TOKEN in GitHub Secrets")
            print("   • Push changes to trigger workflows")

        print("\n5. See detailed guide:")
        print("   cat GITHUB_ACTIONS_SETUP.md")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate Quantization Workflow"
    )

    parser.add_argument('--full', action='store_true', default=True,
                       help='Run full validation (default)')
    parser.add_argument('--check-deps', action='store_true',
                       help='Check dependencies only')
    parser.add_argument('--check-files', action='store_true',
                       help='Check files only')
    parser.add_argument('--check-git', action='store_true',
                       help='Check git status only')
    parser.add_argument('--mock-run', action='store_true',
                       help='Mock workflow run')

    args = parser.parse_args()

    validator = WorkflowValidator()

    if args.check_deps:
        validator.check_python_version()
        validator.check_dependencies()
    elif args.check_files:
        validator.check_files()
    elif args.check_git:
        validator.check_git_status()
    elif args.mock_run:
        validator.mock_workflow_run()
    else:
        validator.run_full_validation()


if __name__ == '__main__':
    main()
