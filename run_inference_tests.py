#!/usr/bin/env python3
"""
Run chatbot_cli inference tests with sample prompts.
Generates a test report with results.
"""
import subprocess
import sys
import os
from datetime import datetime

os.chdir("/home/user/Qubit")

# Test cases
test_cases = [
    {
        "name": "Basic Greeting",
        "prompt": "こんにちは",
        "expected_type": "greeting_response"
    },
    {
        "name": "Math/Code Question",
        "prompt": "Pythonで階乗を計算するコードを書いてください",
        "expected_type": "code_generation"
    },
    {
        "name": "Factual Question",
        "prompt": "日本の首都は何ですか？",
        "expected_type": "factual_answer"
    },
]

def run_inference_test(prompt, timeout=120):
    """Run a single inference test through chatbot_cli"""
    input_text = f"{prompt}\n/exit\n"

    try:
        result = subprocess.run(
            ["python3", "chatbot_cli.py"],
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # Parse response from output
        lines = result.stdout.split("\n")
        response_lines = []
        capture = False

        for line in lines:
            if "▍ NeuroQuantum" in line:
                capture = True
                continue
            if capture:
                if line.strip().startswith("▍") or line.strip().startswith("❯"):
                    break
                if line.strip() and not line.startswith("["):
                    response_lines.append(line.strip())

        response = "\n".join(response_lines).strip()
        return {
            "success": True,
            "response": response,
            "stderr": result.stderr[:500] if result.stderr else None
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout after 120 seconds"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    print("=" * 70)
    print("NeuroQuantum Chat CLI - Inference Test Report")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Check model availability
    ckpt_path = "checkpoints/megabyte_100mb_mathcode_sft_best.pt"
    if not os.path.exists(ckpt_path):
        print(f"❌ Model checkpoint not found: {ckpt_path}")
        print("   Please ensure Git LFS has downloaded the model file.")
        return 1

    ckpt_size = os.path.getsize(ckpt_path)
    if ckpt_size < 1000000:  # Less than 1MB = likely a pointer file
        print(f"⚠️  Checkpoint appears to be a Git LFS pointer (size: {ckpt_size} bytes)")
        print("   Run: git lfs pull")
        return 1

    print(f"✅ Model checkpoint ready ({ckpt_size / 1e9:.1f}GB)")
    print()

    # Run tests
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}/{len(test_cases)}: {test_case['name']}")
        print(f"  Prompt: {test_case['prompt']}")
        print("  Running...", end="", flush=True)

        result = run_inference_test(test_case['prompt'])
        results.append({
            "test": test_case,
            "result": result
        })

        if result["success"]:
            response = result["response"][:100]
            print(f" ✅\n  Output: {response}...")
        else:
            print(f" ❌\n  Error: {result.get('error')}")

        print()

    # Summary
    print("=" * 70)
    success_count = sum(1 for r in results if r["result"]["success"])
    print(f"Results: {success_count}/{len(test_cases)} tests passed")
    print("=" * 70)

    # Save detailed report
    report_path = "inference_test_report.txt"
    with open(report_path, "w") as f:
        f.write("NeuroQuantum Chat CLI - Inference Test Report\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Model: {ckpt_path}\n")
        f.write("=" * 70 + "\n\n")

        for r in results:
            test = r["test"]
            result = r["result"]
            f.write(f"Test: {test['name']}\n")
            f.write(f"Prompt: {test['prompt']}\n")
            if result["success"]:
                f.write(f"Status: ✅ Success\n")
                f.write(f"Response: {result['response']}\n")
            else:
                f.write(f"Status: ❌ Failed\n")
                f.write(f"Error: {result.get('error')}\n")
            if result.get("stderr"):
                f.write(f"Warnings: {result['stderr']}\n")
            f.write("\n")

    print(f"Report saved: {report_path}")
    return 0 if success_count == len(test_cases) else 1


if __name__ == "__main__":
    sys.exit(main())
