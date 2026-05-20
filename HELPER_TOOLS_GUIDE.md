# Helper Tools & Scripts Guide
# ヘルパーツール・スクリプトガイド

Comprehensive guide to helper tools for quantization workflows.

---

## 📚 Tools Overview

| Tool | Purpose | Usage |
|------|---------|-------|
| `demo_quantization_workflow.py` | Interactive demo and learning | `python demo_quantization_workflow.py` |
| `batch_quantize_models.py` | Batch process multiple models | `python batch_quantize_models.py --checkpoint model.pt --all` |
| `validate_workflow_local.py` | Validate setup before GitHub Actions | `python validate_workflow_local.py --full` |
| `compare_quantized_models.py` | Compare and analyze models | `python compare_quantized_models.py --analyze-all` |

---

## 1️⃣ Demo Workflow Tool

### Purpose
Interactive demonstration of the quantization system with:
- Use case explanations
- Compression ratios and performance estimates
- Step-by-step workflow guidance
- Quick reference commands
- Troubleshooting help

### Usage

#### Show everything (no checkpoint required)
```bash
python demo_quantization_workflow.py
```

#### With checkpoint file
```bash
python demo_quantization_workflow.py --checkpoint checkpoint.pt
```

#### For specific bit-width
```bash
python demo_quantization_workflow.py --checkpoint checkpoint.pt --bit-width 2
```

#### Process all bit-widths
```bash
python demo_quantization_workflow.py --checkpoint checkpoint.pt --all
```

#### Comparison only
```bash
python demo_quantization_workflow.py --checkpoint checkpoint.pt --compare-only
```

### Output

```
🚀 NeuroQuantum Quantization Workflow Demo
==================================================

📱 Use Cases & Recommendations:

  1-bit:
    Environments: IoT devices, Embedded systems, Raspberry Pi
    Size: 16 MB, Accuracy: 75-90%, Speed: 10x faster
    Command: python quantize_neuroquantum_1bit.py checkpoint.pt

  2-bit ⭐ RECOMMENDED:
    Environments: Mobile phones, Edge devices, Web browsers
    Size: 31 MB, Accuracy: 90-95%, Speed: 5x faster
    Command: python quantize_neuroquantum_multibit.py checkpoint.pt --bit-width 2

  3-bit:
    Environments: High-end phones, Tablets, Better accuracy
    Size: 47 MB, Accuracy: 95-98%, Speed: 3x faster
    Command: python quantize_neuroquantum_multibit.py checkpoint.pt --bit-width 3

📊 Size Estimates:
  F32:   512.0 MB (1.0x base)
  1-bit: 16.0 MB (32.0x compression)
  2-bit: 31.0 MB (16.0x compression)
  3-bit: 47.0 MB (10.7x compression)

📋 Workflow for 2-bit Quantization:

  Step 1: Load checkpoint
    checkpoint.pt

  Step 2: Quantize
    quantize_neuroquantum_multibit.py checkpoint.pt --bit-width 2

  Step 3: Export to GGUF
    export_multibit_gguf.py model_2bit.pt --bit-width 2

  ...and more
```

---

## 2️⃣ Batch Processing Tool

### Purpose
Generate multiple quantized models with different bit-widths in a single run:
- Process 1-bit, 2-bit, 3-bit in sequence
- Optional GGUF export
- Automatic validation
- Comprehensive reporting
- Batch ID tracking

### Usage

#### Default (2-bit only)
```bash
python batch_quantize_models.py --checkpoint checkpoint.pt
```

#### Specific bit-widths
```bash
python batch_quantize_models.py --checkpoint checkpoint.pt --bit-widths 1,2,3
```

#### With GGUF export
```bash
python batch_quantize_models.py --checkpoint checkpoint.pt \
  --bit-widths 2 --export-gguf
```

#### Complete workflow (all bit-widths + GGUF)
```bash
python batch_quantize_models.py --checkpoint checkpoint.pt --all
```

#### Skip validation
```bash
python batch_quantize_models.py --checkpoint checkpoint.pt --all --no-validate
```

### Output

```
🚀 Batch Quantization Processing
==================================================
Batch ID: 20260520_143022
Checkpoint: checkpoint.pt
Bit-widths: [1, 2, 3]
Export GGUF: True
==================================================

[14:30:22] 📋 Verifying checkpoint: checkpoint.pt
[14:30:22] ✅ ✓ Checkpoint is valid

======================================================================
[14:30:22] Processing 1-bit quantization
======================================================================
[14:30:22] ⚙️  Starting 1-bit quantization
[14:31:45] ✅ ✓ 1-bit quantization complete
[14:31:46] ⚙️  Exporting 1-bit to GGUF
[14:31:52] ✅ ✓ GGUF export complete
[14:31:52] ⚙️  Validating 1-bit outputs
[14:31:53] ✅ ✓ Model file valid (16.0 MB)
[14:31:53] ✅ ✓ GGUF file valid (15.8 MB)

...continues for 2-bit and 3-bit...

📊 Batch Processing Summary
==================================================

Results: 3/3 successful
Total time: 125.5 seconds

Bit-Width | Quantization | GGUF Export | Validation | Duration
----------------------------------------------------------------------
    1     |      ✓       |      ✓      |     ✓      |   45.2s
    2     |      ✓       |      ✓      |     ✓      |   39.8s
    3     |      ✓       |      ✓      |     ✓      |   40.5s

✅ Report saved: batch_report_20260520_143022.json

💡 Next Steps:
==================================================

1. Verify outputs:
   python check_gguf_params.py checkpoint_1bit.gguf
   python check_gguf_params.py checkpoint_2bit.gguf
   python check_gguf_params.py checkpoint_3bit.gguf

2. Upload to Hugging Face:
   python upload_to_huggingface.py checkpoint_*bit.gguf

3. Review documentation:
   • MULTIBIT_QUANTIZATION_COMPARISON.md
   • GGUF_LOADING_TROUBLESHOOTING.md
```

### Batch Report (JSON)

Saved as `batch_report_<timestamp>.json` with:
- Quantization success status
- GGUF export status
- File sizes (MB)
- Validation results
- Timing information

---

## 3️⃣ Workflow Validation Tool

### Purpose
Validate local setup before running GitHub Actions:
- Check Python version
- Verify dependencies installed
- Check required files exist
- Validate YAML syntax in workflows
- Check git status
- Import verification
- Mock workflow execution

### Usage

#### Full validation (default)
```bash
python validate_workflow_local.py
```

#### Check dependencies only
```bash
python validate_workflow_local.py --check-deps
```

#### Check files only
```bash
python validate_workflow_local.py --check-files
```

#### Check git status
```bash
python validate_workflow_local.py --check-git
```

#### Mock run
```bash
python validate_workflow_local.py --mock-run
```

### Output

```
🔍 Local Workflow Validation

Checking Python version...
✅ Python 3.10 is compatible

Checking dependencies...

  Required:
  ✅ PyTorch is installed
  ✅ NumPy is installed

  Optional:
  ✅ GGUF is installed
  ✅ Hugging Face Hub is installed

Checking required files...

  Required:
  ✅ quantize_neuroquantum_1bit.py
  ✅ quantize_neuroquantum_multibit.py
  ✅ export_1bit_gguf.py
  ...

📊 Validation Summary
==================================================

Results: 7/7 checks passed
Warnings: 0

Detailed Results:
  ✅ Python Version
  ✅ Dependencies
  ✅ Files
  ✅ Github Actions
  ✅ Git Status
  ✅ Imports
  ✅ Mock Workflow

💡 Next Steps:
==================================================

3. Ready to run quantization:
   python demo_quantization_workflow.py
   python batch_quantize_models.py --checkpoint model.pt

4. GitHub Actions setup:
   • Configure HF_TOKEN in GitHub Secrets
   • Push changes to trigger workflows

5. See detailed guide:
   cat GITHUB_ACTIONS_SETUP.md
```

---

## 4️⃣ Model Comparison & Analysis Tool

### Purpose
Compare quantization methods with detailed analysis:
- Side-by-side metrics comparison
- Use case recommendations
- Selection decision tree
- Performance benchmarks
- Memory requirements
- Accuracy estimates
- Deployment guidelines

### Usage

#### Show analysis
```bash
python compare_quantized_models.py
```

#### Generate report
```bash
python compare_quantized_models.py --generate-report
```

#### Compare specific models
```bash
python compare_quantized_models.py --models model_1bit.pt model_2bit.pt
```

### Output

```
═══════════════════════════════════════════════════════════════════════════
                        Quantization Comparison
═════════════════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────────────┐
│                        Quantization Comparison                           │
├──────────┬──────────────┬────────────┬────────────┬──────────┬──────────┤
│ Type     │ Size (MB)    │ Compression│ Reduction  │ Accuracy │ Speed    │
├──────────┼──────────────┼────────────┼────────────┼──────────┼──────────┤
│ f32      │        512.0 │       1.0x │       0.0% │   100.0% │     1.0x │
│ 1bit     │         16.0 │      32.0x │      96.9% │    82.5% │    10.0x │
│ 2bit     │         31.0 │      16.0x │      93.9% │    92.5% │     5.0x │
│ 3bit     │         47.0 │      10.7x │      90.8% │    96.5% │     3.0x │
└──────────┴──────────────┴────────────┴────────────┴──────────┴──────────┘

🎯 Quick Selection Guide

Choose your use case:

1️⃣  IoT / Embedded Devices
   → Use 1-bit
   → Command: python quantize_neuroquantum_1bit.py checkpoint.pt

2️⃣  Mobile Apps (⭐ RECOMMENDED)
   → Use 2-bit
   → Command: python quantize_neuroquantum_multibit.py checkpoint.pt --bit-width 2

3️⃣  High-End Devices / High Accuracy
   → Use 3-bit
   → Command: python quantize_neuroquantum_multibit.py checkpoint.pt --bit-width 3

4️⃣  Try All (for comparison)
   → Compare compression & accuracy
   → Command: python quantize_neuroquantum_multibit.py checkpoint.pt --compare
```

---

## 🚀 Typical Workflow

### Scenario: Deploy mobile app with quantized model

```bash
# Step 1: Understand options
python demo_quantization_workflow.py

# Step 2: See detailed comparison
python compare_quantized_models.py

# Step 3: Validate setup
python validate_workflow_local.py --full

# Step 4: Batch quantize all bit-widths
python batch_quantize_models.py --checkpoint checkpoint.pt --all

# Step 5: Verify outputs
python check_gguf_params.py checkpoint_2bit.gguf --diagnose

# Step 6: Test inference
python examples_gguf_client.py checkpoint_2bit.gguf "test input"

# Step 7: Upload to HF
python upload_to_huggingface.py checkpoint_2bit.gguf
```

---

## 📊 Decision Matrix

| Situation | Tool | Command |
|-----------|------|---------|
| Learning the system | demo_quantization_workflow | `python demo_quantization_workflow.py` |
| Generating all models | batch_quantize_models | `python batch_quantize_models.py --checkpoint model.pt --all` |
| Before GitHub Actions | validate_workflow_local | `python validate_workflow_local.py --full` |
| Choosing best option | compare_quantized_models | `python compare_quantized_models.py --analyze-all` |

---

## 💡 Pro Tips

### Tip 1: Start with demo
Always run the demo first to understand options:
```bash
python demo_quantization_workflow.py
```

### Tip 2: Use batch processing
Instead of running quantization 3 times manually, use batch:
```bash
# Batch (recommended)
python batch_quantize_models.py --checkpoint model.pt --all

# vs. Manual (slower)
python quantize_neuroquantum_1bit.py model.pt
python quantize_neuroquantum_multibit.py model.pt --bit-width 2
python quantize_neuroquantum_multibit.py model.pt --bit-width 3
```

### Tip 3: Validate before GitHub Actions
Always run validation locally:
```bash
python validate_workflow_local.py --full
```

### Tip 4: Save batch reports
Batch tool creates JSON reports automatically - keep them for reference:
```bash
cat batch_report_20260520_143022.json
```

### Tip 5: Use comparison for selection
When unsure which bit-width to use:
```bash
python compare_quantized_models.py --analyze-all
# Then follow the "Selection Decision Tree"
```

---

## 🆘 Troubleshooting

### Issue: "No module named 'torch'"
```bash
pip install torch numpy gguf
```

### Issue: "validate_workflow_local.py: command not found"
```bash
python validate_workflow_local.py --full
# not ./validate_workflow_local.py
```

### Issue: Batch processing too slow
Use parallel submission instead:
```bash
# Run each bit-width in separate terminals
python quantize_neuroquantum_1bit.py checkpoint.pt &
python quantize_neuroquantum_multibit.py checkpoint.pt --bit-width 2 &
python quantize_neuroquantum_multibit.py checkpoint.pt --bit-width 3 &
wait
```

### Issue: "checkpoint not found"
Make sure path is correct:
```bash
# Incorrect
python batch_quantize_models.py --checkpoint model.pt

# Correct (absolute or relative path)
python batch_quantize_models.py --checkpoint ./checkpoint.pt
python batch_quantize_models.py --checkpoint /path/to/checkpoint.pt
```

---

## 📚 Related Documentation

- `MULTIBIT_QUANTIZATION_COMPARISON.md` - Detailed comparison guide
- `BINARY_1BIT_QUANTIZATION_GUIDE.md` - 1-bit deep dive
- `GGUF_LOADING_TROUBLESHOOTING.md` - GGUF issues
- `GITHUB_ACTIONS_SETUP.md` - CI/CD setup

---

## ✅ Checklist

- [ ] Run `python demo_quantization_workflow.py`
- [ ] Review output and understand options
- [ ] Run `python validate_workflow_local.py --full`
- [ ] Create checkpoint file for testing
- [ ] Run `python batch_quantize_models.py --checkpoint checkpoint.pt --all`
- [ ] Review `batch_report_*.json` output
- [ ] Run `python compare_quantized_models.py --analyze-all`
- [ ] Choose quantization level (default: 2-bit)
- [ ] Test inference with chosen model
- [ ] Deploy to Hugging Face

---

**Created**: 2026-05-20  
**Status**: ✅ Complete  
**Tools**: 4 helper scripts  
**Documentation**: Comprehensive guide included
