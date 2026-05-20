# 🚀 Quick Start Guide
# クイックスタートガイド

Get quantized models running in 5 minutes.

---

## ⚡ 5-Minute Quick Start

### 1. Install dependencies
```bash
pip install torch numpy gguf huggingface-hub
```

### 2. Understand the options (2 minutes)
```bash
python demo_quantization_workflow.py
```

This shows:
- 📊 Size comparison (1-bit: 16MB, 2-bit: 31MB, 3-bit: 47MB)
- ⚡ Speed improvements (1-bit: 10x, 2-bit: 5x, 3-bit: 3x)
- 💡 Best use cases for each bit-width
- 📋 Step-by-step commands

### 3. Validate your setup
```bash
python validate_workflow_local.py --full
```

Checks: dependencies, files, GitHub Actions workflows, git status

### 4. Generate quantized models
```bash
# Option A: Generate just 2-bit (recommended, fastest)
python quantize_neuroquantum_multibit.py checkpoint.pt --bit-width 2

# Option B: Generate all three (1, 2, 3-bit)
python batch_quantize_models.py --checkpoint checkpoint.pt --all
```

### 5. Export to GGUF (for mobile/edge)
```bash
# Option A: Export 2-bit
python export_multibit_gguf.py model_2bit.pt --bit-width 2

# Option B: Batch export all
python batch_quantize_models.py --checkpoint checkpoint.pt --all --export-gguf
```

**Done!** 🎉

---

## 🎯 Choose Your Path

### Path A: I want to just understand the system
```bash
python demo_quantization_workflow.py
python compare_quantized_models.py --analyze-all
# Takes 2 minutes, no actual model generation
```

### Path B: I want to deploy mobile (⭐ RECOMMENDED)
```bash
python demo_quantization_workflow.py           # Understand options
python validate_workflow_local.py --full        # Check setup
python batch_quantize_models.py \              # Generate all
  --checkpoint checkpoint.pt \
  --bit-widths 2 \
  --export-gguf
python upload_to_huggingface.py model_2bit.gguf  # Deploy
```

### Path C: I want to compare all options
```bash
python compare_quantized_models.py --analyze-all  # See comparison
python batch_quantize_models.py \                 # Generate all
  --checkpoint checkpoint.pt \
  --all
# Compare file sizes, accuracy, speed in your use case
```

### Path D: I want GitHub Actions to automate this
```bash
python validate_workflow_local.py --full        # Validate setup
# Then:
# 1. Add HF_TOKEN to GitHub Secrets
# 2. Push checkpoint.pt to main
# 3. Workflows automatically generate and upload
```

---

## 📊 Which bit-width should I use?

| Device | Bit-Width | File Size | Speed | Accuracy | Command |
|--------|-----------|-----------|-------|----------|---------|
| IoT/Arduino | 1-bit | 16 MB | 10x | 75-90% | `quantize_neuroquantum_1bit.py` |
| Mobile Phone | **2-bit ⭐** | **31 MB** | **5x** | **90-95%** | `--bit-width 2` |
| High-End Phone | 3-bit | 47 MB | 3x | 95-98% | `--bit-width 3` |

**👉 When in doubt, use 2-bit** (best balance of speed, size, and accuracy)

---

## 🛠️ Helper Tools Reference

| Situation | Tool | Command |
|-----------|------|---------|
| Learning | demo | `python demo_quantization_workflow.py` |
| Comparing | compare | `python compare_quantized_models.py --analyze-all` |
| Validating | validate | `python validate_workflow_local.py --full` |
| Batch processing | batch | `python batch_quantize_models.py --checkpoint model.pt --all` |

---

## 📋 Step-by-Step for Beginners

### Step 1: Prepare checkpoint
```bash
# Make sure you have checkpoint.pt in the current directory
ls -lh checkpoint.pt
```

### Step 2: Install & validate
```bash
pip install torch numpy gguf
python validate_workflow_local.py --full
```

### Step 3: Understand options
```bash
python demo_quantization_workflow.py
```

You'll see output like:
```
📱 Use Cases & Recommendations:

  2-bit ⭐ RECOMMENDED:
    Environments: Mobile phones, Edge devices, Web browsers
    Size: 31 MB, Accuracy: 90-95%, Speed: 5x faster
    Command: python quantize_neuroquantum_multibit.py checkpoint.pt --bit-width 2
```

### Step 4: Quantize
```bash
# For mobile (recommended)
python quantize_neuroquantum_multibit.py checkpoint.pt --bit-width 2

# This creates: model_2bit.pt
```

### Step 5: Export to GGUF
```bash
python export_multibit_gguf.py model_2bit.pt --bit-width 2

# This creates: model_2bit_2bit.gguf
```

### Step 6: Verify
```bash
python check_gguf_params.py model_2bit_2bit.gguf

# Output should show:
# ✓ Model parameters
# ✓ Metadata
# ✓ Ready for deployment
```

### Step 7: Use the model
```bash
# Example usage
python examples_gguf_client.py model_2bit_2bit.gguf "Hello, world!"
```

### Step 8: Deploy (optional)
```bash
python upload_to_huggingface.py model_2bit_2bit.gguf
```

---

## 🚨 Common Issues

### Issue: "No module named 'torch'"
**Solution:**
```bash
pip install torch
```

### Issue: "Checkpoint not found"
**Solution:**
```bash
# Make sure checkpoint.pt is in current directory
ls checkpoint.pt

# Or use full path
python quantize_neuroquantum_multibit.py /path/to/checkpoint.pt --bit-width 2
```

### Issue: "GGUF file not created"
**Solution:**
```bash
# Check if model_2bit.pt exists
ls model_2bit.pt

# If not, quantization failed - check output above
# If it exists, try export again:
python export_multibit_gguf.py model_2bit.pt --bit-width 2 -v
```

### Issue: "ImportError in quantization"
**Solution:**
```bash
# Make sure neuroquantum_layered.py exists
ls neuroquantum_layered.py

# If not, check if you're in the right directory
pwd
```

---

## 📚 Where to go next

### Want to learn more?
- `HELPER_TOOLS_GUIDE.md` - Detailed documentation for helper tools
- `MULTIBIT_QUANTIZATION_COMPARISON.md` - Deep comparison of methods
- `BINARY_1BIT_QUANTIZATION_GUIDE.md` - 1-bit deep dive
- `GGUF_LOADING_TROUBLESHOOTING.md` - GGUF issues & solutions

### Want to deploy?
- `GITHUB_ACTIONS_SETUP.md` - Automated deployment with CI/CD
- `upload_to_huggingface.py` - Manual upload to Hugging Face

### Want to benchmark?
```bash
python examples_gguf_client.py model_2bit.gguf "test input" --benchmark
```

### Want to compare all bit-widths?
```bash
python compare_quantized_models.py --analyze-all
```

---

## ✅ Success Checklist

- [ ] Dependencies installed (`pip install torch numpy gguf`)
- [ ] Validation passed (`python validate_workflow_local.py --full`)
- [ ] Understand options (`python demo_quantization_workflow.py`)
- [ ] Model quantized (`python quantize_neuroquantum_multibit.py --bit-width 2`)
- [ ] Exported to GGUF (`python export_multibit_gguf.py model_2bit.pt`)
- [ ] Verified (`python check_gguf_params.py model_2bit_2bit.gguf`)
- [ ] Ready to deploy! 🎉

---

## 🎓 Learning Path

**5 min**: `demo_quantization_workflow.py` → Understand options  
**10 min**: `validate_workflow_local.py` → Verify setup  
**15 min**: `batch_quantize_models.py --checkpoint checkpoint.pt --all` → Generate models  
**5 min**: `compare_quantized_models.py --analyze-all` → Compare results  
**∞**: Deploy and enjoy! 🚀

---

## 💡 Pro Tips

### Tip 1: Use batch processing
```bash
# Fast (generates 1, 2, 3-bit in one go)
python batch_quantize_models.py --checkpoint checkpoint.pt --all

# vs. Manual (3x slower)
python quantize_neuroquantum_1bit.py checkpoint.pt
python quantize_neuroquantum_multibit.py checkpoint.pt --bit-width 2
python quantize_neuroquantum_multibit.py checkpoint.pt --bit-width 3
```

### Tip 2: Validate before CI/CD
```bash
# Always run this first
python validate_workflow_local.py --full

# Saves debugging time in GitHub Actions
```

### Tip 3: Save batch reports
```bash
# Batch creates JSON reports
cat batch_report_20260520_143022.json

# Keep for reference and comparison
```

### Tip 4: Start with demo
```bash
# Always run this first to understand options
python demo_quantization_workflow.py
```

### Tip 5: Default to 2-bit
```bash
# When unsure, use 2-bit
# It's the best balance for most use cases
python quantize_neuroquantum_multibit.py checkpoint.pt --bit-width 2
```

---

## 🆘 Need Help?

### Quick troubleshooting
```bash
# Check everything
python validate_workflow_local.py --full

# See what went wrong
python check_gguf_params.py model.gguf --diagnose

# Try with verbose output
python quantize_neuroquantum_multibit.py checkpoint.pt --bit-width 2 -v
```

### Read documentation
1. `QUICKSTART.md` (this file) - 5 minute overview
2. `HELPER_TOOLS_GUIDE.md` - Tool documentation
3. `MULTIBIT_QUANTIZATION_COMPARISON.md` - Detailed comparison
4. `GGUF_LOADING_TROUBLESHOOTING.md` - GGUF problems

### Common issues
```bash
# PyTorch not installed
pip install torch

# GGUF not installed
pip install gguf

# File not found
ls checkpoint.pt  # Check path

# Import error
python validate_workflow_local.py --check-imports

# Validation failed
python validate_workflow_local.py --full
```

---

## 📞 Quick Commands Cheat Sheet

```bash
# Understand the system
python demo_quantization_workflow.py

# Validate setup
python validate_workflow_local.py --full

# Quantize to 2-bit (recommended)
python quantize_neuroquantum_multibit.py checkpoint.pt --bit-width 2

# Generate all (1, 2, 3-bit)
python batch_quantize_models.py --checkpoint checkpoint.pt --all

# Export to GGUF
python export_multibit_gguf.py model_2bit.pt --bit-width 2

# Check output
python check_gguf_params.py model_2bit_2bit.gguf

# Test inference
python examples_gguf_client.py model_2bit_2bit.gguf "test input"

# Upload to HF
python upload_to_huggingface.py model_2bit_2bit.gguf

# Compare methods
python compare_quantized_models.py --analyze-all
```

---

**Status**: ✅ Ready to use  
**Time to first model**: 5 minutes  
**Recommended**: Start with 2-bit  
**Next step**: Run `python demo_quantization_workflow.py`
