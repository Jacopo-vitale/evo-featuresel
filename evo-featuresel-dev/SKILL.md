---
name: evo-featuresel-dev
description: Expert-level guidance for extending the evo-featuresel framework. Use when adding new models, modifying the Cython core, or orchestrating high-performance experiments.
---

# 🧬 evo-featuresel-dev

This skill provides specialized workflows for the high-performance evolutionary feature selection framework.

## 🚀 Key Workflows

### 1. Extending the Framework
-   **Add a New Model**: Step-by-step instructions for bit-mapping and classifier integration. See [extension_guide.md](references/extension_guide.md).
-   **Custom Fitness Metrics**: How to implement fast Cython-based metrics. See [extension_guide.md](references/extension_guide.md).
-   **Chromosome Architecture**: Detailed map of bit-packed segments. See [architecture.md](references/architecture.md).

### 🛠️ Tooling & Scripts
-   **Build C Extensions**: Use the provided script to recompile and verify `evo.core`.
    ```bash
    python evo-featuresel-dev/scripts/build_ext.py
    ```

### 🧪 Experiment Setup
-   **Stability/Robustness**: Guide for running multi-seed experiments to evaluate feature selection consistency.
-   **Parallel Optimization**: Best practices for balancing OpenMP and `ProcessPoolExecutor`.

## 🧠 Core Principles
1.  **Bit-Packing Integrity**: Never bypass `pack_bits`/`unpack_bits` for genomic data.
2.  **No GIL in Cython**: Ensure heavy genetic operations are optimized with `with nogil` and OpenMP pragmas.
3.  **Reproducibility**: Always use `Setup.seed_all` and independent child seeds for workers.
