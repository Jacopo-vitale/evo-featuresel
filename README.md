# 🚀 evo-featuresel 🧬

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C/Cython Optimized](https://img.shields.io/badge/core-C/Cython-red.svg)](https://cython.org/)
[![OpenMP Parallel](https://img.shields.io/badge/parallel-OpenMP-orange.svg)](https://www.openmp.org/)

**Binary Evolutionary Algorithm** for high-performance joint feature and model selection. Super-powered with **C/Cython** and **OpenMP** for maximum efficiency.

---

## 🌟 Key Features

- **⚡ C-Powered Performance**: Core evolutionary operations (crossover, mutation, decoding) are implemented in C via Cython, delivering up to **20x speedup** over standard Python.
- **📦 Bit-Packed DNA**: Chromosomes are stored as compact bit-packed arrays (1 bit per bit), reducing memory footprint by **8x** and improving cache locality.
- **🔥 OpenMP Parallelization**: Batch crossover and mutation are executed in parallel at the C level, bypassing the Python Global Interpreter Lock (GIL).
- **🧬 Joint Optimization**: Simultaneously optimizes feature subsets, model selection, and discretized hyperparameters in a single evolutionary run.
- **🌱 Edge-Case Seeding**: Initial populations are automatically seeded with extreme "All-Ones" (dense) and "Single-Bit" (sparse) individuals to discover optimal solutions faster.

---

## 🛠️ Supported Models

The algorithm currently optimizes across:
1.  **Random Forest Classifier** 🌲
2.  **Support Vector Classifier (SVC)** 🛰️
3.  **Gradient Boosting Classifier** 🚀
4.  **Extra Trees Classifier** 🌳

---

## ⚙️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Jacopo-vitale/evo-featuresel.git
    cd evo-featuresel
    ```

2.  **Install dependencies and build C extensions**:
    ```bash
    pip install .
    python setup.py build_ext --inplace
    ```
    *Note: Building requires a C compiler (GCC/Clang on Linux/macOS, MSVC on Windows).*

---

## 🚀 Usage

### 🏃 Quick Start
Run the main experiment with a demonstration dataset:
```bash
python main.py
```

### 📊 Performance Benchmarking
Verify the C/Cython speedups on your hardware:
```bash
python benchmark_c_vs_py.py
```

### 🧪 Unit Testing
Run the optimized core tests:
```bash
python -m pytest tests/
```

---

## 🧠 How it Works

The algorithm uses a **Binary Chromosome (DNA)** structured as follows:
-   **Features Segment**: `[f1, f2, ..., fN]` (Presence/Absence of features)
-   **Model Segment**: `[m1, m2]` (Binary choice of the classifier)
-   **Hyperparameter Segment**: `[p1, p2, ..., pK]` (Discretized model parameters)

The **Fitness** is evaluated using the **Matthews Correlation Coefficient (MCC)** to ensure robust performance even on imbalanced datasets.

---

## 📂 Project Structure

```text
evo-featuresel/
├── evo/                # Core Package
│   ├── core.pyx        # ⚡ Cython optimized operations (C-level)
│   ├── individual.py   # Individual DNA/Phenotype logic
│   ├── population.py   # Parallel population management (OpenMP)
│   ├── runner.py       # Evolutionary process orchestration
│   └── utils.py        # Configuration & Setup helpers
├── tests/              # 🧪 Unit test suite
├── main.py             # 🚀 Entry point
├── setup.py            # 🛠️ C extension build configuration
└── pyproject.toml      # Project metadata & dependencies
```

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---
*Developed with ❤️ for high-efficiency machine learning research.*
