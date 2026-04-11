# 🧬 GEMINI.md - evo-featuresel

## 🌟 Project Overview
**evo-featuresel** is a high-performance **Binary Evolutionary Algorithm (EA)** framework designed for joint feature selection and machine learning model optimization. The project is heavily optimized using **C/Cython** and **OpenMP** to bypass the Python Global Interpreter Lock (GIL) and maximize computational efficiency.

### 🛠️ Core Technologies
- **Language**: Python 3.10+
- **Optimization**: Cython, C (OpenMP for multithreading)
- **Data Science**: NumPy, Pandas, Scikit-learn, Joblib
- **GUI**: PySide6, Matplotlib
- **Testing**: Pytest

### 🏗️ Architecture
The project follows a modular evolutionary architecture:
- **`evo/core.pyx`**: The performance heart. Contains bit-packed genetic operators (crossover, mutation) and fast fitness metrics (MCC, Accuracy) implemented in Cython.
- **`evo/individual.py`**: Defines candidate solutions. Chromosomes are stored as **bit-packed uint8 arrays** to minimize memory footprint and improve cache locality.
- **`evo/population.py`**: Manages the evolution cycle. It uses `ProcessPoolExecutor` for parallel fitness evaluations and calls Cython/OpenMP batch functions for genetic operations.
- **`evo/runner.py`**: Orchestrates the entire experiment lifecycle, including logging, robustness runs, and result serialization.
- **`evo/gui/`**: Interactive dashboard for real-time monitoring and parameter tuning.

---

## ⚙️ Building and Running

### 🛠️ Build C Extensions
Before running the algorithm, you must compile the Cython core:
```bash
python setup.py build_ext --inplace
```
*Requires a C compiler (GCC, Clang, or MSVC).*

### 🏃 Running the Project
- **CLI Mode**: `python main.py` (Main entry point for experiments)
- **GUI Dashboard**: `python gui_main.py` (Interactive UI)
- **Benchmarking**: `python benchmark_c_vs_py.py` (Compare Python vs. C performance)
- **Parallelism Verification**: `python verify_parallelism.py`

### 🧪 Testing
Run the unit test suite to ensure core logic integrity:
```bash
python -m pytest tests/
```

---

## 📜 Development Conventions

### ⚡ Performance-First Mindset
- **Bit-Packing**: Always use bit-packed representations (`pack_bits`, `unpack_bits`) for genomic data.
- **Cython for Hot Loops**: Any operation that iterates over large populations or long chromosomes should be moved to `evo/core.pyx`.
- **Parallelism**: Use OpenMP in Cython for batch operations. Use `ProcessPoolExecutor` for model training/evaluation to leverage multiple cores.

### 📂 Experiment Management
- Results are automatically saved to the `experiment/` directory, organized by timestamp.
- Each run generates:
    - `experiment.log`: Detailed execution logs.
    - `iron_man.joblib`: Serialized best model and its metadata.
    - `detailed_metrics.csv`: Performance metrics for the entire final population.
    - `config.evoconf`: JSON summary of the experiment parameters.

### 🧬 Evolutionary Logic
- **Fitness Metric**: Primarily uses **Matthews Correlation Coefficient (MCC)** to handle imbalanced datasets.
- **Seeding**: The population is initialized with "edge cases" (all-ones and single-bit individuals) to improve convergence on sparse/dense optima.
- **Joint Optimization**: The chromosome encodes features, model choice, and discretized hyperparameters.

---
*This file serves as a guide for Gemini CLI and developers to maintain consistency and performance standards in the evo-featuresel repository.*
