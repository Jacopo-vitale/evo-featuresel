# evo-featuresel

Binary Evolutionary Algorithm for automatic features and model selection.

## Overview

This repository provides a custom Evolutionary Algorithm (EA) designed to optimize both feature selection and machine learning model selection (including hyperparameters) simultaneously.

The algorithm uses a binary representation (DNA/filament) where:
- A portion of the bits represents the presence or absence of specific features.
- A portion represents the choice of the machine learning model (e.g., Random Forest, SVM, etc.).
- A portion represents the discretized hyperparameters for the selected model.

## Features

- **Simultaneous Optimization**: Optimizes features and model parameters in a single evolutionary run.
- **Multithreading**: Uses `ThreadPoolExecutor` for parallel fitness evaluation of individuals.
- **Modular Design**: Easy to extend with new models or custom fitness functions.
- **Logging**: Detailed logging of experiments, including top individuals and fitness trends.

## Supported Models

1. **Random Forest Classifier**
2. **Support Vector Classifier (SVC)**
3. **Gradient Boosting Classifier**
4. **Extra Trees Classifier**

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jacopo/evo-featuresel.git
   cd evo-featuresel
   ```

2. Install dependencies:
   ```bash
   pip install .
   ```
   *Note: Requires Python 3.10+*

## Usage

### Basic Example

You can run the `main.py` script to see a demonstration using a generated dummy dataset:

```bash
python main.py
```

### Custom Dataset

To use your own dataset, ensure it's a CSV file and update the `preprocessing` logic in `main.py` or provide your own data loading routine that fits the `Setup` object's requirements.

The `Setup` object expects:
- `DATA`: A tuple `(X_train, X_test)` as numpy arrays.
- `LABELS`: A tuple `(y_train, y_test)` as numpy arrays.
- `BITS`: A dictionary defining how many bits are allocated to features, model selection, and parameters.

## Project Structure

- `evo/`: Core package containing the EA implementation.
  - `individual.py`: Definition of an individual (DNA to phenotype mapping).
  - `population.py`: Population management, crossover, and mutation logic.
  - `runner.py`: Orchestrates the evolutionary process.
  - `utils.py`: Setup and configuration helpers.
- `main.py`: Entry point for running experiments.
- `pyproject.toml`: Project metadata and dependency definitions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
