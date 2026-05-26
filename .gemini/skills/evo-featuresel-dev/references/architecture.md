# 🧬 Chromosome Architecture & Bit-Mapping

The chromosome is stored as a **packed uint8 array** (bit-packed) for maximum performance and cache locality.

## 📦 Chromosome Structure
The `Individual.bits` dictionary defines the segments:
1.  **Features**: `bits['features']` (e.g., 100 bits) - Presence/Absence of each feature.
2.  **Model Selection**: `bits['model_selection']` (e.g., 2 bits) - Index of the classifier.
3.  **Hyperparameters**: `bits['model_params']` (e.g., 11 bits) - Discretized values for parameters.

## ⚡ Decoding Logic
Decoding is handled in `evo/core.pyx` for performance:
-   `decode_individual(genes, bits)`: Returns a tuple `(model_selection_index, model_params_dict)`.
-   The bits are mapped linearly:
    -   Bits `[0 : bits['features']]` -> Feature Selection
    -   Bits `[bits['features'] : bits['features'] + bits['model_selection']]` -> Model Selection
    -   Bits `[bits['features'] + bits['model_selection'] : end]` -> Parameters

## 🔥 OpenMP & Multithreading
-   **Batch Crossover/Mutation**: Parallelized at the C level across the population.
-   **Fitness Evaluation**: Uses `pebble.ProcessPool` in `evo/population.py` to leverage multiple cores for model training and enforces strict evaluation timeouts (`PATIENCE`).
-   **Thread Oversubscription Control**: Dynamic models MUST be instantiated with `n_jobs=1` to prevent Scikit-learn from spawning OpenMP threads that clash with the process pool.
-   **GIL Management**: Cython functions use `with nogil` where possible to allow true parallel genetic operations.
