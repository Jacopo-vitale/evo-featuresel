# 🚀 Extension Guide: Models & Metrics

## 1. Adding a New Classifier
To add a new model (e.g., XGBoost):

1.  **Update `Individual.fitness_eval`**:
    -   Add a new case in the `match (self.model_sel):` block.
    -   Import the classifier in `evo/individual.py`.
2.  **Update `evo/core.pyx` (if needed)**:
    -   Adjust `decode_individual` to handle bits for the new model's parameters.
3.  **Update `Setup.BITS`**:
    -   Increase `model_selection` bits if necessary.
    -   Allocate bits for the new model's hyperparameters in `model_params`.

## 2. Adding a New Fitness Metric
To add a new metric (e.g., PR AUC):

1.  **Modify `evo/core.pyx`**:
    -   Add a Cython implementation in `fast_binary_metrics` or a new standalone function.
    -   Recompile with `python setup.py build_ext --inplace`.
2.  **Update `Individual.fitness_eval`**:
    -   Call the new Cython metric and assign it to `self._fitness`.
3.  **Update `Runner.save_detailed_report`**:
    -   Add the new metric to the CSV logging and summary report.

## 3. Custom Seeding Strategy
To add a new seeding strategy (e.g., "Always-Half" seeds):

1.  **Modify `Population.init_population`**:
    -   Add the new seed logic in the `edge_cases_genes` list.
    -   Ensure the seed is correctly bit-packed using `pack_bits`.
