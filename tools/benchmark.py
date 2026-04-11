import time
import numpy as np
import os
from evo.core import (
    fast_binary_to_decimal, 
    pack_bits, 
    unpack_bits, 
    decode_individual,
    fast_crossover_packed, 
    fast_mutation_packed
)

# --- Original Python Implementations (Simplified for Benchmarking) ---

def binaryToDecimal_py(binary):
    if len(binary) == 0:
        return 0
    return int("".join(map(str, map(int, binary))), 2)

def crossover_py(p1, p2, cp):
    return np.concatenate([p1[:cp], p2[cp:]])

def mutation_py(genes, rate):
    mutation_mask = np.random.uniform(0, 1, size=len(genes)) < rate
    genes[mutation_mask] = 1 - genes[mutation_mask]
    return genes

def to_phenotype_py(genes, bits):
    # This mimics the old individual.py logic
    feat_bits = bits['features']
    model_sel_bits = bits['model_selection']
    
    features = genes[:feat_bits]
    model_selection = binaryToDecimal_py(genes[feat_bits:feat_bits + model_sel_bits])
    param_bits = genes[feat_bits + model_sel_bits:]
    
    model_param = {}
    if model_selection == 0:
        n_estimators = binaryToDecimal_py(param_bits[:9])
        model_param['n_estimators'] = n_estimators if n_estimators > 2 else 2
        criterion_selector = binaryToDecimal_py(param_bits[9:11])
        model_param['criterion'] = 'gini' if criterion_selector == 0 else ('entropy' if criterion_selector == 1 else 'log_loss')
    # ... other models omitted for brevity as they follow same pattern
    return features, model_selection, model_param

# --- Benchmark Config ---
N_ITER = 1000000
FILAMENT_LEN = 500
BITS = {'features': 400, 'model_selection': 2, 'model_params': 98}

def run_benchmark():
    print(f"--- Benchmarking Python vs C (Cython) over {N_ITER} iterations ---")
    print(f"Filament Length: {FILAMENT_LEN} bits\n")

    # Prepare data
    p1_unpacked = np.random.choice([0, 1], size=FILAMENT_LEN).astype(np.int8)
    p2_unpacked = np.random.choice([0, 1], size=FILAMENT_LEN).astype(np.int8)
    p1_packed = pack_bits(p1_unpacked)
    p2_packed = pack_bits(p2_unpacked)
    cp = FILAMENT_LEN // 2
    mutation_rate = 0.1
    random_values = np.random.uniform(0, 1, size=FILAMENT_LEN)

    # 1. Binary to Decimal
    start = time.time()
    for _ in range(N_ITER):
        binaryToDecimal_py(p1_unpacked[:32])
    py_time = time.time() - start

    start = time.time()
    from evo.core import fast_binary_to_decimal_packed
    for _ in range(N_ITER):
        fast_binary_to_decimal_packed(p1_packed, 0, 32)
    c_time = time.time() - start
    print(f"Binary-to-Decimal (32 bits):")
    print(f"  Python: {py_time:.4f}s")
    print(f"  C:      {c_time:.4f}s (Speedup: {py_time/c_time:.1f}x)")

    # 2. Crossover
    start = time.time()
    for _ in range(N_ITER):
        crossover_py(p1_unpacked, p2_unpacked, cp)
    py_time = time.time() - start

    start = time.time()
    for _ in range(N_ITER):
        fast_crossover_packed(p1_packed, p2_packed, cp, FILAMENT_LEN)
    c_time = time.time() - start
    print(f"\nCrossover:")
    print(f"  Python: {py_time:.4f}s")
    print(f"  C:      {c_time:.4f}s (Speedup: {py_time/c_time:.1f}x)")

    # 3. Mutation
    start = time.time()
    for _ in range(N_ITER):
        mutation_py(p1_unpacked.copy(), mutation_rate)
    py_time = time.time() - start

    start = time.time()
    for _ in range(N_ITER):
        fast_mutation_packed(p1_packed.copy(), mutation_rate, random_values, FILAMENT_LEN)
    c_time = time.time() - start
    print(f"\nMutation:")
    print(f"  Python: {py_time:.4f}s")
    print(f"  C:      {c_time:.4f}s (Speedup: {py_time/c_time:.1f}x)")

    # 4. Phenotype Decoding (The "Big Win")
    start = time.time()
    for _ in range(N_ITER):
        to_phenotype_py(p1_unpacked, BITS)
    py_time = time.time() - start

    start = time.time()
    for _ in range(N_ITER):
        decode_individual(p1_packed, BITS)
    c_time = time.time() - start
    print(f"\nUnified Phenotype Decoding:")
    print(f"  Python: {py_time:.4f}s")
    print(f"  C:      {c_time:.4f}s (Speedup: {py_time/c_time:.1f}x)")

if __name__ == "__main__":
    run_benchmark()
