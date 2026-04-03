import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from evo.individual import Individual
from evo.utils import Setup
import os

def mock_fitness_eval(individual, data, labels):
    # Simulate CPU-bound work that might or might not release the GIL
    # In reality, this would be model.fit()
    start = time.time()
    while time.time() - start < 0.1: # 100ms of work
        _ = 2**1000
    return 0.5

def task_module(args):
    genes, filament_len, bits, folder, seed, data, labels = args
    ind = Individual(filament_len, genes, bits, folder, seed)
    mock_fitness_eval(ind, data, labels)
    return ind

def run_test(executor_class, pop_size, data, labels, bits, filament_len):
    all_genes = [np.random.choice([0, 1], size=filament_len).astype(np.int8) for _ in range(pop_size)]
    task_args = [(g, filament_len, bits, ".", 42, data, labels) for g in all_genes]

    start = time.time()
    if executor_class is None:
        results = [task_module(arg) for arg in task_args]
    else:
        with executor_class(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(task_module, task_args))
    return time.time() - start

if __name__ == "__main__":
    POP_SIZE = 20
    BITS = {'features': 100, 'model_selection': 2, 'model_params': 11}
    FILAMENT_LEN = sum(BITS.values())
    DATA = (np.random.randn(100, 100), np.random.randn(20, 100))
    LABELS = (np.random.randint(0, 2, 100), np.random.randint(0, 2, 20))

    print(f"Checking parallelism for POP_SIZE={POP_SIZE}...")
    
    t_seq = run_test(None, POP_SIZE, DATA, LABELS, BITS, FILAMENT_LEN)
    print(f"Sequential: {t_seq:.4f}s")
    
    t_thread = run_test(ThreadPoolExecutor, POP_SIZE, DATA, LABELS, BITS, FILAMENT_LEN)
    print(f"ThreadPool: {t_thread:.4f}s (Speedup: {t_seq/t_thread:.2f}x)")
    
    # ProcessPool might fail if things aren't pickleable, but Individual should be okay
    try:
        t_proc = run_test(ProcessPoolExecutor, POP_SIZE, DATA, LABELS, BITS, FILAMENT_LEN)
        print(f"ProcessPool: {t_proc:.4f}s (Speedup: {t_seq/t_proc:.2f}x)")
    except Exception as e:
        print(f"ProcessPool failed: {e}")
