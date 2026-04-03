import numpy as np
import pytest
from evo.core import fast_binary_to_decimal, fast_crossover, fast_mutation

def test_fast_binary_to_decimal():
    # Test cases: (binary_array, expected_decimal)
    test_cases = [
        (np.array([0, 0, 0, 0], dtype=np.int8), 0),
        (np.array([0, 0, 0, 1], dtype=np.int8), 1),
        (np.array([1, 0, 1, 0], dtype=np.int8), 10),
        (np.array([1, 1, 1, 1], dtype=np.int8), 15),
        (np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8), 256),
    ]
    
    for binary, expected in test_cases:
        assert fast_binary_to_decimal(binary) == expected

def test_fast_crossover():
    p1 = np.array([1, 1, 1, 1, 1], dtype=np.int8)
    p2 = np.array([0, 0, 0, 0, 0], dtype=np.int8)
    
    # Crossover point at index 2
    child = fast_crossover(p1, p2, 2)
    expected = np.array([1, 1, 0, 0, 0], dtype=np.int8)
    np.testing.assert_array_equal(child, expected)
    
    # Crossover point at index 0 (all p2)
    child = fast_crossover(p1, p2, 0)
    expected = np.array([0, 0, 0, 0, 0], dtype=np.int8)
    np.testing.assert_array_equal(child, expected)
    
    # Crossover point at index 5 (all p1)
    child = fast_crossover(p1, p2, 5)
    expected = np.array([1, 1, 1, 1, 1], dtype=np.int8)
    np.testing.assert_array_equal(child, expected)

def test_fast_mutation():
    genes = np.array([0, 1, 0, 1, 0], dtype=np.int8)
    mutation_rate = 0.5
    
    # Case 1: All random values < mutation_rate (all bits flip)
    rnd_all_flip = np.array([0.1, 0.2, 0.3, 0.4, 0.45], dtype=np.float64)
    mutated = fast_mutation(genes.copy(), mutation_rate, rnd_all_flip)
    expected = np.array([1, 0, 1, 0, 1], dtype=np.int8)
    np.testing.assert_array_equal(mutated, expected)
    
    # Case 2: All random values > mutation_rate (no bits flip)
    rnd_no_flip = np.array([0.6, 0.7, 0.8, 0.9, 0.95], dtype=np.float64)
    mutated = fast_mutation(genes.copy(), mutation_rate, rnd_no_flip)
    expected = np.array([0, 1, 0, 1, 0], dtype=np.int8)
    np.testing.assert_array_equal(mutated, expected)
    
    # Case 3: Mixed
    rnd_mixed = np.array([0.1, 0.7, 0.1, 0.9, 0.1], dtype=np.float64)
    mutated = fast_mutation(genes.copy(), mutation_rate, rnd_mixed)
    expected = np.array([1, 1, 1, 1, 1], dtype=np.int8)
    np.testing.assert_array_equal(mutated, expected)

if __name__ == "__main__":
    # If pytest is not available, run manually
    try:
        test_fast_binary_to_decimal()
        test_fast_crossover()
        test_fast_mutation()
        print("All tests passed successfully!")
    except Exception as e:
        print(f"Tests failed: {e}")
        exit(1)
