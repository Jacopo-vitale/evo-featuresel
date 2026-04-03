# cython: language_level=3
import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint8_t, int8_t

from cython.parallel import prange

# --- Bit-Packed Operations ---

def pack_bits(cnp.ndarray[int8_t, ndim=1] unpacked):
    """
    Packs an int8 array of 0s and 1s into a uint8 array (1 bit per bit).
    """
    cdef int n = unpacked.shape[0]
    cdef int packed_n = (n + 7) // 8
    cdef cnp.ndarray[uint8_t, ndim=1] packed = np.zeros(packed_n, dtype=np.uint8)
    cdef int i, byte_idx, bit_idx
    
    for i in range(n):
        byte_idx = i // 8
        bit_idx = i % 8
        if unpacked[i]:
            packed[byte_idx] |= (1 << (7 - bit_idx))
            
    return packed

def unpack_bits(cnp.ndarray[uint8_t, ndim=1] packed, int original_n):
    """
    Unpacks a uint8 array back into an int8 array of 0s and 1s.
    """
    cdef cnp.ndarray[int8_t, ndim=1] unpacked = np.zeros(original_n, dtype=np.int8)
    cdef int i, byte_idx, bit_idx
    
    for i in range(original_n):
        byte_idx = i // 8
        bit_idx = i % 8
        if (packed[byte_idx] >> (7 - bit_idx)) & 1:
            unpacked[i] = 1
            
    return unpacked

def fast_binary_to_decimal_packed(cnp.ndarray[uint8_t, ndim=1] packed, int start_bit, int n_bits):
    """
    Extracts a decimal value from a bit-packed array given a start bit and length.
    """
    cdef long long res = 0
    cdef int i, byte_idx, bit_idx
    
    for i in range(n_bits):
        byte_idx = (start_bit + i) // 8
        bit_idx = (start_bit + i) % 8
        res = (res << 1) | ((packed[byte_idx] >> (7 - bit_idx)) & 1)
        
    return res

# --- Unified Decoder ---

def decode_individual(cnp.ndarray[uint8_t, ndim=1] packed, dict bits):
    """
    Decodes all phenotype parameters from a bit-packed array in one pass.
    """
    cdef int feat_bits = bits['features']
    cdef int model_sel_bits = bits['model_selection']
    
    # 1. Model Selection
    cdef int model_selection = fast_binary_to_decimal_packed(packed, feat_bits, model_sel_bits)
    
    # 2. Model Parameters
    cdef int param_start = feat_bits + model_sel_bits
    cdef dict model_param = {}
    cdef int n_estimators, criterion_selector, kernel_selector, degree_bits
    cdef double mantissa, segno, esponente
    
    if model_selection == 0: # RandomForest
        n_estimators = fast_binary_to_decimal_packed(packed, param_start, 9)
        model_param['n_estimators'] = n_estimators if n_estimators > 2 else 2
        criterion_selector = fast_binary_to_decimal_packed(packed, param_start + 9, 2)
        model_param['criterion'] = 'gini' if criterion_selector == 0 else ('entropy' if criterion_selector == 1 else 'log_loss')
        
    elif model_selection == 1: # SVC
        mantissa = fast_binary_to_decimal_packed(packed, param_start, 3) * 0.1
        segno = 1.0 if fast_binary_to_decimal_packed(packed, param_start + 3, 1) == 0 else -1.0
        esponente = fast_binary_to_decimal_packed(packed, param_start + 4, 3)
        model_param['C'] = (1.0 + mantissa) * (10 ** (segno * esponente))
        
        kernel_selector = fast_binary_to_decimal_packed(packed, param_start + 8, 2)
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        model_param['kernel'] = kernels[kernel_selector] if kernel_selector < 4 else 'rbf'
        model_param['degree'] = fast_binary_to_decimal_packed(packed, param_start + 10, 3) + 1 # Assuming 3 bits for degree if available
        
    elif model_selection == 2: # GradientBoosting
        n_estimators = fast_binary_to_decimal_packed(packed, param_start, 9)
        model_param['n_estimators'] = n_estimators if n_estimators > 2 else 2
        model_param['criterion'] = 'friedman_mse' if fast_binary_to_decimal_packed(packed, param_start + 9, 1) == 0 else 'squared_error'
        model_param['loss'] = 'log_loss' if fast_binary_to_decimal_packed(packed, param_start + 10, 1) == 0 else 'exponential'
        
    elif model_selection == 3: # ExtraTrees
        n_estimators = fast_binary_to_decimal_packed(packed, param_start, 9)
        model_param['n_estimators'] = n_estimators if n_estimators > 2 else 2
        criterion_selector = fast_binary_to_decimal_packed(packed, param_start + 9, 2)
        model_param['criterion'] = 'gini' if criterion_selector == 0 else ('entropy' if criterion_selector == 1 else 'log_loss')

    return model_selection, model_param

# --- Evolution Operations on Packed Data ---

def fast_crossover_packed(cnp.ndarray[uint8_t, ndim=1] p1, 
                          cnp.ndarray[uint8_t, ndim=1] p2, 
                          int crossover_bit,
                          int total_bits):
    """
    Fast bit-level crossover on packed uint8 arrays.
    """
    cdef int n_bytes = p1.shape[0]
    cdef cnp.ndarray[uint8_t, ndim=1] child = np.empty(n_bytes, dtype=np.uint8)
    cdef int i, byte_idx, bit_idx
    cdef int cross_byte = crossover_bit // 8
    cdef int cross_bit = crossover_bit % 8
    cdef uint8_t mask
    
    # 1. Bytes before crossover byte
    for i in range(cross_byte):
        child[i] = p1[i]
        
    # 2. The crossover byte itself
    if cross_byte < n_bytes:
        mask = 0xFF << (8 - cross_bit)
        child[cross_byte] = (p1[cross_byte] & mask) | (p2[cross_byte] & ~mask)
        
    # 3. Bytes after crossover byte
    for i in range(cross_byte + 1, n_bytes):
        child[i] = p2[i]
        
    return child

def fast_mutation_packed(cnp.ndarray[uint8_t, ndim=1] packed, 
                         double mutation_rate,
                         cnp.ndarray[cnp.float64_t, ndim=1] random_values,
                         int total_bits):
    """
    Fast bit-level mutation on packed uint8 arrays.
    """
    cdef int i, byte_idx, bit_idx
    
    for i in range(total_bits):
        if random_values[i] < mutation_rate:
            byte_idx = i // 8
            bit_idx = i % 8
            packed[byte_idx] ^= (1 << (7 - bit_idx))
            
    return packed

# --- Parallel Batch Operations (OpenMP) ---

def batch_crossover_packed(cnp.ndarray[uint8_t, ndim=2] parents_pool,
                           cnp.ndarray[cnp.int32_t, ndim=1] p1_indices,
                           cnp.ndarray[cnp.int32_t, ndim=1] p2_indices,
                           cnp.ndarray[cnp.int32_t, ndim=1] crossover_bits,
                           int total_bits):
    """
    Parallel crossover of the entire population using OpenMP.
    """
    cdef int n_offspring = p1_indices.shape[0]
    cdef int n_bytes = parents_pool.shape[1]
    cdef cnp.ndarray[uint8_t, ndim=2] offspring = np.empty((n_offspring, n_bytes), dtype=np.uint8)
    
    cdef int i, j, cross_byte, cross_bit
    cdef uint8_t mask
    
    # We release the GIL to let OpenMP threads run in parallel
    with nogil:
        for i in prange(n_offspring, schedule='static'):
            cross_byte = crossover_bits[i] // 8
            cross_bit = crossover_bits[i] % 8
            mask = 0xFF << (8 - cross_bit)
            
            # 1. Bytes before crossover
            for j in range(cross_byte):
                offspring[i, j] = parents_pool[p1_indices[i], j]
            
            # 2. Crossover byte
            if cross_byte < n_bytes:
                offspring[i, cross_byte] = (parents_pool[p1_indices[i], cross_byte] & mask) | \
                                            (parents_pool[p2_indices[i], cross_byte] & ~mask)
            
            # 3. Bytes after crossover
            for j in range(cross_byte + 1, n_bytes):
                offspring[i, j] = parents_pool[p2_indices[i], j]
                
    return offspring

def batch_mutation_packed(cnp.ndarray[uint8_t, ndim=2] offspring_pool,
                          double mutation_rate,
                          cnp.ndarray[cnp.float64_t, ndim=2] random_matrix,
                          int total_bits):
    """
    Parallel mutation of the entire population using OpenMP.
    """
    cdef int n_pop = offspring_pool.shape[0]
    cdef int n_bytes = offspring_pool.shape[1]
    
    cdef int i, j, byte_idx, bit_idx
    
    with nogil:
        for i in prange(n_pop, schedule='static'):
            for j in range(total_bits):
                if random_matrix[i, j] < mutation_rate:
                    byte_idx = j // 8
                    bit_idx = j % 8
                    offspring_pool[i, byte_idx] ^= (1 << (7 - bit_idx))
                    
    return offspring_pool

# --- Original int8 Helpers (for backward compatibility) ---

def fast_binary_to_decimal(cnp.ndarray[int8_t, ndim=1] binary):
    cdef long long res = 0
    cdef int i, n = binary.shape[0]
    for i in range(n):
        res = (res << 1) | binary[i]
    return res

def fast_crossover(cnp.ndarray[int8_t, ndim=1] parent1, 
                   cnp.ndarray[int8_t, ndim=1] parent2, 
                   int crossover_point):
    cdef int n = parent1.shape[0]
    cdef cnp.ndarray[int8_t, ndim=1] child = np.empty(n, dtype=np.int8)
    cdef int i
    for i in range(crossover_point):
        child[i] = parent1[i]
    for i in range(crossover_point, n):
        child[i] = parent2[i]
    return child

def fast_mutation(cnp.ndarray[int8_t, ndim=1] genes, 
                  double mutation_rate,
                  cnp.ndarray[cnp.float64_t, ndim=1] random_values):
    cdef int n = genes.shape[0]
    cdef int i
    for i in range(n):
        if random_values[i] < mutation_rate:
            genes[i] = 1 - genes[i]
    return genes
