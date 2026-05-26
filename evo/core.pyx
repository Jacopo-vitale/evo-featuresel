# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdint cimport uint8_t, int8_t, uint64_t, int32_t, int64_t
from libc.math cimport sqrt
from cython.parallel import prange

# --- Bit-Packed Operations ---

@cython.boundscheck(False)
@cython.wraparound(False)
def pack_bits(const int8_t[:] unpacked):
    """
    Packs an int8 array of 0s and 1s into a uint8 array (1 bit per bit).
    Optimized to process 8 elements at a time using bitwise shifts.
    """
    cdef int n = unpacked.shape[0]
    cdef int packed_n = (n + 7) >> 3
    cdef uint8_t[:] packed = np.zeros(packed_n, dtype=np.uint8)
    cdef int i, j, byte_idx
    cdef uint8_t byte_val
    
    # Process complete bytes
    for byte_idx in range(n >> 3):
        byte_val = 0
        for j in range(8):
            if unpacked[(byte_idx << 3) + j]:
                byte_val |= (1 << (7 - j))
        packed[byte_idx] = byte_val
            
    # Process remaining bits
    if n & 7:
        byte_idx = n >> 3
        byte_val = 0
        for j in range(n & 7):
            if unpacked[(byte_idx << 3) + j]:
                byte_val |= (1 << (7 - j))
        packed[byte_idx] = byte_val
            
    return np.asarray(packed)

@cython.boundscheck(False)
@cython.wraparound(False)
def unpack_bits(const uint8_t[:] packed, int original_n):
    """
    Unpacks a uint8 array back into an int8 array of 0s and 1s.
    Optimized to process bits using bitwise extraction.
    """
    cdef int8_t[:] unpacked = np.zeros(original_n, dtype=np.int8)
    cdef int i, byte_idx, bit_idx
    cdef uint8_t byte_val
    
    for byte_idx in range(packed.shape[0]):
        byte_val = packed[byte_idx]
        for bit_idx in range(8):
            i = (byte_idx << 3) + bit_idx
            if i < original_n:
                unpacked[i] = (byte_val >> (7 - bit_idx)) & 1
            
    return np.asarray(unpacked)

def fast_binary_to_decimal_packed(const uint8_t[:] packed, int start_bit, int n_bits):
    """
    Extracts a decimal value from a bit-packed array given a start bit and length.
    """
    cdef long long res = 0
    cdef int i, byte_idx, bit_idx
    
    for i in range(n_bits):
        byte_idx = (start_bit + i) >> 3
        bit_idx = (start_bit + i) & 7
        res = (res << 1) | ((packed[byte_idx] >> (7 - bit_idx)) & 1)
        
    return res

# --- Unified Dynamic Decoder ---

@cython.boundscheck(False)
@cython.wraparound(False)
def decode_individual(const uint8_t[:] packed, 
                      int feat_bits, 
                      int model_sel_bits,
                      list param_names,
                      list param_categories,
                      const int32_t[:, :] model_layouts):
    """
    Decodes phenotype parameters dynamically based on a layout array.
    model_layouts structure: [num_params, p1_name_idx, p1_bits, p1_type, p1_e1, p1_e2, p1_e3, ...]
    """
    # 1. Model Selection
    cdef int model_selection = 0
    if model_sel_bits > 0:
        model_selection = fast_binary_to_decimal_packed(packed, feat_bits, model_sel_bits)
    
    # 2. Extract layout for the selected model
    if model_selection >= model_layouts.shape[0]:
        model_selection = 0 # Fallback
        
    cdef int num_params = model_layouts[model_selection, 0]
    cdef dict model_param = {}
    cdef int param_start = feat_bits + model_sel_bits
    
    cdef int i, name_idx, p_bits, p_type, e1, e2, e3
    cdef long long val, m_val, e_val, s_bit
    cdef double s_val
    cdef list cat_list
    
    cdef int current_bit = param_start
    
    for i in range(num_params):
        name_idx = model_layouts[model_selection, 1 + i * 6]
        p_bits = model_layouts[model_selection, 2 + i * 6]
        p_type = model_layouts[model_selection, 3 + i * 6]
        e1 = model_layouts[model_selection, 4 + i * 6]
        e2 = model_layouts[model_selection, 5 + i * 6]
        e3 = model_layouts[model_selection, 6 + i * 6]
        
        # 0 = INT (e1 is min_val)
        if p_type == 0:
            val = fast_binary_to_decimal_packed(packed, current_bit, p_bits)
            model_param[param_names[name_idx]] = val + e1
            current_bit += p_bits
            
        # 1 = CATEGORICAL (e1 is category_list_idx)
        elif p_type == 1:
            val = fast_binary_to_decimal_packed(packed, current_bit, p_bits)
            cat_list = param_categories[e1]
            if val < len(cat_list):
                model_param[param_names[name_idx]] = cat_list[val]
            else:
                model_param[param_names[name_idx]] = cat_list[0]
            current_bit += p_bits
                
        # 2 = FLOAT (e1=m_bits, e2=e_bits, e3=s_bits)
        elif p_type == 2:
            m_val = fast_binary_to_decimal_packed(packed, current_bit, e1)
            current_bit += e1
            s_val = 1.0
            if e3 > 0:
                s_bit = fast_binary_to_decimal_packed(packed, current_bit, e3)
                s_val = 1.0 if s_bit == 0 else -1.0
                current_bit += e3
            e_val = fast_binary_to_decimal_packed(packed, current_bit, e2)
            current_bit += e2
            model_param[param_names[name_idx]] = (1.0 + m_val * 0.1) * (10.0 ** (s_val * e_val))

    return model_selection, model_param

# --- Evolution Operations on Packed Data ---

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_crossover_packed(const uint8_t[:] p1, 
                          const uint8_t[:] p2, 
                          int crossover_bit,
                          int total_bits):
    """
    Fast bit-level crossover on packed uint8 arrays.
    """
    cdef int n_bytes = p1.shape[0]
    cdef uint8_t[:] child = np.empty(n_bytes, dtype=np.uint8)
    cdef int i
    cdef int cross_byte = crossover_bit >> 3
    cdef int cross_bit = crossover_bit & 7
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
        
    return np.asarray(child)

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_mutation_packed(uint8_t[:] packed, 
                         double mutation_rate,
                         const double[:] random_values,
                         int total_bits):
    """
    Fast bit-level mutation on packed uint8 arrays.
    """
    cdef int i, byte_idx, bit_idx
    
    for i in range(total_bits):
        if random_values[i] < mutation_rate:
            byte_idx = i >> 3
            bit_idx = i & 7
            packed[byte_idx] ^= (1 << (7 - bit_idx))
            
    return np.asarray(packed)

# --- Parallel Batch Operations (OpenMP) ---

@cython.boundscheck(False)
@cython.wraparound(False)
def batch_crossover_packed(const uint8_t[:, :] parents_pool,
                           const int32_t[:] p1_indices,
                           const int32_t[:] p2_indices,
                           const int32_t[:] crossover_bits,
                           int total_bits):
    """
    Parallel crossover of the entire population using OpenMP.
    """
    cdef int n_offspring = p1_indices.shape[0]
    cdef int n_bytes = parents_pool.shape[1]
    cdef uint8_t[:, :] offspring = np.empty((n_offspring, n_bytes), dtype=np.uint8)
    
    cdef int i, j, cross_byte, cross_bit
    cdef uint8_t mask
    
    with nogil:
        for i in prange(n_offspring, schedule='static'):
            cross_byte = crossover_bits[i] >> 3
            cross_bit = crossover_bits[i] & 7
            mask = 0xFF << (8 - cross_bit)
            
            for j in range(cross_byte):
                offspring[i, j] = parents_pool[p1_indices[i], j]
            
            if cross_byte < n_bytes:
                offspring[i, cross_byte] = (parents_pool[p1_indices[i], cross_byte] & mask) | \
                                            (parents_pool[p2_indices[i], cross_byte] & ~mask)
            
            for j in range(cross_byte + 1, n_bytes):
                offspring[i, j] = parents_pool[p2_indices[i], j]
                
    return np.asarray(offspring)

@cython.boundscheck(False)
@cython.wraparound(False)
def batch_mutation_packed(uint8_t[:, :] offspring_pool,
                          double mutation_rate,
                          const double[:, :] random_matrix,
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
                    byte_idx = j >> 3
                    bit_idx = j & 7
                    offspring_pool[i, byte_idx] ^= (1 << (7 - bit_idx))
                    
    return np.asarray(offspring_pool)

# --- Original int8 Helpers (for backward compatibility) ---

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_binary_to_decimal(const int8_t[:] binary):
    cdef long long res = 0
    cdef int i, n = binary.shape[0]
    for i in range(n):
        res = (res << 1) | binary[i]
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_crossover(const int8_t[:] parent1, 
                   const int8_t[:] parent2, 
                   int crossover_point):
    cdef int n = parent1.shape[0]
    cdef int8_t[:] child = np.empty(n, dtype=np.int8)
    cdef int i
    for i in range(crossover_point):
        child[i] = parent1[i]
    for i in range(crossover_point, n):
        child[i] = parent2[i]
    return np.asarray(child)

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_mutation(int8_t[:] genes, 
                  double mutation_rate,
                  const double[:] random_values):
    cdef int n = genes.shape[0]
    cdef int i
    for i in range(n):
        if random_values[i] < mutation_rate:
            genes[i] = 1 - genes[i]
    return np.asarray(genes)

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_binary_metrics(const int64_t[:] y_true, const int64_t[:] y_pred):
    """
    Ultra-fast C-level calculation of binary classification metrics.
    Calculates TP, TN, FP, FN in a single pass to avoid Python loop overhead
    and multiple scikit-learn function calls.
    Returns: mcc, acc, f1, prec, recall, confusion_matrix
    """
    cdef int n = y_true.shape[0]
    cdef long long tp = 0, tn = 0, fp = 0, fn = 0
    cdef int i
    
    with nogil:
        for i in range(n):
            if y_true[i] == 1:
                if y_pred[i] == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if y_pred[i] == 1:
                    fp += 1
                else:
                    tn += 1
                
    cdef double acc = 0.0
    cdef double prec = 0.0
    cdef double recall = 0.0
    cdef double f1 = 0.0
    cdef double mcc = 0.0
    cdef double mcc_num, mcc_den_sq
    
    if n > 0:
        acc = (tp + tn) / float(n)
        
    if (tp + fp) > 0:
        prec = tp / float(tp + fp)
        
    if (tp + fn) > 0:
        recall = tp / float(tp + fn)
        
    if (prec + recall) > 0:
        f1 = (2.0 * prec * recall) / (prec + recall)
        
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den_sq = float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    
    if mcc_den_sq > 0:
        mcc = mcc_num / sqrt(mcc_den_sq)
    else:
        mcc = 0.0
        
    # Return cm as list of lists
    return mcc, acc, f1, prec, recall, [[tn, fp], [fn, tp]]
