import numpy as np
import pytest
from evo.core import pack_bits, unpack_bits, decode_individual, fast_crossover_packed, fast_mutation_packed

def test_packing():
    unpacked = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8)
    packed = pack_bits(unpacked)
    
    # Check length (2 bytes for 10 bits)
    assert len(packed) == 2
    
    # Check values
    # Byte 1: 10110010 -> 0xB2 (178)
    # Byte 2: 11000000 -> 0xC0 (192)
    assert packed[0] == 178
    assert packed[1] == 192
    
    unpacked_back = unpack_bits(packed, 10)
    np.testing.assert_array_equal(unpacked, unpacked_back)

def test_decode_individual():
    # Setup bits structure
    bits = {'features': 8, 'model_selection': 2}
    # 8 bits features, 2 bits model selection, rest params
    # Let's target RandomForest (00)
    # Features: 10101010 (170)
    # Model: 00
    # Params (n_estimators): 000001010 (10 bits total, first 9 for n_estimators) -> 10
    
    unpacked = np.zeros(30, dtype=np.int8)
    unpacked[:8] = [1, 0, 1, 0, 1, 0, 1, 0] # Features
    unpacked[8:10] = [0, 0] # Model 0
    unpacked[10:19] = [0, 0, 0, 0, 0, 1, 0, 1, 0] # n_estimators = 10
    
    packed = pack_bits(unpacked)
    model_sel, model_param = decode_individual(packed, bits)
    
    assert model_sel == 0
    assert model_param['n_estimators'] == 10

def test_crossover_packed():
    p1_unpacked = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int8)
    p2_unpacked = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    
    p1 = pack_bits(p1_unpacked)
    p2 = pack_bits(p2_unpacked)
    
    # Crossover at bit 4
    child = fast_crossover_packed(p1, p2, 4, 8)
    child_unpacked = unpack_bits(child, 8)
    
    np.testing.assert_array_equal(child_unpacked, [1, 1, 1, 1, 0, 0, 0, 0])

if __name__ == "__main__":
    test_packing()
    test_decode_individual()
    test_crossover_packed()
    print("Bit-packed tests passed!")
