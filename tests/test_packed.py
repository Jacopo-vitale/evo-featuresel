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
    from evo.utils import Setup
    setup = Setup()
    setup.BITS = {'features': 8}
    setup.calculate_filament_len()

    # RandomForest (0) is default enabled
    # Features: 10101010
    # Model: 0 (if only 1 model, bits=0, but default has multiple)
    # n_models = 4 -> bits = 2

    unpacked = np.zeros(setup.FILAMENT_LEN, dtype=np.int8)
    unpacked[:8] = [1, 0, 1, 0, 1, 0, 1, 0] # Features
    # Default bits: features=8, model_sel=2, params=...
    # Model 0 (RandomForest) bits = [0, 0]
    unpacked[8:10] = [0, 0] 
    # RF params: n_estimators (9 bits) = 10
    unpacked[10:19] = [0, 0, 0, 0, 0, 1, 0, 1, 0] 

    packed = pack_bits(unpacked)

    param_names, param_categories, layout = setup.get_cython_layout()

    model_sel, model_param = decode_individual(
        packed, 
        setup.BITS['features'], 
        setup.BITS['model_selection'],
        param_names,
        param_categories,
        layout
    )

    assert model_sel == 0
    assert model_param['n_estimators'] == 10 + 2 # min_val is 2
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
