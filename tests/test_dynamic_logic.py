import numpy as np
import pytest
import os
import json
from evo.utils import Setup, get_default_individual_config
from evo.individual import Individual, _MODEL_CACHE, _FITNESS_FUNC_CACHE
from evo.models_registry import reload_registry, MODELS_REGISTRY, USER_MODELS_PATH

def test_user_model_registry_merging(tmp_path):
    # Backup existing user_models.json if it exists
    backup_path = "user_models.json.bak"
    if os.path.exists(USER_MODELS_PATH):
        os.rename(USER_MODELS_PATH, backup_path)
    
    try:
        # Create a mock user model
        custom_model = {
            "name": "MockModel",
            "enabled": True,
            "import_path": "sklearn.linear_model.Ridge",
            "params": {
                "alpha": {"enabled": True, "bits": 5, "type": "int", "min": 1, "max": 32}
            }
        }
        
        with open(USER_MODELS_PATH, "w") as f:
            json.dump([custom_model], f)
            
        reload_registry()
        
        # Check if MockModel is in registry
        names = [m['name'] for m in MODELS_REGISTRY]
        assert "MockModel" in names
        
        # Check ID assignment
        mock_entry = next(m for m in MODELS_REGISTRY if m['name'] == "MockModel")
        assert mock_entry['id'] >= 6 # Default has 0-5
        
    finally:
        # Cleanup and restore
        if os.path.exists(USER_MODELS_PATH):
            os.remove(USER_MODELS_PATH)
        if os.path.exists(backup_path):
            os.rename(backup_path, USER_MODELS_PATH)
        reload_registry()

def test_custom_fitness_execution_and_caching():
    _FITNESS_FUNC_CACHE.clear()
    
    custom_code = """
def custom_fitness(y_true, y_pred, n_total, n_selected, metrics):
    return 999.0
"""
    setup = Setup()
    setup.BITS = {'features': 10}
    setup.calculate_filament_len()
    layout = setup.get_cython_layout()
    enabled_models = [m for m in setup.INDIVIDUAL_CONFIG['models'] if m['enabled']]
    
    # Set one bit to 1 to avoid "Every gene is zero" guard
    unpacked = np.zeros(setup.FILAMENT_LEN, dtype=np.int8)
    unpacked[0] = 1
    
    from evo.core import pack_bits
    genes = pack_bits(unpacked)
    
    ind = Individual(setup.FILAMENT_LEN, genes, setup.BITS, ".", 42, 
                     cython_layout=layout, enabled_models=enabled_models,
                     fitness_code=custom_code)
    
    # Mock data for eval
    DATA = (np.random.randn(10, 10), np.random.randn(5, 10))
    LABELS = (np.random.randint(0, 2, 10), np.random.randint(0, 2, 5))
    
    ind.fitness_eval(DATA, LABELS)
    
    assert ind.fitness == 999.0
    
    # Check cache
    code_hash = hash(custom_code)
    assert code_hash in _FITNESS_FUNC_CACHE
    assert _FITNESS_FUNC_CACHE[code_hash].__name__ == 'custom_fitness'

def test_model_instantiation_caching():
    _MODEL_CACHE.clear()
    
    import_path = "sklearn.ensemble.RandomForestClassifier"
    setup = Setup()
    setup.BITS = {'features': 10}
    setup.calculate_filament_len()
    layout = setup.get_cython_layout()
    enabled_models = [m for m in setup.INDIVIDUAL_CONFIG['models'] if m['enabled']]
    
    # Set one bit to 1
    unpacked = np.zeros(setup.FILAMENT_LEN, dtype=np.int8)
    unpacked[0] = 1
    from evo.core import pack_bits
    genes = pack_bits(unpacked)
    
    # Eval first individual
    ind1 = Individual(setup.FILAMENT_LEN, genes, setup.BITS, ".", 42, 
                      cython_layout=layout, enabled_models=enabled_models)
    
    DATA = (np.random.randn(10, 10), np.random.randn(5, 10))
    LABELS = (np.random.randint(0, 2, 10), np.random.randint(0, 2, 5))
    
    ind1.fitness_eval(DATA, LABELS)
    
    assert import_path in _MODEL_CACHE
    cached_class, supports_rs, supports_n_jobs = _MODEL_CACHE[import_path]
    assert cached_class.__name__ == "RandomForestClassifier"
    assert supports_rs is True

def test_float_decoding_precision():
    setup = Setup()
    # Enable only SVC to make bit-mapping predictable for the test
    for m in setup.INDIVIDUAL_CONFIG['models']:
        m['enabled'] = (m['name'] == "SVC")
    
    setup.BITS = {'features': 8}
    setup.calculate_filament_len()
    
    # SVC C param uses type 'float': m=3, e=3, s=1
    # model_sel_bits = 0 (only 1 model)
    # param_start = 8 + 0 = 8
    # Bits: [0-7 features] [8-10 m] [11 s] [12-14 e] ...
    # Let's set:
    # m = 5 (binary 101) -> bits 8, 9, 10
    # s = 1 (negative) -> bit 11
    # e = 2 (binary 010) -> bits 12, 13, 14
    # Expected C = (1.0 + 0.5) * (10 ** -2) = 0.015
    
    start = setup.BITS['features'] + setup.BITS['model_selection']
    
    unpacked = np.zeros(setup.FILAMENT_LEN, dtype=np.int8)
    unpacked[start : start+3] = [1, 0, 1] # m=5
    unpacked[start+3] = 1                 # s=1 (negative)
    unpacked[start+4 : start+7] = [0, 1, 0] # e=2
    
    packed = np.zeros((setup.FILAMENT_LEN + 7)//8, dtype=np.uint8)
    for i, val in enumerate(unpacked):
        if val: packed[i // 8] |= (1 << (7 - (i % 8)))
        
    layout = setup.get_cython_layout()
    enabled_models = [m for m in setup.INDIVIDUAL_CONFIG['models'] if m['enabled']]
    
    ind = Individual(setup.FILAMENT_LEN, packed, setup.BITS, ".", 42, 
                     cython_layout=layout, enabled_models=enabled_models)
    
    ind.ensure_phenotype()
    
    assert ind.model_param['C'] == pytest.approx(0.015)

if __name__ == "__main__":
    pytest.main([__file__])
