import numpy as np
import os
import pandas as pd
import pytest
from evo.utils import preprocessing_general
from evo.core import fast_binary_metrics, pack_bits, unpack_bits
from evo.individual import Individual
from evo.gui.worker import run_single_fold

def test_preprocessing_separate_labels(tmp_path):
    # Create dummy feature and label files
    feat_path = tmp_path / "feat.csv"
    label_path = tmp_path / "label.csv"
    
    df_feat = pd.DataFrame(np.random.randn(10, 5), columns=[f'f{i}' for i in range(5)])
    df_label = pd.DataFrame(np.random.randint(0, 2, 10), columns=['label'])
    
    df_feat.to_csv(feat_path, index=False)
    df_label.to_csv(label_path, index=False)
    
    # Test loading with separate labels
    data, labels = preprocessing_general(str(feat_path), None, None, train_labels_path=str(label_path))
    
    X_train, X_val, X_test = data
    y_train, y_val, y_test = labels
    
    assert X_train.shape == (10, 5)
    assert y_train.shape == (10,)
    assert X_val is None
    assert X_test is None

def test_fast_binary_metrics_accuracy():
    # Simple case: perfect match
    y_true = np.array([1, 0, 1, 0, 1], dtype=np.int64)
    y_pred = np.array([1, 0, 1, 0, 1], dtype=np.int64)
    
    mcc, acc, f1, prec, recall, cm = fast_binary_metrics(y_true, y_pred)
    
    assert acc == 1.0
    assert mcc == 1.0
    assert f1 == 1.0
    assert prec == 1.0
    assert recall == 1.0
    assert cm == [[2, 0], [0, 3]] # TN, FP, FN, TP

def test_fast_binary_metrics_mismatch():
    y_true = np.array([1, 1, 1, 1, 0], dtype=np.int64)
    y_pred = np.array([0, 0, 0, 0, 1], dtype=np.int64)
    
    mcc, acc, f1, prec, recall, cm = fast_binary_metrics(y_true, y_pred)
    
    assert acc == 0.0
    assert mcc == -1.0
    assert f1 == 0.0
    assert prec == 0.0
    assert recall == 0.0
    assert cm == [[0, 1], [4, 0]]

def test_individual_fitness_eval_uses_cython_metrics():
    # Mock data
    X = np.random.randn(20, 10)
    y = np.random.randint(0, 2, 20)
    DATA = (X, X)
    LABELS = (y, y)
    
    from evo.utils import Setup
    setup = Setup()
    setup.BITS = {'features': 10}
    setup.calculate_filament_len()
    layout = setup.get_cython_layout()
    enabled_models = [m for m in setup.INDIVIDUAL_CONFIG['models'] if m['enabled']]
    
    genes = np.ones(setup.FILAMENT_LEN, dtype=np.int8)
    
    ind = Individual(setup.FILAMENT_LEN, genes, setup.BITS, ".", 42, 
                     cython_layout=layout, enabled_models=enabled_models)
    ind.fitness_eval(DATA, LABELS)
    
    assert ind.fitness >= -1.0 and ind.fitness <= 1.0
    assert hasattr(ind, 'cm')
    assert isinstance(ind.cm, list) 

def test_run_single_fold_logic():
    # Smoke test for the helper function used in Outer CV
    X = np.random.randn(50, 10)
    y = np.random.randint(0, 2, 50)
    
    from evo.utils import get_default_individual_config
    params = {
        'pop_size': 10,
        'generations': 2,
        'seed': 42,
        'alpha': 0.5,
        'individual_config': get_default_individual_config()
    }
    
    # Simulating one fold
    res = run_single_fold((0, X[:40], y[:40], X[40:], y[40:], params))
    
    assert 'fitness' in res
    assert 'fold' in res
    assert res['fold'] == 0
    assert res['model_type'] is not None

if __name__ == "__main__":
    # If run directly
    pytest.main([__file__])
