import numpy as np
import os
import pandas as pd
from main import preprocessing_general

def test_pca_preprocessing(tmp_path):
    # Create dummy feature file with high correlation to ensure PCA reduces dimensions
    feat_path = tmp_path / "feat_pca.csv"
    
    # 100 samples, 10 features, but only 2 independent ones
    x1 = np.random.randn(100)
    x2 = np.random.randn(100)
    data = np.zeros((100, 10))
    for i in range(5):
        data[:, i] = x1 + np.random.normal(0, 0.01, 100)
    for i in range(5, 10):
        data[:, i] = x2 + np.random.normal(0, 0.01, 100)
        
    labels = np.random.randint(0, 2, 100)
    df = pd.DataFrame(data, columns=[f'f{i}' for i in range(10)])
    df['label'] = labels
    df.to_csv(feat_path, index=False)
    
    # Test PCA - 95% variance should result in fewer than 10 features
    data_res, labels_res = preprocessing_general(str(feat_path), None, None, pca=True)
    X_train, X_val, X_test = data_res
    
    print(f"Original features: 10, PCA features: {X_train.shape[1]}")
    assert X_train.shape[1] < 10
    assert X_train.shape[0] == 100

def test_lda_preprocessing(tmp_path):
    # Create dummy feature file
    feat_path = tmp_path / "feat_lda.csv"
    
    # 100 samples, 10 features
    data = np.random.randn(100, 10)
    labels = np.random.randint(0, 2, 100)
    # Make labels somewhat dependent on features to avoid LDA issues
    data[labels == 1] += 2.0
    
    df = pd.DataFrame(data, columns=[f'f{i}' for i in range(10)])
    df['label'] = labels
    df.to_csv(feat_path, index=False)
    
    # Test LDA - for 2 classes, LDA should result in 1 component
    data_res, labels_res = preprocessing_general(str(feat_path), None, None, lda=True)
    X_train, X_val, X_test = data_res
    
    print(f"Original features: 10, LDA features: {X_train.shape[1]}")
    assert X_train.shape[1] == 1 # 2 classes -> 1 component
    assert X_train.shape[0] == 100

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
