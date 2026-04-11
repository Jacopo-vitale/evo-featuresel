import os
import json

# --- Default Registry ---
MODELS_REGISTRY = [
# ... (existing models)
    {
        "id": 0,
        "name": "RandomForestClassifier",
        "import_path": "sklearn.ensemble.RandomForestClassifier",
        "params": {
            "n_estimators": {"enabled": True, "bits": 9, "type": "int", "min": 2, "max": 513},
            "criterion": {"enabled": True, "bits": 2, "type": "categorical", "values": ["gini", "entropy", "log_loss"]}
        }
    },
    {
        "id": 1,
        "name": "SVC",
        "import_path": "sklearn.svm.SVC",
        "params": {
            "C": {"enabled": True, "type": "float", "bits_mantissa": 3, "bits_exponent": 3, "bits_sign": 1},
            "kernel": {"enabled": True, "bits": 2, "type": "categorical", "values": ["linear", "poly", "rbf", "sigmoid"]},
            "degree": {"enabled": True, "bits": 3, "type": "int", "min": 1, "max": 8}
        }
    },
    {
        "id": 2,
        "name": "GradientBoostingClassifier",
        "import_path": "sklearn.ensemble.GradientBoostingClassifier",
        "params": {
            "n_estimators": {"enabled": True, "bits": 9, "type": "int", "min": 2, "max": 513},
            "criterion": {"enabled": True, "bits": 1, "type": "categorical", "values": ["friedman_mse", "squared_error"]},
            "loss": {"enabled": True, "bits": 1, "type": "categorical", "values": ["log_loss", "exponential"]}
        }
    },
    {
        "id": 3,
        "name": "ExtraTreesClassifier",
        "import_path": "sklearn.ensemble.ExtraTreesClassifier",
        "params": {
            "n_estimators": {"enabled": True, "bits": 9, "type": "int", "min": 2, "max": 513},
            "criterion": {"enabled": True, "bits": 2, "type": "categorical", "values": ["gini", "entropy", "log_loss"]}
        }
    },
    {
        "id": 4,
        "name": "LogisticRegression",
        "import_path": "sklearn.linear_model.LogisticRegression",
        "params": {
            "C": {"enabled": True, "type": "float", "bits_mantissa": 3, "bits_exponent": 3, "bits_sign": 1},
            "penalty": {"enabled": True, "bits": 1, "type": "categorical", "values": ["l2", "none"]},
            "max_iter": {"enabled": True, "bits": 7, "type": "int", "min": 50, "max": 250}
        }
    },
    {
        "id": 5,
        "name": "KNeighborsClassifier",
        "import_path": "sklearn.neighbors.KNeighborsClassifier",
        "params": {
            "n_neighbors": {"enabled": True, "bits": 5, "type": "int", "min": 1, "max": 32},
            "weights": {"enabled": True, "bits": 1, "type": "categorical", "values": ["uniform", "distance"]},
            "metric": {"enabled": True, "bits": 2, "type": "categorical", "values": ["euclidean", "manhattan", "minkowski", "chebyshev"]}
        }
    }
]

# --- Load User Models ---
USER_MODELS_PATH = "user_models.json"

def reload_registry():
    """Reloads the global MODELS_REGISTRY by merging defaults with user_models.json."""
    global MODELS_REGISTRY
    # Re-declare defaults to avoid doubling on re-call
    defaults = [
        {"id": 0, "name": "RandomForestClassifier", "import_path": "sklearn.ensemble.RandomForestClassifier", "params": {"n_estimators": {"enabled": True, "bits": 9, "type": "int", "min": 2, "max": 513}, "criterion": {"enabled": True, "bits": 2, "type": "categorical", "values": ["gini", "entropy", "log_loss"]}}},
        {"id": 1, "name": "SVC", "import_path": "sklearn.svm.SVC", "params": {"C": {"enabled": True, "type": "float", "bits_mantissa": 3, "bits_exponent": 3, "bits_sign": 1}, "kernel": {"enabled": True, "bits": 2, "type": "categorical", "values": ["linear", "poly", "rbf", "sigmoid"]}, "degree": {"enabled": True, "bits": 3, "type": "int", "min": 1, "max": 8}}},
        {"id": 2, "name": "GradientBoostingClassifier", "import_path": "sklearn.ensemble.GradientBoostingClassifier", "params": {"n_estimators": {"enabled": True, "bits": 9, "type": "int", "min": 2, "max": 513}, "criterion": {"enabled": True, "bits": 1, "type": "categorical", "values": ["friedman_mse", "squared_error"]}, "loss": {"enabled": True, "bits": 1, "type": "categorical", "values": ["log_loss", "exponential"]}}},
        {"id": 3, "name": "ExtraTreesClassifier", "import_path": "sklearn.ensemble.ExtraTreesClassifier", "params": {"n_estimators": {"enabled": True, "bits": 9, "type": "int", "min": 2, "max": 513}, "criterion": {"enabled": True, "bits": 2, "type": "categorical", "values": ["gini", "entropy", "log_loss"]}}},
        {"id": 4, "name": "LogisticRegression", "import_path": "sklearn.linear_model.LogisticRegression", "params": {"C": {"enabled": True, "type": "float", "bits_mantissa": 3, "bits_exponent": 3, "bits_sign": 1}, "penalty": {"enabled": True, "bits": 1, "type": "categorical", "values": ["l2", "none"]}, "max_iter": {"enabled": True, "bits": 7, "type": "int", "min": 50, "max": 250}}},
        {"id": 5, "name": "KNeighborsClassifier", "import_path": "sklearn.neighbors.KNeighborsClassifier", "params": {"n_neighbors": {"enabled": True, "bits": 5, "type": "int", "min": 1, "max": 32}, "weights": {"enabled": True, "bits": 1, "type": "categorical", "values": ["uniform", "distance"]}, "metric": {"enabled": True, "bits": 2, "type": "categorical", "values": ["euclidean", "manhattan", "minkowski", "chebyshev"]}}}
    ]
    
    if os.path.exists(USER_MODELS_PATH):
        try:
            with open(USER_MODELS_PATH, 'r') as f:
                user_models = json.load(f)
                # Ensure user IDs start after defaults
                next_id = len(defaults)
                for m in user_models:
                    m['id'] = next_id
                    next_id += 1
                defaults.extend(user_models)
        except Exception as e:
            print(f"Error loading user_models.json: {e}")
            
    MODELS_REGISTRY[:] = defaults

# Initial load
reload_registry()
