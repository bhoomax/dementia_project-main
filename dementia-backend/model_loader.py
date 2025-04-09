import joblib

def load_models():
    """Load ANN and RFC models from disk."""
    ann_model = joblib.load("models/ann_model.pkl")
    # rfc_model = joblib.load("models/rfc_model.pkl")
    return ann_model
