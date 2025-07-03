from .model_loader import load_model, Model
from functools import lru_cache

@lru_cache()
def get_model():
    return load_model()
