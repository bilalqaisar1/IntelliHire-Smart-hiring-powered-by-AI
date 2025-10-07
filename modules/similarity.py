import numpy as np

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two vectors."""
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
