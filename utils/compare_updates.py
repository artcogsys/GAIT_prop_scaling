import numpy as np

def get_angles(w1, w2):
    assert w1.shape == w2.shape
    w1, w2 = w1.numpy().flatten(), w2.numpy().flatten()
    dot = np.dot(w1 / np.linalg.norm(w1), w2 / np.linalg.norm(w2))
    angle = np.rad2deg(np.arccos(dot))
    if np.isnan(angle):
        return 0.0
    return angle

def get_cors_per_layer(updates1, updates2):
    cors = []
    for index, update1 in enumerate(updates1):
        if index % 2 == 0: # we only compare weights, not biases
            update2 = updates2[index]
            cor = get_angles(update1, update2)
            cors.append(cor)

    return np.array(cors)

def average_cors_over_batches(cors_per_layer):
    cors_per_layer = np.array(cors_per_layer)
    return np.mean(cors_per_layer, axis=0)