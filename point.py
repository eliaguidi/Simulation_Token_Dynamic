import numpy as np


class Point:

    # Random Initialization on Unit Sphere
    def __init__(self):
        temp = np.random.randn(3)
        self.point = temp / np.linalg.norm(temp)


