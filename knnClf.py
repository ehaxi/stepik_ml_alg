import pandas as pd
import numpy as np

class MyKNNClf():
    def __init__(self, k=3, n_samples=None, n_features=None, n_informative=None):
        self.k = k
        self.train_size = None
        self.X = None
        self.y = None
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative

    def __str__(self):
        return f"MyKNNClf class: k={self.k}"
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.train_size = self.X.shape[0], self.X.shape[1]
    

def main():
    clf = MyKNNClf()
    print(clf)

if __name__ == "__main__":
    main()