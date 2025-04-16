import pandas as pd
import numpy as np

class MyKNNClf():
    def __init__(self, k=3):
        self.k = k

    def __str__(self):
        return f"MyKNNClf class: k={self.k}"
    

def main():
    clf = MyKNNClf()
    print(clf)

if __name__ == "__main__":
    main()