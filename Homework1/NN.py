import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import torch
def func(x,a0,a1,a2,a3):
    return (np.exp(a0*x)+a1)+1.3*np.sin(a2*np.pi*x+a3)
if __name__ == "__main__":    
    train = pd.read_excel("complex_nonlinear_data.xlsx")
    test = pd.read_excel("new_complex_nonlinear_data.xlsx")
    x_train = train["x"].to_numpy()
    y_train = train["y_complex"].to_numpy()
    x_test = test["x_new"].to_numpy()
    y_test = test["y_new_complex"].to_numpy()