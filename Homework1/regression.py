import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
# def func(x,a0,a1,a2):
#     return (np.exp(a0*x)+a1)+1.3*np.sin(2*np.pi*x+a2)
def func(x,a0,a1,a2,a3,a4,a5):
    return a0*np.power(x,5)+a1*np.power(x,4)+a2*np.power(x,3)+a3*x*x+a4*x+a5*np.sin(x)
if __name__ == "__main__":    
    train = pd.read_excel("complex_nonlinear_data.xlsx")
    test = pd.read_excel("new_complex_nonlinear_data.xlsx")
    x_train = train["x"].to_numpy()
    y_train = train["y_complex"].to_numpy()
    x_test = test["x_new"].to_numpy()
    y_test = test["y_new_complex"].to_numpy()
    popt, pcov = curve_fit(func,x_train,y_train)
    print(popt)
    print(pcov)
    a0, a1, a2, a3, a4, a5 = popt
    x_plot = np.linspace(0,10,1000)
    y_plot = func(x_plot,a0,a1,a2,a3,a4,a5)
    plt.plot(x_plot,y_plot)
    # MSE = 0.0
    # MSE = np.float_power()
    plt.scatter(x_test,y_test,c="red")
    # plt.scatter(x_train,y_train,c="blue")
    xy = (y_test-func(x_test,a0,a1,a2,a3,a4,a5))**2
    MSE = np.sum(xy)/len(x_test)
    print("MSE: {}".format(MSE))
    plt.show()
