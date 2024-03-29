import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
if __name__ == "__main__":    
    train = pd.read_excel("complex_nonlinear_data.xlsx")
    test = pd.read_excel("new_complex_nonlinear_data.xlsx")
    x_train = train["x"].to_numpy()
    y_train = train["y_complex"].to_numpy()
    x_test = test["x_new"].to_numpy()
    y_test = test["y_new_complex"].to_numpy()
    a = np.polyfit(x_train,y_train,20)
    print(a)
    x_plot = np.linspace(0,10,1000)
    y_plot = np.polyval(a,x_plot)
    plt.plot(x_plot,y_plot)
    # plt.scatter(x_test,y_test,c="red")
    plt.scatter(x_train,y_train,c="blue")
    xy = (y_test-np.polyval(a,x_test))**2
    MSE = np.sum(xy)/len(x_test)
    print("MSE: {}".format(MSE))
    plt.show()