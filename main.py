import numpy as np
from neuralnet import NeuralNet
import matplotlib.pyplot as plt

def main():
    model=NeuralNet(None)
    model.load("layers.pkl")
    x_test = np.linspace(0, 2*np.pi, 200).reshape(-1, 1)
    pred = model.predict(x_test)

    
    plt.plot(x_test, np.sin(x_test), label="sine wave", linewidth=2)
    plt.plot(x_test, pred, label="model prediction", linestyle="--")
    plt.legend()
    # plt.show()
    plt.savefig("model vs true.png")  
    plt.close()  
if __name__=="__main__":
    main()