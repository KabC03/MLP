import matplotlib.pyplot as plt
import numpy as np

numOutputs = 2;


def main():

    with open("./data/out.txt", "r") as file:
        xLine = file.readline();
        xVals = [float(value) for value in xLine.split()];

        for i in range(0, numOutputs):
            yLine = file.readline();
            yVals = [float(value) for value in yLine.split()];
            plt.plot(xVals, yVals, label = "f(x" + str(i) + ") vs x");


        xCopy = np.array(xVals);
        xCopy *= (np.pi);
        plt.plot(xVals, 0.2 * np.sin(xCopy), label = "f(x) = 0.2 * sin(x) vs x");
        plt.plot(xVals, 0.5 * np.cos(xCopy), label = "f(x) = 0.5 * cos(x) vs x");


        plt.title("f(xn) vs x");
        plt.xlabel("x");
        plt.ylabel("f(xn)");


        plt.legend(loc = "best");
        plt.show();





if __name__ == "__main__":
    main();



