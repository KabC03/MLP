import matplotlib.pyplot as plt
import numpy as np

numOutputs = 1;

def main():

    with open("./data/out.txt", "r") as file:
        xLine = file.readline();




        xVals = [float(value) for value in xLine.split()];

        for i in range(0, numOutputs):
            yLine = file.readline();
            yVals = [float(value) for value in yLine.split()];
            plt.plot(xVals, yVals, label = "x" + str(i) + " vs x");


        plt.title("f(x) vs x");
        plt.xlabel("x");
        plt.ylabel("f(x)");


        plt.legend(loc = "best");
        plt.show();





if __name__ == "__main__":
    main();



