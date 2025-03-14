import matplotlib.pyplot as plt
import numpy as np


def main():

    with open("./data/out.txt", "r") as file:
        xLine = file.readline();
        yLine = file.readline();

        xVals = [float(value) for value in xLine.split()];
        yVals = [float(value) for value in yLine.split()];



        plt.title("x vs y");
        plt.xlabel("x");
        plt.ylabel("y");

        plt.plot(xVals, yVals);
        plt.show();






if __name__ == "__main__":
    main();



