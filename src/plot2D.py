import matplotlib.pyplot as plt;
import numpy as np;
import sys;




def main():

    with open('./data/out.txt', 'r') as file:

        xSeries = [];
        ySeriesArray = []; #List of list of y/z/w/etc values
        for line in file:

            lineSeries = line.strip().split(', ');
            xSeries.append(float(lineSeries[0]));

            for i in range(1, len(lineSeries)):
                if len(ySeriesArray) < i:
                    ySeriesArray.append([]);
                
                ySeriesArray[i - 1].append((float(lineSeries[i])));


        plt.xlabel("x");
        plt.ylabel("F[x]");
        plt.title("F[x] vs x");

        for i in range(0, len(ySeriesArray)):
            plt.plot(xSeries, ySeriesArray[i], label = "F" + str(i) + "[x]");

        plt.legend(loc = 'best');
        plt.show();



if __name__ == "__main__":
    sys.exit(main());




