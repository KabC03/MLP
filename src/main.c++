#include <iostream>
#include <vector>
#include <unistd.h>
#include <fstream>
#include "matrix.h++"
#include "MLP.h++"

using namespace std;
using namespace matrix;
using namespace mlp;

#define RANGE_STOP 100
#define OUT_STOP 100

double act(double x) {
    if(x > 0) {
        return x;
    } else {
        return 0.1 * x;
    }
}

double act_deriv(double x) {
    if(x > 0) {
        return 1;
    } else {
        return 0.1;
    }

}



double loss(double expected, double actual) {
    return (expected - actual) * (expected - actual);
}

double loss_deriv(double expected, double actual) {
    return 2 * (actual - expected);
}




double func(double x) {
    return tan(x) * 0.3;
}




int main(void) {

    vector<size_t> dims = {1,4,4,4,1};

    
    MLP<double> network(dims, -0.05, 0.05, act, act_deriv, loss, loss_deriv);
    //network.print();


    Matrix<double> input = {1,1};
    Matrix<double> expected = {1,1};
    Matrix<double> result = {1,1};
    Matrix<double> norm = {1,1};

    vector<double> x;
    vector<double> y;

    ofstream file("./data/out.txt", ios::trunc);
    if(!file) {
        cout << "Cannot open file" << endl;
        exit(1);
    }
    file.close();

    ofstream appendFile("./data/out.txt", ios::app);
    if(!appendFile) {
        cout << "Cannot open file" << endl;
        exit(1);
    }





    file << "" << endl;

    for(size_t i = 0; i < OUT_STOP; i++) {
        //double lossMean = 0;

        double start = -3.14159;
        while(start < 3.14159) {

            start += 0.01;
            input.at(0,0) = start/3.14159;
            expected.at(0,0) = func((double)(start/3.14159));
    
            result = network.run(input);

            network.backpropagate(expected, 0.01);

            x.push_back(start/3.14159);
            y.push_back(result.at(0,0));
        }



        if(i == OUT_STOP - 1) {

            appendFile << "x = [";
            for(size_t i = 0; i < x.size(); i++) {
                appendFile << x[i] << ", ";
            }
            appendFile << "]" << endl;

            appendFile << "y = [";
            for(size_t i = 0; i < x.size(); i++) {
                appendFile << x[i] << ", ";
            }
            appendFile << "]" << endl;
        }

        //cout << "Expected: " << expected.at(0,0) << " || Calculated: " << result.at(0,0) << endl;
        //lossMean += network.loss(expected).at(0,0);
        //cout << "LOSS: " << lossMean/1000 << endl;
    }
    

    //network.print();


    appendFile.close();
    cout << "Program complete" << endl;
    return 0;
}




