#include <iostream>
#include <vector>
#include <unistd.h>
#include "matrix.h++"
#include "MLP.h++"

using namespace std;
using namespace matrix;
using namespace mlp;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_deriv(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}



double loss(double expected, double actual) {
    return (expected - actual) * (expected - actual);
}

double loss_deriv(double expected, double actual) {
    return 2 * (actual - expected);
}




double func(double x) {
    return x * x * x * x;
}




int main(void) {

    vector<size_t> dims = {1,5,5,1};

    
    MLP<double> network(dims, -0.5, 0.5, sigmoid, sigmoid_deriv, loss, loss_deriv);
    //network.print();


    Matrix<double> input = {1,1};
    Matrix<double> expected = {1,1};
    Matrix<double> result = {1,1};

    for(size_t i = 0; i < 100; i++) {
        double lossMean = 0;

        for(size_t j = 0; j < 1000; j++) {

            input.at(0,0) = (double)(j);
            expected.at(0,0) = func((double)(j));
    
            result = network.run(input);

            network.backpropagate(expected, 0.001);
        }
        cout << "Expected: " << expected.at(0,0) << " || Calculated: " << result.at(0,0) << endl;
        lossMean += network.loss(expected).at(0,0);
        cout << "LOSS: " << lossMean/1000 << endl;
    }
    

    network.print();



    cout << "Program complete" << endl;
    return 0;
}




