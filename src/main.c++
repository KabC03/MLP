#include <iostream>
#include <vector>
#include <unistd.h>
#include "matrix.h++"
#include "MLP.h++"

using namespace std;
using namespace matrix;
using namespace mlp;

#define RANGE_STOP 100
#define OUT_STOP 1000

double act(double x) {
    if(x > 0) {
        return x;
    } else {
        return x/10;
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
    return x * x * x;
}




int main(void) {

    vector<size_t> dims = {1,5,5,1};

    
    MLP<double> network(dims, -0.5, 0.5, act, act_deriv, loss, loss_deriv);
    //network.print();


    Matrix<double> input = {1,1};
    Matrix<double> expected = {1,1};
    Matrix<double> result = {1,1};

    vector<double> x;
    x.resize(RANGE_STOP);
    vector<double> y;
    y.resize(RANGE_STOP);

    for(size_t i = 0; i < OUT_STOP; i++) {
        //double lossMean = 0;

        for(size_t j = 0; j < RANGE_STOP; j++) {

            input.at(0,0) = double(j) / RANGE_STOP;
            expected.at(0,0) = func((double)(j));
    
            result = network.run(input);

            network.backpropagate(expected, 0.001);

            x[j] = double(j) / RANGE_STOP;
            y[j] = expected.at(0,0);

        }

        if(i == OUT_STOP - 1) {
            for(size_t i = 0; i < x.size(); i++) {
                cout << x[i] << ", ";
            }
            cout << endl;
            cout << endl;
            cout << endl;
            cout << endl;
            cout << endl;
            cout << endl;
            for(size_t i = 0; i < y.size(); i++) {
                cout << y[i] << ", ";
            }
            cout << endl;
        }

        //cout << "Expected: " << expected.at(0,0) << " || Calculated: " << result.at(0,0) << endl;
        //lossMean += network.loss(expected).at(0,0);
        //cout << "LOSS: " << lossMean/1000 << endl;
    }
    

    //network.print();



    cout << "Program complete" << endl;
    return 0;
}




