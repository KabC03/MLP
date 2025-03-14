#include <iostream>
#include <vector>
#include "matrix.h++"
#include "MLP.h++"

using namespace std;
using namespace matrix;
using namespace mlp;

float relu(float x) {
    if(x > 0) {
        return x;
    } else {
        return 0;
    }
    return 0;
}
float relu_deriv(float x) {
    if(x > 0) {
        return 1;
    } else {
        return 0;
    }
}

float loss(float x, float y) {
    return x - y;
}

float loss_deriv(float x, float y) {
    return 1;
}



int main(void) {

    vector<size_t> dims = {5,3,5};


    MLP<float> network(dims, -0.01, 0.01, relu, relu_deriv, loss, loss_deriv);
    //network.print();


    Matrix<float> input(5,1);
    input.randomise(-0.01, 0.01);
    Matrix<float> result = network.run(input);
    network.print();
    //network.print();


    network.print();

    cout << "Program complete" << endl;
    return 0;
}




