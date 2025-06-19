#include <iostream>
#include <vector>
#include <unistd.h>
#include <fstream>
#include <time.h>
#include "matrix.h++"
#include "MLP.h++"

using namespace std;
using namespace matrix;
using namespace mlp;


float relu(float x) {
    return x > 0 ? x : 0;
}

float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}


float mse(float expected, float actual) {
    float diff = expected - actual;
    return 0.5f * diff * diff;
}

float mse_derivative(float expected, float actual) {
    return actual - expected;
}

int main() {
    vector<size_t> dims = {3, 2, 1};
    MLP<float> net(dims, -1.0f, 1.0f, relu, relu_derivative, relu, relu_derivative, mse, mse_derivative);
    
    if(net.save("./data/network.txt") == false) {
        cout << "Export failed" << endl;
    }
    cout << "Network:" << endl;
    net.print();


    net.load("./data/network.txt");

    cout << "Loading network:" << endl;
    net.print();

    return 0;
}




