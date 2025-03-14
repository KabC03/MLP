#include <iostream>
#include <vector>
#include "matrix.h++"
#include "MLP.h++"

using namespace std;
using namespace matrix;
using namespace mlp;

float func(float x) {
    return 0;
}
float func2(float x) {
    return x;
}

int main(void) {

    vector<size_t> dims = {};

    MLP<float> network(dims, -0.01, 0.01, func, func2);
    //network.print();


    Matrix<float> input(3,1);
    input.randomise(-0.01, 0.01);
    Matrix<float> result = network.run(input);
    //network.print();
    result.print();

    cout << "Program complete" << endl;
    return 0;
}




