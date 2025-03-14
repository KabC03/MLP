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


    Matrix<float> m1(1,5);
    m1.randomise(0, 100);
    Matrix<float> m2 = m1.transpose();
    m1.print();
    m2.print();
    exit(1);

    vector<size_t> dims = {3,2,3,4};

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




