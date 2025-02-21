#include <iostream>
#include <vector>
#include "matrix.h++"
#include "MLP.h++"

using namespace std;
using namespace matrix;
using namespace mlp;

int main(void) {

    vector<size_t> dims = {3,2,3,4};

    MLP<float> network(dims, -0.01, 0.01);
    //network.print();


    Matrix<float> input(3,1);
    input.randomise(-0.01, 0.01);
    Matrix<float> result = network.run(input);
    //network.print();
    result.print();

    cout << "Program complete" << endl;
    return 0;
}




