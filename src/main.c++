#include <iostream>
#include <vector>
#include <unistd.h>
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
        return 0.01;
    }
}

float loss(float x, float y) {
    return (x - y) * (x - y);
}

float loss_deriv(float y, float x) {
    return 2 * (x - y);
}




float func(float x) {
    return x * x * x * x;
}




int main(void) {

    vector<size_t> dims = {1,5,1};

    
    MLP<float> network(dims, -0.5, 0.5, relu, relu_deriv, loss, loss_deriv);
    //network.print();


    Matrix<float> input = {1,1};
    Matrix<float> expected = {1,1};


    for(size_t i = 0; i < SIZE_MAX; i++) {

        input.at(0,0) = (float)(i)/255;
        expected.at(0,0) = func((float)(i)/255);

        Matrix<float> result = network.run(input);
        
        network.backpropagate(expected, 0.0001);



        usleep(50000);
        //result.print();
        cout << "Expected: " << expected.at(0,0) << " || Calculated: " << result.at(0,0) << endl;
        cout << "LOSS:" << endl;
        network.loss(expected).print();
    }

    cout << "Program complete" << endl;
    return 0;
}




