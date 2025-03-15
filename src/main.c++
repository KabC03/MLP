#include <iostream>
#include <vector>
#include <unistd.h>
#include <fstream>
#include "matrix.h++"
#include "MLP.h++"

using namespace std;
using namespace matrix;
using namespace mlp;

#define OUT_STOP 10000

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




double func1(double x) {
    return 0.2 * sin(x) * x;
}
double func2(double x) {
    return 0.5 * cos(x) * x;
}
double func3(double x, double y) {
    return 0.5 * cos(x) * y;
}


int main(void) {

    vector<size_t> dims = {2, 5, 5, 5, 2};

    
    MLP<double> network(dims, -0.05, 0.05, act, act_deriv, loss, loss_deriv);
    //network.print();


    Matrix<double> input = {2,1};
    Matrix<double> expected = {2,1};
    Matrix<double> result = {2,1};

    vector<double> x;
    vector<double> y;
    vector<double> z;

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
        size_t count = 0;
        while(start < 3.14159) {

            start += 0.0001;
            input.at(0,0) = start/3.14159;
            expected.at(0,0) = func1((double)(start));
            expected.at(1,0) = func2((double)(start));
    
            result = network.run(input);

            network.backpropagate(expected, 0.01);

            if(i == OUT_STOP - 1) {
                x.push_back(start/3.14159);
                y.push_back(result.at(0,0));
                z.push_back(result.at(1,0));
            }
            if(count++ % 10000 == 0) {
                cout << "Expected: " << expected.at(0,0) << " || Calculated: " << result.at(0,0) << endl;
            }

        }


        if(i == OUT_STOP - 1) {

            for(size_t i = 0; i < x.size(); i++) {
                appendFile << x[i] << " ";
            }
            appendFile << endl;

            for(size_t i = 0; i < y.size(); i++) {
                appendFile << y[i] << " ";
            }
            appendFile << endl;

            
            for(size_t i = 0; i < z.size(); i++) {
                appendFile << z[i] << " ";
            }
            
            appendFile << endl;
            break;
        }



    }
    

    network.print();


    appendFile.close();
    cout << "Program complete" << endl;
    return 0;
}




