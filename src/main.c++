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

#define OUT_STOP 10

float act(float x) {
    if(x > 0) {
        return x;
    } else {
        return 0.1 * x;
    }
}

float act_deriv(float x) {
    if(x > 0) {
        return 1;
    } else {
        return 0.1;
    }

}



float loss(float expected, float actual) {
    return (expected - actual) * (expected - actual);
}

float loss_deriv(float expected, float actual) {
    return 2 * (actual - expected);
}




float func1(float x) {
    return 0.2 * sin(x) * x;
}
float func2(float x) {
    return 0.5 * cos(x) * x;
}
float func3(float x, float y) {
    return 0.5 * cos(x);
}


int main(void) {

    Matrix<float> m1;
    Matrix<float> m2;
    m1.randomise_in_place(0.01, 0.05);
    m2.randomise_in_place(0.01, 0.05);

    Matrix<float> m3 = m1 + m2;
    m3.print();

    return 0;
    vector<size_t> dims = {2, 500, 2};

    
    MLP<float> network(dims, -0.005, 0.005, act, act_deriv, loss, loss_deriv);
    //network.print();


    Matrix<float> input = {2,1};
    Matrix<float> expected = {2,1};
    Matrix<float> result = {2,1};

    vector<float> x;
    vector<float> y;
    vector<float> z;

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
        //float lossMean = 0;

        float start = -3.14159;
        size_t count = 0;

        clock_t startTime = 0;
        clock_t endTime = 0;
        float timeRun = 0;
        float timeTrain = 0;

        while(start < 3.14159) {

            start += 0.001;
            input.at(0,0) = start/3.14159;
            expected.at(0,0) = func1((float)(start));
            expected.at(1,0) = func2((float)(start));
    
            startTime = clock();
            result = network.run(input);
            endTime = clock();
            timeRun = ((float) (endTime - startTime)) / CLOCKS_PER_SEC;


            startTime = clock();
            network.backpropagate(input, expected, 0.01);
            endTime = clock();
            timeTrain = ((float) (endTime - startTime)) / CLOCKS_PER_SEC;


            if(i == OUT_STOP - 1) {
                x.push_back(start/3.14159);
                y.push_back(result.at(0,0));
                z.push_back(result.at(1,0));
            }
            /*
            if(count++ % 100000 == 0) {
                cout << "Expected: " << expected.at(0,0) << " || Calculated: " << result.at(0,0) << endl;
            }
            */
            if(count++ % 10000 == 0) {
                cout << "\tRun: " << timeRun << " || Train: " << timeTrain << endl;
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
    

    //network.print();


    appendFile.close();
    cout << "Program complete" << endl;
    return 0;
}




