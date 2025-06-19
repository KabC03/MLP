// test_function.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "MLP.h++"      // your header
#include "matrix.h++"   // your matrix class

using namespace std;
using namespace matrix;
using namespace mlp;

// ---------- helpers ----------------------------------------------------------
float relu(float x)                   { return x > 0.f ? x : 0.f; }
float relu_d(float x)                 { return x > 0.f ? 1.f : 0.f; }
float identity(float x)               { return x; }
float identity_d(float)               { return 1.f; }
float mse(float y, float ŷ)           { float d=y-ŷ; return 0.5f*d*d; }
float mse_d(float y, float ŷ)         { return ŷ-y; }            // dL/dŷ

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// -----------------------------------------------------------------------------

int main()
{
    /* -----------------------------------------------------------------------
       1. Create training set
       -------------------------------------------------------------------- */
    const size_t S = 1024;                 // #samples
    vector<float> xs(S), ys(S);
    const float xmin = -static_cast<float>(M_PI);
    const float xmax =  static_cast<float>(M_PI);

    for (size_t i = 0; i < S; ++i) {
        float x = xmin + (xmax - xmin) * i / (S - 1);
        xs[i] = x;
        ys[i] = 0.5f * sinf(3.f * x) + 0.5f * cosf(5.f * x);    // target
    }

    /* -----------------------------------------------------------------------
       2. Build network: 1-32-32-1
       -------------------------------------------------------------------- */
    vector<size_t> dims = {1, 64, 64, 64, 1};
    MLP<float> net(dims,                     // layer sizes
                   -0.5f, 0.5f,              // weight/bias init range
                   relu,  relu_d,            // hidden activation
                   identity, identity_d,     // output activation
                   mse,   mse_d);            // loss

    /* -----------------------------------------------------------------------
       3. Train
       -------------------------------------------------------------------- */
    const float   lr     = 0.00001f;
    const size_t  epochs = 10000;

    Matrix<float> in (1,1);
    Matrix<float> tgt(1,1);

    for (size_t e = 0; e < epochs; ++e) {
        float epoch_loss = 0.f;

        for (size_t i = 0; i < S; ++i) {
            in .at(0,0) = xs[i];
            tgt.at(0,0) = ys[i];

            net.backpropagate(in, tgt, lr);
            epoch_loss += net.loss(tgt).at(0,0);
        }

        if (e % 50 == 0) {
            cout << "Epoch " << e
                 << " | mean L = " << epoch_loss / S << '\n';
        }
    }

    /* -----------------------------------------------------------------------
       4. Dump predictions -> ./data/out.txt
       -------------------------------------------------------------------- */
    ofstream out("./data/out.txt", ios::trunc);
    if (!out) { cerr << "Could not open ./data/out.txt\n"; return 1; }

    Matrix<float> pred;
    for (size_t i = 0; i < S; ++i) {
        in.at(0,0) = xs[i];
        pred = net.run(in);
        out << xs[i] << ", " << pred.at(0,0) << ", " << ys[i] << '\n';
    }
    cout << "Finished. Results written to ./data/out.txt\n";
    net.save("./data/0.5f * sinf(3.f * x) + 0.5f * cosf(5.f * x).txt");
    return 0;
}
