//AI generated

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
float relu     (float x)        { return x > 0.f ? x : 0.f; }
float relu_d   (float x)        { return x > 0.f ? 1.f : 0.f; }
float identity (float x)        { return x; }
float identity_d(float /*x*/)   { return 1.f; }
float mse      (float y, float ŷ) { float d = y - ŷ; return 0.5f * d * d; }
float mse_d    (float y, float ŷ) { return ŷ - y; }            // dL/dŷ
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// -----------------------------------------------------------------------------


// ============================================================================
//                        ⚙️  EDIT *ONLY* THIS BLOCK ⚙️
// ============================================================================
namespace cfg {
    //Target
    float target(float x) { return sinf(3*x); }

    // Dataset
    constexpr size_t SAMPLES = 1000;
    constexpr float  XMIN    = -M_PI;
    constexpr float  XMAX    =  M_PI;

    // Network topology
    const vector<size_t> DIMS = {1, 16, 16, 1};

    // Init range for weights / biases
    constexpr float INIT_MIN = -0.5f;
    constexpr float INIT_MAX =  0.5f;

    // Training hyper-parameters
    constexpr float  LR      = 0.01f;
    constexpr size_t EPOCHS  = 2000;

    // Callbacks (activations & loss)
    constexpr auto H_ACT     = relu;
    constexpr auto H_ACT_D   = relu_d;
    constexpr auto OUT_ACT   = identity;
    constexpr auto OUT_ACT_D = identity_d;
    constexpr auto LOSS      = mse;
    constexpr auto LOSS_D    = mse_d;
} // namespace cfg
// ============================================================================


int main()
{
    /* -----------------------------------------------------------------------
       1. Create training set
       -------------------------------------------------------------------- */
    const size_t S = cfg::SAMPLES;
    vector<float> xs(S), ys(S);

    for (size_t i = 0; i < S; ++i) {
        float x = cfg::XMIN + (cfg::XMAX - cfg::XMIN) * i / (S - 1);
        xs[i] = x;
        ys[i] = cfg::target(x);          // target
    }

    /* -----------------------------------------------------------------------
       2. Build network
       -------------------------------------------------------------------- */
    auto dims = cfg::DIMS;               // mutable copy – satisfies the ctor
    MLP<float> net(dims,                 // layer sizes
                cfg::INIT_MIN, cfg::INIT_MAX,
                cfg::H_ACT, cfg::H_ACT_D,
                cfg::OUT_ACT, cfg::OUT_ACT_D,
                cfg::LOSS, cfg::LOSS_D);

    /* -----------------------------------------------------------------------
       3. Train
       -------------------------------------------------------------------- */
    Matrix<float> in (1,1);
    Matrix<float> tgt(1,1);

    for (size_t e = 0; e < cfg::EPOCHS; ++e) {
        float epoch_loss = 0.f;

        for (size_t i = 0; i < S; ++i) {
            in .at(0,0) = xs[i];
            tgt.at(0,0) = ys[i];

            net.backpropagate(in, tgt, cfg::LR);
            epoch_loss += net.loss(tgt).at(0,0);
        }
        if (e % 50 == 0) {
            cout << "Epoch " << e
                 << " | mean L = " << epoch_loss / S << '\n';
        }
    }

    /* -----------------------------------------------------------------------
       4. Dump predictions
       -------------------------------------------------------------------- */
    ofstream out("./data/out.txt", ios::trunc);
    if (!out) { cerr << "Could not open ./data/out.txt\n"; return 1; }

    Matrix<float> pred;
    for (size_t i = 0; i < S; ++i) {
        in.at(0,0) = xs[i];
        pred = net.run(in);
        out << xs[i] << ", " << pred.at(0,0) << ", " << ys[i] << '\n';
    }
    cout << "Finished. Results written to ./data/network.txt\n";

    net.save("out.txt");
    return 0;
}



