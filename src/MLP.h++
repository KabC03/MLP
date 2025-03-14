#ifndef MLP_H
#define MLP_H
#include "matrix.h++"

using namespace std;
using namespace matrix;

namespace mlp {
    
    template <typename Type>
    class MLP {
        public:

        //Constructor
        MLP(vector<size_t> &dimensions, Type min, Type max, Type (*newActivationFunction)(Type arg), Type (*newActivationFunctionDerivative)(Type arg)) {
            //[1,2,3] creates a network with 1 input neuron, 1 hidden layer (2 neurons) and 3 output neurons 

            activationFunction = newActivationFunction;
            activationFunctionDerivative = newActivationFunctionDerivative;
            weights.resize(dimensions.size() - 1);
            biases.resize(dimensions.size() - 1);

            preActivation.resize(dimensions.size() - 1);
            outputs.resize(dimensions.size() - 1);

            numLayers = dimensions.size() - 1;

            for(size_t i = 0; i < numLayers; i++) {
                weights[i].resize(dimensions[i + 1], dimensions[i]);
                biases[i].resize(dimensions[i + 1], 1);

                preActivation[i].resize(dimensions[i + 1], 1);
                outputs[i].resize(dimensions[i + 1], 1);

                weights[i].randomise(min, max);
                biases[i].randomise(min, max);
            }
        }


        //Run network
        Matrix<Type> run(Matrix<Type> &input) {
            networkInput = input;

            Matrix<Type> *prevOutput = &networkInput; //Use pointer instead of reference
            for(size_t i = 0; i < numLayers; i++) {

                //cout << "\t\t\t\t HERE" << endl;
                //weights[i].print();
                //(*prevOutput).print();
                //biases[i].print();

                //Matrix<Type> temp = (weights[i] * *prevOutput);
                //temp.print();

                preActivation[i] = (weights[i] * (*prevOutput)) + biases[i]; 
                outputs[i] = preActivation[i].activate(activationFunction);

                prevOutput = &(outputs[i]);
            }


            return outputs.back();
        }

        //Backpropagate
        void backpropagate(Matrix<Type> &input) {
            
            return;
        }

        //Print network structure
        void print() {
            cout << "Layers: " << numLayers << endl;
            cout << "Input: " << endl;
            networkInput.print();
            for(size_t i = 0; i < numLayers; i++) {
                cout << "\t\tLayer: " << i << endl;
                cout << "Weights: " << endl;
                weights[i].print(); 
                cout << "Biases: " << endl;
                biases[i].print(); 
                cout << "Outputs: " << endl;
                outputs[i].print(); 
            }
        }




        private:

        //Attributes

        size_t numLayers;
        vector<Matrix<Type>> weights;
        vector<Matrix<Type>> biases;
        vector<Matrix<Type>> preActivation;
        vector<Matrix<Type>> outputs;

         
        Matrix<Type> networkInput;
        Type (*activationFunction)(Type arg);
        Type (*activationFunctionDerivative)(Type arg);

    };
}

/*
    TODO:

        - ADD ACTIVATION FUNCTION TO NETWORK RUN FUNCTION
        - Backpropagate method
        void MLP_backprop(Network &network, vector<float> &expected);
*/






#endif

