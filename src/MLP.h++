#ifndef MLP_H
#define MLP_H
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>
#include <thread>
#include "matrix.h++"


using namespace std;
using namespace matrix;

namespace mlp {

    template <typename Type>
    class MLP {

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

        Type (*lossFunction)(Type expected, Type actual);
        Type (*lossFunctionDerivative)(Type expected, Type actual); //Derivative with respect to actual output NOT expected
    

        Type (*lastLayeractivationFunction)(Type arg);
        Type (*lastLayeractivationFunctionDerivative)(Type arg); //Derivative with respect to actual output NOT expected


        std::vector<std::string> tokenize(const std::string& input) {
            istringstream iss(input);
            vector<std::string> tokens;
            string word;

            while (iss >> word) {
                tokens.push_back(word);
            }

            return tokens;
        }       

    
        public:

        //Constructor
        MLP(vector<size_t> &dimensions, Type min, Type max, 
        Type (*newActivationFunction)(Type arg), 
        Type (*newActivationFunctionDerivative)(Type arg),  
        Type (*newLastLayerActivationFunction)(Type arg), 
        Type (*newLastLayerActivationFunctionDerivative)(Type arg), 
        Type (*newLossFunction)(Type arg1, Type arg2), 
        Type (*newLossFunctionDerivative)(Type arg1, Type arg2)) {
            //[1,2,3] creates a network with 1 input neuron, 1 hidden layer (2 neurons) and 3 output neurons 

            activationFunction = newActivationFunction;
            activationFunctionDerivative = newActivationFunctionDerivative;

            lastLayeractivationFunction = newLastLayerActivationFunction;
            lastLayeractivationFunctionDerivative = newLastLayerActivationFunctionDerivative;

            lossFunction = newLossFunction;
            lossFunctionDerivative = newLossFunctionDerivative;

            weights.resize(dimensions.size() - 1);
            biases.resize(dimensions.size() - 1);

            preActivation.resize(dimensions.size() - 1);
            outputs.resize(dimensions.size() - 1);

            numLayers = dimensions.size() - 1;

            networkInput.resize(dimensions[0], 1);
            networkInput.randomise_in_place(min, max);

            for(size_t i = 0; i < numLayers; i++) {
                weights[i].resize(dimensions[i + 1], dimensions[i]);
                biases[i].resize(dimensions[i + 1], 1);

                preActivation[i].resize(dimensions[i + 1], 1);
                outputs[i].resize(dimensions[i + 1], 1);

                weights[i].randomise_in_place(min, max);
                biases[i].randomise_in_place(min, max);
            }
        }



        //Import network params froma file
        bool load(string fileName) {
            
            ifstream file(fileName);

            if(!file) {
                return false;
            }
            string line = "";
            typedef enum STATE {
                WEIGHT,
                BIAS,
            } STATE;
            STATE state = WEIGHT;
            Matrix<Type> currentMatrix = (1,1);

            while(1) {

                getline(file, line);



                vector<string> tokens = tokenise(line);
            }
            file.close();
            return true;
        }

        //Export network parametrs to a file
        bool save(string fileName) {

            //Clear file
            std::ofstream file(fileName, ios::trunc);
            if (!file) {
                return false;
            }
            file.close();

            for(size_t i = 0; i < numLayers; i++) {

                if(weights[i].append_file(fileName) == false) {
                    return false;
                }
                if(biases[i].append_file(fileName) == false) {
                    return false;
                }
            }

            return true;
        }

        


        //Run network
        Matrix<Type> run(Matrix<Type> &input) {
            networkInput = input;

            Matrix<Type> *prevOutput = &networkInput; //Use pointer instead of reference
            for(size_t i = 0; i < numLayers; i++) {


                outputs[i] = ((weights[i] * (*prevOutput)) + (biases[i])).activate(activationFunction); //Slower...
                //outputs[i] = ((weights[i] * (*prevOutput)) + biases[i]).activate(activationFunction); //Also slower
                //outputs[i] = ((weights[i] * (*prevOutput)) + biases[i]).activate(activationFunction);

                prevOutput = &(outputs[i]);
            }
            return outputs.back();
        }




        //Backpropagate
        void backpropagate(Matrix<Type> &input, Matrix<Type> &expectedOutput, Type learningRate) {


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

            Matrix<Type> error(expectedOutput.rows, expectedOutput.cols);
            for(size_t i = 0; i < expectedOutput.rows; i++) {
                for(size_t j = 0; j < expectedOutput.cols; j++) {
                    error.at(i,j) = lossFunctionDerivative(expectedOutput.at(i,j), outputs.back().at(i,j));
                }
            }

            for(size_t i = numLayers - 1; i != SIZE_MAX; --i) {
                Matrix<Type> delta = error.hadamard(preActivation[i].activate(activationFunctionDerivative));

                Matrix<Type> inputTransposed;
                if(i == 0) {
                    inputTransposed = networkInput.transpose();
                } else {
                    inputTransposed = outputs[i - 1].transpose();
                }

                Matrix<Type> deltaWeights = delta * inputTransposed;
                Matrix<Type> deltaBiases = delta;

                Matrix<Type> &currentWeights = weights[i];
                weights[i] = weights[i] - (deltaWeights.scale(learningRate));
                biases[i] = biases[i] - (deltaBiases.scale(learningRate));

                if(i > 0) {
                    error = currentWeights.transpose() * delta;
                }
            }

            return;
        }

        //Calculate loss
        Matrix<Type> loss(Matrix<Type> &expectedOutput) {
            Matrix<Type> result(expectedOutput.rows, expectedOutput.cols);

            for(size_t i = 0; i < expectedOutput.rows; i++) {
                for(size_t j = 0; j < expectedOutput.cols; j++) {
                    result.at(i,j) = lossFunction(expectedOutput.at(i,j), outputs.back().at(i,j));
                }
            }

            return result;
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
                cout << "Pre-activation: " << endl;
                preActivation[i].print();
                cout << "Outputs: " << endl;
                outputs[i].print(); 
            }
        }
        void print_dimensions() {
            cout << "Layers: " << numLayers << endl;
            cout << "Input: " << endl;
            networkInput.print_dimensions();
            for(size_t i = 0; i < numLayers; i++) {
                cout << "\t\tLayer: " << i << endl;
                cout << "Weights: " << endl;
                cout << "Biases: " << endl;
                cout << "Pre-activation: " << endl;
                preActivation[i].print_dimensions();
                cout << "Outputs: " << endl;
                outputs[i].print_dimensions(); 
            }
        }
    };
}






#endif


