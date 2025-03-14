#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>

using namespace std;
namespace matrix {

    template <typename Type>
    class Matrix {
        public:

        size_t rows;
        size_t cols;

        vector<Type> data;


        //Constructor
        Matrix() {

        }
        Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
            data.resize(rows * cols);
        }

        //Resize 
        void resize(size_t nrows, size_t ncols) {
            rows = nrows;
            cols = ncols;
            data.resize(rows * cols);
        }


        //At
        Type& at(size_t row, size_t col) {
            return data[(this->row * this->cols) + this->col];
        }
        const Type& at(size_t row, size_t col) const{
            return data[(row * cols) + col];
        }

        //Overloading
        Matrix operator+(const Matrix<Type>& rhs) const {

            Matrix result(rows, cols);
            for(size_t i = 0; i < rows * cols; i++) {
                result.data[i] = this->data[i] + rhs.data[i];
            }
            return result;
        }
        Matrix operator-(const Matrix<Type>& rhs) const {

            Matrix result(rows, this->cols);
            for(size_t i = 0; i < rows * cols; i++) {
                result.data[i] = this->data[i] - rhs.data[i];
            }
            return result;
        }
        Matrix operator*(const Matrix<Type>& rhs) const {

            Matrix result(rows, rhs.cols);
            for(size_t i = 0; i < rows; i++) {
                for(size_t j = 0; j < rhs.cols; j++) {

                    result.at(i,j) = 0;
                    for(size_t k = 0; k < cols; k++) {
                        result.at(i,j) += at(i,k) * rhs.at(k,j);
                    }
                }
            }
            return result;
        }

        //Randomise
        void randomise(Type min, Type max) {
            random_device rd;
            mt19937 gen(rd());

            if constexpr(is_integral<Type>::value) {
                uniform_int_distribution<Type> dist(min, max);
                for(size_t i = 0; i < rows; i++) {
                    for(size_t j = 0; j < cols; j++) {
                        at(i,j) = dist(gen);
                    }
                }
            } else if constexpr(is_floating_point<Type>::value) {
                uniform_real_distribution<Type> dist(min, max);
                for(size_t i = 0; i < rows; i++) {
                    for(size_t j = 0; j < cols; j++) {
                        at(i,j) = dist(gen);
                    }
                }
            } else {
                static_assert(is_floating_point<Type>::value || is_integral<Type>::value, "Unsupported type");
            }

        }

        //Activate
        Matrix activate(Type (*activationFunction)(Type arg)) {
            Matrix result(rows, cols);
            for(size_t i = 0; i < rows; i++) {
                for(size_t j = 0; j < cols; j++) {
                    result.at(i,j) = activationFunction(at(i,j));
                }
            }
            return result;
        }

        //Scalar multiplication
        Matrix scale(Type scalar) {
            Matrix result(rows, cols);
            for(size_t i = 0; i < rows; i++) {
                for(size_t j = 0; j < cols; j++) {
                    result.at(i,j) = scalar * at(i,j);
                }
            }
            return result;
        }


        //Transpose
        Matrix transpose(void) {
            Matrix result(cols, rows);
            for(size_t i = 0; i < rows; i++) {
                for(size_t j = 0; j < cols; j++) {
                    result.at(j,i) = at(i,j);
                }
            }
            return result;
        }


        //Hadamard product
        Matrix hadamard(Matrix matrix) {
            Matrix result(rows, cols);
            for(size_t i = 0; i < rows; i++) {
                for(size_t j = 0; j < cols; j++) {
                    result.at(i,j) = at(i,j) * matrix.at(i,j);
                }
            }
            return result;
        }


        //Print
        void print() const {
            cout << "Rows :: " << rows << " || Cols :: " << cols << "\n" << endl;
            for(size_t i = 0; i < rows; i++) {
                for(size_t j = 0; j < cols; j++) {
                    cout << at(i, j) << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
    };
}









#endif


