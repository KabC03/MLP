#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>
#include <thread>

using namespace std;
static unsigned int numThreads = thread::hardware_concurrency();

namespace matrix {

    template <typename Type>
    static void add_vector(vector<Type> &dest ,vector<Type> &src1, vector<Type> &src2, size_t start, size_t stop) {
        for(size_t i = start; i < stop; i++) {
            dest[i] = src1[i] + src2[i];
        }
        return;
    }
    template <typename Type>
    static void sub_vector(vector<Type> &dest ,vector<Type> &src1, vector<Type> &src2, size_t start, size_t stop) {
        for(size_t i = start; i < stop; i++) {
            dest[i] = src1[i] - src2[i];
        }
        return;
    }


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
        Matrix(size_t rows, size_t cols, vector<Type> inputData) : rows(rows), cols(cols) {
            this->data = inputData;
        }

        //Convert to vector
        vector<float> vectorise() {
            vector<Type> result;
            result.resize(rows * cols);
            result = data;
            return result;
        }

        //Resize
        void resize(const size_t nrows, const size_t ncols) {
            rows = nrows;
            cols = ncols;
            data.resize(rows * cols);
        }

        //Fill
        void fill(const vector<Type> newData) {
            data = newData;
        }

        //At
        Type& at(size_t row, size_t col) {
            return data[(row * cols) + col];
        }
        const Type& at(size_t row, size_t col) const{
            return data[(row * cols) + col];
        }

        //Overloading
        Matrix operator+(const Matrix<Type>& rhs) const {

            Matrix result(rows, cols);
            vector<thread> threads(numThreads);



            size_t matrixSize = this->rows * this->cols;
            size_t evenWork = matrixSize/numThreads;
            //size_t remainder = matrixSize % numThreads;

            for(size_t i = 0; i < numThreads - 1; i++) { //First threads do even work
                threads.emplace_back(thread(add_vector<Type>, ref(result.data), cref(this->data), cref(rhs.data), 
                i * evenWork, (i + 1) * evenWork));
            }
            threads.emplace_back(thread(add_vector<Type>, ref(result.data), cref(this->data), cref(rhs.data),
             (numThreads - 1) * evenWork, matrixSize));

            for(size_t i = 0; i < numThreads; i++) {
                threads[i].join();
            }


            
            return result;
        }
        Matrix operator-(const Matrix<Type>& rhs) const {

            Matrix result(rows, cols);
            vector<thread> threads(numThreads);

            size_t matrixSize = this->rows * this->cols;
            size_t evenWork = matrixSize/numThreads;
            //size_t remainder = matrixSize % numThreads;

            for(size_t i = 0; i < numThreads - 1; i++) { //First threads do even work
                threads.emplace_back(thread(sub_vector<Type>, ref(result.data), cref(this->data), cref(rhs.data), 
                i * evenWork, (i + 1) * evenWork));
            }
            threads.emplace_back(thread(sub_vector<Type>, ref(result.data), cref(this->data), cref(rhs.data),
             (numThreads - 1) * evenWork, matrixSize));

            for(size_t i = 0; i < numThreads; i++) {
                threads[i].join();
            }
            return result;
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


        //In place arithmatic
        Matrix multiply_in_place(const Matrix<Type> &lhs, const Matrix<Type> &rhs) {
            for(size_t i = 0; i < lhs.rows; i++) {
                for(size_t j = 0; j < rhs.cols; j++) {

                    at(i,j) = 0;
                    for(size_t k = 0; k < lhs.cols; k++) {
                        at(i,j) += lhs.at(i,k) * rhs.at(k,j);
                    }
                }
            }
            return *this;
        }

        Matrix add_in_place(const Matrix<Type> &rhs) {
            for(size_t i = 0; i < rows * cols; i++) {
                data[i] = data[i] + rhs.data[i];
            }
            return *this;
        }
        Matrix sub_in_place(const Matrix<Type> &rhs) {
            for(size_t i = 0; i < rows * cols; i++) {
                data[i] = data[i] - rhs.data[i];
            }
            return *this;
        }


        //Randomise
        Matrix randomise_in_place(Type min, Type max) {
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
                static_assert(is_floating_point<Type>::value || is_integral<Type>::value, "Unsupported type for matrix randomisation");
            }
            return *this;
        }

        //Activate
        Matrix activate(Type (*activationFunction)(Type arg)) const {
            Matrix result(rows, cols);
            for(size_t i = 0; i < rows; i++) {
                for(size_t j = 0; j < cols; j++) {
                    result.at(i,j) = activationFunction(at(i,j));
                }
            }
            return result;
        }

        Matrix activate_in_place(Type (*activationFunction)(Type arg)) {
            for(size_t i = 0; i < rows; i++) {
                for(size_t j = 0; j < cols; j++) {
                    at(i,j) = activationFunction(at(i,j));
                }
            }
            return *this;
        }



        //Scalar multiplication
        Matrix scale(const Type scalar) const {
            Matrix result(rows, cols);
            for(size_t i = 0; i < rows; i++) {
                for(size_t j = 0; j < cols; j++) {
                    result.at(i,j) = scalar * at(i,j);
                }
            }
            return result;
        }

        Matrix scale_in_place(const Type scalar) {
            for(size_t i = 0; i < rows; i++) {
                for(size_t j = 0; j < cols; j++) {
                    at(i,j) = scalar * at(i,j);
                }
            }
            return *this;
        }


        //Transpose
        Matrix transpose(void) const {
            Matrix result(cols, rows);
            for(size_t i = 0; i < rows; i++) {
                for(size_t j = 0; j < cols; j++) {
                    result.at(j,i) = at(i,j);
                }
            }
            return result;
        }

        Matrix transpose_in_place(void) {
            for(size_t i = 0; i < rows; i++) {
                for(size_t j = 0; j < cols; j++) {
                    Type temp = at(j,i);
                    at(j,i) = at(i,j);
                    at(i,j) = temp;
                }
            }
            return *this;
        }

        //Min max norm
        Matrix normalise() const {
            Matrix result(rows, cols);

            //Find min and max in matrix
            Type minVal = 0;
            Type maxVal = 0;
            for(size_t i = 0; i < rows; i++) {
                for(size_t j = 0; j < cols; j++) {
                    if(at(i,j) > maxVal) {
                        maxVal = at(i,j);
                    }
                    if(minVal > at(i,j)) {
                        minVal = at(i,j);
                    }
                }
            }

            //Normalise
            for(size_t i = 0; i < rows; i++) {
                for(size_t j = 0; j < cols; j++) {
                    result.at(i,j) = (2 * (at(i,j) - minVal) / (maxVal - minVal)) - 1;
                }
            }
            return result;
        }

        //Hadamard product
        Matrix hadamard(const Matrix &matrix) const {
            Matrix result(rows, cols);
            for(size_t i = 0; i < rows; i++) {
                for(size_t j = 0; j < cols; j++) {
                    result.at(i,j) = at(i,j) * matrix.at(i,j);
                }
            }
            return result;
        }


        Matrix hadamard_in_place(const Matrix &matrix) {
            for(size_t i = 0; i < rows; i++) {
                for(size_t j = 0; j < cols; j++) {
                    at(i,j) = at(i,j) * matrix.at(i,j);
                }
            }
            return *this;
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


