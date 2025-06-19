#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>
#include <thread>


#define MACRO_MATRIX_THREAD_ACTIVATION(rows, cols, destArray, src1Array, func, thrededFunction) { \
    if(rows * cols < threadSizeThreshold) { \
        for(size_t i = 0; i < rows * cols; i++) { \
            destArray[i] = func(src1Array[i]); \
        } \
    } else { \
        vector<thread> threads; \
        threads.reserve(numThreads); \
        size_t matrixSize = rows * cols; \
        size_t evenWork = matrixSize/numThreads; \
        for(size_t i = 0; i < numThreads; i++) { \
            size_t start = i * evenWork; \
            size_t stop  = (i == numThreads - 1) ? matrixSize : (i + 1) * evenWork; \
            threads.emplace_back([&, start, stop] { \
                thrededFunction<Type, decltype(func)>(destArray, src1Array, func, start, stop); \
            }); \
        } \
        for(size_t i = 0; i < numThreads; i++) { \
            threads[i].join(); \
        } \
    } \
};


//Thread addition, subtraction, etc (basically vector addition)
#define MACRO_MATRIX_THREAD_OPERATION(rows, cols, destArray, src1Array, src2Array, nonThreadedOperation, thrededFunction) { \
    if(rows * cols < threadSizeThreshold) { \
        for(size_t i = 0; i < rows * cols; i++) { \
            destArray[i] = src1Array[i] nonThreadedOperation src2Array[i]; \
        } \
    } else { \
        vector<thread> threads; \
        threads.reserve(numThreads); \
        size_t matrixSize = rows * cols; \
        size_t evenWork = matrixSize/numThreads; \
        for(size_t i = 0; i < numThreads; i++) { \
            size_t start = i * evenWork; \
            size_t stop = 0; \
            if (i == numThreads - 1) { \
                stop = matrixSize; \
            } else { \
                stop = (i + 1) * evenWork; \
            } \
            threads.emplace_back([&, start, stop] { \
                thrededFunction<Type>(ref(destArray), cref(src1Array), cref(src2Array), start, stop); \
            }); \
        } \
        for(size_t i = 0; i < numThreads; i++) { \
            threads[i].join(); \
        } \
    } \
};



using namespace std;


namespace matrix {

    static const size_t threadSizeThreshold = 1000; //If a matrix is larger than this use threading
    static unsigned int numThreads = thread::hardware_concurrency();

    template <typename Type>
    static void add_vector(vector<Type> &dest ,const vector<Type> &src1, const vector<Type> &src2, size_t start, size_t stop) {
        for(size_t i = start; i < stop; i++) {
            dest[i] = src1[i] + src2[i];
        }
        return;
    }
    template <typename Type>
    static void sub_vector(vector<Type> &dest ,const vector<Type> &src1, const vector<Type> &src2, size_t start, size_t stop) {
        for(size_t i = start; i < stop; i++) {
            dest[i] = src1[i] - src2[i];
        }
        return;
    }
    template <typename Type>
    static void mul_vector(vector<Type> &dest ,const vector<Type> &src1, const vector<Type> &src2, size_t start, size_t stop) {
        for(size_t i = start; i < stop; i++) {
            dest[i] = src1[i] * src2[i];
        }
        return;
    }
    template <typename Type, typename Func>
    static void activate_vector(std::vector<Type>& dest, const std::vector<Type>& src, Func activationFunction, size_t start, size_t stop) {
        for(size_t i = start; i < stop; i++) {
            dest[i] = activationFunction(src[i]);
        }
    }
    template <typename Type>
    static void rand_vector_float(vector<Type> &dest, Type val, size_t start, size_t stop) {
        for(size_t i = start; i < stop; i++) {
            dest[i] = val;
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

            Matrix result(this->rows, this->cols);
            MACRO_MATRIX_THREAD_OPERATION(this->rows, this->cols, result.data, this->data, rhs.data, +, add_vector);
            
            return result;
        }
        Matrix operator-(const Matrix<Type>& rhs) const {

            Matrix result(this->rows, this->cols);
            MACRO_MATRIX_THREAD_OPERATION(this->rows, this->cols, result.data, this->data, rhs.data, -, sub_vector);
            
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
            MACRO_MATRIX_THREAD_OPERATION(this->rows, this->cols, this->data, this->data, rhs.data, +, add_vector);
            return *this;
        }
        Matrix sub_in_place(const Matrix<Type> &rhs) {
            MACRO_MATRIX_THREAD_OPERATION(this->rows, this->cols, this->data, this->data, rhs.data, -, sub_vector);
            return *this;
        }


        //Randomise
        Matrix randomise_in_place(Type min, Type max) {
            static random_device rd;
            static mt19937 gen(rd());
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
            MACRO_MATRIX_THREAD_ACTIVATION(this->rows, this->cols, result.data, this->data, activationFunction, activate_vector);
            return result;
        }

        Matrix activate_in_place(Type (*activationFunction)(Type arg)) {
            MACRO_MATRIX_THREAD_ACTIVATION(this->rows, this->cols, this->data, this->data, activationFunction, activate_vector);
            return *this;
        }



        //Scalar multiplication
        Matrix scale(const Type scalar) const {
            Matrix result(rows, cols);

            auto scale_element = [scalar](Type arg) {
                return arg * scalar;
            };
            
            MACRO_MATRIX_THREAD_ACTIVATION(this->rows, this->cols, result.data, this->data, scale_element, activate_vector);
            return result;
        }

        Matrix scale_in_place(const Type scalar) {
            auto scale_element = [scalar](Type arg) {
                return arg * scalar;
            };
            MACRO_MATRIX_THREAD_ACTIVATION(this->rows, this->cols, this->data, this->data, scale_element, activate_vector);
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
            Matrix result(this->rows, this->cols);

            MACRO_MATRIX_THREAD_OPERATION(this->rows, this->cols, result.data, this->data, matrix.data, *, mul_vector);

            return result;
        }


        Matrix hadamard_in_place(const Matrix &matrix) {

            MACRO_MATRIX_THREAD_OPERATION(this->rows, this->cols, this->data, this->data, matrix.data, *, mul_vector);

            return *this;
        }

        //Print
        void print_dimensions() const {
            cout << "Rows :: " << rows << " || Cols :: " << cols << "\n" << endl;
            return;
        }

        void print() const {
            this->print_dimensions();
            for(size_t i = 0; i < rows; i++) {
                for(size_t j = 0; j < cols; j++) {
                    cout << this->at(i, j) << " ";
                }
                cout << endl;
            }
            cout << endl;
            return;
        }


        bool append_file(string fileName) {
            ofstream file(fileName, ios::app);
            if(!file) {
                return false;
            }

            for(size_t i = 0; i < rows; i++) {
                for(size_t j = 0; j < cols; j++) {
                    file << this->at(i, j) << " ";
                }
                file << endl;
            }
            file << endl;

            file.close();
            return true;
        }
    };
}









#endif


