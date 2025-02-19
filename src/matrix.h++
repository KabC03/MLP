#
#
earqqqqqqqqfndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace matrix {

    template <typename Type>
    class Matrix {

        //Constructor
        Matrix<Type>::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
            data.resize(rows * cols);
        }


        //At
        Type& Matrix<Type>::at(size_t row, size_t col) {
            return data[(row * cols) + col];
        }
        const Type& Matrix<Type>::at(size_t row, size_t col) const {
            return data[(row * cols) + col];
        }


        //Overloading
        Matrix<Type> Matrix<Type>::operator+(const Matrix<Type>& rhs) const {

            Matrix result(rows, cols);
            for(size_t i = 0; i < rows * cols; i++) {
                result.data[i] = this->data[i] + rhs.data[i];
            }
            return result;
        }
        Matrix<Type> Matrix<Type>::operator-(const Matrix<Type>& rhs) const {

            Matrix result(rows, cols);
            for(size_t i = 0; i < rows * cols; i++) {
                result.data[i] = this->data[i] - rhs.data[i];
            }
            return result;
        }
        Matrix<Type> Matrix<Type>::operator*(const Matrix<Type>& rhs) const {

            Matrix result(rows, cols);
            for(size_t i = 0; i < rows; i++) {
                for(size_t j = 0; j < cols; j++) {
                    for(size_t k = 0; k < cols; k++) {
                        result.at(i,j) = at(i,k) * rhs.at(j,k);
                    }
                }
            }
            return result;
        }


        //Print
        void Matrix<Type>::print() const {
            cout << "Rows :: " << rows << " || Cols :: " << cols << "\n" << endl;
            for(size_t i = 0; i < rows; i++) {
                for(size_t j = 0; j < cols; j++) {
                    cout << at(i, j) << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
    }
}









#endif


