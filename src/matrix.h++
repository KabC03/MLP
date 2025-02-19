#ifndef MATRIX_H
#define MATRIX_H
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstddef>
using namespace std;
namespace matrix {

    template <typename Type>
    class Matrix {

    public:
        
        size_t rows;
        size_t cols;

        vector<Type> data; //Use vector not array since array is stack allocated


        //Constructor
        Matrix(size_t rows, size_t cols);

        
        //At function (get or set)
        Type& at(size_t row, size_t col); //Non constant
        const Type& at(size_t row, size_t col) const; //Constant access


        //Overload operators
        Matrix<Type> operator+(const Matrix<Type>& rhs) const;
        Matrix<Type> operator-(const Matrix<Type>& rhs) const;
        Matrix<Type> operator*(const Matrix<Type>& rhs) const;


        //Print
        void print() const;
    };

}




#endif





