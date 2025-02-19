#ifndef MATRIX_H
#define MATRIX_H
#include <vector>
#include <cstdint>
#include <cstddef>

namespace matrix {

    template <typename type>
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
        Matrix<Type> operator+(const Matrix<type>& rhs) const;
        Matrix<Type> operator-(const Matrix<type>& rhs) const;
        Matrix<Type> operator*(const Matrix<type>& rhs) const;


        //Print
        void print() const;
    };

}



Matrix matrix_create(size_t rows, size_t cols);
void matrix_randomise(Matrix matrix, 

#endif





