#include <iostream>
#include <vector>
#include "matrix.h++"

using namespace std;
using namespace matrix;

int main(void) {

    Matrix<int> a(3,3);
    a.randomise(0,10); 

    a.print();
    cout << "Program complete" << endl;
    return 0;
}




