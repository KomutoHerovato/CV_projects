#include <iostream>
#include <vector>

template <typename T>
class Matrix {
private:
    std::vector<std::vector<T>> data;
    size_t rows, cols;

public:
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
        data.resize(rows, std::vector<T>(cols));
    }

    void print() const {
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                std::cout << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrices have different sizes");
        }

        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrices have different sizes");
        }

        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] - other.data[i][j];
            }
        }
        return result;
    }

    Matrix operator*(const T& scalar) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * scalar;
            }
        }
        return result;
    }

    T& operator()(size_t row, size_t col) {
        if (row >= rows || col >= cols) {
            throw std::out_of_range("Index out of range");
        }
        return data[row][col];
    }

    const T& operator()(size_t row, size_t col) const {
        if (row >= rows || col >= cols) {
            throw std::out_of_range("Index out of range");
        }
        return data[row][col];
    }
};

int main() {
    Matrix<int> m(2, 3);
    m(0, 0) = 1;
    m(0, 1) = 2;
    m(0, 2) = 3;
    m(1, 0) = 4;
    m(1, 1) = 5;
    m(1, 2) = 6;

    std::cout << "Matrix m:" << std::endl;
    m.print();

    Matrix<int> n(2, 3);
    n(0, 0) = 7;
    n(0, 1) = 8;
    n(0, 2) = 9;
    n(1, 0) = 10;
    n(1, 1) = 11;
    n(1, 2) = 12;

    std::cout << "Matrix n:" << std::endl;
    n.print();

    Matrix<int> sum = m + n;
    std::cout << "Matrix sum = m + n:" << std::endl;
    sum.print();

    Matrix<int> diff = m - n;
    std::cout << "Matrix diff = m - n:" << std::endl;
    diff.print();

    Matrix<int> scaled = m * 2;
    std::cout << "Matrix scaled = m * 2:" << std::endl;
    scaled.print();

    std::cout << "Element (1, 2) of matrix m: " << m(1, 2) << std::endl;

    return 0;
}
