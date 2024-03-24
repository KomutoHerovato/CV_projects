#include "Circle.h"
#include <cmath>

Circle::Circle(Point O, int R) : O(O){
    this->R = R;
}

double Circle::getArea() const {
    double a = (M_PI*R*R);
    return a;
}

double Circle::getPerLength() const {
    double a = (2*M_PI*R);
    return a;
}

Point Circle::getMass() {
    return O;
}

Point Circle::getVertices() {
    return O;
}
