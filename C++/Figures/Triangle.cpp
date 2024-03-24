//
// Created by Дамир Мухаметзянов on 10/27/23.
//

#include "Triangle.h"
#include <cmath>

Triangle::Triangle(Point p1, Point p2, Point p3): p1(p1), p2(p2), p3(p3) {

}

double Triangle::getPerLength() const {
    double a = sqrt(pow(p1.getX() - p2.getX(), 2) + pow(p1.getY() - p2.getY(), 2));
    double b = sqrt(pow(p2.getX() - p3.getX(), 2) + pow(p2.getY() - p3.getY(), 2));
    double c = sqrt(pow(p1.getX() - p3.getX(), 2) + pow(p1.getY() - p3.getY(), 2));
    double P = a+b+c;
    return P;
}

double Triangle::getArea() const {
    double a = sqrt(pow(p1.getX() - p2.getX(), 2) + pow(p1.getY() - p2.getY(), 2));
    double b = sqrt(pow(p2.getX() - p3.getX(), 2) + pow(p2.getY() - p3.getY(), 2));
    double c = sqrt(pow(p3.getX() - p1.getX(), 2) + pow(p3.getY() - p1.getY(), 2));
    double p = getPerLength()/2;
    double S = sqrt(p*(p-a)*(p-b)*(p-c));
    return S;

}

double Triangle::getSinus() const {
    double a = sqrt(pow(p1.getX() - p2.getX(), 2) + pow(p1.getY() - p2.getY(), 2));
    double b = sqrt(pow(p2.getX() - p3.getX(), 2) + pow(p2.getY() - p3.getY(), 2));
    double sin = getArea()/(a*b);
    return sin;
}

Point Triangle::getMass() const {
    double x = (p1.getX()+p2.getX()+p3.getX())/3;
    double y = (p1.getY()+p2.getY()+p3.getY())/3;

    return Point(x, y);
}

Point Triangle::getVertices(int i) const {
    if (i%3 == 1){
        return p1;
    }
    if (i%3 == 2){
        return p2;
    }
    if (i%3 == 0){
        return p3;
    }
}



