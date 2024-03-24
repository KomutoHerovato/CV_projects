#include "Figures.h"
#include <cmath>


Point::Point(double x, double y) : x(x), y(y) {}

double Point::getX() const {
    return x;
}

double Point::getY() const {
    return y;
}

bool Point::dotLower(Point p1, Point p2) const {
    double k = (p2.getY()-p1.getY()) / (p2.getX()-p1.getX());
    double b = p2.getY() - k*p2.getX();
    if(this->getY() < k*this->getX() + b){
        return true;
    }
    return false;
}

double Point::distanceToLine(Point p1, Point p2) {
    int k = (p2.getY()-p1.getY()) / (p2.getX()-p1.getX());
    int b = p2.getY() - k*p2.getX();
    return (k * this->getX() + getY() +b)/(sqrt(pow(k,2)+1));
}

Triangle::Triangle(Point p1, Point p2, Point p3): p1(p1), p2(p2), p3(p3) {

}



double Triangle::getSinus()  {
    double a = sqrt(pow(p1.getX() - p2.getX(), 2) + pow(p1.getY() - p2.getY(), 2));
    double b = sqrt(pow(p2.getX() - p3.getX(), 2) + pow(p2.getY() - p3.getY(), 2));
    double sin = getArea()/(a*b);
    return sin;
}




Quadrilateral::Quadrilateral(Point p1, Point p2, Point p3, Point p4) : p1(p1), p2(p2), p3(p3), p4(p4) {

}



Circle::Circle(Point O, int R) : O(O){
    this->R = R;
}







Square::Square(Point p1, Point p2) : p1(p1), p2(p2) {}






