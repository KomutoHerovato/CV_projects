#include "square.h"
#include <cmath>

Square::Square(Point p1, Point p2) : p1(p1), p2(p2) {}

double Square::getPerLength() const {
    double per = (4*(sqrt(pow(p1.getX() - p2.getX(), 2) + pow(p1.getY() - p2.getY(), 2))))/sqrt(2);
    return per;
}

double Square::getArea() {
    double side = (getPerLength() / 4);
    double area = pow(side, 2);
    return area;
}

Point Square::getMass() {
    double x = (p1.getX() + p2.getX())/2;
    double y = (p1.getY() + p2.getY())/2;
    return Point(x, y);
}

Point Square::getVertices(int i) {
    if (i == 1){
        return p1;
    }
    if (i == 3){
        return p2;
    }
    if(i == 2){
        double x = (p1.getX()+p2.getX())/2;
        double y = (p1.getY()+p2.getY())/2;
        Point v (-y+p2.getY(),x-p2.getX());
        double C_x = x + v.getX();
        double C_y = y + v.getY();
        return Point(C_x, C_y);}
    if(i == 4){
        double x = (p1.getX()+p2.getX())/2;
        double y = (p1.getY()+p2.getY())/2;
        Point v (-y+p1.getY(),x-p1.getX());
        double C_x = x + v.getX();
        double C_y = y + v.getY();
        return Point(C_x, C_y);} else{
        return Point(0,0);
    }
}

