//
// Created by Дамир Мухаметзянов on 10/27/23.
//
#include <cmath>
#include "Quadrilateral.h"
#include "Triangle.h"

Quadrilateral::Quadrilateral(Point p1, Point p2, Point p3, Point p4) : p1(p1), p2(p2), p3(p3), p4(p4) {

}

double Quadrilateral::getArea() const {
    double d1 = sqrt(pow(p1.getX() - p3.getX(), 2) + pow(p1.getY() - p3.getY(), 2));
    double d2 = sqrt(pow(p2.getX() - p4.getX(), 2) + pow(p2.getY() - p4.getY(), 2));
    Triangle t(p1, p2, p3);
    double s = d1*d2 * t.getSinus()/2;
    return s;
}

double Quadrilateral::getPerLength() const {
    double a = sqrt(pow(p1.getX() - p2.getX(), 2) + pow(p1.getY() - p2.getY(), 2));
    double b = sqrt(pow(p2.getX() - p3.getX(), 2) + pow(p2.getY() - p3.getY(), 2));
    double c = sqrt(pow(p3.getX() - p4.getX(), 2) + pow(p3.getY() - p4.getY(), 2));
    double d = sqrt(pow(p4.getX() - p1.getX(), 2) + pow(p4.getY() - p1.getY(), 2));
    double p = a+b+c+d;
    return p;

}

Point Quadrilateral::getMass() const {
    Triangle t1(p1,p2,p3);
    Triangle t2(p4,p2,p3);
    double C_x = (t1.getArea()*(t1.getMass().getX()) + t2.getArea()*(t1.getMass().getX()))/getArea();
    double C_y = (t1.getArea()*(t1.getMass().getY()) + t2.getArea()*(t1.getMass().getY()))/getArea();

    //double C_x = ((p1.getX()+p2.getX()+p3.getX())/3*((p1.getX()*p2.getX() + p1.getY()*p2.getY())*(p1.getX()*p3.getX() + p1.getY()*p3.getY()))+(p3.getX()+p4.getX()+p1.getX())/3*((p1.getX()*p3.getX() + p1.getY()*p3.getY())*(p1.getX()*p4.getX() + p1.getY()*p4.getY())))/((((p1.getX()*p2.getX() + p1.getY()*p2.getY()))*(p1.getX()*p3.getX() + p1.getY()*p3.getY()))+((p1.getX()*p3.getX() + p1.getY()*p3.getY())*(p1.getX()*p4.getX() + p1.getY()*p4.getY())));
    //double C_y = ((p1.getY()+p2.getY()+p3.getY())/3*((p1.getX()*p2.getX() + p1.getY()*p2.getY())*(p1.getX()*p3.getX() + p1.getY()*p3.getY()))+(p3.getY()+p4.getY()+p1.getY())/3*((p1.getX()*p3.getX() + p1.getY()*p3.getY())*(p1.getX()*p4.getX() + p1.getY()*p4.getY())))/((((p1.getX()*p2.getX() + p1.getY()*p2.getY()))*(p1.getX()*p3.getX() + p1.getY()*p3.getY()))+((p1.getX()*p3.getX() + p1.getY()*p3.getY())*(p1.getX()*p4.getX() + p1.getY()*p4.getY())));

    return Point(C_x, C_y);
}

Point Quadrilateral::getVertices(int i) {
        if (i%4 == 1){
            return p1;
        }
        if (i%4 == 2){
            return p2;
        }
        if (i%4 == 3){
            return p3;
        }
        if (i%4 == 0){
            return p4;
        }
}

