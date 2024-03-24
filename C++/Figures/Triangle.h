//
// Created by Дамир Мухаметзянов on 10/27/23.
//

#ifndef FIGURES_TRIANGLE_H
#define FIGURES_TRIANGLE_H
#include "dot.h"


class Triangle {
private:
    Point p1;
    Point p2;
    Point p3;
public:
    Triangle(Point p1, Point p2, Point p3);
    double getPerLength() const;
    double getArea() const;
    double getSinus() const;
    Point getMass() const;
    Point getVertices(int i) const;
};


#endif //FIGURES_TRIANGLE_H
