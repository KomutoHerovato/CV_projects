//
// Created by Дамир Мухаметзянов on 10/27/23.
//

#ifndef FIGURES_QUADRILATERAL_H
#define FIGURES_QUADRILATERAL_H
#include "dot.h"


class Quadrilateral {
private:
    Point p1;
    Point p2;
    Point p3;
    Point p4;
public:
    Quadrilateral(Point p1, Point p2,Point p3, Point p4);
    double getArea() const;
    double getPerLength() const;
    Point getMass() const;
    Point getVertices(int i);

};


#endif //FIGURES_QUADRILATERAL_H
