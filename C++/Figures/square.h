//
// Created by Дамир Мухаметзянов on 10/27/23.
//

#ifndef SQUARE_H
#define SQUARE_H

#include "dot.h"

class Square {
private:
    Point p1, p2;

public:
    Square(Point p1, Point p2);

    double getPerLength() const;
    Point getMass();
    double getArea();
    Point getVertices(int i);
};

#endif

