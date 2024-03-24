//
// Created by Дамир Мухаметзянов on 10/27/23.
//

#ifndef FIGURES_CIRCLE_H
#define FIGURES_CIRCLE_H
#include "dot.h"


class Circle {
private:
    Point O;
    int R;
public:
    Circle(Point O, int R);
    double getArea() const;
    double getPerLength() const;
    Point getMass();
    Point getVertices();
};


#endif //FIGURES_CIRCLE_H
