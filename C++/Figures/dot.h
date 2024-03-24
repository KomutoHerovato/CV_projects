#ifndef POINT_H
#define POINT_H

class Point {
private:
    double x, y;

public:
    Point(double x, double y);

    double getX() const;
    double getY() const;
};

#endif

