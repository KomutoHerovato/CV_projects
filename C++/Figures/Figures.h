//
// Created by Дамир Мухаметзянов on 11/2/23.
//

#ifndef FIGURES_FIGURES_H
#define FIGURES_FIGURES_H
#include <cmath>

class Point {
private:
    double x, y;

public:
    Point(double x, double y);
    double distanceToLine(Point p1, Point p2);
    bool dotLower(Point p1, Point p2) const;
    double getX() const;
    double getY() const;
};

class Figures {
public:
    virtual double calc() = 0;
    virtual double getPerLength() = 0;
    virtual void Turn(int a) = 0;
    virtual void Move(Point P) = 0;
    virtual void Scale(int a) = 0;
    virtual Point getMass() = 0;
    virtual Point getVertices(int i) = 0;
    virtual double getArea() = 0;
    virtual int isDotInFigure(Point P) = 0;
};

class Triangle : public Figures{
private:
    Point p1;
    Point p2;
    Point p3;
public:
    Triangle(Point p1, Point p2, Point p3);
    double calc(){return 0;}
    double getPerLength() {
        double a = sqrt(pow(p1.getX() - p2.getX(), 2) + pow(p1.getY() - p2.getY(), 2));
        double b = sqrt(pow(p2.getX() - p3.getX(), 2) + pow(p2.getY() - p3.getY(), 2));
        double c = sqrt(pow(p1.getX() - p3.getX(), 2) + pow(p1.getY() - p3.getY(), 2));
        double P = a+b+c;
        return P;
    };
    double getArea(){
        double a = sqrt(pow(p1.getX() - p2.getX(), 2) + pow(p1.getY() - p2.getY(), 2));
        double b = sqrt(pow(p2.getX() - p3.getX(), 2) + pow(p2.getY() - p3.getY(), 2));
        double c = sqrt(pow(p3.getX() - p1.getX(), 2) + pow(p3.getY() - p1.getY(), 2));
        double p = getPerLength()/2;
        double S = sqrt(p*(p-a)*(p-b)*(p-c));
        return S;
    };
    double getSinus();
    Point getMass() {
        double x = (p1.getX()+p2.getX()+p3.getX())/3;
        double y = (p1.getY()+p2.getY()+p3.getY())/3;

        return Point(x, y);
    };
    Point getVertices(int i) {
        if (i%3 == 1){
            return p1;
        }
        if (i%3 == 2){
            return p2;
        }
        if (i%3 == 0){
            return p3;
        }
    };
    int isDotInFigure(Point P){
        if (p3.dotLower(p1,p2) == P.dotLower(p1,p2)){
            if (p1.dotLower(p2,p3) == P.dotLower(p2,p3)){
                if (p2.dotLower(p1,p3) == P.dotLower(p1,p3)){
                    return 1;
                }
            }
        }else{
            return 0;
        }
    };
    void Scale(int a){
        Point P = this->getMass();
        Point p1x = Point(-P.getX()+p1.getX(),-P.getY()+p1.getY());
        Point p2x = Point(-P.getX()+p2.getX(),-P.getY()+p2.getY());
        Point p3x = Point(-P.getX()+p3.getX(),-P.getY()+p3.getY());
        p1 = Point(P.getX()+(p1x.getX()*a),P.getY()+(p1x.getY()*a));
        p2 = Point(P.getX()+(p2x.getX()*a),P.getY()+(p2x.getY()*a));
        p3 = Point(P.getX()+(p3x.getX()*a),P.getY()+(p3x.getY()*a));
    }
    void Move(Point a){
        p1 = Point(p1.getX()+a.getX(),p1.getY()+a.getY());
        p2 = Point(p2.getX()+a.getX(),p2.getY()+a.getY());
        p3 = Point(p3.getX()+a.getX(),p3.getY()+a.getY());
    }
    void Turn(int a){
        p1 = Point((p1.getX()*cos((a*M_PI) /180))-(p1.getY()*sin((a*M_PI) /180)), (p1.getX()*sin((a*M_PI) /180))+(p1.getY()*cos((a*M_PI) /180)));
        p2 = Point((p2.getX()*cos((a*M_PI) /180))-(p2.getY()*sin((a*M_PI) /180)), (p2.getX()*sin((a*M_PI) /180))+(p2.getY()*cos((a*M_PI) /180)));
        p3 = Point((p3.getX()*cos((a*M_PI) /180))-(p3.getY()*sin((a*M_PI) /180)), (p3.getX()*sin((a*M_PI) /180))+(p3.getY()*cos((a*M_PI) /180)));
    }
};

class Quadrilateral : public Figures{
private:
    Point p1;
    Point p2;
    Point p3;
    Point p4;
public:
    double calc(){return 0;}
    Quadrilateral(Point p1, Point p2,Point p3, Point p4);
    double getArea() {
        double d1 = sqrt(pow(p1.getX() - p3.getX(), 2) + pow(p1.getY() - p3.getY(), 2));
        double d2 = sqrt(pow(p2.getX() - p4.getX(), 2) + pow(p2.getY() - p4.getY(), 2));
        Triangle t (p1, p2, p3);
        double s = d1*d2 * t.getSinus()/2;
        return s;
    };
    double getPerLength() {
        double a = sqrt(pow(p1.getX() - p2.getX(), 2) + pow(p1.getY() - p2.getY(), 2));
        double b = sqrt(pow(p2.getX() - p3.getX(), 2) + pow(p2.getY() - p3.getY(), 2));
        double c = sqrt(pow(p3.getX() - p4.getX(), 2) + pow(p3.getY() - p4.getY(), 2));
        double d = sqrt(pow(p4.getX() - p1.getX(), 2) + pow(p4.getY() - p1.getY(), 2));
        double p = a+b+c+d;
        return p;
    }
    Point getMass() {
        Triangle t1(p1,p2,p3);
        Triangle t2(p4,p2,p3);
        double C_x = (t1.getArea()*(t1.getMass().getX()) + t2.getArea()*(t1.getMass().getX()))/getArea();
        double C_y = (t1.getArea()*(t1.getMass().getY()) + t2.getArea()*(t1.getMass().getY()))/getArea();

        //double C_x = ((p1.getX()+p2.getX()+p3.getX())/3*((p1.getX()*p2.getX() + p1.getY()*p2.getY())*(p1.getX()*p3.getX() + p1.getY()*p3.getY()))+(p3.getX()+p4.getX()+p1.getX())/3*((p1.getX()*p3.getX() + p1.getY()*p3.getY())*(p1.getX()*p4.getX() + p1.getY()*p4.getY())))/((((p1.getX()*p2.getX() + p1.getY()*p2.getY()))*(p1.getX()*p3.getX() + p1.getY()*p3.getY()))+((p1.getX()*p3.getX() + p1.getY()*p3.getY())*(p1.getX()*p4.getX() + p1.getY()*p4.getY())));
        //double C_y = ((p1.getY()+p2.getY()+p3.getY())/3*((p1.getX()*p2.getX() + p1.getY()*p2.getY())*(p1.getX()*p3.getX() + p1.getY()*p3.getY()))+(p3.getY()+p4.getY()+p1.getY())/3*((p1.getX()*p3.getX() + p1.getY()*p3.getY())*(p1.getX()*p4.getX() + p1.getY()*p4.getY())))/((((p1.getX()*p2.getX() + p1.getY()*p2.getY()))*(p1.getX()*p3.getX() + p1.getY()*p3.getY()))+((p1.getX()*p3.getX() + p1.getY()*p3.getY())*(p1.getX()*p4.getX() + p1.getY()*p4.getY())));

        return Point(C_x, C_y);
    };
    Point getVertices(int i){
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
    };
    int isDotInFigure(Point P){
        Triangle t1 = Triangle(p1, p2, p3);
        Triangle t2 = Triangle(p1, p2, p4);
        if (!t1.isDotInFigure(P) and !t2.isDotInFigure(P)){
            return 0;
        }else{
            return 1;
        }
    };

    void Scale(int a){
        Point P = this->getMass();
        Point p1x = Point(-P.getX()+p1.getX(),-P.getY()+p1.getY());
        Point p2x = Point(-P.getX()+p2.getX(),-P.getY()+p2.getY());
        Point p3x = Point(-P.getX()+p3.getX(),-P.getY()+p3.getY());
        Point p4x = Point(-P.getX()+p4.getX(),-P.getY()+p4.getY());
        p1 = Point(P.getX()+(p1x.getX()*a),P.getY()+(p1x.getY()*a));
        p2 = Point(P.getX()+(p2x.getX()*a),P.getY()+(p2x.getY()*a));
        p3 = Point(P.getX()+(p3x.getX()*a),P.getY()+(p3x.getY()*a));
        p4 = Point(P.getX()+(p4x.getX()*a),P.getY()+(p4x.getY()*a));
    }
    void Move(Point a){
        p1 = Point(p1.getX()+a.getX(),p1.getY()+a.getY());
        p2 = Point(p2.getX()+a.getX(),p2.getY()+a.getY());
        p3 = Point(p3.getX()+a.getX(),p3.getY()+a.getY());
        p4 = Point(p4.getX()+a.getX(),p4.getY()+a.getY());
    }
    void Turn(int a){
        p1 = Point((p1.getX()*cos((a*M_PI) /180))-(p1.getY()*sin((a*M_PI) /180)), (p1.getX()*sin((a*M_PI) /180))+(p1.getY()*cos((a*M_PI) /180)));
        p2 = Point((p2.getX()*cos((a*M_PI) /180))-(p2.getY()*sin((a*M_PI) /180)), (p2.getX()*sin((a*M_PI) /180))+(p2.getY()*cos((a*M_PI) /180)));
        p3 = Point((p3.getX()*cos((a*M_PI) /180))-(p3.getY()*sin((a*M_PI) /180)), (p3.getX()*sin((a*M_PI) /180))+(p3.getY()*cos((a*M_PI) /180)));
        p4 = Point((p4.getX()*cos((a*M_PI) /180))-(p4.getY()*sin((a*M_PI) /180)), (p4.getX()*sin((a*M_PI) /180))+(p4.getY()*cos((a*M_PI) /180)));
    };
};

class Circle : public Figures{
private:
    Point O;
    int R;
public:
    double calc(){return 0;}
    Circle(Point O, int R);
    double getArea() {
        double a = (M_PI*R*R);
        return a;
    };
    double getPerLength() {
        double a = (2*M_PI*R);
        return a;
    }
    Point getMass(){
        return O;
    };
    Point getVertices(int i){
        return O;
    };
    int isDotInFigure(Point P){
        if (sqrt(pow(O.getX()-P.getX(),2)+pow(O.getY()-P.getY(),2)) < R){
            return 1;
        }else{
            return 0;
        }
    };
    void Scale(int a){
        R = R * a;
    };
    void Move(Point a){
        O = Point(O.getX()+a.getX(),O.getY()+a.getY());
    };
    void Turn(int a){
        O = Point(((O.getX()*cos((a*M_PI) /180))-(O.getY()*sin((a*M_PI) /180))), ((O.getX()*sin((a*M_PI) /180))+(O.getY()*cos((a*M_PI) /180))));
    };
};



class Square : public Figures{
private:
    Point p1, p2;

public:
    Square(Point p1, Point p2);
    double calc(){return 0;}
    double getPerLength() {
        double per = (4*(sqrt(pow(p1.getX() - p2.getX(), 2) + pow(p1.getY() - p2.getY(), 2))))/sqrt(2);
        return per;
    };
    Point getMass(){
        double x = (p1.getX() + p2.getX())/2;
        double y = (p1.getY() + p2.getY())/2;
        return Point(x, y);
    };
    double getArea(){
        double side = (getPerLength() / 4);
        double area = pow(side, 2);
        return area;
    };
    Point getVertices(int i){
        if (i == 1){
            return p1;
        }
        if (i == 3){
            return p2;
        }
        if(i == 2){
            double x = (p1.getX()+p2.getX())/2;
            double y = (p1.getY()+p2.getY())/2;
            Point v (-y+p1.getY(),-x+p1.getX());
            double C_x = x + v.getX();
            double C_y = y - v.getY();
            return Point(C_x, C_y);}
        if(i == 4){
            double x = (p1.getX()+p2.getX())/2;
            double y = (p1.getY()+p2.getY())/2;
            Point v (-y+p1.getY(),-x+p1.getX());
            double C_x = x - v.getX();
            double C_y = y + v.getY();
            return Point(C_x, C_y);} else{
            return Point(0,0);
        }
    };
    int isDotInFigure(Point P){
        Triangle t1 = Triangle(p1, this->getVertices(2), p2);
        Triangle t2 = Triangle(p1, this->getVertices(4), p2);
        if (!t1.isDotInFigure(P) and !t2.isDotInFigure(P)){
            return 0;
        }else{
            return 1;
        }
    };
    void Scale(int a){
        Point P = this->getMass();
        Point p1x = Point(-P.getX()+p1.getX(),-P.getY()+p1.getY());
        Point p2x = Point(-P.getX()+p2.getX(),-P.getY()+p2.getY());
        p1 = Point(P.getX()+(p1x.getX()*a),P.getY()+(p1x.getY()*a));
        p2 = Point(P.getX()+(p2x.getX()*a),P.getY()+(p2x.getY()*a));
    }
    void Move(Point a){
        p1 = Point(p1.getX()+a.getX(),p1.getY()+a.getY());
        p2 = Point(p2.getX()+a.getX(),p2.getY()+a.getY());
    }
    void Turn(int a){
        p1 = Point((p1.getX()*cos((a*M_PI) /180))-(p1.getY()*sin((a*M_PI) /180)), (p1.getX()*sin((a*M_PI) /180))+(p1.getY()*cos((a*M_PI) /180)));
        p2 = Point((p2.getX()*cos((a*M_PI) /180))-(p2.getY()*sin((a*M_PI) /180)), (p2.getX()*sin((a*M_PI) /180))+(p2.getY()*cos((a*M_PI) /180)));
    };
};




#endif //FIGURES_FIGURES_H
