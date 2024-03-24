#include "Figures.h"
#include <iostream>

int main() {
    Point p1(0, 0);
    Point p3(1, 1);
    Point p2(2, 0);
    Point p4(13,13);

    std::vector<Figures*> figure;
    figure.push_back(new Square(p1,p3));
    figure.push_back(new Circle(p2, 3));
    figure.push_back(new Triangle(p1,p2,p3));
    figure.push_back(new Quadrilateral(p1,p2,p3,p4));

    Square square(p1, p3);
    std::cout << "Square: " << std::endl;
    figure[0]= new Square(p1,p3);
    std::cout << "Per length: " << figure[0]->getPerLength() << std::endl;
    std::cout << "Area: " << figure[0]->getArea() << std::endl;
    std::cout << "Mass: " << figure[0]->getMass().getX() << ", " << figure[0]->getMass().getY() << std::endl;
    std::cout << "Vertices :";
    for (int i = 1; i < 5; ++i) {
        std::cout << i << ": "<< figure[0]->getVertices(i).getX() << ", " << figure[0]->getVertices(i).getY() << std::endl;
    }
    figure[0]->Turn(60);
    std::cout << "After rotation:" << std::endl;
    for (int i = 1; i < 5; ++i) {
        std::cout << i << ": "<< figure[0]->getVertices(i).getX() << ", " << figure[0]->getVertices(i).getY() << std::endl;
    }

    std::cout << std::endl;

    Circle circle(p2, 3);
    std::cout << "Circle: " << std::endl;
    figure[1] = new Circle(p2,3);

    std::cout << "Per length: " << figure[1]->getPerLength() << std::endl;
    std::cout << "Area: " << figure[1]->getArea() << std::endl;
    std::cout << "Mass: " << figure[1]->getMass().getX() << ", " << figure[1]->getMass().getY() << std::endl;
    std::cout << "Vertices: " << figure[1]->getVertices(1).getX() <<", " << figure[1]->getVertices(1).getY() << std::endl;
    std::cout << "After rotation:" << std::endl;
    circle.Turn(60);
    std::cout << "Vertices: " << figure[1]->getVertices(1).getX() <<", " << figure[1]->getVertices(1).getY() << std::endl;

    std::cout << std::endl;


    Triangle triangle(p1, p2, p3);
    std::cout << "Triangle: " << std::endl;
    figure[2] = new Triangle(p1,p2,p3);
    std::cout << "Per length: " << figure[2]->getPerLength() << std::endl;
    std::cout << "Area: " << figure[2]->getArea() << std::endl;
    std::cout << "Mass: " << figure[2]->getMass().getX() << ", " << figure[2]->getMass().getY() << std::endl;
    std::cout << "Vertices :";
    for (int i = 1; i < 4; ++i) {
        std::cout << i << ": "<< figure[2]->getVertices(i).getX() << ", " << figure[2]->getVertices(i).getY() << std::endl;
    }
    std::cout << "After rotation:" << std::endl;
    figure[2]->Turn(60);
    for (int i = 1; i < 4; ++i) {
        std::cout << i << ": "<< figure[2]->getVertices(i).getX() << ", " << figure[2]->getVertices(i).getY() << std::endl;
    }

    std::cout << std::endl;

    Quadrilateral quadrilateral(p1, p2, p3,p4);
    figure[3] = new Quadrilateral(p1,p2,p3,p4);
    std::cout << "Quadrilateral: " << std::endl;

    std::cout << "Per length: " << figure[3]->getPerLength() << std::endl;
    std::cout << "Area: " << figure[3]->getArea() << std::endl;
    std::cout << "Mass: " << figure[3]->getMass().getX() << ", " << figure[3]->getMass().getY() << std::endl;
    std::cout << "Vertices :";
    for (int i = 1; i < 5; ++i) {
        std::cout << i << ": "<< figure[3]->getVertices(i).getX() << ", " << figure[3]->getVertices(i).getY() << std::endl;
    }
    figure[3]->Turn(60);
    std::cout << "After rotation:" << std::endl;
    for (int i = 1; i < 5; ++i) {
        std::cout << i << ": "<< figure[3]->getVertices(i).getX() << ", " << figure[3]->getVertices(i).getY() << std::endl;
    }
    std::cout << "After move on (1,1):" << std::endl;
    figure[3]->Move(p3);
    for (int i = 1; i < 5; ++i) {
        std::cout << i << ": "<< figure[3]->getVertices(i).getX() << ", " << figure[3]->getVertices(i).getY() << std::endl;
    }
    std::cout << "After scale on 3 (triangele):" << std::endl;
    figure[2]->Scale(3);
    for (int i = 1; i < 4; ++i) {
        std::cout << i << ": "<< figure[2]->getVertices(i).getX() << ", " << figure[2]->getVertices(i).getY() << std::endl;
    }
    std::cout << "is dot (0.5, 0.5) in square p1 = 0,0 ; p2 = 1,1?"<< std::endl;
    std::cout << "(Yes/No == 1/0) : " << figure[2]->isDotInFigure(Point(0.5,0.5));
    std::cout << std::endl;




    for (int i = 0; i < figure.size(); ++i) {
        delete figure[i];
    }
    return 0;
}
