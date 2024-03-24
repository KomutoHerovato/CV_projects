#include <iostream>
using namespace std;
#include "List.h"

int main()
{
    list l;
    list l1;
    list l2;
    cout << "Лист пустой" << endl;
    cout << l.is_empty() << std::endl;
    cout << "Добавляем элементы в список" << endl;
    l.push(28);
    l.push(37);
    l.push(21);
    l.push(4);
    l.print();
    cout << "Добовляем элемент в начале и наоборот" << endl;
    l.push_re(17);
    l.print_re();
    cout << "Удаляем последний элемент" << endl;
    l.delet();
    l.print();
    cout << "Удаляем первый элемент" << endl;
    l.delet_re();
    l.print();
    cout << "Пустой ли список и указатель" << endl;
    cout << l.is_empty() << " // " << l.first << endl << endl;

    cout << "3я неделя" << endl;
    l.print();
    l.push_p(33);
    l.print();
    cout << "Передвижение на 1 и добавление после указанного" << endl;
    l.move(1);
    l.push_p(22);
    l.print();
    cout << "Удаление выдранного (предыдушего)" << endl;
    l.delet_p();
    l.print();
    l.push_p(333);
    l.print();
    cout << endl << endl;

    cout << "4я неделя" << endl;
    l.partition(l1, l2);
    cout << endl << endl;

    cout << "6я неделя и 8ая неделя" << endl;
    list l_copy = l;
    cout << "Конструктор копирования: ";
    l_copy.print();
    cout << "Конструктор перемещения: ";
    list l_move = std::move(l_copy);
    l_move.print();
    cout << "Конструктор перемещения без std::move : ";
    list l_move2 = l_move;
    l_move2.print();

    return 0;
}