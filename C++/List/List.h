//
// Created by Дамир Мухаметзянов on 10/23/23.
//

#ifndef LIST_LIST_H
#define LIST_LIST_H
#include <iostream>
using namespace std;

struct list_item {
public:
    int data;
    list_item* next;
    list_item* previous;
    list_item* p;
    list_item(int _data);
};

class list {
public:
    list_item* first;
    list_item* p;
    list_item* last;
    list();

    list(const list& other);

    list(list&& other) noexcept;

    bool is_empty();

    void push(int data);

    void push_re(int data);

    void print();

    void print_re();

    void delet();

    void delet_re();

    void push_p(int data);

    void move(int n);

    void delet_p();

    void partition(list l1, list l2);

    ~list();

};


#endif //LIST_LIST_H
