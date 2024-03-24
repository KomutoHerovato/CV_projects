#include "List.h"
#include <iostream>
using namespace std;

list::list() : first(nullptr), last(nullptr), p(nullptr) {};

bool list::is_empty() {
    if (first == nullptr)
    {
        return true;
    }
    return false;
}

void list::push(int data) {
    list_item* p_tmp = new list_item(data);
    if (is_empty()) {
        first = p_tmp;
        last = p_tmp;
        return;
    }
    p = last;
    last->next = p_tmp;
    last = p_tmp;
    last->previous = p;
}

void list::push_re(int data) {
    list_item* p_tmp = new list_item(data);
    if (is_empty()) {
        first = p_tmp;
        last = p_tmp;
        return;
    }
    p = first;
    first->previous = p_tmp;
    first = p_tmp;
    first->next = p;
}

void list::print() {
    list_item* p_tmp = first;
    while (p_tmp != nullptr) {
        std::cout << p_tmp->data << " ";
        p_tmp = p_tmp->next;
    }
    std::cout << std::endl;
}

void list::print_re() {
    list_item* p_tmp = last;
    while (p_tmp != nullptr) {
        std::cout << p_tmp->data << " ";
        p_tmp = p_tmp->previous;
    }
    std::cout << std::endl;
}

void list::delet() {
    p = last->previous;
    delete(last);
    last = p;
    last->next = nullptr;
}

void list::delet_re() {
    p = first->next;
    delete(first);
    first = p;
    first->previous = nullptr;
}

void list::push_p(int data) {
    list_item* p_tmp = new list_item(data);
    if (is_empty()) {
        first = p_tmp;
        last = p_tmp;
        return;
    }
    if (p == last) {
        push(data);
    }
    else if (p == first) {
        push_re(data);
    }
    else {
        p->next->previous = p_tmp;
        p->next->previous->next = p->next;
        p->next = p_tmp;
        p->next->previous = p;
    }
}

void list::move(int n) {
    if (n > 0) {
        for (int i = 0; i < n; i++) {
            if (p == last) {
                p = first;
            }
            else
            {
                p = p->next;
            }
        }
    }
    else {
        n = unsigned(n);
        for (int i = 0; i < n; i++) {
            if (p == first) {
                p = last;
            }
            else
            {
                p = p->previous;
            }
        }
    }
}

void list::delet_p() {
    if (first == last)
    {
        first = nullptr;
        last = nullptr;
        p = nullptr;
        return;
    }
    if (p == last) {
        delet();
    }
    else if (p == first) {
        delet_re();
    }
    else {
        p->previous->next = p->next;
        p = p->previous;
        delete(p->next->previous);
        p->next->previous = p;
    }

}

void list::partition(list l1, list l2) {
    l1.first = first;
    l1.last = p;
    l2.first = p->next;
    l2.last = last;
    l1.last->next = nullptr;
    l2.first->previous = nullptr;
    l1.print();
    l2.print();
}


list::~list() {
    list_item* p = first;
    while (p != last) {
        list_item* tmp = p;
        p = p->next;
        if (p->next != nullptr) {
            delete tmp;
        }
    }
}

list::list(const list &other) {
    first = nullptr;
    last = nullptr;
    list_item* p = other.first;
    while (p != nullptr) {
        list_item* newNode = new list_item(*p);
        if (first == nullptr) {
            first = newNode;
            last = newNode;
        } else {
            last->next = newNode;
            newNode->previous = last;
            last = newNode;
        }
        p = p->next;
    }
}

list::list(list &&other) noexcept {
    first = other.first;
    last = other.last;
    other.first = nullptr;
    other.last = nullptr;

}
