//
// Created by Дамир Мухаметзянов on 11/22/23.
//
#pragma once

#include <iostream>
#include <functional>

template<typename T>
class MergeSort {
public:
    T* sort(T* arr, int len, bool (*comp)(T,T) = nullptr);
private:
    T* merge(T* arr1, int len1, T* arr2, int len2, bool (*comp)(T,T));
};

template<typename T>
T* MergeSort<T>::sort(T* arr, int len, bool (*comp)(T,T))
{
    if (len <= 1) {
        return arr;
    }

    int mid = len / 2;
    T* left = sort(arr, mid, comp);
    T* right = sort(arr + mid, len - mid, comp);

    return merge(left, mid, right, len - mid, comp);
}

template<typename T>
T* MergeSort<T>::merge(T* arr1, int len1, T* arr2, int len2, bool (*comp)(T,T))
{
    if (!comp) {
        comp = [](T a, T b) { return a < b; };
    }

    T* result = new T[len1 + len2];
    int i = 0, j = 0, k = 0;

    while (i < len1 && j < len2) {
        if (comp(arr1[i], arr2[j])) {
            result[k++] = arr1[i++];
        } else {
            result[k++] = arr2[j++];
        }
    }

    while (i < len1) {
        result[k++] = arr1[i++];
    }

    while (j < len2) {
        result[k++] = arr2[j++];
    }

    return result;
}

