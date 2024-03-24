#include "merge.h"

int main()
{
    int arr[] = {5, 2, 8, false, 1, 6, 9, 4, 7, '13'};
    int len = sizeof(arr) / sizeof(arr[0]);

    MergeSort<int> mergeSort;
    int* sortedArr = mergeSort.sort(arr, len);

    for (int i = 0; i < len; i++) {
        std::cout << sortedArr[i] << " ";
    }
    std::cout << std::endl;

    delete[] sortedArr;

    return 0;
}
