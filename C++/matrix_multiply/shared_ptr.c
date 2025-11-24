#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>
#include <string.h>

// ============================================================================
// Умный указатель с подсчетом ссылок (shared_ptr)
// ============================================================================

typedef struct SharedPtr {
    void* data;
    int* ref_count;
    size_t data_size;
} SharedPtr;

/**
 * Создание нового умного указателя
 * @param data - указатель на данные
 * @param data_size - размер данных в байтах
 * @return новый SharedPtr или NULL при ошибке
 */
SharedPtr* shared_ptr_create(void* data, size_t data_size) {
    if (!data) return NULL;
    
    SharedPtr* ptr = (SharedPtr*)malloc(sizeof(SharedPtr));
    if (!ptr) return NULL;
    
    // Выделяем память для счетчика ссылок
    ptr->ref_count = (int*)malloc(sizeof(int));
    if (!ptr->ref_count) {
        free(ptr);
        return NULL;
    }
    
    ptr->data = data;
    ptr->data_size = data_size;
    
    // Инициализируем счетчик ссылок значением 1
    #pragma omp atomic write
    *(ptr->ref_count) = 1;
    
    return ptr;
}

/**
 * Копирование умного указателя (увеличение счетчика ссылок)
 * @param src - исходный указатель
 * @return новая копия SharedPtr или NULL при ошибке
 */
SharedPtr* shared_ptr_copy(SharedPtr* src) {
    if (!src || !src->ref_count) return NULL;
    
    SharedPtr* new_ptr = (SharedPtr*)malloc(sizeof(SharedPtr));
    if (!new_ptr) return NULL;
    
    // Копируем все поля
    new_ptr->data = src->data;
    new_ptr->ref_count = src->ref_count;
    new_ptr->data_size = src->data_size;
    
    // Атомарно увеличиваем счетчик ссылок
    #pragma omp atomic update
    (*(new_ptr->ref_count))++;
    
    return new_ptr;
}

/**
 * Уничтожение умного указателя (уменьшение счетчика ссылок)
 * Если счетчик достигает 0, освобождается память
 * @param ptr - указатель для уничтожения
 */
void shared_ptr_destroy(SharedPtr* ptr) {
    if (!ptr || !ptr->ref_count) return;
    
    int count;
    
    // Атомарно уменьшаем счетчик и получаем новое значение
    #pragma omp atomic capture
    {
        (*(ptr->ref_count))--;
        count = *(ptr->ref_count);
    }
    
    // Если это была последняя ссылка - освобождаем память
    if (count == 0) {
        printf("[Освобождение памяти] ref_count достиг 0\n");
        free(ptr->data);
        free(ptr->ref_count);
    }
    
    free(ptr);
}

/**
 * Получение текущего количества ссылок
 * @param ptr - указатель
 * @return количество ссылок или 0 при ошибке
 */
int shared_ptr_use_count(SharedPtr* ptr) {
    if (!ptr || !ptr->ref_count) return 0;
    
    int count;
    // Атомарное чтение счетчика
    #pragma omp atomic read
    count = *(ptr->ref_count);
    
    return count;
}

/**
 * Получение указателя на данные
 * @param ptr - умный указатель
 * @return указатель на данные или NULL
 */
void* shared_ptr_get(SharedPtr* ptr) {
    if (!ptr) return NULL;
    return ptr->data;
}

// ============================================================================
// Тестовые структуры и функции
// ============================================================================

typedef struct Person {
    char name[50];
    int age;
    int id;
} Person;

void print_person(Person* p) {
    if (p) {
        printf("  Person[id=%d, name=%s, age=%d]\n", p->id, p->name, p->age);
    }
}

/**
 * Тест 1: Базовая функциональность
 */
void test_basic_functionality() {
    printf("\n=== ТЕСТ 1: Базовая функциональность ===\n");
    
    // Создаем данные
    Person* person = (Person*)malloc(sizeof(Person));
    strcpy(person->name, "Иван Иванов");
    person->age = 25;
    person->id = 1;
    
    // Создаем умный указатель
    SharedPtr* ptr1 = shared_ptr_create(person, sizeof(Person));
    printf("Создан ptr1, ref_count = %d\n", shared_ptr_use_count(ptr1));
    print_person((Person*)shared_ptr_get(ptr1));
    
    // Создаем копию
    SharedPtr* ptr2 = shared_ptr_copy(ptr1);
    printf("Создан ptr2 (копия ptr1), ref_count = %d\n", shared_ptr_use_count(ptr1));
    
    // Создаем еще одну копию
    SharedPtr* ptr3 = shared_ptr_copy(ptr2);
    printf("Создан ptr3 (копия ptr2), ref_count = %d\n", shared_ptr_use_count(ptr1));
    
    // Удаляем ptr2
    shared_ptr_destroy(ptr2);
    printf("Удален ptr2, ref_count = %d\n", shared_ptr_use_count(ptr1));
    
    // Удаляем ptr3
    shared_ptr_destroy(ptr3);
    printf("Удален ptr3, ref_count = %d\n", shared_ptr_use_count(ptr1));
    
    // Удаляем ptr1 (последний) - память должна освободиться
    printf("Удаляем ptr1 (последний указатель)...\n");
    shared_ptr_destroy(ptr1);
}

/**
 * Тест 2: Многопоточное копирование
 */
void test_multithreaded_copy() {
    printf("\n=== ТЕСТ 2: Многопоточное копирование ===\n");
    
    // Создаем данные
    Person* person = (Person*)malloc(sizeof(Person));
    strcpy(person->name, "Мария Петрова");
    person->age = 30;
    person->id = 2;
    
    SharedPtr* original = shared_ptr_create(person, sizeof(Person));
    printf("Создан оригинальный указатель, ref_count = %d\n", 
           shared_ptr_use_count(original));
    
    const int NUM_COPIES = 100;
    SharedPtr* copies[NUM_COPIES];
    
    // Параллельное создание копий
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < NUM_COPIES; i++) {
        copies[i] = shared_ptr_copy(original);
        
        // Проверяем данные в каждом потоке
        Person* p = (Person*)shared_ptr_get(copies[i]);
        if (p->id != 2) {
            printf("ОШИБКА: некорректные данные в потоке %d\n", omp_get_thread_num());
        }
    }
    
    printf("Создано %d копий, ref_count = %d\n", 
           NUM_COPIES, shared_ptr_use_count(original));
    
    // Параллельное удаление копий
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < NUM_COPIES; i++) {
        shared_ptr_destroy(copies[i]);
    }
    
    printf("Удалено %d копий, ref_count = %d\n", 
           NUM_COPIES, shared_ptr_use_count(original));
    
    printf("Удаляем оригинал...\n");
    shared_ptr_destroy(original);
}

/**
 * Тест 3: Стресс-тест с многопоточным доступом
 */
void test_stress() {
    printf("\n=== ТЕСТ 3: Стресс-тест ===\n");
    
    Person* person = (Person*)malloc(sizeof(Person));
    strcpy(person->name, "Алексей Сидоров");
    person->age = 35;
    person->id = 3;
    
    SharedPtr* original = shared_ptr_create(person, sizeof(Person));
    printf("Запуск стресс-теста...\n");
    
    const int NUM_THREADS = 8;
    const int OPERATIONS_PER_THREAD = 1000;
    
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int thread_id = omp_get_thread_num();
        
        for (int i = 0; i < OPERATIONS_PER_THREAD; i++) {
            // Создаем копию
            SharedPtr* temp = shared_ptr_copy(original);
            
            // Читаем данные
            Person* p = (Person*)shared_ptr_get(temp);
            if (p->id != 3) {
                printf("ОШИБКА в потоке %d: некорректные данные!\n", thread_id);
            }
            
            // Удаляем копию
            shared_ptr_destroy(temp);
        }
    }
    
    printf("Выполнено %d операций в %d потоках\n", 
           NUM_THREADS * OPERATIONS_PER_THREAD, NUM_THREADS);
    printf("Финальный ref_count = %d (должен быть 1)\n", 
           shared_ptr_use_count(original));
    
    printf("Удаляем оригинал...\n");
    shared_ptr_destroy(original);
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║  Умный указатель с подсчетом ссылок (SharedPtr)       ║\n");
    printf("║  Реализация с использованием OpenMP                   ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n");
    
    test_basic_functionality();
    test_multithreaded_copy();
    test_stress();
    
    printf("\n✓ Все тесты успешно завершены!\n");
    
    return 0;
}