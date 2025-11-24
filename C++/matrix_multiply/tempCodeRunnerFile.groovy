#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>
#include <unistd.h>

// ============================================================================
// ЗАДАНИЕ 1: Потокобезопасный счетчик с максимальным пределом
// ============================================================================

typedef struct {
    int value;
    int max_value;
} SafeCounter;

void counter_init(SafeCounter* counter, int max_value) {
    counter->value = 0;
    counter->max_value = max_value;
}

bool counter_increment(SafeCounter* counter) {
    int old_val, new_val;
    bool success = false;
    
    #pragma omp atomic capture
    {
        old_val = counter->value;
        if (old_val < counter->max_value) {
            counter->value++;
            new_val = counter->value;
        } else {
            new_val = old_val;
        }
    }
    
    success = (new_val > old_val);
    return success;
}

bool counter_decrement(SafeCounter* counter) {
    int old_val, new_val;
    bool success = false;
    
    #pragma omp atomic capture
    {
        old_val = counter->value;
        if (old_val > 0) {
            counter->value--;
            new_val = counter->value;
        } else {
            new_val = old_val;
        }
    }
    
    success = (new_val < old_val);
    return success;
}

int counter_get(SafeCounter* counter) {
    int val;
    #pragma omp atomic read
    val = counter->value;
    return val;
}

void test_safe_counter() {
    printf("=== ТЕСТ 1: Потокобезопасный счетчик ===\n");
    
    SafeCounter counter;
    counter_init(&counter, 100);
    
    int success_count = 0;
    int fail_count = 0;
    
    #pragma omp parallel num_threads(8)
    {
        #pragma omp for reduction(+:success_count, fail_count)
        for (int i = 0; i < 150; i++) {
            if (counter_increment(&counter)) {
                success_count++;
            } else {
                fail_count++;
            }
        }
    }
    
    printf("Финальное значение счетчика: %d\n", counter_get(&counter));
    printf("Успешных инкрементов: %d\n", success_count);
    printf("Неудачных инкрементов: %d\n", fail_count);
    printf("Ожидаемое значение: 100 (достигнут предел)\n\n");
}