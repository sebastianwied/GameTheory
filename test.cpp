#include <omp.h>
#include <iostream>

int main() {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::cout << "Hello from thread " << tid << "\n";
    }
    return 0;
}
