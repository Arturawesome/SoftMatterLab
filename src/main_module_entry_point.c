#include <stdio.h>

#ifdef PRINT_LOG
#include "print_module.h"
#endif

#ifdef DOCUMENTATION_MODULE
#include "documentation_module.h"
#endif

int main() {
#ifdef PRINT_LOG
    print_log(print_char, Module_load_success_message);
#endif

#ifdef DOCUMENTATION_MODULE
    int* availability_mask = check_available_documentation_module(validate, Documents_count, Documents);
    char* documents[] = {Documents};
    for (int i = 0; i != Documents_count; ++i) {
        printf("[%-15s : %s]", documents[i], (availability_mask[i] ? "available" : "unavailable"));
        if (i != Documents_count - 1) printf("\n");
    }
    free(availability_mask);
#endif

    return 0;
}
