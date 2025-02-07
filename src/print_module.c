#include "print_module.h"

#include <stdio.h>
#include <time.h>

#define BUFF_SIZE 1024

int print_char(char ch) { return putchar(ch); }

void print_log(int (*print)(char), char* message) {
    char buffer[BUFF_SIZE];
    time_t secs = time(NULL);
    struct tm* local = localtime(&secs);
    sprintf(buffer, "%s %02d:%02d:%02d %s", Log_prefix, local->tm_hour, local->tm_min, local->tm_sec,
            message);

    char* ptr = buffer;
    while (*ptr) {
        print(*ptr++);
    }
}