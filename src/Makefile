CC = gcc
CFLAGS = -c -Wall -Werror -Wextra

EXECUTABLE1 = ../build/Quest_1
SOURCES1 = main_module_entry_point.c print_module.c
OBJECTS1 = $(SOURCES1:.c=.o)

EXECUTABLE2 = ../build/Quest_2
SOURCES2 = main_module_entry_point.c documentation_module.c
OBJECTS2 = $(SOURCES2:.c=.o)

EXECUTABLE3 = ../build/Quest_3
SOURCES3 = bst_create_test.c bst.c
OBJECTS3 = $(SOURCES3:.c=.o)

EXECUTABLE4 = ../build/Quest_4
SOURCES4 = bst_insert_test.c bst.c
OBJECTS4 = $(SOURCES4:.c=.o)

EXECUTABLE5 = ../build/Quest_5
SOURCES5 = bst_traverse_test.c bst.c
OBJECTS5 = $(SOURCES5:.c=.o)


%.o: %.c
	$(CC) $(CFLAGS) $< -o $@


print_module: CFLAGS += -DPRINT_LOG
print_module: $(OBJECTS1)
	$(CC) $(OBJECTS1) -o $(EXECUTABLE1)

documentation_module: CFLAGS += -DDOCUMENTATION_MODULE
documentation_module: $(OBJECTS2)
	$(CC) $(OBJECTS2) -o $(EXECUTABLE2)

bst_create_test: $(OBJECTS3)
	$(CC) $(OBJECTS3) -o $(EXECUTABLE3)

bst_insert_test: $(OBJECTS4)
	$(CC) $(OBJECTS4) -o $(EXECUTABLE4)

bst_traverse_test: $(OBJECTS5)
	$(CC) $(OBJECTS5) -o $(EXECUTABLE5)

.PHONY: all
all: $(SOURCES1) $(SOURCES2) $(SOURCES3) $(SOURCES4) $(SOURCES5) $(EXECUTABLE1) $(EXECUTABLE2) $(EXECUTABLE3) $(EXECUTABLE4) $(EXECUTABLE5)

.PHONY: clean
clean:
	rm -f *.o $(EXECUTABLE1) $(EXECUTABLE2) $(EXECUTABLE3) $(EXECUTABLE4) $(EXECUTABLE5)