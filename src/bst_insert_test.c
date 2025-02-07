#include <stdio.h>

#include "bst.h"

#define INPUTS_COUNT 4
#define INPUTS_SIZE 5

int cmp(int a, int b);

int main() {
    int inputs[INPUTS_COUNT][INPUTS_SIZE] = {
        {32, 13432, 12, -54, 1}, {5, 4, 3, 2, 1}, {1, 2, 3, 4, 5}, {1, 1, 1, 1, 1}};

    for (int i = 0; i != INPUTS_COUNT; ++i) {
        printf("TEST %d\n", i + 1);
        t_btree* root = bstree_create_node(inputs[i][0]);
        printf("ROOT: %d\n", root->item);
        for (int j = 1; j != INPUTS_SIZE; ++j) {
            bstree_insert(root, inputs[i][j], cmp);
            printf("INSERTED %d. PATH:", inputs[i][j]);
            print_path(root, inputs[i][j]);
            printf("\n");
        }
        destroy(root);
        printf("\n");
    }

    return 0;
}

int cmp(int a, int b) { return (a > b) - (a < b); }