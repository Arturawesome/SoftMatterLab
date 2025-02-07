#include <stdio.h>

#include "bst.h"

#define INPUTS_COUNT 5
#define INPUTS_SIZE 5

int cmp(int a, int b);
void print(int a);

int main() {
    int inputs[INPUTS_COUNT][INPUTS_SIZE] = {{32, 13432, 12, -54, 1},
                                             {5, 4, 3, 2, 1},
                                             {1, 2, 3, 4, 5},
                                             {1, 1, 1, 1, 1},
                                             {432, 143, 5345, 14, 563}};

    for (int i = 0; i != INPUTS_COUNT; ++i) {
        printf("TEST %d\n", i + 1);
        t_btree* root = bstree_create_node(inputs[i][0]);
        printf("ROOT: %d\n", root->item);
        for (int j = 1; j != INPUTS_SIZE; ++j) {
            bstree_insert(root, inputs[i][j], cmp);
        }
        printf("INFIX: ");
        bstree_apply_infix(root, print);
        printf("\nPOSTFIX: ");
        bstree_apply_postfix(root, print);
        printf("\nPREFIX: ");
        bstree_apply_prefix(root, print);
        destroy(root);
        printf("\n\n");
    }

    return 0;
}

int cmp(int a, int b) { return (a > b) - (a < b); }

void print(int a) { printf(" %d", a); }