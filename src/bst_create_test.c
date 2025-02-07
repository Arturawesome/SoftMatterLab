#include <stdio.h>

#include "bst.h"

#define INPUTS_COUNT 5

int main() {
    int inputs[INPUTS_COUNT] = {32, 13432, 12, -54, 1};
    for (int i = 0; i != INPUTS_COUNT; ++i) {
        t_btree* node = bstree_create_node(inputs[i]);
        printf("input: %d\ntree node: %d\n", inputs[i], node->item);
        printf("result: %s", (inputs[i] == node->item ? "SUCCESS" : "FAIL"));
        if (i != INPUTS_COUNT - 1) printf("\n\n");
        free(node);
    }

    return 0;
}