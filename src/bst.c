#include "bst.h"

t_btree* bstree_create_node(int item) {
    t_btree* node = malloc(sizeof(t_btree));
    if (node != NULL) {
        node->item = item;
        node->left = NULL;
        node->right = NULL;
    }
    return node;
}

void bstree_insert(t_btree* root, int item, int (*cmpf)(int, int)) {
    if (root == NULL) return;

    int cmp = 0;
    t_btree* cur = root;
    t_btree* prv = NULL;
    while (cur) {
        prv = cur;
        cmp = cmpf(item, cur->item);
        if (cmp < 0) {
            cur = cur->left;
        } else if (cmp > 0) {
            cur = cur->right;
        } else {
            return;
        }
    }

    cur = bstree_create_node(item);
    if (cmp < 0) {
        prv->left = cur;
    } else {
        prv->right = cur;
    }
}

void destroy(t_btree* node) {
    if (node == NULL) return;
    destroy(node->left);
    destroy(node->right);
    free(node);
}

void print_path(t_btree* node, int item) {
    if (node == NULL) {
        printf(" DIDN'T FIND %d", item);
        return;
    }
    if (node->item == item) return;

    if (item < node->item) {
        printf(" left");
        print_path(node->left, item);
    } else {
        printf(" right");
        print_path(node->right, item);
    }
}

void bstree_apply_infix(t_btree* root, void (*applyf)(int)) {
    if (root == NULL) return;
    bstree_apply_infix(root->left, applyf);
    applyf(root->item);
    bstree_apply_infix(root->right, applyf);
}

void bstree_apply_prefix(t_btree* root, void (*applyf)(int)) {
    if (root == NULL) return;
    applyf(root->item);
    bstree_apply_prefix(root->left, applyf);
    bstree_apply_prefix(root->right, applyf);
}

void bstree_apply_postfix(t_btree* root, void (*applyf)(int)) {
    if (root == NULL) return;
    bstree_apply_postfix(root->left, applyf);
    bstree_apply_postfix(root->right, applyf);
    applyf(root->item);
}