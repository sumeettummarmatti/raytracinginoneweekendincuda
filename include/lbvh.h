#pragma once
#include <cuda_runtime.h>
#include "aabb.h"
#include "sphere.h"

struct BVHNode {
    AABB bounds;
    int  leftChild;   // index into nodes[]; >= numLeaves means leaf
    int  rightChild;
    int  parent;
    int  primIdx;     // valid only for leaves
    bool isLeaf;
};

struct LBVH {
    BVHNode* nodes;      // device pointer, size = 2*numPrims - 1
    int*     sortedIdx;  // device pointer, size = numPrims
    int      numPrims;
};

// host-callable build — allocates and returns a ready LBVH
LBVH buildLBVH(sphere* d_spheres, int n);
void freeLBVH(LBVH& bvh);

// Inline device traversal — compiled into every .cu that includes this header
// (avoids cross-TU device symbol linking issues with nvlink)
__device__ inline bool traverseLBVH(const ray& r, const LBVH& bvh,
                                     sphere* spheres,
                                     float tMin, float tMax,
                                     hit_record& rec) {
    if (bvh.numPrims == 0) return false;

    int stack[64];
    int top = 0;
    stack[top++] = 0;
    bool hit = false;

    while (top > 0) {
        int idx = stack[--top];
        BVHNode& node = bvh.nodes[idx];
        if (!node.bounds.hit(r, tMin, tMax)) continue;

        if (node.isLeaf) {
            hit_record tmp;
            if (spheres[node.primIdx].hit(r, tMin, tMax, tmp)) {
                tMax = tmp.t;
                rec  = tmp;
                hit  = true;
            }
        } else {
            stack[top++] = node.leftChild;
            stack[top++] = node.rightChild;
        }
    }
    return hit;
}
