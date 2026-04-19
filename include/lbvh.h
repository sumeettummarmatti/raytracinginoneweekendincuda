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

__device__ bool traverseLBVH(const ray& r, const LBVH& bvh,
                              sphere* spheres,
                              float tMin, float tMax,
                              hit_record& rec);
