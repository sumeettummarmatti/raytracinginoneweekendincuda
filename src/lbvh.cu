#include "lbvh.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

// ── Morton code helpers ───────────────────────────────────────────────────────

__device__ uint32_t expandBits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ uint32_t morton3D(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    return (expandBits((uint32_t)x) << 2) |
           (expandBits((uint32_t)y) << 1) |
           (expandBits((uint32_t)z));
}

// ── Kernels ───────────────────────────────────────────────────────────────────

__global__ void k_computeMorton(sphere* spheres, uint32_t* codes,
                                 int* indices, AABB sceneBounds, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vec3 c   = spheres[i].center;
    vec3 ext = sceneBounds.mx - sceneBounds.mn;
    // protect against division by zero for a flat scene
    float ex = fmaxf(ext.x(), 1e-6f);
    float ey = fmaxf(ext.y(), 1e-6f);
    float ez = fmaxf(ext.z(), 1e-6f);
    vec3 norm((c.x() - sceneBounds.mn.x()) / ex,
              (c.y() - sceneBounds.mn.y()) / ey,
              (c.z() - sceneBounds.mn.z()) / ez);
    codes[i]   = morton3D(norm.x(), norm.y(), norm.z());
    indices[i] = i;
}

__device__ int deltaLeading(uint32_t* codes, int i, int j, int n) {
    if (j < 0 || j >= n) return -1;
    if (codes[i] == codes[j])
        return __clz(i ^ j) + 32;
    return __clz(codes[i] ^ codes[j]);
}

__global__ void k_buildInternal(uint32_t* codes, BVHNode* nodes,
                                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;

    int d = (deltaLeading(codes,i,i+1,n) - deltaLeading(codes,i,i-1,n)) >= 0 ? 1 : -1;
    int dMin = deltaLeading(codes, i, i - d, n);
    int lMax = 2;
    while (deltaLeading(codes, i, i + lMax * d, n) > dMin) lMax <<= 1;

    int l = 0;
    for (int t = lMax >> 1; t >= 1; t >>= 1)
        if (deltaLeading(codes, i, i + (l + t) * d, n) > dMin) l += t;

    int j = i + l * d;
    int lo = min(i, j), hi = max(i, j);

    int dNode = deltaLeading(codes, i, j, n);
    int s = 0;
    for (int t = (hi - lo + 1) >> 1; t >= 1; t >>= 1)
        if (deltaLeading(codes, lo, lo + s + t, n) > dNode) s += t;
    int split = lo + s;

    int leftIdx  = (split     == lo) ? (n - 1) + split     : split;
    int rightIdx = (split + 1 == hi) ? (n - 1) + split + 1 : split + 1;

    nodes[i].leftChild  = leftIdx;
    nodes[i].rightChild = rightIdx;
    nodes[i].isLeaf     = false;

    if (leftIdx  >= n - 1) nodes[leftIdx].parent  = i;
    if (rightIdx >= n - 1) nodes[rightIdx].parent = i;
}

__global__ void k_initLeaves(sphere* spheres, int* sortedIdx,
                              BVHNode* nodes, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int leafNode = (n - 1) + i;
    sphere& s = spheres[sortedIdx[i]];
    vec3 r(s.radius, s.radius, s.radius);
    nodes[leafNode].bounds   = AABB(s.center - r, s.center + r);
    nodes[leafNode].isLeaf   = true;
    nodes[leafNode].primIdx  = sortedIdx[i];
    nodes[leafNode].parent   = -1;
}

__global__ void k_fitBounds(BVHNode* nodes, int* flags, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int cur = (n - 1) + i;
    int par = nodes[cur].parent;
    while (par != -1) {
        if (atomicAdd(&flags[par], 1) == 0) return;
        __threadfence();
        nodes[par].bounds = merge(nodes[nodes[par].leftChild].bounds,
                                   nodes[nodes[par].rightChild].bounds);
        par = nodes[par].parent;
    }
}

// ── Host build ────────────────────────────────────────────────────────────────

LBVH buildLBVH(sphere* d_spheres, int n) {
    // 1. compute scene AABB on host (small n, fine to do serially)
    std::vector<sphere> h_spheres(n);
    cudaMemcpy(h_spheres.data(), d_spheres, n * sizeof(sphere),
               cudaMemcpyDeviceToHost);
    AABB sceneBounds;
    sceneBounds.mn = vec3(1e30f, 1e30f, 1e30f);
    sceneBounds.mx = vec3(-1e30f,-1e30f,-1e30f);
    for (auto& s : h_spheres)
        sceneBounds = merge(sceneBounds,
            AABB(s.center - vec3(s.radius,s.radius,s.radius),
                 s.center + vec3(s.radius,s.radius,s.radius)));

    // 2. Morton codes
    thrust::device_vector<uint32_t> d_codes(n);
    thrust::device_vector<int>      d_idx(n);
    k_computeMorton<<<(n+127)/128,128>>>(
        d_spheres, d_codes.data().get(), d_idx.data().get(), sceneBounds, n);

    // 3. sort
    thrust::sort_by_key(d_codes.begin(), d_codes.end(), d_idx.begin());

    // 4. allocate node array (2n-1 nodes)
    int totalNodes = 2 * n - 1;
    BVHNode* d_nodes;
    cudaMalloc(&d_nodes, totalNodes * sizeof(BVHNode));
    cudaMemset(d_nodes, 0, totalNodes * sizeof(BVHNode));

    int* d_sortedIdx;
    cudaMalloc(&d_sortedIdx, n * sizeof(int));
    cudaMemcpy(d_sortedIdx, d_idx.data().get(), n * sizeof(int),
               cudaMemcpyDeviceToDevice);

    // 5. init leaves
    k_initLeaves<<<(n+127)/128,128>>>(d_spheres, d_sortedIdx, d_nodes, n);

    // 6. build internal nodes
    if (n > 1) {
        k_buildInternal<<<(n+127)/128,128>>>(
            d_codes.data().get(), d_nodes, n);

        // 7. fit bounds bottom-up
        int* d_flags;
        cudaMalloc(&d_flags, totalNodes * sizeof(int));
        cudaMemset(d_flags, 0, totalNodes * sizeof(int));
        k_fitBounds<<<(n+127)/128,128>>>(d_nodes, d_flags, n);
        cudaFree(d_flags);
    }

    return LBVH{d_nodes, d_sortedIdx, n};
}

void freeLBVH(LBVH& bvh) {
    cudaFree(bvh.nodes);
    cudaFree(bvh.sortedIdx);
}

// ── Device traversal ─────────────────────────────────────────────────────────

__device__ bool traverseLBVH(const ray& r, const LBVH& bvh,
                              sphere* spheres,
                              float tMin, float tMax, hit_record& rec) {
    int stack[64];
    int top = 0;
    stack[top++] = 0;
    bool hit = false;
    
    if (bvh.numPrims == 0) return false;

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
