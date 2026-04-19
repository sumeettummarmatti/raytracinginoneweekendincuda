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
    if (isnan(x) || isnan(y) || isnan(z)) return 0;
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    return (expandBits((uint32_t)x) << 2) |
           (expandBits((uint32_t)y) << 1) |
           (expandBits((uint32_t)z));
}

// ── Kernels ───────────────────────────────────────────────────────────────────

__global__ void k_initNodes(BVHNode* nodes, int totalNodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= totalNodes) return;
    nodes[i].parent     = -1;
    nodes[i].leftChild  = -1;
    nodes[i].rightChild = -1;
    nodes[i].isLeaf     = false;
    nodes[i].primIdx    = -1;
    // init bounds to invalid so we can detect unfilled nodes
    nodes[i].bounds.mn = vec3( 1e30f,  1e30f,  1e30f);
    nodes[i].bounds.mx = vec3(-1e30f, -1e30f, -1e30f);
}

__global__ void k_computeMorton(sphere* spheres, uint32_t* codes,
                                 int* indices, AABB sceneBounds, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vec3 c   = spheres[i].center;
    vec3 ext = sceneBounds.mx - sceneBounds.mn;
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

__global__ void k_buildInternal(uint32_t* codes, BVHNode* nodes, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;

    int d = (deltaLeading(codes,i,i+1,n) - deltaLeading(codes,i,i-1,n)) >= 0 ? 1 : -1;
    int dMin = deltaLeading(codes, i, i - d, n);
    int lMax = 2;
    while (deltaLeading(codes, i, i + lMax * d, n) > dMin) lMax <<= 1;

    int l = 0;
    for (int t = lMax >> 1; t >= 1; t >>= 1)
        if (deltaLeading(codes, i, i + (l + t) * d, n) > dMin) l += t;

    int j    = i + l * d;
    int lo   = min(i, j), hi = max(i, j);

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

    nodes[leftIdx].parent  = i;
    nodes[rightIdx].parent = i;
}

__global__ void k_initLeaves(sphere* spheres, int* sortedIdx,
                              BVHNode* nodes, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int leafNode = (n - 1) + i;
    sphere& s = spheres[sortedIdx[i]];
    vec3 r(s.radius, s.radius, s.radius);
    nodes[leafNode].bounds  = AABB(s.center - r, s.center + r);
    nodes[leafNode].isLeaf  = true;
    nodes[leafNode].primIdx = sortedIdx[i];
}

// FIX: The old k_fitBounds never wrote the root (node 0) because it
// exits when par == -1 BEFORE merging. The correct pattern:
// - each leaf thread walks up
// - at each internal node, the SECOND thread to arrive does the merge
//   and continues upward
// - the root (parent == -1) is reached by the very last surviving thread,
//   which merges its two children and writes node 0's bounds.
__global__ void k_fitBounds(BVHNode* nodes, int* flags, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // start at this leaf
    int cur = (n - 1) + i;

    while (true) {
        int par = nodes[cur].parent;

        // cur is the root (no parent) — we are done
        if (par == -1) break;

        // Race to be the second thread at this parent.
        // First thread (old==0) exits — its sibling hasn't computed bounds yet.
        // Second thread (old==1) proceeds to merge both children.
        int old = atomicAdd(&flags[par], 1);
        if (old == 0) return;

        // Both children's bounds are now guaranteed written (first thread wrote
        // its child before calling atomicAdd, and __threadfence makes it visible).
        __threadfence();

        nodes[par].bounds = merge(nodes[nodes[par].leftChild].bounds,
                                   nodes[nodes[par].rightChild].bounds);
        __threadfence();  // make parent bounds visible before moving up

        cur = par;
    }

    // If we reach here, cur == 0 (root). Its bounds were just written above
    // (or cur started as 0 for the n==1 case handled separately).
}

// For n==1: the single sphere IS the root. Leaf init covers it.
// For n>1: the root is internal node 0; k_fitBounds above handles it.
// Safety net: one thread re-merges root from its children after k_fitBounds.
__global__ void k_fixupRoot(BVHNode* nodes, int n) {
    // only one thread needed
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (n <= 1) return;
    nodes[0].bounds = merge(nodes[nodes[0].leftChild].bounds,
                             nodes[nodes[0].rightChild].bounds);
}

// ── Host build ────────────────────────────────────────────────────────────────

LBVH buildLBVH(sphere* d_spheres, int n) {
    // 1. scene AABB on host
    std::vector<sphere> h_spheres(n);
    cudaMemcpy(h_spheres.data(), d_spheres, n * sizeof(sphere),
               cudaMemcpyDeviceToHost);

    AABB sceneBounds;
    sceneBounds.mn = vec3( 1e30f,  1e30f,  1e30f);
    sceneBounds.mx = vec3(-1e30f, -1e30f, -1e30f);
    for (auto& s : h_spheres)
        sceneBounds = merge(sceneBounds,
            AABB(s.center - vec3(s.radius, s.radius, s.radius),
                 s.center + vec3(s.radius, s.radius, s.radius)));

    std::cerr << "[LBVH] scene bounds min=("
              << sceneBounds.mn.x() << "," << sceneBounds.mn.y() << "," << sceneBounds.mn.z()
              << ") max=("
              << sceneBounds.mx.x() << "," << sceneBounds.mx.y() << "," << sceneBounds.mx.z()
              << ") n=" << n << "\n";

    // 2. Morton codes
    thrust::device_vector<uint32_t> d_codes(n);
    thrust::device_vector<int>      d_idx(n);
    k_computeMorton<<<(n+127)/128, 128>>>(
        d_spheres, d_codes.data().get(), d_idx.data().get(), sceneBounds, n);
    cudaDeviceSynchronize();

    // 3. sort by Morton code
    thrust::sort_by_key(d_codes.begin(), d_codes.end(), d_idx.begin());

    // 4. allocate & zero-init node array
    int totalNodes = 2 * n - 1;
    BVHNode* d_nodes;
    cudaMalloc(&d_nodes, totalNodes * sizeof(BVHNode));
    k_initNodes<<<(totalNodes+127)/128, 128>>>(d_nodes, totalNodes);
    cudaDeviceSynchronize();

    int* d_sortedIdx;
    cudaMalloc(&d_sortedIdx, n * sizeof(int));
    cudaMemcpy(d_sortedIdx, d_idx.data().get(), n * sizeof(int),
               cudaMemcpyDeviceToDevice);

    // 5. init leaves (sets bounds, isLeaf, primIdx)
    k_initLeaves<<<(n+127)/128, 128>>>(d_spheres, d_sortedIdx, d_nodes, n);
    cudaDeviceSynchronize();

    if (n == 1) {
        // Single primitive: root IS the leaf, already set up. Done.
        std::cerr << "[LBVH] single-prim tree, skipping internal build\n";
        return LBVH{d_nodes, d_sortedIdx, n};
    }

    // 6. build internal nodes (sets leftChild, rightChild, parent)
    k_buildInternal<<<(n+127)/128, 128>>>(d_codes.data().get(), d_nodes, n);
    cudaDeviceSynchronize();

    // 7. fit AABBs bottom-up
    int* d_flags;
    cudaMalloc(&d_flags, totalNodes * sizeof(int));
    cudaMemset(d_flags, 0, totalNodes * sizeof(int));
    k_fitBounds<<<(n+127)/128, 128>>>(d_nodes, d_flags, n);
    cudaDeviceSynchronize();

    // 8. safety fixup: re-merge root from its direct children
    //    (handles any edge case where the last surviving thread exited early)
    k_fixupRoot<<<1, 1>>>(d_nodes, n);
    cudaDeviceSynchronize();
    cudaFree(d_flags);

    // verification
    BVHNode h_root;
    cudaMemcpy(&h_root, d_nodes, sizeof(BVHNode), cudaMemcpyDeviceToHost);
    std::cerr << "[LBVH] root bounds min=("
              << h_root.bounds.mn.x() << "," << h_root.bounds.mn.y() << "," << h_root.bounds.mn.z()
              << ") max=("
              << h_root.bounds.mx.x() << "," << h_root.bounds.mx.y() << "," << h_root.bounds.mx.z()
              << ") isLeaf=" << h_root.isLeaf
              << " left=" << h_root.leftChild
              << " right=" << h_root.rightChild << "\n";

    return LBVH{d_nodes, d_sortedIdx, n};
}

void freeLBVH(LBVH& bvh) {
    cudaFree(bvh.nodes);
    cudaFree(bvh.sortedIdx);
}
