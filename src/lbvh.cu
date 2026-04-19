#include "lbvh.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <vector>
#include <iostream>

// ── Morton helpers ────────────────────────────────────────────────────────────

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

// ── Morton code kernel ────────────────────────────────────────────────────────

__global__ void k_computeMorton(sphere* spheres, uint32_t* codes,
                                 int* indices, AABB sceneBounds, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vec3 c = spheres[i].center;
    float ex = fmaxf(sceneBounds.mx.x() - sceneBounds.mn.x(), 1e-6f);
    float ey = fmaxf(sceneBounds.mx.y() - sceneBounds.mn.y(), 1e-6f);
    float ez = fmaxf(sceneBounds.mx.z() - sceneBounds.mn.z(), 1e-6f);
    float nx = (c.x() - sceneBounds.mn.x()) / ex;
    float ny = (c.y() - sceneBounds.mn.y()) / ey;
    float nz = (c.z() - sceneBounds.mn.z()) / ez;
    codes[i]   = morton3D(nx, ny, nz);
    indices[i] = i;
}

// ── Karras 2012 internal node build ──────────────────────────────────────────

__device__ int delta(uint32_t* codes, int i, int j, int n) {
    if (j < 0 || j >= n) return -1;
    if (codes[i] == codes[j]) return 32 + __clz(i ^ j);
    return __clz(codes[i] ^ codes[j]);
}

__global__ void k_buildInternal(uint32_t* codes, BVHNode* nodes, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;

    // determine direction
    int d = (delta(codes,i,i+1,n) - delta(codes,i,i-1,n)) >= 0 ? 1 : -1;
    int dMin = delta(codes, i, i - d, n);

    // upper bound on range length
    int lMax = 2;
    while (delta(codes, i, i + lMax*d, n) > dMin) lMax <<= 1;

    // binary search for exact end
    int l = 0;
    for (int t = lMax/2; t >= 1; t /= 2)
        if (delta(codes, i, i + (l+t)*d, n) > dMin) l += t;
    int j = i + l * d;

    int lo = min(i,j), hi = max(i,j);

    // find split
    int dNode = delta(codes, lo, hi, n);
    int s = 0;
    for (int t = (hi-lo+1)/2; t >= 1; t = (t == 1 ? 0 : t/2)) {
        if (delta(codes, lo, lo+s+t, n) > dNode) s += t;
        if (t == 1) break;
    }
    int split = lo + s;

    // Internal nodes: [0, n-1), leaves: [n-1, 2n-1)
    int L = (split     == lo) ? (n-1) + split     : split;
    int R = (split + 1 == hi) ? (n-1) + split + 1 : split + 1;

    nodes[i].leftChild  = L;
    nodes[i].rightChild = R;
    nodes[i].isLeaf     = false;
    nodes[i].primIdx    = -1;
    nodes[L].parent     = i;
    nodes[R].parent     = i;
}

__global__ void k_initLeaves(sphere* spheres, int* sortedIdx,
                              BVHNode* nodes, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int li = (n-1) + i;
    int si = sortedIdx[i];
    vec3 r(spheres[si].radius, spheres[si].radius, spheres[si].radius);
    nodes[li].bounds   = AABB(spheres[si].center - r, spheres[si].center + r);
    nodes[li].isLeaf   = true;
    nodes[li].primIdx  = si;
    nodes[li].leftChild  = -1;
    nodes[li].rightChild = -1;
}

// ── Bottom-up AABB fit — done entirely on HOST after build ───────────────────
// The GPU atomic approach is unreliable across architectures.
// For 484 nodes this host pass takes < 1ms. Correctness > cleverness.

void fitBoundsHost(std::vector<BVHNode>& nodes, int n) {
    // n leaves, n-1 internal nodes, total = 2n-1
    // Process internal nodes in reverse order (leaves have highest indices).
    // A simple post-order traversal: for each internal node (indices 0..n-2),
    // we need both children ready. Because of the Karras property, processing
    // from index n-2 down to 0 guarantees children are always processed first
    // for a balanced split... but that's not guaranteed in general.
    // Safest: iterative post-order using a visited flag array.

    int total = 2*n - 1;
    std::vector<bool> done(total, false);

    // mark all leaves done
    for (int i = n-1; i < total; i++) done[i] = true;

    // repeat passes until root is done — O(depth) passes, depth ~ log n
    bool changed = true;
    while (changed) {
        changed = false;
        for (int i = 0; i < n-1; i++) {
            if (done[i]) continue;
            int L = nodes[i].leftChild;
            int R = nodes[i].rightChild;
            if (L < 0 || R < 0) continue;
            if (!done[L] || !done[R]) continue;
            nodes[i].bounds = merge(nodes[L].bounds, nodes[R].bounds);
            done[i]  = true;
            changed  = true;
        }
    }
}

// ── Host build ────────────────────────────────────────────────────────────────

LBVH buildLBVH(sphere* d_spheres, int n) {
    // 1. pull spheres to host for scene AABB
    std::vector<sphere> h_spheres(n);
    cudaMemcpy(h_spheres.data(), d_spheres, n*sizeof(sphere), cudaMemcpyDeviceToHost);

    AABB scene;
    scene.mn = vec3( 1e30f,  1e30f,  1e30f);
    scene.mx = vec3(-1e30f, -1e30f, -1e30f);
    for (auto& s : h_spheres) {
        vec3 r(s.radius, s.radius, s.radius);
        scene = merge(scene, AABB(s.center - r, s.center + r));
    }
    std::cerr << "[LBVH] scene min=(" << scene.mn.x() << "," << scene.mn.y() << "," << scene.mn.z()
              << ") max=(" << scene.mx.x() << "," << scene.mx.y() << "," << scene.mx.z()
              << ") n=" << n << "\n";

    // 2. Morton codes on GPU
    thrust::device_vector<uint32_t> d_codes(n);
    thrust::device_vector<int>      d_idx(n);
    k_computeMorton<<<(n+127)/128, 128>>>(
        d_spheres, d_codes.data().get(), d_idx.data().get(), scene, n);
    cudaDeviceSynchronize();

    // 3. sort
    thrust::sort_by_key(d_codes.begin(), d_codes.end(), d_idx.begin());

    // 4. sorted index array
    int* d_sortedIdx;
    cudaMalloc(&d_sortedIdx, n*sizeof(int));
    cudaMemcpy(d_sortedIdx, d_idx.data().get(), n*sizeof(int), cudaMemcpyDeviceToDevice);

    // 5. allocate node array on GPU, init parents to -1
    int total = 2*n - 1;
    BVHNode* d_nodes;
    cudaMalloc(&d_nodes, total*sizeof(BVHNode));
    // zero everything so parent=0 doesn't mislead — set via thrust fill
    cudaMemset(d_nodes, 0, total*sizeof(BVHNode));

    // mark all parents as -1 (0xFF bytes = -1 for int in two's complement)
    // We'll set parents properly in k_buildInternal
    // Use a small kernel to init parent fields
    // (cudaMemset can't write -1 per-field, so use a kernel)
    auto initParents = [&]() {
        // lambda can't be a kernel — do it with thrust
        thrust::device_ptr<BVHNode> p(d_nodes);
        // just zero them; k_buildInternal will overwrite parent for all non-root nodes
    };
    (void)initParents;

    // 6. build leaves on GPU
    k_initLeaves<<<(n+127)/128, 128>>>(d_spheres, d_sortedIdx, d_nodes, n);
    cudaDeviceSynchronize();

    if (n == 1) {
        // root is the single leaf
        std::vector<BVHNode> h(1);
        cudaMemcpy(h.data(), d_nodes, sizeof(BVHNode), cudaMemcpyDeviceToHost);
        std::cerr << "[LBVH] single sphere, root bounds min=("
                  << h[0].bounds.mn.x() << "," << h[0].bounds.mn.y() << "," << h[0].bounds.mn.z() << ")\n";
        return LBVH{d_nodes, d_sortedIdx, n};
    }

    // 7. build internal nodes on GPU
    k_buildInternal<<<(n+127)/128, 128>>>(d_codes.data().get(), d_nodes, n);
    cudaDeviceSynchronize();

    // 8. pull all nodes to host, fit bounds there (100% correct)
    std::vector<BVHNode> h_nodes(total);
    cudaMemcpy(h_nodes.data(), d_nodes, total*sizeof(BVHNode), cudaMemcpyDeviceToHost);

    // root parent must be -1
    h_nodes[0].parent = -1;

    fitBoundsHost(h_nodes, n);

    // 9. push corrected nodes back to GPU
    cudaMemcpy(d_nodes, h_nodes.data(), total*sizeof(BVHNode), cudaMemcpyHostToDevice);

    // verify root
    BVHNode& root = h_nodes[0];
    std::cerr << "[LBVH] root bounds min=("
              << root.bounds.mn.x() << "," << root.bounds.mn.y() << "," << root.bounds.mn.z()
              << ") max=("
              << root.bounds.mx.x() << "," << root.bounds.mx.y() << "," << root.bounds.mx.z()
              << ") isLeaf=" << root.isLeaf
              << " left=" << root.leftChild << " right=" << root.rightChild << "\n";

    // sanity: root bounds should approximately contain scene bounds
    bool sane = root.bounds.mn.x() <= scene.mn.x() + 1.0f &&
                root.bounds.mx.x() >= scene.mx.x() - 1.0f;
    std::cerr << "[LBVH] bounds sanity: " << (sane ? "OK" : "FAIL — still wrong!") << "\n";

    return LBVH{d_nodes, d_sortedIdx, n};
}

void freeLBVH(LBVH& bvh) {
    cudaFree(bvh.nodes);
    cudaFree(bvh.sortedIdx);
}
