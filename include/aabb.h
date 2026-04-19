#pragma once
#include <cuda_runtime.h>
#include "vec3.h"
#include "ray.h"

struct AABB {
    vec3 mn, mx;   // min/max corners

    __host__ __device__ AABB() {}
    __host__ __device__ AABB(vec3 a, vec3 b) : mn(a), mx(b) {}

    __device__ bool hit(const ray& r, float tMin, float tMax) const {
        for (int a = 0; a < 3; a++) {
            float invD = 1.0f / r.direction()[a];
            float t0   = (mn[a] - r.origin()[a]) * invD;
            float t1   = (mx[a] - r.origin()[a]) * invD;
            if (invD < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
            tMin = t0 > tMin ? t0 : tMin;
            tMax = t1 < tMax ? t1 : tMax;
            if (tMax <= tMin) return false;
        }
        return true;
    }
};

__host__ __device__ inline AABB merge(AABB a, AABB b) {
    vec3 mn(fminf(a.mn.x(), b.mn.x()),
            fminf(a.mn.y(), b.mn.y()),
            fminf(a.mn.z(), b.mn.z()));
    vec3 mx(fmaxf(a.mx.x(), b.mx.x()),
            fmaxf(a.mx.y(), b.mx.y()),
            fmaxf(a.mx.z(), b.mx.z()));
    return AABB(mn, mx);
}
