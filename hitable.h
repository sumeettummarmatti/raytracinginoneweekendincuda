#ifndef HITABLEH
#define HITABLEH

#include "ray.h"

// Compatibility macros for plain C++ compilers (host-only)
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

class material;

enum MaterialType { MAT_LAMBERTIAN = 0, MAT_DIELECTRIC, MAT_METAL };

struct MaterialData {
    MaterialType type;
    vec3         albedo;
    float        fuzz;
    float        ref_idx;
};

struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
    material* mat_ptr;
    MaterialData mat;
};

class hitable  {
    public:
        __host__ __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif
