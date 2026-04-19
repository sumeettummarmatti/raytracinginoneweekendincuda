#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

class sphere: public hitable  {
    public:
        __host__ __device__ sphere() {}
        __host__ __device__ sphere(vec3 cen, float r, MaterialData m) : center(cen), radius(r), mat(m), mat_ptr(nullptr) {}
        __host__ __device__ sphere(vec3 cen, float r, material* m) : center(cen), radius(r), mat_ptr(m) {
            // Baseline only uses mat_ptr, so we don't need to populate 'mat' here.
        }
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        vec3 center;
        float radius;
        material* mat_ptr;
        MaterialData mat;
};

// __device__ inline gives weak linkage — safe to include in multiple .cu files
// without RDC, and still works in single-TU baseline build.
__device__ inline bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrtf(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat = mat;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrtf(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat = mat;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}

#endif
