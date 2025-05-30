#pragma once
#ifndef BEZ_CUBIC_SPLINE_HELPERS_H
#define BEZ_CUBIC_SPLINE_HELPERS_H
#include <Eigen/Dense>
typedef unsigned int uint32;

namespace bez
{
    struct WorldSpace
    {
        float x, y, z;

        WorldSpace() : x(0), y(0), z(0) {}
        WorldSpace(const float f) : x(f), y(f), z(f) {}
        WorldSpace(
            const float x, 
            const float y, 
            const float z) : x(x), y(y), z(z) {}
        WorldSpace(const float* d) : x(d[0]), y(d[1]), z(d[2]) {
        }
        WorldSpace(const Eigen::Vector3f& v) : x(float(v[0])), y(float(v[1])), z(float(v[2])) {}
        WorldSpace(const Eigen::Vector3f&& v) : x(float(v[0])), y(float(v[1])), z(float(v[2])) {}
        WorldSpace(const Eigen::Array3d& v) : x(float(v[0])), y(float(v[1])), z(float(v[2])) {}
        WorldSpace(const Eigen::Array3f& v) : x(float(v[0])), y(float(v[1])), z(float(v[2])) {}

    };

    // sometimes marking things static fixes "already-defined" linker errors
    // sometimes it's marking them inline
    // I've given up trying to understand  
    static WorldSpace operator* (float c, const WorldSpace& v)
    {
        WorldSpace cv = { c * v.x, c * v.y, c * v.z };
        return cv;
    }

    static WorldSpace operator+ (const WorldSpace& lhs, const WorldSpace& rhs)
    {
        WorldSpace v = { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z };
        return v;
    }

    static WorldSpace operator- (const WorldSpace& lhs, const WorldSpace& rhs)
    {
        WorldSpace v = { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z };
        return v;
    }

    static float Dot(const WorldSpace& lhs, const WorldSpace& rhs)
    {
        return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
    }

    static float LengthSquared(const WorldSpace& v)
    {
        return Dot(v, v);
    }

    static Eigen::Vector3f toEig(WorldSpace& v) {
        return Eigen::Map<Eigen::Vector3f>(&v.x);
    }



}
#endif