#pragma once

#include <Eigen/Dense>

namespace bez
{
    struct WorldSpace
    {
        float x, y, z;

        WorldSpace() {}
        WorldSpace(const float f) : x(f), y(f), z(f) {}
        WorldSpace(
            const float x, 
            const float y, 
            const float z) : x(x), y(y), z(z) {}

        WorldSpace(const Eigen::Vector3d& v) : x(float(v[0])), y(float(v[1])), z(float(v[2])) {}
        WorldSpace(const Eigen::Vector3f& v) : x(float(v[0])), y(float(v[1])), z(float(v[2])) {}
        WorldSpace(const Eigen::Array3d& v) : x(float(v[0])), y(float(v[1])), z(float(v[2])) {}
        WorldSpace(const Eigen::Array3f& v) : x(float(v[0])), y(float(v[1])), z(float(v[2])) {}
        
    };

    WorldSpace operator* (float c, const WorldSpace& v)
    {
        WorldSpace cv = { c * v.x, c * v.y, c * v.z };
        return cv;
    }

    WorldSpace operator+ (const WorldSpace& lhs, const WorldSpace& rhs)
    {
        WorldSpace v = { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z };
        return v;
    }

    WorldSpace operator- (const WorldSpace& lhs, const WorldSpace& rhs)
    {
        WorldSpace v = { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z };
        return v;
    }

    float Dot(const WorldSpace& lhs, const WorldSpace& rhs)
    {
        return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
    }

    float LengthSquared(const WorldSpace& v)
    {
        return Dot(v, v);
    }

    Eigen::Vector3f toEig(WorldSpace& v) {
        return Eigen::Map<Eigen::Vector3f>(&v.x);
    }
}