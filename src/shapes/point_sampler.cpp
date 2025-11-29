// point_sampler.cpp - CPU code to generate shell points (placeholder)
#include <vector>
#include <cmath>

namespace demgpu {

// Simple Fibonacci sphere sampler
std::vector<float> sampleSpherePoints(int n) {
    std::vector<float> pts;
    pts.reserve(n * 3);
    const float phi = (1.f + std::sqrt(5.f)) * 0.5f;
    for (int i = 0; i < n; ++i) {
        float y = 1.f - (2.f * i + 1.f) / n;
        float r = std::sqrt(std::max(0.f, 1.f - y*y));
        float theta = 2.f * 3.1415926535f * i / phi;
        float x = r * std::cos(theta);
        float z = r * std::sin(theta);
        pts.push_back(x);
        pts.push_back(y);
        pts.push_back(z);
    }
    return pts;
}

} // namespace demgpu
