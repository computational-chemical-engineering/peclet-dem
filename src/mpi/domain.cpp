// domain.cpp - Z-curve (Morton) partitioning placeholder
#include <vector>
#include <cstdint>
#include <algorithm>

namespace demgpu {

struct Particle {
    float x, y, z;
    uint64_t morton;
};

static inline uint64_t expandBits(uint32_t v) {
    uint64_t x = v & 0x1fffff; // 21 bits
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8)  & 0x100f00f00f00f00f;
    x = (x | x << 4)  & 0x10c30c30c30c30c3;
    x = (x | x << 2)  & 0x1249249249249249;
    return x;
}

uint64_t morton3D(float x, float y, float z) {
    auto clampf = [](float v){ return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); };
    uint32_t xx = static_cast<uint32_t>(clampf(x) * (1<<21));
    uint32_t yy = static_cast<uint32_t>(clampf(y) * (1<<21));
    uint32_t zz = static_cast<uint32_t>(clampf(z) * (1<<21));
    return (expandBits(xx) << 2) | (expandBits(yy) << 1) | expandBits(zz);
}

void assignMorton(std::vector<Particle>& particles) {
    for (auto& p : particles) {
        p.morton = morton3D(p.x, p.y, p.z);
    }
    std::sort(particles.begin(), particles.end(), [](const Particle& a, const Particle& b){ return a.morton < b.morton; });
}

} // namespace demgpu
