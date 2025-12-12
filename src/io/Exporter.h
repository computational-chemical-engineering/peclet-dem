#pragma once
#include <string>
#include <vector>
#include <vector_types.h> // for float3, float4

// Function to export particle state to a LAMMPS Dump Custom file
void export_lammps_dump(const std::string &filename, int step,
                        const std::vector<float3> &pos,
                        const std::vector<float3> &vel,
                        const std::vector<float4> &quats, // w, x, y, z
                        const std::vector<float> &radii,
                        const float3 *box_min_ptr = nullptr,
                        const float3 *box_max_ptr = nullptr,
                        bool pbc_enabled = false);
