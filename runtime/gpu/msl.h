#pragma once
// The renderers' generated HLSL as Metal Shading Language, compiled at run time like D3DCompile (gpu/metal.mm)
#include <string>
#include <utility>
#include <vector>

namespace gpu {

// a generated vertex shader (functions vmain, and vpoint for a point drawn as a quad, its inputs per instance) or
// pixel shader (function fmain); constants in buffer 0, VC then DC, as one block
std::string VertexMsl(const std::string& hlsl);
std::string PixelMsl(const std::string& hlsl);
// a full screen pass's source, with its globals carried in a value: entries by name ("vs", "psmain"), each function
// named as its entry; constants in buffer 0
std::string PassMsl(const std::string& hlsl, const std::vector<std::pair<std::string, bool /*vertex*/>>& entries);

}  // namespace gpu
