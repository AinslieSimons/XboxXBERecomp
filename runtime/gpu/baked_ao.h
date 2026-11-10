// Baked ambient occlusion (Metal): the static world's occlusion, ray traced in the background from the triangles a title
// draws and kept per scene beside the shader cache (metalcache/ao), then read by those draws' pixel shaders.
//
// A title says which draws are static world geometry in world coordinates (g_eligible, on the thread that issues draws)
// and which scene they belong to (g_scene: draws of different scenes may overlap in space). Each such draw is known by
// its vertex data, its indices' address and its count; the first time one is drawn its triangles are read, and the
// store bakes every triangle not yet baked against all the scene's triangles seen so far. Each triangle gets a small
// grid of values across it (one every kTexel, at most kCap along an edge), which the pixel shader reads by
// [[primitive_id]] and [[barycentric_coord]]. A later run rebakes what was baked against far fewer triangles than the
// scene has since shown, so coverage and quality fill in as places are seen; nothing is ever waited for.
#pragma once

#include <atomic>
#include <cstdint>

namespace gpu {
struct State;
namespace ao {

// the title's: whether a draw is static world geometry in world coordinates (may set g_scene)
inline bool (*g_eligible)(const State&) = nullptr;
inline std::atomic<uint32_t> g_scene{0};     // the scene being drawn, 0 none
inline std::atomic<float> g_strength{0.0f};  // 0 off, 1 the occlusion as baked
// the draw being executed: its entry from Lookup (0 none), set by the render thread around the backend's Draw
inline uint32_t g_draw = 0;

// on the thread that issues draws, before they are queued: the draw's first triangle in Triangles() plus one, 0 none
uint32_t Lookup(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices);

}  // namespace ao
}  // namespace gpu

#ifdef __OBJC__
#import <Metal/Metal.h>
namespace gpu::ao {
// per triangle (uint2): its first value in Values() and its grid size n, (n + 1)(n + 2) / 2 values in rows of n + 1 - i
id<MTLBuffer> Triangles();
id<MTLBuffer> Values();  // uchar, 255 open
}  // namespace gpu::ao
#endif
