// Baked ambient occlusion: gpu/baked_ao.h says what it is.
#import <Metal/Metal.h>

#include "gpu/baked_ao.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "app/app.h"
#include "core/log.h"
#include "gpu/shaders.h"
#include "gpu/state.h"
#include "gpu/vertex.h"
#include "kernel/kernel.h"

namespace gpu::ao {
namespace {

using Clock = std::chrono::steady_clock;

// the bake: a value every kTexel metres along a triangle's longest edge, at most kCap; kRays cosine-weighted rays each,
// kRadius metres long, from kEps off the surface on the side the camera was on when the mesh was first drawn (the game's
// meshes are wound either way: roads face down, barriers out, and every visible face reports front facing). A hit on a surface facing within about 18 degrees of the
// point's own and less than kLayer from its plane is not an occluder: the city's roads are laid in overlapping layers
// (nearly half of Hong Kong's ground cells hold two within 30 cm), which otherwise darken each other a flat grey
constexpr float kTexel = 0.25f, kRadius = 8.0f, kEps = 0.02f, kLayer = 0.25f, kLayerCos = 0.95f;
constexpr uint32_t kCap = 16, kRays = 128;
constexpr uint32_t kFormat = 4;                                      // the file's layout and the bake's method
constexpr size_t kTriCapacity = size_t(4) << 20, kValCapacity = size_t(128) << 20;
constexpr uint64_t kRaysPerBuffer = uint64_t(4) << 20;               // one command buffer's rays: about a millisecond
constexpr size_t kBatchTriangles = 20000;                            // a bake starts at this many new triangles
constexpr auto kBatchAge = std::chrono::seconds(3);                  // or when the oldest has waited this long
constexpr char kMagic[8] = {'X', 'B', 'E', 'A', 'O', 'B', 'K', '1'};

uint32_t GridOf(const float* p) {
    float longest = 0;
    for (int a = 0; a < 3; a++) {
        const float* u = p + a * 3;
        const float* v = p + ((a + 1) % 3) * 3;
        const float d[3] = {u[0] - v[0], u[1] - v[1], u[2] - v[2]};
        longest = std::max(longest, std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
    }
    return std::clamp(uint32_t(std::ceil(longest / kTexel)), 1u, kCap);
}
size_t ValuesOf(uint32_t n) { return size_t(n + 1) * (n + 2) / 2; }

struct Mesh {
    std::vector<float> xyz;  // 9 a triangle, as the draw lists them
    std::vector<uint8_t> grid, values;  // each triangle's n; a value a point, empty until baked
    float eye[3] = {};        // the camera when first drawn: each triangle is baked on the side facing it
    uint32_t baked_with = 0;  // the scene's triangles when baked
    uint32_t gpu = 0;         // first triangle in the buffers plus one, 0 not there
};

struct Store {
    std::mutex mu;
    std::condition_variable cv;
    uint32_t want = 0, scene = 0;            // asked for, loaded
    std::atomic<uint32_t> ready{0};          // the scene loaded and taking draws
    std::atomic<uint32_t> gen{1};            // counts changes a waiting draw should look again for
    std::unordered_map<uint64_t, Mesh> meshes;
    std::deque<uint64_t> pending;
    size_t pending_tris = 0, total_tris = 0;
    Clock::time_point oldest;
    id<MTLDevice> dev;
    id<MTLBuffer> tri, val;
    size_t tri_at = 0, val_at = 0;
    bool started = false;
};
Store& S() {
    static Store s;
    return s;
}

std::filesystem::path FileOf(uint32_t scene) {
    char name[64];
    std::snprintf(name, sizeof name, "scene_%08X.bin", scene);
    // XBE_AO_DIR: elsewhere (a test keeping its bakes across fresh copies of a game folder)
    if (const char* dir = std::getenv("XBE_AO_DIR"); dir && *dir) return std::filesystem::path(dir) / name;
    return app::ExeDir() / "metalcache" / "ao" / name;
}

uint64_t Params() {
    const struct {
        float texel, radius, eps, layer, layer_cos;
        uint32_t cap, rays, format;
    } p{kTexel, kRadius, kEps, kLayer, kLayerCos, kCap, kRays, kFormat};
    return Hash(&p, sizeof p);
}

// the scene's file: a header, then a record a baked mesh (the last of one content counts)
std::unordered_map<uint64_t, Mesh> Load(uint32_t scene, size_t& records) {
    std::unordered_map<uint64_t, Mesh> out;
    records = 0;
    FILE* f = std::fopen(FileOf(scene).string().c_str(), "rb");
    if (!f) return out;
    char magic[8];
    uint64_t params = 0;
    if (std::fread(magic, 8, 1, f) != 1 || std::memcmp(magic, kMagic, 8) || std::fread(&params, 8, 1, f) != 1 || params != Params()) {
        std::fclose(f);
        std::error_code ec;
        std::filesystem::remove(FileOf(scene), ec);  // another bake's
        return out;
    }
    for (;;) {
        uint64_t content;
        uint32_t head[3];  // triangles, baked with, values
        if (std::fread(&content, 8, 1, f) != 1 || std::fread(head, 12, 1, f) != 1) break;
        Mesh m;
        m.xyz.resize(size_t(head[0]) * 9);
        m.grid.resize(head[0]);
        m.values.resize(head[2]);
        if (std::fread(m.xyz.data(), 4, m.xyz.size(), f) != m.xyz.size() || std::fread(m.grid.data(), 1, m.grid.size(), f) != m.grid.size() ||
            std::fread(m.values.data(), 1, m.values.size(), f) != m.values.size())
            break;  // cut off
        if (std::fread(m.eye, 4, 3, f) != 3) break;
        m.baked_with = head[1];
        out[content] = std::move(m);
        records++;
    }
    std::fclose(f);
    return out;
}

void Append(uint32_t scene, const std::vector<std::pair<uint64_t, const Mesh*>>& meshes, bool whole = false) {
    namespace fs = std::filesystem;
    const fs::path path = FileOf(scene), tmp = fs::path(path).concat(".new");
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    const bool fresh = whole || !fs::exists(path, ec);
    FILE* f = std::fopen((whole ? tmp : path).string().c_str(), whole ? "wb" : "ab");
    if (!f) return;
    if (fresh) {
        const uint64_t params = Params();
        std::fwrite(kMagic, 8, 1, f);
        std::fwrite(&params, 8, 1, f);
    }
    for (auto& [content, m] : meshes) {
        const uint32_t head[3] = {uint32_t(m->grid.size()), m->baked_with, uint32_t(m->values.size())};
        std::fwrite(&content, 8, 1, f);
        std::fwrite(head, 12, 1, f);
        std::fwrite(m->xyz.data(), 4, m->xyz.size(), f);
        std::fwrite(m->grid.data(), 1, m->grid.size(), f);
        std::fwrite(m->values.data(), 1, m->values.size(), f);
        std::fwrite(m->eye, 4, 3, f);
    }
    std::fclose(f);
    if (whole) fs::rename(tmp, path, ec);
}

// into the buffers the pixel shaders read, with the store locked; false when they are full
bool Upload(Store& s, Mesh& m) {
    if (m.gpu) return true;
    const size_t n = m.grid.size();
    if (!n || m.values.empty() || s.tri_at + n > kTriCapacity || s.val_at + m.values.size() > kValCapacity) return false;
    uint32_t* t = static_cast<uint32_t*>(s.tri.contents) + s.tri_at * 2;
    uint32_t off = uint32_t(s.val_at);
    for (size_t i = 0; i < n; i++) {
        t[i * 2] = off;
        t[i * 2 + 1] = m.grid[i];
        off += uint32_t(ValuesOf(m.grid[i]));
    }
    std::memcpy(static_cast<uint8_t*>(s.val.contents) + s.val_at, m.values.data(), m.values.size());
    m.gpu = uint32_t(s.tri_at) + 1;
    s.tri_at += n;
    s.val_at += m.values.size();
    return true;
}

const char* kBakeSource = R"(
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;
struct Point { uint tri; float a, b; };
struct Params { uint rays; float radius, eps, layer, layer_cos; };
float radical(uint b) { return float(reverse_bits(b)) * 2.3283064365386963e-10; }
kernel void bake(uint id [[thread_position_in_grid]], const device Point* pts [[buffer(0)]], const device packed_float3* v [[buffer(1)]],
                 device float* out [[buffer(2)]], constant Params& p [[buffer(3)]], primitive_acceleration_structure as [[buffer(4)]]) {
    Point x = pts[id];
    const bool back = (x.tri >> 31) != 0;  // the camera on the other side of the triangle's winding
    x.tri &= 0x7FFFFFFFu;
    float3 A = v[x.tri * 3], B = v[x.tri * 3 + 1], C = v[x.tri * 3 + 2];
    float3 n = cross(B - A, C - A);
    float l = length(n);
    if (l < 1e-9) { out[id] = 1; return; }
    n /= l;
    float3 o = A + (B - A) * x.a + (C - A) * x.b;
    float3 u = normalize(abs(n.x) > 0.5 ? float3(n.y, -n.x, 0) : float3(0, n.z, -n.y)), w = cross(n, u);
    intersection_params ip;
    ip.set_triangle_cull_mode(triangle_cull_mode::none);
    intersection_query<triangle_data> q;
    ray r;
    r.min_distance = p.eps;
    r.max_distance = p.radius;
    float j1 = fract(float(id) * 0.7548776662), j2 = fract(float(id) * 0.5698402910);
    {
        float3 m = back ? -n : n;
        float occluded = 0;
        for (uint i = 0; i < p.rays; i++) {
            float r1 = fract((float(i) + 0.5) / float(p.rays) + j1), r2 = fract(radical(i) + j2);
            float s = sqrt(r1), ph = 6.2831853 * r2;
            r.origin = o + m * p.eps;
            r.direction = normalize(u * (s * cos(ph)) + w * (s * sin(ph)) + m * sqrt(max(0.0, 1 - r1)));
            q.reset(r, as, ip);
            while (q.next()) {
                if (q.get_candidate_intersection_type() != intersection_type::triangle) continue;
                const uint t = q.get_candidate_primitive_id();
                float3 h = cross(v[t * 3 + 1] - v[t * 3], v[t * 3 + 2] - v[t * 3]);
                const float hl = length(h);
                const float3 at = r.origin + r.direction * q.get_candidate_triangle_distance();
                if (hl > 1e-9 && abs(dot(h / hl, n)) > p.layer_cos && abs(dot(at - o, n)) < p.layer) continue;  // a layer of this surface
                occluded += 1;
                q.abort();
            }
        }
        out[id] = 1 - occluded / float(p.rays);
    }
}
)";

struct Baker {
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pso;
    bool Init(id<MTLDevice> dev) {
        queue = [dev newCommandQueue];
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@(kBakeSource) options:nil error:&err];
        if (!lib) {
            LOGR("baked ao: the bake's shader did not build: %s", err.localizedDescription.UTF8String);
            return false;
        }
        pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bake"] error:&err];
        return pso != nil;
    }

    // every triangle of the batch, against every triangle of the scene (scene: 9 floats a triangle; first: each
    // batch mesh's first triangle in it)
    void Run(id<MTLDevice> dev, const std::vector<float>& scene, const std::vector<std::pair<Mesh*, uint32_t>>& batch) {
        const auto t0 = Clock::now();
        const uint32_t tris = uint32_t(scene.size() / 9);
        id<MTLBuffer> vb = [dev newBufferWithBytes:scene.data() length:scene.size() * 4 options:MTLResourceStorageModeShared];
        MTLAccelerationStructureTriangleGeometryDescriptor* g = [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
        g.vertexBuffer = vb;
        g.vertexStride = 12;
        g.triangleCount = tris;
        g.opaque = NO;  // every hit to the bake's query, which leaves out layers
        MTLPrimitiveAccelerationStructureDescriptor* d = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        d.geometryDescriptors = @[ g ];
        const MTLAccelerationStructureSizes sz = [dev accelerationStructureSizesWithDescriptor:d];
        id<MTLAccelerationStructure> as = [dev newAccelerationStructureWithSize:sz.accelerationStructureSize];
        id<MTLBuffer> scratch = [dev newBufferWithLength:std::max<NSUInteger>(sz.buildScratchBufferSize, 256) options:MTLResourceStorageModePrivate];
        if (!as || !scratch) return;
        {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLAccelerationStructureCommandEncoder> e = [cb accelerationStructureCommandEncoder];
            [e buildAccelerationStructure:as descriptor:d scratchBuffer:scratch scratchBufferOffset:0];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        // the points: each triangle's grid, pulled a hair inside so a point on an edge is on the triangle
        struct Point {
            uint32_t tri;
            float a, b;
        };
        std::vector<Point> pts;
        size_t back = 0, all = 0;
        for (auto& [m, first] : batch) {
            const size_t n = m->xyz.size() / 9;
            m->grid.resize(n);
            for (size_t i = 0; i < n; i++) {
                const float* q = &m->xyz[i * 9];
                const float e1[3] = {q[3] - q[0], q[4] - q[1], q[5] - q[2]}, e2[3] = {q[6] - q[0], q[7] - q[1], q[8] - q[2]};
                const float nx = e1[1] * e2[2] - e1[2] * e2[1], ny = e1[2] * e2[0] - e1[0] * e2[2], nz = e1[0] * e2[1] - e1[1] * e2[0];
                const float to[3] = {m->eye[0] - (q[0] + q[3] + q[6]) / 3, m->eye[1] - (q[1] + q[4] + q[7]) / 3, m->eye[2] - (q[2] + q[5] + q[8]) / 3};
                const uint32_t side = nx * to[0] + ny * to[1] + nz * to[2] < 0 ? 0x80000000u : 0u;
                back += side != 0, all++;
                const uint32_t k = GridOf(q);
                m->grid[i] = uint8_t(k);
                for (uint32_t r = 0; r <= k; r++)
                    for (uint32_t c = 0; c <= k - r; c++) {
                        constexpr float kIn = 0.999f, kMid = 1.0f / 3;
                        pts.push_back({(first + uint32_t(i)) | side, kMid + (float(r) / k - kMid) * kIn, kMid + (float(c) / k - kMid) * kIn});
                    }
            }
        }
        if (pts.empty()) return;
        id<MTLBuffer> pb = [dev newBufferWithBytes:pts.data() length:pts.size() * sizeof(Point) options:MTLResourceStorageModeShared];
        id<MTLBuffer> ob = [dev newBufferWithLength:pts.size() * 4 options:MTLResourceStorageModeShared];
        const struct {
            uint32_t rays;
            float radius, eps, layer, layer_cos;
        } params{kRays, kRadius, kEps, kLayer, kLayerCos};
        const size_t per = size_t(std::max<uint64_t>(1, kRaysPerBuffer / kRays));
        for (size_t at = 0; at < pts.size(); at += per) {
            const size_t n = std::min(per, pts.size() - at);
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pso];
            [e setBuffer:pb offset:at * sizeof(Point) atIndex:0];
            [e setBuffer:vb offset:0 atIndex:1];
            [e setBuffer:ob offset:at * 4 atIndex:2];
            [e setBytes:&params length:sizeof params atIndex:3];
            [e setAccelerationStructure:as atBufferIndex:4];
            [e useResource:vb usage:MTLResourceUsageRead];
            [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(pso.threadExecutionWidth, 1, 1)];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            std::this_thread::sleep_for(std::chrono::milliseconds(1));  // the frames' GPU time first
        }
        const float* v = static_cast<const float*>(ob.contents);
        size_t at = 0;
        for (auto& [m, first] : batch) {
            size_t count = 0;
            for (uint8_t k : m->grid) count += ValuesOf(k);
            m->values.resize(count);
            for (size_t i = 0; i < count; i++) m->values[i] = uint8_t(std::lround(std::clamp(v[at + i], 0.0f, 1.0f) * 255.0f));
            at += count;
        }
        LOGR("baked ao: %zu meshes (%.0f%% of triangles seen from behind their winding), %zu points against %u triangles in %.0f ms",
             batch.size(), all ? 100.0 * double(back) / double(all) : 0.0, pts.size(), tris,
             std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
};

void Worker() {
    Store& s = S();
    Baker baker;
    if (!baker.Init(s.dev)) return;
    std::unique_lock lock(s.mu);
    for (;;) {
        s.cv.wait_for(lock, std::chrono::milliseconds(500));
        if (s.want != s.scene) {
            const uint32_t want = s.want;
            s.ready = 0;
            s.scene = want;
            s.meshes.clear();
            s.pending.clear();
            s.pending_tris = s.total_tris = 0;
            s.tri_at = s.val_at = 0;
            lock.unlock();
            size_t records = 0;
            auto loaded = Load(want, records);
            // rebakes add records: the file written again with only the latest of each once half of it is old
            if (records > loaded.size() * 2) {
                std::vector<std::pair<uint64_t, const Mesh*>> all;
                for (auto& [content, m] : loaded) all.emplace_back(content, &m);
                Append(want, all, true);
            }
            lock.lock();
            if (s.want != want) continue;
            s.meshes = std::move(loaded);
            size_t stale = 0;
            for (auto& [content, m] : s.meshes) s.total_tris += m.grid.size();
            // baked against a fifth fewer triangles than the scene has now: again, keeping the old values meanwhile
            for (auto& [content, m] : s.meshes)
                if (uint64_t(m.baked_with) * 5 < uint64_t(s.total_tris) * 4) s.pending.push_back(content), s.pending_tris += m.grid.size(), stale++;
            s.oldest = Clock::now();
            s.ready = want;
            s.gen++;
            LOGR("baked ao: scene %08X: %zu meshes, %zu triangles from the cache, %zu to bake again", want, s.meshes.size(), s.total_tris, stale);
            continue;
        }
        if (s.pending.empty() || (s.pending_tris < kBatchTriangles && Clock::now() - s.oldest < kBatchAge)) continue;
        // a copy of the batch's meshes and the scene, baked unlocked, then put back if the scene is the same
        const uint32_t scene = s.scene;
        std::vector<float> all;
        all.reserve(s.total_tris * 9);
        std::unordered_map<uint64_t, uint32_t> first;
        for (auto& [content, m] : s.meshes) {
            first[content] = uint32_t(all.size() / 9);
            all.insert(all.end(), m.xyz.begin(), m.xyz.end());
        }
        std::vector<std::pair<uint64_t, Mesh>> work;
        for (uint64_t content : s.pending) {
            auto it = s.meshes.find(content);
            if (it == s.meshes.end()) continue;
            Mesh copy;
            copy.xyz = it->second.xyz;
            work.emplace_back(content, std::move(copy));
        }
        s.pending.clear();
        s.pending_tris = 0;
        const uint32_t with = uint32_t(all.size() / 9);
        lock.unlock();
        std::vector<std::pair<Mesh*, uint32_t>> batch;
        for (auto& [content, m] : work) batch.emplace_back(&m, first[content]);
        @autoreleasepool {
            baker.Run(s.dev, all, batch);
        }
        lock.lock();
        if (s.scene != scene) continue;
        std::vector<std::pair<uint64_t, const Mesh*>> done;
        for (auto& [content, m] : work) {
            auto it = s.meshes.find(content);
            if (it == s.meshes.end() || m.values.empty()) continue;
            Mesh& to = it->second;
            to.grid = std::move(m.grid);
            to.values = std::move(m.values);
            to.baked_with = with;
            done.emplace_back(content, &to);
        }
        Append(scene, done);
        s.gen++;
    }
}

bool Start() {
    Store& s = S();
    if (s.started) return s.tri != nil;
    s.started = true;
    s.dev = MTLCreateSystemDefaultDevice();
    if (!s.dev || !s.dev.supportsRaytracing) {
        LOGR("baked ao: no ray tracing on this GPU");
        return false;
    }
    s.tri = [s.dev newBufferWithLength:kTriCapacity * 8 options:MTLResourceStorageModeShared];
    s.val = [s.dev newBufferWithLength:kValCapacity options:MTLResourceStorageModeShared];
    if (!s.tri || !s.val) return false;
    std::thread(Worker).detach();
    return true;
}

struct Key {
    uint32_t data, iaddr, count, base, prim, start;
    bool operator==(const Key&) const = default;
};
struct KeyHash {
    size_t operator()(const Key& k) const { return Hash(&k, sizeof k); }
};
struct Draw {
    uint64_t content = 0;
    uint32_t gpu = 0, gen = 0;
    uint8_t state = 0;  // 0 not read, 1 waiting for its bake, 2 ready, 3 never
};

}  // namespace

id<MTLBuffer> Triangles() { return S().tri; }
id<MTLBuffer> Values() { return S().val; }

uint32_t Lookup(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices) {
    if (g_strength.load(std::memory_order_relaxed) <= 0 || !g_eligible || prim < 5 || prim > 10 || !s.streams[0].vb || !(s.vs_handle & 1) ||
        s.input_active || !g_eligible(s))
        return 0;
    const uint32_t scene = g_scene.load(std::memory_order_relaxed);
    if (!scene || !Start()) return 0;
    Store& st = S();
    static uint32_t current = 0;
    static std::unordered_map<Key, Draw, KeyHash> draws;
    if (scene != current) {
        current = scene;
        draws.clear();
        {
            std::lock_guard lock(st.mu);
            st.want = scene;
        }
        st.cv.notify_one();
    }
    if (st.ready.load(std::memory_order_acquire) != scene) return 0;
    const Key key{kernel::R32(s.streams[0].vb + 4), indices ? uint32_t(indices - kernel::g_base) : 0, count, s.base_vertex, prim, indices ? 0 : start};
    Draw& d = draws[key];
    if (d.state == 2) return d.gpu;
    if (d.state == 3) return 0;
    const uint32_t gen = st.gen.load(std::memory_order_acquire);
    if (d.state == 1 && d.gen == gen) return 0;
    thread_local std::vector<float> xyz;
    if (d.state == 0) {
        xyz.clear();
        if (!DrawTriangles(s, prim, start, count, indices, 0, xyz) || xyz.size() < 9) {
            d.state = 3;
            return 0;
        }
        d.content = Hash(xyz.data(), xyz.size() * 4);
    }
    std::lock_guard lock(st.mu);
    if (st.scene != scene) return 0;
    d.gen = gen;
    auto it = st.meshes.find(d.content);
    if (it == st.meshes.end()) {
        if (d.state != 0) return 0;  // read once: its mesh went with a scene change, and comes back with the draw's next read
        Mesh m;
        m.xyz = xyz;
        for (int k = 0; k < 3; k++) m.eye[k] = s.vsc[126][size_t(k)];  // the city programs' eye (as the car programs': src/game/car_reflections.cpp)
        if (st.pending.empty()) st.oldest = Clock::now();
        st.pending.push_back(d.content);
        st.pending_tris += xyz.size() / 9;
        st.total_tris += xyz.size() / 9;
        st.meshes.emplace(d.content, std::move(m));
        d.state = 1;
        return 0;
    }
    if (it->second.values.empty()) {
        d.state = 1;
        return 0;
    }
    if (!Upload(st, it->second)) {
        d.state = 3;
        return 0;
    }
    d.state = 2;
    d.gpu = it->second.gpu;
    return d.gpu;
}

}  // namespace gpu::ao
