// The title's Direct3D 8 calls drawn with Metal, on macOS. The console path of d3d11.cpp, what does not depend on the
// API shared through d3d_common, the generated HLSL translated by gpu/msl.cpp. Not here yet: MSAA, SMAA, a sampler's LOD
// bias, renderer extensions, F3 captures.
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <imgui.h>
#include <backends/imgui_impl_metal.h>

#include "app/app.h"
#include "core/diag.h"
#include "core/log.h"
#include "gpu/backend.h"
#include "gpu/d3d_common.h"
#include "gpu/msl.h"
#include "gpu/shaders.h"
#include "gpu/texture.h"
#include "gpu/vertex.h"
#include "kernel/kernel.h"
#include "kernel/memory.h"

namespace app {
void* MetalLayer();     // mac/main.mm: the window's CAMetalLayer
Mode DrawablePixels();  // the window's size in pixels
}

extern const char kShaderSourceHash[];  // written by CMake

namespace gpu {

namespace {

// ---- what was built in earlier runs: every shader and pipeline a run needed is recorded in metalcache/records.bin beside
// the executable, built again on background threads at the next start, before a draw needs it. Pipelines also go in a
// binary archive, so that their machine code need not be made again. The generators' output and PsoKey's layout belong to
// those sources, so records made by other ones are dropped.
constexpr char kCacheMagic[8] = {'X', 'B', 'E', 'M', 'T', 'L', '1', 0};

MTLCompileOptions* ShaderOptions() {
    MTLCompileOptions* opt = [MTLCompileOptions new];
    opt.mathMode = MTLMathModeSafe;  // as the console: no assumptions about NaN and infinity, no reordering
    opt.preserveInvariance = YES;    // EQUAL depth passes need bit exact positions
    return opt;
}

// the generators' sources hashed by CMake: records outlive a rebuild and go only when what makes shaders changes
uint64_t BuildIdentity() { return Hash(kShaderSourceHash, std::strlen(kShaderSourceHash)); }

struct CacheFiles {
    dispatch_queue_t archive_queue = nullptr;  // adds to the archive and writes of it, in order
    id<MTLBinaryArchive> archive;
    NSURL* archive_url = nil;
    FILE* records = nullptr;
    bool dirty = false;  // archive_queue's
    std::chrono::steady_clock::time_point saved;
};
CacheFiles* g_cache_files = nullptr;

void SaveArchiveNow(CacheFiles& c) {  // on the archive's queue
    if (!c.dirty || !c.archive) return;
    c.dirty = false;
    NSError* err = nil;
    NSURL* temp = [c.archive_url URLByAppendingPathExtension:@"new"];
    [NSFileManager.defaultManager removeItemAtURL:temp error:nil];
    if (![c.archive serializeToURL:temp error:&err]) {
        LOGW("metal: pipeline archive not saved: %s", err.localizedDescription.UTF8String);
        return;
    }
    [NSFileManager.defaultManager removeItemAtURL:c.archive_url error:nil];
    [NSFileManager.defaultManager moveItemAtURL:temp toURL:c.archive_url error:nil];
    c.saved = std::chrono::steady_clock::now();
}

constexpr MTLPixelFormat kBgra = MTLPixelFormatBGRA8Unorm;
constexpr MTLPixelFormat kDepth = MTLPixelFormatDepth32Float_Stencil8;
constexpr uint64_t kStaleFrames = 8;       // frames a target may go unused and still be read back
constexpr uint32_t kRingBytes = 96u << 20;  // per frame: vertices, indices, constants, uploads
constexpr int kFramesInFlight = 3;
constexpr int kLateSlots = 3;              // copies in flight per surface locked without a wait
constexpr uint32_t kVisSlots = 1u << 16;   // visibility counters, 8 bytes each
constexpr uint32_t kVertexBufferBase = 1;  // buffer 0 holds the constants

// a console surface drawn into
struct Target {
    id<MTLTexture> res;
    id<MTLTexture> fb;  // copy read by draws that sample the target they render to
    bool depth = false;
    uint32_t gw = 0, gh = 0, w = 0, h = 0;
    uint32_t data = 0, bytes = 0;
    uint64_t epoch = 0, sig = 0;  // last lock seen, guest memory as last matched
    bool drawn = false;           // the GPU wrote it since the last readback
    bool linear32 = true;         // a 32 bit linear surface in guest memory, written back row by row
    bool swizzled32 = false;      // a 32 bit swizzled one, written back in Morton order
    std::vector<uint8_t> read;    // what Sync left in guest memory
};

// a texture from guest memory
struct Tex {
    id<MTLTexture> res;
    TextureDesc d;
    MTLPixelFormat fmt = MTLPixelFormatInvalid;
    uint32_t levels = 1;
    uint64_t sig = 0;
    uint64_t frame = 0;
    uint64_t palette = 0;   // hash of the colours a P8 texture was decoded with
    uint64_t faces = 0;     // a cube's frame its faces drawn on the GPU were last copied in
    bool uploaded = false;  // no draw can have used it before its first upload
};

// a draw's four texture stages
struct DrawTextures {
    TexInfo info[4];
    id<MTLTexture> tex[4];
    uint32_t levels[4] = {1, 1, 1, 1};
    bool replaced[4] = {};  // a title's replacement picture (FindReplacement)
    StageReads reads;
};

struct Pipeline {
    id<MTLRenderPipelineState> state;
    bool points = false;
};

// what a pipeline state object is made from; culling, bias, fill mode, depth and stencil are set per draw in Metal, and a
// polygon offset or a render scale in the key would make a new pipeline (a compile, a stall) for each value
PsoKey PipelineOnly(PsoKey k) {
    k.zen = k.zwrite = k.zfunc = k.sten = k.sfunc = k.smask = k.swmask = k.sfail = k.szfail = k.spass = 0;
    k.cull = k.ccw = k.fill = 0;
    k.zbias = 0;
    k.zslope = 0;
    return k;
}

// Blit's settings past the source and its rectangle
struct BlitOptions {
    bool gamma = false;  // through the gamma lookup
    bool alpha = false;  // the source's alpha, else 1
    uint32_t taps = 1;   // n x n box filter per pixel
    bool over = false;   // premultiplied over the target, which keeps its alpha
    bool final = false;  // the finished frame: psfinal with the settings' FXAA and sharpening
};

// a full screen triangle that writes a colour or a depth, for clears of part of a target
const char* kClearHlsl = R"(
cbuffer B : register(b1) { float4 colour; float4 depth; };
float4 vs(uint id : SV_VertexID) {
    float2 t = float2((id << 1) & 2, id & 2);
    return float4(t * float2(2, -2) + float2(-1, 1), 0, 1);
}
float4 pscolour(float4 p : SV_Position) : SV_Target { return colour; }
float psdepth(float4 p : SV_Position) : SV_Depth { return depth.x; }
)";

// after a Sync every byte is compared: the signature samples a big surface and misses small writes (XFONT text)
bool Changed(const Target& t) {
    if (t.drawn || t.read.empty()) return false;
    uint32_t row = t.gw * 4;
    uint32_t pitch = GuestPitch(t.gw);
    const uint8_t* p = static_cast<const uint8_t*>(kernel::HostPtr(t.data));
    for (uint32_t y = 0; y < t.gh; y++) {
        if (std::memcmp(p + size_t(y) * pitch, &t.read[size_t(y) * row], row)) return true;
    }
    return false;
}

// a lock alone proves nothing, reads lock too: only changed bytes mean the CPU wrote it
bool LatchCpuWrite(Target& t) {
    uint64_t e = DirtyEpoch(t.data);
    if (e <= t.epoch) return false;
    t.epoch = e;
    uint64_t sig = SurfaceSig(t.data, t.bytes);
    if (sig == t.sig && !Changed(t)) return false;
    t.sig = sig;
    return true;
}

// the pixels the CPU changed since the last Sync, opaque, the rest 0; t.read takes them
std::vector<uint8_t> LatchCpuChanges(Target& t) {
    std::vector<uint8_t> px(t.read.size(), 0);
    uint32_t guest_pitch = GuestPitch(t.gw);
    const uint8_t* guest = static_cast<const uint8_t*>(kernel::HostPtr(t.data));
    for (uint32_t y = 0; y < t.gh; y++) {
        for (uint32_t x = 0; x < t.gw; x++) {
            const uint8_t* c = guest + size_t(y) * guest_pitch + x * 4;
            size_t i = (size_t(y) * t.gw + x) * 4;
            if (!std::memcmp(c, &t.read[i], 4)) continue;
            std::memcpy(&t.read[i], c, 4);
            std::memcpy(&px[i], c, 3);
            px[i + 3] = 255;
        }
    }
    return px;
}

uint64_t TargetMapKey(uint32_t data, bool depth) { return uint64_t(data) << 1 | (depth ? 1 : 0); }

// a smaller depth surface's console size, above the address bits
constexpr int kKeyWidthShift = 34;
constexpr int kKeyHeightShift = 48;

MTLPixelFormat MetalFormat(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_BC1_UNORM: return MTLPixelFormatBC1_RGBA;
        case DXGI_FORMAT_BC2_UNORM: return MTLPixelFormatBC2_RGBA;
        case DXGI_FORMAT_BC3_UNORM: return MTLPixelFormatBC3_RGBA;
        case DXGI_FORMAT_R32_FLOAT: return MTLPixelFormatR32Float;
        default: return kBgra;
    }
}

// bytes per block of a compressed host format, 0 for the 4 byte texel formats
uint32_t BlockBytes(MTLPixelFormat f) {
    switch (f) {
        case MTLPixelFormatBC1_RGBA: return 8;
        case MTLPixelFormatBC2_RGBA: case MTLPixelFormatBC3_RGBA: return 16;
        default: return 0;
    }
}

MTLVertexFormat VertexFormat(uint32_t f) {
    switch (DXGI_FORMAT(f)) {
        case DXGI_FORMAT_R32_FLOAT: return MTLVertexFormatFloat;
        case DXGI_FORMAT_R32G32_FLOAT: return MTLVertexFormatFloat2;
        case DXGI_FORMAT_R32G32B32_FLOAT: return MTLVertexFormatFloat3;
        case DXGI_FORMAT_R32G32B32A32_FLOAT: return MTLVertexFormatFloat4;
        case DXGI_FORMAT_B8G8R8A8_UNORM: return MTLVertexFormatUChar4Normalized_BGRA;
        case DXGI_FORMAT_R8G8B8A8_UNORM: return MTLVertexFormatUChar4Normalized;
        case DXGI_FORMAT_R8G8_UNORM: return MTLVertexFormatUChar2Normalized;
        case DXGI_FORMAT_R8_UNORM: return MTLVertexFormatUCharNormalized;
        case DXGI_FORMAT_R16_SNORM: return MTLVertexFormatShortNormalized;
        case DXGI_FORMAT_R16G16_SNORM: return MTLVertexFormatShort2Normalized;
        case DXGI_FORMAT_R16G16B16A16_SNORM: return MTLVertexFormatShort4Normalized;
        case DXGI_FORMAT_R16_SINT: return MTLVertexFormatShort;
        case DXGI_FORMAT_R16G16_SINT: return MTLVertexFormatShort2;
        case DXGI_FORMAT_R16G16B16A16_SINT: return MTLVertexFormatShort4;
        case DXGI_FORMAT_R32_UINT: return MTLVertexFormatUInt;
        default: return MTLVertexFormatInvalid;
    }
}

MTLBlendFactor BlendFactorOf(D3D12_BLEND b) {
    switch (b) {
        case D3D12_BLEND_ZERO: return MTLBlendFactorZero;
        case D3D12_BLEND_ONE: return MTLBlendFactorOne;
        case D3D12_BLEND_SRC_COLOR: return MTLBlendFactorSourceColor;
        case D3D12_BLEND_INV_SRC_COLOR: return MTLBlendFactorOneMinusSourceColor;
        case D3D12_BLEND_SRC_ALPHA: return MTLBlendFactorSourceAlpha;
        case D3D12_BLEND_INV_SRC_ALPHA: return MTLBlendFactorOneMinusSourceAlpha;
        case D3D12_BLEND_DEST_ALPHA: return MTLBlendFactorDestinationAlpha;
        case D3D12_BLEND_INV_DEST_ALPHA: return MTLBlendFactorOneMinusDestinationAlpha;
        case D3D12_BLEND_DEST_COLOR: return MTLBlendFactorDestinationColor;
        case D3D12_BLEND_INV_DEST_COLOR: return MTLBlendFactorOneMinusDestinationColor;
        case D3D12_BLEND_SRC_ALPHA_SAT: return MTLBlendFactorSourceAlphaSaturated;
        case D3D12_BLEND_BLEND_FACTOR: return MTLBlendFactorBlendColor;
        case D3D12_BLEND_INV_BLEND_FACTOR: return MTLBlendFactorOneMinusBlendColor;
    }
    return MTLBlendFactorOne;
}

MTLBlendOperation BlendOpOf(D3D12_BLEND_OP o) {
    switch (o) {
        case D3D12_BLEND_OP_ADD: return MTLBlendOperationAdd;
        case D3D12_BLEND_OP_SUBTRACT: return MTLBlendOperationSubtract;
        case D3D12_BLEND_OP_REV_SUBTRACT: return MTLBlendOperationReverseSubtract;
        case D3D12_BLEND_OP_MIN: return MTLBlendOperationMin;
        case D3D12_BLEND_OP_MAX: return MTLBlendOperationMax;
    }
    return MTLBlendOperationAdd;
}

// D3D12_COMPARISON_FUNC 1 to 8 in the same order as MTLCompareFunction 0 to 7
MTLCompareFunction CompareOf(D3D12_COMPARISON_FUNC f) { return MTLCompareFunction(int(f) - 1); }

MTLStencilOperation StencilOf(D3D12_STENCIL_OP o) {
    switch (o) {
        case D3D12_STENCIL_OP_KEEP: return MTLStencilOperationKeep;
        case D3D12_STENCIL_OP_ZERO: return MTLStencilOperationZero;
        case D3D12_STENCIL_OP_REPLACE: return MTLStencilOperationReplace;
        case D3D12_STENCIL_OP_INCR_SAT: return MTLStencilOperationIncrementClamp;
        case D3D12_STENCIL_OP_DECR_SAT: return MTLStencilOperationDecrementClamp;
        case D3D12_STENCIL_OP_INVERT: return MTLStencilOperationInvert;
        case D3D12_STENCIL_OP_INCR: return MTLStencilOperationIncrementWrap;
        case D3D12_STENCIL_OP_DECR: return MTLStencilOperationDecrementWrap;
    }
    return MTLStencilOperationKeep;
}

MTLSamplerAddressMode AddressOf(D3D12_TEXTURE_ADDRESS_MODE m) {
    switch (m) {
        case D3D12_TEXTURE_ADDRESS_MODE_WRAP: return MTLSamplerAddressModeRepeat;
        case D3D12_TEXTURE_ADDRESS_MODE_MIRROR: return MTLSamplerAddressModeMirrorRepeat;
        case D3D12_TEXTURE_ADDRESS_MODE_CLAMP: return MTLSamplerAddressModeClampToEdge;
        case D3D12_TEXTURE_ADDRESS_MODE_BORDER: return MTLSamplerAddressModeClampToBorderColor;
        case D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE: return MTLSamplerAddressModeMirrorClampToEdge;
    }
    return MTLSamplerAddressModeRepeat;
}

class MetalBackend final : public Backend {
public:
    ~MetalBackend() override {
        if (ui_) ImGui_ImplMetal_Shutdown();
    }

    bool Init(uint32_t width, uint32_t height) override;
    void Clear(const State& s, uint32_t count, const uint8_t* rects, uint32_t flags, uint32_t color, float z,
               uint32_t stencil) override;
    void Draw(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices) override;
    void Copy(const Surface& src, const Surface& dst, uint32_t count, uint32_t src_rects, uint32_t dst_points) override;
    void Present(const Surface& front) override;
    void Sync(uint32_t data, uint32_t size) override;
    void SyncLate(uint32_t data) override {
        if (!DeliverLate(data)) Sync(data, 0);
        RecordLate(data);
    }
    bool DeliverLate(uint32_t data) override;
    void RecordLate(uint32_t data) override;
    void Flush() override {
        @autoreleasepool {
            id<MTLCommandBuffer> sent = Commit();
            if (!reports_.empty() && sent) [sent waitUntilCompleted];
            DeliverReports();
        }
    }
    void SetGamma(const uint8_t* ramp) override {
        std::memcpy(ramp_, ramp, sizeof(ramp_));
        ramp_dirty_ = true;
    }
    void ClearReport() override { counted_.clear(); }
    void GetReport(uint32_t report) override { reports_.push_back({report, counted_}); }

private:
    struct Query {
        uint32_t slot;
        uint64_t serial;  // the command buffer counting it, 0 while it is being recorded
        float samples_per_pixel;
    };
    struct Report {
        uint32_t va;
        std::vector<Query> queries;
    };
    struct Late {
        id<MTLBuffer> staging[kLateSlots];
        uint64_t serial[kLateSlots] = {};  // 0 free
        uint64_t frame[kLateSlots] = {};
        std::vector<uint8_t> cpu;  // rows packed
        uint64_t cpu_frame = 0, delivered = 0;
        uint32_t gw = 0, gh = 0, next = 0;
    };
    struct Alloc {
        id<MTLBuffer> buf;
        uint32_t offset = 0;
        uint8_t* p = nullptr;
    };
    struct PassFn {
        id<MTLFunction> vs, fs;
    };
    // built on a background thread, handed to the render thread
    struct ReadyShader {
        bool vertex;
        uint64_t key;
        id<MTLLibrary> lib;
        id<MTLFunction> main, point;
    };
    struct ReadyPipeline {
        uint64_t hash;
        Pipeline p;
    };
    struct RecordedPipeline {
        PsoKey k;
        uint32_t strides[16];
    };

    // recording
    id<MTLCommandBuffer> Cb();
    id<MTLCommandBuffer> Commit();  // what was recorded goes to the GPU; returns it
    void EndEncoder();
    void OpenPass(id<MTLTexture> colour, id<MTLTexture> depth);
    id<MTLBlitCommandEncoder> Blitter();  // ends the render pass; end it after use
    Alloc Reserve(uint32_t bytes, uint32_t align = 16);
    // targets
    uint64_t SizedTargetKey(uint32_t data, bool depth, uint32_t gw, uint32_t gh) const;
    Target* FindTarget(const Surface& s, bool depth);
    Target* EnsureTarget(const Surface& s, bool depth, uint32_t gw = 0, uint32_t gh = 0);
    Target* DepthTarget(const State& s);
    bool CreateTarget(Target& t);
    bool TargetStale(uint32_t data) const;
    bool BindTargets(const State& s);
    void Unbind() { bound_rt_ = bound_ds_ = nullptr; }
    void UploadTarget(Target& t, const Surface& s);
    id<MTLTexture> Shrink(Target& t, uint32_t w, uint32_t h);
    id<MTLTexture> PrepareCopySource(Target& t);
    bool ReadBack(id<MTLTexture> src, uint32_t w, uint32_t h, std::vector<uint8_t>& out);
    // full screen passes
    id<MTLLibrary> Library(const std::string& msl);
    PassFn PassFunctions(const char* hlsl, const char* vs, const char* fs);
    // the cache across runs
    void OpenCache();
    void Precompile(std::vector<std::pair<uint64_t, std::string>> vertex, std::vector<std::pair<uint64_t, std::string>> pixel,
                    std::vector<RecordedPipeline> pipelines, bool fill_archive);
    void TakeReady();
    void RecordShader(char kind, uint64_t key, const std::string& msl);
    void RecordPipeline(const PsoKey& k, const uint32_t strides[16]);
    MTLRenderPipelineDescriptor* PipelineDescriptor(const PsoKey& k, const uint32_t strides[16], id<MTLFunction> vs, id<MTLFunction> fs);
    id<MTLRenderPipelineState> PassPipeline(const PassFn& f, MTLPixelFormat colour, MTLPixelFormat depth, bool over);
    void SetPassTarget(id<MTLTexture> colour, id<MTLTexture> depth, uint32_t w, uint32_t h);
    void FullScreen(const PassFn& f, const float* cb, uint32_t cb_floats, id<MTLTexture> const* tex, uint32_t n_tex, bool over,
                    id<MTLDepthStencilState> depth, uint32_t stencil_ref = 0);
    void Blit(id<MTLTexture> src, const float uvrect[4], const BlitOptions& o);
    bool DepthCopy(const State& s);
    void ClearWhole(Target& t, uint32_t flags, const float colour[4], float z, uint32_t stencil);
    void ClearRects(Target& t, const std::vector<RECT>& rs, uint32_t flags, const float colour[4], float z, uint32_t stencil);
    void CopyByDraw(Target& src_t, Target& dst_t, uint32_t count, uint32_t src_rects, uint32_t dst_points);
    void CopyByRegion(Target& src_t, Target& dst_t, uint32_t count, uint32_t src_rects, uint32_t dst_points);
    // a draw's steps, in call order
    void EnsureVertexShader(const State& s, const Assembled& a, DrawShaders& sh);
    id<MTLTexture> BindTexture(const State& s, int stage, DrawTextures& stages);
    id<MTLTexture> BindTargetTexture(const State& s, int stage, const TextureDesc& d, Target& t, DrawTextures& stages);
    id<MTLTexture> BindGuestTexture(const State& s, int stage, const TextureDesc& d, const FormatInfo& fi, DrawTextures& stages);
    bool CreateTexture(Tex& tx, const TextureDesc& d, const FormatInfo& fi);
    void UploadLevel(Tex& tx, uint32_t level, uint32_t slice, uint32_t lw, uint32_t lh, uint32_t depth, const uint8_t* p,
                     uint32_t row, uint32_t image);
    void UploadTexture(Tex& tx, const TextureDesc& d, const uint8_t* p, const uint32_t* colours);
    id<MTLTexture> BindReplacement(const Replacement& r, const TextureDesc& d, uint32_t& levels);
    void CopyCubeTargets(Tex& tx, const TextureDesc& d);
    void EnsurePixelShader(const PixelKey& pk, DrawShaders& sh);
    Pipeline* EnsurePipeline(const PsoKey& key, const Assembled& a, const DrawShaders& sh);
    id<MTLDepthStencilState> DepthState(const PsoKey& k);
    id<MTLSamplerState> Sampler(const D3D12_SAMPLER_DESC& sd);
    void UploadConstants(const State& s, const DrawShaders& sh, const PsoKey& k, const DrawTextures& stages, GuestExtent size,
                         DrawConstants& c);
    id<MTLTexture> Dummy(const TexInfo& info);
    void DrawGeometry(const Assembled& a, const Pipeline& p);
    // visibility tests
    float SamplesPerPixel() const;
    void DeliverReports();
    // present
    void UploadGammaRamp();
    void RenderUi(id<MTLRenderCommandEncoder> enc, MTLRenderPassDescriptor* desc);
    void ReleaseOldTextures();
    void CountFps();
    void ApplySettings();
    void RescaleTargets();

    id<MTLDevice> dev_;
    id<MTLCommandQueue> queue_;
    CAMetalLayer* layer_ = nil;
    id<MTLCommandBuffer> cb_;
    id<MTLRenderCommandEncoder> enc_;
    id<MTLTexture> enc_colour_, enc_depth_;
    id<MTLBuffer> rings_[kFramesInFlight];
    id<MTLBuffer> ring_;  // this frame's, or a bigger one when it filled
    uint32_t ring_used_ = 0;
    int frame_slot_ = 0;
    dispatch_semaphore_t frames_ = nullptr;
    std::atomic<uint64_t> completed_serial_{0};
    uint64_t next_serial_ = 1;
    id<MTLBuffer> vis_;
    uint32_t vis_next_ = 0;
    // constants: the block in the ring holding what the last draw used
    std::vector<uint8_t> const_last_;
    Alloc const_at_;
    bool const_bound_ = false;
    // set in the open encoder
    id<MTLRenderPipelineState> enc_pipeline_;
    id<MTLDepthStencilState> enc_depth_state_;
    MTLScissorRect enc_scissor_{};
    bool enc_has_scissor_ = false;
    MTLViewport enc_viewport_{};
    bool enc_has_viewport_ = false;

    id<MTLTexture> lut_;
    id<MTLSamplerState> pass_samplers_[4];
    id<MTLDepthStencilState> no_depth_, depth_always_;
    id<MTLTexture> dummy_2d_, dummy_3d_, dummy_cube_, dummy_depth_;
    std::unordered_map<std::string, id<MTLLibrary>> libraries_;
    std::unordered_map<uint64_t, id<MTLRenderPipelineState>> pass_pipelines_;
    std::unordered_map<uint64_t, id<MTLLibrary>> vs_libs_, ps_libs_;  // by key
    std::unordered_map<uint64_t, Pipeline> pipelines_;
    std::unordered_map<uint64_t, id<MTLDepthStencilState>> depth_states_;
    std::unordered_map<uint64_t, id<MTLSamplerState>> samplers_;
    std::unordered_map<uint64_t, Target> targets_;
    std::unordered_map<uint32_t, uint64_t> target_used_;  // target address to the frame it was last used
    std::unordered_map<uint64_t, Tex> textures_;          // by data address, a P8 texture's with its palette's above
    std::unordered_map<uint32_t, uint32_t> reloads_;      // targets reloaded from guest memory, for the log
    std::unordered_map<uint32_t, Late> late_;
    std::mutex late_lock_;                    // DeliverLate runs on the title's thread
    std::atomic<uint64_t> presents_seen_{0};  // presents_ for DeliverLate
    std::vector<Query> counted_;              // since the last ClearReport
    std::deque<Report> reports_;
    Assembled asm_;  // kept between draws for its buffers
    bool draw_2d_ = false;
    Target* bound_rt_ = nullptr;
    Target* bound_ds_ = nullptr;
    Target* draw_rt_ = nullptr;  // the draw's targets, the bound ones go when a texture's copy is drawn
    Target* draw_ds_ = nullptr;
    uint32_t guest_w_ = 640, guest_h_ = 480;
    float scale_ = 1;
    float aspect_ = 0;  // app::FrameAspect() the targets were made for
    uint8_t ramp_[768] = {};
    bool ramp_dirty_ = false;
    bool gamma_ = false;
    bool ui_ = false;
    uint64_t presents_ = 0;
    double fps_ = 0;
    std::chrono::steady_clock::time_point fps_t_ = std::chrono::steady_clock::now();
    uint64_t fps_n_ = 0;
    std::unordered_map<std::string, PassFn> pass_functions_;
    std::mutex ready_lock_;
    std::vector<ReadyShader> ready_shaders_;
    std::vector<ReadyPipeline> ready_pipelines_;
    std::atomic<bool> precompiling_{false};
    CacheFiles cache_;
    // the time a frame spent, for the hitch lines
    struct FrameCost {
        double shaders_ms = 0, pipelines_ms = 0, uploads_ms = 0, readback_ms = 0, drawable_ms = 0;
        uint32_t shaders = 0, pipelines = 0, uploads = 0, readbacks = 0;
    } cost_;
    std::chrono::steady_clock::time_point last_present_ = std::chrono::steady_clock::now();
    uint32_t hitch_lines_ = 0;
    std::unordered_map<uint64_t, id<MTLFunction>> fn_vs_, fn_vp_, fn_ps_;
};

// ---- device and fixed objects

bool MetalBackend::Init(uint32_t width, uint32_t height) {
    @autoreleasepool {
        guest_w_ = width;
        guest_h_ = height;
        aspect_ = app::FrameAspect();
        scale_ = app::RenderScale(guest_h_);
        layer_ = (__bridge CAMetalLayer*)app::MetalLayer();
        dev_ = layer_.device ?: MTLCreateSystemDefaultDevice();
        if (!dev_) {
            LOGE("metal: no device");
            return false;
        }
        layer_.device = dev_;
        layer_.pixelFormat = kBgra;
        layer_.framebufferOnly = NO;
        layer_.maximumDrawableCount = 3;
        queue_ = [dev_ newCommandQueue];
        frames_ = dispatch_semaphore_create(kFramesInFlight);
        for (auto& r : rings_) r = [dev_ newBufferWithLength:kRingBytes options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
        ring_ = rings_[0];
        vis_ = [dev_ newBufferWithLength:kVisSlots * 8 options:MTLResourceStorageModeShared];
        for (int i = 0; i < 4; i++) {
            MTLSamplerDescriptor* sd = [MTLSamplerDescriptor new];
            sd.minFilter = sd.magFilter = i == 0 ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
            sd.mipFilter = i == 0 ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
            pass_samplers_[i] = [dev_ newSamplerStateWithDescriptor:sd];
        }
        MTLDepthStencilDescriptor* dd = [MTLDepthStencilDescriptor new];
        no_depth_ = [dev_ newDepthStencilStateWithDescriptor:dd];
        dd.depthWriteEnabled = YES;
        depth_always_ = [dev_ newDepthStencilStateWithDescriptor:dd];
        MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kBgra width:256 height:1 mipmapped:NO];
        lut_ = [dev_ newTextureWithDescriptor:td];
        // what a stage with nothing bound reads: zeros
        const uint32_t zero[6] = {};
        td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kBgra width:1 height:1 mipmapped:NO];
        dummy_2d_ = [dev_ newTextureWithDescriptor:td];
        [dummy_2d_ replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:zero bytesPerRow:4];
        td.textureType = MTLTextureType3D;
        dummy_3d_ = [dev_ newTextureWithDescriptor:td];
        [dummy_3d_ replaceRegion:MTLRegionMake3D(0, 0, 0, 1, 1, 1) mipmapLevel:0 slice:0 withBytes:zero bytesPerRow:4 bytesPerImage:4];
        td = [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:kBgra size:1 mipmapped:NO];
        dummy_cube_ = [dev_ newTextureWithDescriptor:td];
        for (NSUInteger f = 0; f < 6; f++)
            [dummy_cube_ replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 slice:f withBytes:zero bytesPerRow:4 bytesPerImage:4];
        td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kDepth width:1 height:1 mipmapped:NO];
        td.storageMode = MTLStorageModePrivate;
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        dummy_depth_ = [dev_ newTextureWithDescriptor:td];
        ui_ = ImGui_ImplMetal_Init(dev_);
        OpenCache();
        LOGR("metal: drawing with %s, %ux%u guest, render scale %.3f", dev_.name.UTF8String, width, height, scale_);
        return true;
    }
}

// ---- recording

id<MTLCommandBuffer> MetalBackend::Cb() {
    if (!cb_) cb_ = [queue_ commandBuffer];
    return cb_;
}

void MetalBackend::EndEncoder() {
    if (!enc_) return;
    [enc_ endEncoding];
    enc_ = nil;
    enc_colour_ = enc_depth_ = nil;
    enc_pipeline_ = nil;
    enc_depth_state_ = nil;
    enc_has_scissor_ = enc_has_viewport_ = const_bound_ = false;
}

id<MTLCommandBuffer> MetalBackend::Commit() {
    EndEncoder();
    id<MTLCommandBuffer> sent = cb_;
    if (!sent) return nil;
    uint64_t serial = next_serial_++;
    for (Query& q : counted_)
        if (!q.serial) q.serial = serial;
    for (Report& r : reports_)
        for (Query& q : r.queries)
            if (!q.serial) q.serial = serial;
    auto* done = &completed_serial_;
    [sent addCompletedHandler:^(id<MTLCommandBuffer>) {
        uint64_t seen = done->load();
        while (seen < serial && !done->compare_exchange_weak(seen, serial)) {}
    }];
    [sent commit];
    cb_ = nil;
    return sent;
}

void MetalBackend::OpenPass(id<MTLTexture> colour, id<MTLTexture> depth) {
    if (enc_ && enc_colour_ == colour && enc_depth_ == depth) return;
    EndEncoder();
    MTLRenderPassDescriptor* d = [MTLRenderPassDescriptor renderPassDescriptor];
    if (colour) {
        d.colorAttachments[0].texture = colour;
        d.colorAttachments[0].loadAction = MTLLoadActionLoad;
        d.colorAttachments[0].storeAction = MTLStoreActionStore;
    }
    if (depth) {
        d.depthAttachment.texture = depth;
        d.depthAttachment.loadAction = MTLLoadActionLoad;
        d.depthAttachment.storeAction = MTLStoreActionStore;
        d.stencilAttachment.texture = depth;
        d.stencilAttachment.loadAction = MTLLoadActionLoad;
        d.stencilAttachment.storeAction = MTLStoreActionStore;
    }
    d.visibilityResultBuffer = vis_;
    enc_ = [Cb() renderCommandEncoderWithDescriptor:d];
    enc_colour_ = colour;
    enc_depth_ = depth;
}

id<MTLBlitCommandEncoder> MetalBackend::Blitter() {
    EndEncoder();
    return [Cb() blitCommandEncoder];
}

MetalBackend::Alloc MetalBackend::Reserve(uint32_t bytes, uint32_t align) {
    uint32_t at = (ring_used_ + align - 1) & ~(align - 1);
    if (uint64_t(at) + bytes > ring_.length) {
        // this frame's ring is full: a bigger or fresh one for the rest of it, kept alive by the command buffers using it
        ring_ = [dev_ newBufferWithLength:std::max(kRingBytes, bytes) options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
        at = 0;
    }
    ring_used_ = at + bytes;
    return {ring_, at, static_cast<uint8_t*>(ring_.contents) + at};
}

// ---- targets

uint64_t MetalBackend::SizedTargetKey(uint32_t data, bool depth, uint32_t gw, uint32_t gh) const {
    uint64_t key = TargetMapKey(data, depth);
    bool small_depth = depth && (gw != guest_w_ || gh != guest_h_);
    if (!small_depth) return key;
    return key | uint64_t(gw) << kKeyWidthShift | uint64_t(gh) << kKeyHeightShift;
}

Target* MetalBackend::FindTarget(const Surface& s, bool depth) {
    if (!s.header) return nullptr;
    uint32_t data = mem::VirtualFromPhysical(s.data);
    uint32_t gw, gh;
    SurfaceSize(s, gw, gh);
    auto it = targets_.find(SizedTargetKey(data, depth, gw, gh));
    if (it == targets_.end() || it->second.gw != gw || it->second.gh != gh) return nullptr;
    target_used_[data] = presents_;
    return &it->second;
}

bool MetalBackend::CreateTarget(Target& t) {
    t.w = std::max(1u, uint32_t(std::lround(t.gw * scale_ * XStretch(aspect_, guest_h_, t.gw, t.gh))));
    t.h = std::max(1u, uint32_t(std::lround(t.gh * scale_)));
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:t.depth ? kDepth : kBgra width:t.w height:t.h mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    t.res = [dev_ newTextureWithDescriptor:td];
    if (!t.res) return false;
    // as a new resource reads in D3D: cleared, depth to the far plane
    EndEncoder();
    MTLRenderPassDescriptor* d = [MTLRenderPassDescriptor renderPassDescriptor];
    if (t.depth) {
        d.depthAttachment.texture = t.res;
        d.depthAttachment.loadAction = MTLLoadActionClear;
        d.depthAttachment.storeAction = MTLStoreActionStore;
        d.depthAttachment.clearDepth = 1;
        d.stencilAttachment.texture = t.res;
        d.stencilAttachment.loadAction = MTLLoadActionClear;
        d.stencilAttachment.storeAction = MTLStoreActionStore;
    } else {
        d.colorAttachments[0].texture = t.res;
        d.colorAttachments[0].loadAction = MTLLoadActionClear;
        d.colorAttachments[0].storeAction = MTLStoreActionStore;
        d.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
    }
    [[Cb() renderCommandEncoderWithDescriptor:d] endEncoding];
    return true;
}

Target* MetalBackend::EnsureTarget(const Surface& s, bool depth, uint32_t gw, uint32_t gh) {
    if (!s.header) return nullptr;
    uint32_t data = mem::VirtualFromPhysical(s.data);
    uint32_t sw, sh;
    SurfaceSize(s, sw, sh);
    if (gw && gh) sw = gw, sh = gh;
    uint64_t key = SizedTargetKey(data, depth, sw, sh);
    target_used_[data] = presents_;
    auto it = targets_.find(key);
    if (it != targets_.end()) {
        if (it->second.gw == sw && it->second.gh == sh) return &it->second;
        if (bound_rt_ == &it->second || bound_ds_ == &it->second) Unbind();
        targets_.erase(it);
    }
    Target t;
    t.depth = depth;
    t.gw = sw;
    t.gh = sh;
    if (!CreateTarget(t)) {
        LOG_FIRST(8, "metal: target %08X %ux%u could not be made", data, sw, sh);
        return nullptr;
    }
    t.data = data;
    TextureDesc d;
    t.linear32 = depth || (Describe(s.header, d) && d.linear && Format(d.format).bits == 32);
    t.swizzled32 = !depth && !t.linear32 && Describe(s.header, d) && Format(d.format).swizzled &&
                   !Format(d.format).compressed && Format(d.format).bits == 32;
    t.bytes = depth ? sw * sh * 4 : GuestPitch(sw) * sh;
    t.epoch = DirtyEpoch(data);
    t.sig = SurfaceSig(data, t.bytes);
    return &(targets_[key] = std::move(t));
}

// a smaller colour target drawn with a bigger depth buffer uses its top left part: a depth target of its size stands in
Target* MetalBackend::DepthTarget(const State& s) {
    uint32_t rw = 0, rh = 0, zw = 0, zh = 0;
    if (s.rt.header && s.zs.header) {
        SurfaceSize(s.rt, rw, rh);
        SurfaceSize(s.zs, zw, zh);
        if (rw <= zw && rh <= zh && (rw != zw || rh != zh)) return EnsureTarget(s.zs, true, rw, rh);
    }
    return EnsureTarget(s.zs, true);
}

bool MetalBackend::TargetStale(uint32_t data) const {
    auto it = target_used_.find(data);
    uint64_t used = it == target_used_.end() ? 0 : it->second;
    return presents_ - used > kStaleFrames;
}

bool MetalBackend::BindTargets(const State& s) {
    bool depth = (s.rs[kRsZEnable] || s.rs[kRsStencilEnable]) && !DepthOnlySampled(s);
    // the colour target on the depth buffer's own memory: what the console writes there alone (d3d11.cpp BindTargets)
    bool shared = s.rt.header && s.zs.header && s.rt.data == s.zs.data;
    bool colour = SimpleState(s, kMColorWriteEnable) != 0;
    Target* rt = shared && depth && !colour ? nullptr : EnsureTarget(s.rt, false);
    Target* ds = depth && !(shared && colour) ? DepthTarget(s) : nullptr;
    if (rt && ds && (rt->w != ds->w || rt->h != ds->h)) ds = nullptr;
    if (!rt && !ds) return false;
    if (rt && LatchCpuWrite(*rt)) {
        if (++reloads_[rt->data] <= 3)
            LOGR("metal: target %08X %ux%u reloaded from guest memory before drawing (frame %llu)", rt->data, rt->gw, rt->gh,
                 (unsigned long long)presents_);
        UploadTarget(*rt, s.rt);
    }
    if (rt) rt->drawn = true;
    if (ds) ds->drawn = true;
    bound_rt_ = rt;
    bound_ds_ = ds;
    return true;
}

// CPU pixels of a colour surface into its target, scaled
void MetalBackend::UploadTarget(Target& t, const Surface& s) {
    TextureDesc d;
    if (!Describe(s.header, d) || Format(d.format).depth) return;
    std::vector<uint8_t> px;
    uint32_t pitch;
    // after a Sync only the pixels the CPU changed go over the picture, which stays at render scale
    bool over = !t.drawn && d.width == t.gw && d.height == t.gh && t.read.size() == size_t(t.gw) * t.gh * 4;
    if (over) {
        pitch = t.gw * 4;
        px = LatchCpuChanges(t);
    } else {
        DecodeLevel(d, 0, 0, static_cast<const uint8_t*>(kernel::HostPtr(d.data)), nullptr, px, pitch);
    }
    if (px.size() < size_t(pitch) * d.height) px.resize(size_t(pitch) * d.height);
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kBgra width:d.width height:d.height mipmapped:NO];
    id<MTLTexture> tex = [dev_ newTextureWithDescriptor:td];
    if (!tex) return;
    [tex replaceRegion:MTLRegionMake2D(0, 0, d.width, d.height) mipmapLevel:0 withBytes:px.data() bytesPerRow:pitch];
    SetPassTarget(t.res, nil, t.w, t.h);
    float uv[4] = {0, 0, 1, 1};
    Blit(tex, uv, {.alpha = true, .over = over});
    t.epoch = DirtyEpoch(t.data);
    t.sig = SurfaceSig(t.data, t.bytes);
}

id<MTLTexture> MetalBackend::Shrink(Target& t, uint32_t w, uint32_t h) {
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kBgra width:w height:h mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    id<MTLTexture> down = [dev_ newTextureWithDescriptor:td];
    if (!down) return nil;
    SetPassTarget(down, nil, w, h);
    float uv[4] = {0, 0, 1, 1};
    Blit(t.res, uv, {.alpha = true});
    return down;
}

id<MTLTexture> MetalBackend::PrepareCopySource(Target& t) {
    if (t.w != t.gw || t.h != t.gh) return Shrink(t, t.gw, t.gh);
    return t.res;
}

// a BGRA texture to the CPU, rows packed; waits for the GPU
bool MetalBackend::ReadBack(id<MTLTexture> src, uint32_t w, uint32_t h, std::vector<uint8_t>& out) {
    const auto start = std::chrono::steady_clock::now();
    id<MTLBuffer> st = [dev_ newBufferWithLength:size_t(w) * h * 4 options:MTLResourceStorageModeShared];
    if (!st) return false;
    id<MTLBlitCommandEncoder> b = Blitter();
    [b copyFromTexture:src sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1)
              toBuffer:st destinationOffset:0 destinationBytesPerRow:w * 4 destinationBytesPerImage:size_t(w) * h * 4];
    [b endEncoding];
    id<MTLCommandBuffer> sent = Commit();
    [sent waitUntilCompleted];
    out.resize(size_t(w) * h * 4);
    std::memcpy(out.data(), st.contents, out.size());
    cost_.readbacks++;
    cost_.readback_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return true;
}

// the CPU reads a surface the GPU drew: bring the pixels back
void MetalBackend::Sync(uint32_t data, uint32_t) {
    @autoreleasepool {
        auto it = targets_.find(TargetMapKey(data, false));
        if (it == targets_.end() || !it->second.drawn || TargetStale(data)) return;
        Target& t = it->second;
        if (!t.linear32 && !t.swizzled32) {
            LOG_FIRST(8, "metal: read-back of %08X skipped, not a 32 bit surface", data);
            return;
        }
        id<MTLTexture> src = PrepareCopySource(t);
        std::vector<uint8_t> px;
        if (!src || !ReadBack(src, t.gw, t.gh, px)) return;
        uint32_t pitch = GuestPitch(t.gw);
        uint8_t* guest = static_cast<uint8_t*>(kernel::HostPtr(data));
        if (t.swizzled32) {
            for (uint32_t y = 0; y < t.gh; y++)
                for (uint32_t x = 0; x < t.gw; x++)
                    std::memcpy(guest + size_t(SwizzledOffset(x, y, t.gw, t.gh)) * 4, &px[(size_t(y) * t.gw + x) * 4], 4);
        } else {
            for (uint32_t y = 0; y < t.gh; y++) std::memcpy(guest + size_t(y) * pitch, &px[size_t(y) * t.gw * 4], t.gw * 4);
        }
        t.read = std::move(px);
        t.drawn = false;
        t.sig = SurfaceSig(t.data, t.bytes);
    }
}

// A NOOVERWRITE lock: the console reads whatever the GPU has finished, a frame or two old (d3d11.cpp DeliverLate)
bool MetalBackend::DeliverLate(uint32_t data) {
    std::lock_guard lock(late_lock_);
    auto it = late_.find(data);
    if (it == late_.end() || it->second.cpu.empty()) return false;
    Late& l = it->second;
    if (presents_seen_.load() - l.cpu_frame > uint64_t(kLateSlots) + 1) return false;
    if (l.cpu_frame > l.delivered) {
        uint32_t pitch = GuestPitch(l.gw);
        uint8_t* dst = static_cast<uint8_t*>(kernel::HostPtr(data));
        for (uint32_t y = 0; y < l.gh; y++) std::memcpy(dst + size_t(y) * pitch, &l.cpu[size_t(y) * l.gw * 4], l.gw * 4);
        l.delivered = l.cpu_frame;
    }
    return true;
}

void MetalBackend::RecordLate(uint32_t data) {
    @autoreleasepool {
        auto it = targets_.find(TargetMapKey(data, false));
        if (it == targets_.end() || TargetStale(data) || !it->second.linear32) return;
        Target& t = it->second;
        t.sig = SurfaceSig(t.data, t.bytes);
        t.read.clear();
        std::lock_guard lock(late_lock_);
        Late& l = late_[data];
        if (l.gw != t.gw || l.gh != t.gh) {
            l = Late{};
            l.gw = t.gw;
            l.gh = t.gh;
        }
        // finished copies to the CPU, the newest kept
        for (int i = 0; i < kLateSlots; i++) {
            if (!l.serial[i] || completed_serial_.load() < l.serial[i]) continue;
            if (l.frame[i] >= l.cpu_frame) {
                l.cpu.resize(size_t(l.gw) * l.gh * 4);
                std::memcpy(l.cpu.data(), l.staging[i].contents, l.cpu.size());
                l.cpu_frame = l.frame[i];
            }
            l.serial[i] = 0;
        }
        // a copy for a later lock, unless every slot is still in flight
        uint32_t slot = l.next;
        if (l.serial[slot]) return;
        if (!l.staging[slot]) l.staging[slot] = [dev_ newBufferWithLength:size_t(l.gw) * l.gh * 4 options:MTLResourceStorageModeShared];
        id<MTLTexture> src = PrepareCopySource(t);
        if (!src || !l.staging[slot]) return;
        id<MTLBlitCommandEncoder> b = Blitter();
        [b copyFromTexture:src sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(l.gw, l.gh, 1)
                  toBuffer:l.staging[slot] destinationOffset:0 destinationBytesPerRow:l.gw * 4 destinationBytesPerImage:size_t(l.gw) * l.gh * 4];
        [b endEncoding];
        l.serial[slot] = next_serial_;  // the command buffer being recorded
        l.frame[slot] = presents_ + 1;
        l.next = (slot + 1) % kLateSlots;
    }
}

// ---- full screen passes

id<MTLLibrary> MetalBackend::Library(const std::string& msl) {
    auto it = libraries_.find(msl);
    if (it != libraries_.end()) return it->second;
    const auto start = std::chrono::steady_clock::now();
    NSError* err = nil;
    id<MTLLibrary> lib = [dev_ newLibraryWithSource:[NSString stringWithUTF8String:msl.c_str()] options:ShaderOptions() error:&err];
    g_shaders_compiled++;
    cost_.shaders++;
    cost_.shaders_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (!lib) {
        LOGE("metal: shader failed: %s", err.localizedDescription.UTF8String);
        LOGE("%s", msl.c_str());
    }
    libraries_[msl] = lib;
    return lib;
}

// every source's two functions made once: a pass runs many times a frame
MetalBackend::PassFn MetalBackend::PassFunctions(const char* hlsl, const char* vs, const char* fs) {
    std::string key = std::to_string(reinterpret_cast<uintptr_t>(hlsl)) + "/" + vs + "/" + fs;
    auto it = pass_functions_.find(key);
    if (it != pass_functions_.end()) return it->second;
    id<MTLLibrary> lib = Library(PassMsl(hlsl, {{vs, true}, {fs, false}}));
    PassFn f{[lib newFunctionWithName:@(vs)], [lib newFunctionWithName:@(fs)]};
    pass_functions_[key] = f;
    return f;
}

// ---- the cache across runs

void MetalBackend::OpenCache() {
    if (std::getenv("XBE_NO_SHADER_CACHE")) {
        LOGR("metal: shader cache off (XBE_NO_SHADER_CACHE)");
        return;
    }
    namespace fs = std::filesystem;
    const fs::path dir = app::ExeDir() / "metalcache";
    std::error_code ec;
    fs::create_directories(dir, ec);
    const fs::path file = dir / "records.bin", archive_file = dir / "pipelines.bin";
    const uint64_t identity = BuildIdentity();
    std::vector<std::pair<uint64_t, std::string>> vertex, pixel;
    std::vector<RecordedPipeline> pipelines;
    std::string all;
    if (std::ifstream in{file, std::ios::binary}) all.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    uint64_t was = 0;
    size_t good = 0;
    if (all.size() >= 16 && !std::memcmp(all.data(), kCacheMagic, 8)) std::memcpy(&was, all.data() + 8, 8);
    if (was == identity) {
        size_t at = good = 16;
        while (at + 13 <= all.size()) {
            const char kind = all[at];
            uint64_t key;
            uint32_t n;
            std::memcpy(&key, all.data() + at + 1, 8);
            std::memcpy(&n, all.data() + at + 9, 4);
            if (at + 13 + n > all.size()) break;
            const char* body = all.data() + at + 13;
            if (kind == 'V') vertex.emplace_back(key, std::string(body, n));
            else if (kind == 'P') pixel.emplace_back(key, std::string(body, n));
            else if (kind == 'L' && n == sizeof(RecordedPipeline)) {
                pipelines.emplace_back();
                std::memcpy(&pipelines.back(), body, n);
            }
            good = at += 13 + n;
        }
    } else {
        fs::remove(archive_file, ec);  // made from other shaders
    }
    if (good == 0) {
        if (FILE* f = std::fopen(file.string().c_str(), "wb")) {
            std::fwrite(kCacheMagic, 1, 8, f);
            std::fwrite(&identity, 8, 1, f);
            std::fclose(f);
        }
    } else if (good != all.size()) {
        fs::resize_file(file, good, ec);  // a run cut off in the middle of a record
    }
    cache_.records = std::fopen(file.string().c_str(), "ab");
    cache_.archive_queue = dispatch_queue_create("metal pipeline archive", DISPATCH_QUEUE_SERIAL);
    cache_.saved = std::chrono::steady_clock::now();
    cache_.archive_url = [NSURL fileURLWithPath:@(archive_file.c_str())];
    const bool had_archive = fs::exists(archive_file, ec);
    MTLBinaryArchiveDescriptor* ad = [MTLBinaryArchiveDescriptor new];
    if (had_archive) ad.url = cache_.archive_url;
    NSError* err = nil;
    cache_.archive = [dev_ newBinaryArchiveWithDescriptor:ad error:&err];
    bool fresh_archive = !had_archive;
    if (!cache_.archive && had_archive) {  // not readable: start it again
        LOGW("metal: pipeline archive not read: %s", err.localizedDescription.UTF8String);
        fs::remove(archive_file, ec);
        cache_.archive = [dev_ newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&err];
        fresh_archive = true;
    }
    g_cache_files = &cache_;
    LOGR("metal: cache has %zu vertex shaders, %zu pixel shaders, %zu pipelines from earlier runs%s", vertex.size(), pixel.size(),
         pipelines.size(), was == identity || all.empty() ? "" : " (the shader generators changed: older ones dropped)");
    if (vertex.empty() && pixel.empty()) return;
    precompiling_ = true;
    std::thread([this, vertex = std::move(vertex), pixel = std::move(pixel), pipelines = std::move(pipelines), fresh_archive]() mutable {
        Precompile(std::move(vertex), std::move(pixel), std::move(pipelines), fresh_archive);
    }).detach();
}

// on its own thread, the work spread over most of the cores; a game thread may be drawing the same shaders meanwhile
void MetalBackend::Precompile(std::vector<std::pair<uint64_t, std::string>> vertex, std::vector<std::pair<uint64_t, std::string>> pixel,
                              std::vector<RecordedPipeline> pipelines, bool fill_archive) {
    @autoreleasepool {
        const auto start = std::chrono::steady_clock::now();
        // blocks copy what they capture: the shared things by pointer
        auto* built_lock = new std::mutex;
        auto* vs = new std::unordered_map<uint64_t, id<MTLFunction>>;
        auto* vp = new std::unordered_map<uint64_t, id<MTLFunction>>;
        auto* ps = new std::unordered_map<uint64_t, id<MTLFunction>>;
        dispatch_queue_t work = dispatch_get_global_queue(QOS_CLASS_UTILITY, 0);
        MTLCompileOptions* opt = ShaderOptions();
        auto shaders = [&](const std::vector<std::pair<uint64_t, std::string>>& source, bool is_vertex) {
            const auto* list = &source;
            dispatch_apply(source.size(), work, ^(size_t i) {
                @autoreleasepool {
                    NSError* err = nil;
                    id<MTLLibrary> lib = [dev_ newLibraryWithSource:[NSString stringWithUTF8String:(*list)[i].second.c_str()] options:opt error:&err];
                    if (!lib) return;
                    ReadyShader r{is_vertex, (*list)[i].first, lib, [lib newFunctionWithName:is_vertex ? @"vmain" : @"fmain"],
                                  is_vertex ? [lib newFunctionWithName:@"vpoint"] : nil};
                    std::lock_guard lock(*built_lock);
                    (*(is_vertex ? vs : ps))[r.key] = r.main;
                    if (is_vertex) (*vp)[r.key] = r.point;
                    std::lock_guard ready(ready_lock_);
                    ready_shaders_.push_back(std::move(r));
                }
            });
        };
        shaders(vertex, true);
        shaders(pixel, false);
        const double shaders_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        auto* made = new std::atomic<uint32_t>{0};
        CacheFiles* files = &cache_;
        const RecordedPipeline* recorded = pipelines.data();
        dispatch_apply(pipelines.size(), work, ^(size_t i) {
            @autoreleasepool {
                const RecordedPipeline& r = recorded[i];
                const bool points = r.k.topo == uint32_t(Topology::Points);
                id<MTLFunction> vf, pf;
                {
                    std::lock_guard lock(*built_lock);
                    auto& map = *(points ? vp : vs);
                    if (auto it = map.find(r.k.vs); it != map.end()) vf = it->second;
                    if (auto it = ps->find(r.k.ps); it != ps->end()) pf = it->second;
                }
                if (!vf || !pf) return;
                MTLRenderPipelineDescriptor* d = PipelineDescriptor(r.k, r.strides, vf, pf);
                NSError* err = nil;
                id<MTLRenderPipelineState> state = [dev_ newRenderPipelineStateWithDescriptor:d error:&err];
                if (!state) return;
                if (fill_archive && files->archive)
                    dispatch_async(files->archive_queue, ^{
                        [files->archive addRenderPipelineFunctionsWithDescriptor:d error:nil];
                        files->dirty = true;
                    });
                ReadyPipeline ready{Hash(r.strides, sizeof(r.strides), Hash(&r.k, sizeof(r.k))), {state, points}};
                std::lock_guard lock(ready_lock_);
                ready_pipelines_.push_back(std::move(ready));
                (*made)++;
            }
        });
        precompiling_ = false;
        delete built_lock;
        LOGR("metal: cache built %zu shaders in %.1f s, %u of %zu pipelines in %.1f s more", vertex.size() + pixel.size(), shaders_s, made->load(),
             pipelines.size(), std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() - shaders_s);
    }
}

// what the background thread has finished, into the maps only the render thread touches
void MetalBackend::TakeReady() {
    std::vector<ReadyShader> shaders;
    std::vector<ReadyPipeline> pipes;
    {
        std::lock_guard lock(ready_lock_);
        shaders.swap(ready_shaders_);
        pipes.swap(ready_pipelines_);
    }
    for (ReadyShader& r : shaders) {
        if (r.vertex) {
            if (vs_libs_.emplace(r.key, r.lib).second) {
                fn_vs_[r.key] = r.main;
                fn_vp_[r.key] = r.point;
            }
        } else if (ps_libs_.emplace(r.key, r.lib).second) {
            fn_ps_[r.key] = r.main;
        }
    }
    for (ReadyPipeline& r : pipes) pipelines_.emplace(r.hash, std::move(r.p));
}

void MetalBackend::RecordShader(char kind, uint64_t key, const std::string& msl) {
    if (!cache_.records) return;
    const uint32_t n = uint32_t(msl.size());
    std::fwrite(&kind, 1, 1, cache_.records);
    std::fwrite(&key, 8, 1, cache_.records);
    std::fwrite(&n, 4, 1, cache_.records);
    std::fwrite(msl.data(), 1, n, cache_.records);
    std::fflush(cache_.records);
}

void MetalBackend::RecordPipeline(const PsoKey& k, const uint32_t strides[16]) {
    if (!cache_.records) return;
    RecordedPipeline r{};
    r.k = k;
    std::memcpy(r.strides, strides, sizeof(r.strides));
    const char kind = 'L';
    const uint64_t key = 0;
    const uint32_t n = sizeof(r);
    std::fwrite(&kind, 1, 1, cache_.records);
    std::fwrite(&key, 8, 1, cache_.records);
    std::fwrite(&n, 4, 1, cache_.records);
    std::fwrite(&r, 1, n, cache_.records);
    std::fflush(cache_.records);
}

id<MTLRenderPipelineState> MetalBackend::PassPipeline(const PassFn& f, MTLPixelFormat colour, MTLPixelFormat depth, bool over) {
    if (!f.vs || !f.fs) return nil;
    uint64_t key = uint64_t((__bridge void*)f.vs) * 31 + uint64_t((__bridge void*)f.fs);
    (void)key;
    std::string name = std::string(f.vs.name.UTF8String) + "/" + f.fs.name.UTF8String;
    key = Hash(name.data(), name.size(), uint64_t(colour) * 1000 + uint64_t(depth) * 3 + (over ? 1 : 0));
    auto it = pass_pipelines_.find(key);
    if (it != pass_pipelines_.end()) return it->second;
    MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = f.vs;
    d.fragmentFunction = f.fs;
    d.colorAttachments[0].pixelFormat = colour;
    if (over) {
        auto* c = d.colorAttachments[0];
        c.blendingEnabled = YES;
        c.sourceRGBBlendFactor = MTLBlendFactorOne;
        c.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        c.sourceAlphaBlendFactor = MTLBlendFactorZero;
        c.destinationAlphaBlendFactor = MTLBlendFactorOne;
    }
    d.depthAttachmentPixelFormat = depth;
    if (depth == kDepth) d.stencilAttachmentPixelFormat = depth;
    NSError* err = nil;
    id<MTLRenderPipelineState> state = [dev_ newRenderPipelineStateWithDescriptor:d error:&err];
    if (!state) LOGE("metal: pass pipeline %s failed: %s", name.c_str(), err.localizedDescription.UTF8String);
    return pass_pipelines_[key] = state;
}

// one colour or depth target for a pass, the draw's bound targets forgotten
void MetalBackend::SetPassTarget(id<MTLTexture> colour, id<MTLTexture> depth, uint32_t w, uint32_t h) {
    OpenPass(colour, depth);
    MTLViewport vp{0, 0, double(w), double(h), 0, 1};
    [enc_ setViewport:vp];
    enc_has_viewport_ = false;
    MTLScissorRect sc{0, 0, w, h};
    [enc_ setScissorRect:sc];
    enc_has_scissor_ = false;
    Unbind();
}

void MetalBackend::FullScreen(const PassFn& f, const float* cb, uint32_t cb_floats, id<MTLTexture> const* tex, uint32_t n_tex, bool over,
                              id<MTLDepthStencilState> depth, uint32_t stencil_ref) {
    MTLPixelFormat cf = enc_colour_ ? enc_colour_.pixelFormat : MTLPixelFormatInvalid;
    MTLPixelFormat df = enc_depth_ ? enc_depth_.pixelFormat : MTLPixelFormatInvalid;
    id<MTLRenderPipelineState> state = PassPipeline(f, cf, df, over);
    if (!state) return;
    [enc_ setRenderPipelineState:state];
    enc_pipeline_ = state;
    [enc_ setDepthStencilState:depth];
    enc_depth_state_ = depth;
    [enc_ setStencilReferenceValue:stencil_ref];
    [enc_ setCullMode:MTLCullModeNone];
    [enc_ setTriangleFillMode:MTLTriangleFillModeFill];
    [enc_ setDepthBias:0 slopeScale:0 clamp:0];
    if (cb) {
        [enc_ setVertexBytes:cb length:cb_floats * sizeof(float) atIndex:0];
        [enc_ setFragmentBytes:cb length:cb_floats * sizeof(float) atIndex:0];
        const_bound_ = false;
    }
    for (int i = 0; i < 4; i++) [enc_ setFragmentSamplerState:pass_samplers_[i] atIndex:i];
    for (uint32_t i = 0; i < n_tex; i++) [enc_ setFragmentTexture:tex[i] atIndex:i];
    [enc_ drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
}

// src over the bound viewport and scissor
void MetalBackend::Blit(id<MTLTexture> src, const float uvrect[4], const BlitOptions& o) {
    const app::Settings& st = app::Get();
    const bool final = o.final && (st.fxaa || st.sharpen || st.scale_filter);
    PassFn f = PassFunctions(kBlitHlsl, "vs", final ? "psfinal" : "psmain");
    float c[8] = {uvrect[0], uvrect[1], uvrect[2], uvrect[3], o.gamma ? 1.0f : 0.0f, o.alpha ? 1.0f : 0.0f, float(o.taps), 1.0f};
    if (final) c[5] = float((st.fxaa ? 1 : 0) | (st.scale_filter ? 2 : 0)), c[6] = float(std::min<uint32_t>(st.sharpen, 100)) / 100.0f;
    id<MTLTexture> tex[2] = {src, lut_};
    FullScreen(f, c, 8, tex, 2, o.over, no_depth_);
}

// a depth buffer drawn as colour into another depth buffer's memory: copied as depth
bool MetalBackend::DepthCopy(const State& s) {
    TextureDesc d;
    if (!s.rt.header || !s.textures[0] || !Describe(s.textures[0], d) || Format(d.format).depth || d.cube || d.depth != 1) return false;
    uint32_t gw, gh;
    SurfaceSize(s.rt, gw, gh);
    auto di = targets_.find(SizedTargetKey(mem::VirtualFromPhysical(s.rt.data), true, gw, gh));
    auto si = targets_.find(SizedTargetKey(d.data, true, d.width, d.height));
    if (di == targets_.end() || si == targets_.end() || di == si) return false;
    Target& dst = di->second;
    Target& src = si->second;
    PassFn f = PassFunctions(kDepthHlsl, "vs", "psmain");
    if (!f.vs || !f.fs) return false;
    SetPassTarget(nil, dst.res, dst.w, dst.h);
    id<MTLTexture> tex = src.res;
    FullScreen(f, nullptr, 0, &tex, 1, false, depth_always_);
    dst.drawn = true;
    return true;
}

// ---- clears

void MetalBackend::ClearWhole(Target& t, uint32_t flags, const float colour[4], float z, uint32_t stencil) {
    EndEncoder();
    MTLRenderPassDescriptor* d = [MTLRenderPassDescriptor renderPassDescriptor];
    if (t.depth) {
        d.depthAttachment.texture = t.res;
        d.depthAttachment.loadAction = (flags & kClearZ) ? MTLLoadActionClear : MTLLoadActionLoad;
        d.depthAttachment.storeAction = MTLStoreActionStore;
        d.depthAttachment.clearDepth = std::clamp(z, 0.0f, 1.0f);
        d.stencilAttachment.texture = t.res;
        d.stencilAttachment.loadAction = (flags & kClearStencil) ? MTLLoadActionClear : MTLLoadActionLoad;
        d.stencilAttachment.storeAction = MTLStoreActionStore;
        d.stencilAttachment.clearStencil = stencil & 0xFF;
    } else {
        d.colorAttachments[0].texture = t.res;
        d.colorAttachments[0].loadAction = MTLLoadActionClear;
        d.colorAttachments[0].storeAction = MTLStoreActionStore;
        d.colorAttachments[0].clearColor = MTLClearColorMake(colour[0], colour[1], colour[2], colour[3]);
    }
    [[Cb() renderCommandEncoderWithDescriptor:d] endEncoding];
}

// a clear of part of a target: the full screen triangle in each rect
void MetalBackend::ClearRects(Target& t, const std::vector<RECT>& rs, uint32_t flags, const float colour[4], float z, uint32_t stencil) {
    PassFn f = PassFunctions(kClearHlsl, "vs", t.depth ? "psdepth" : "pscolour");
    id<MTLDepthStencilState> dss = no_depth_;
    if (t.depth) {
        MTLDepthStencilDescriptor* dd = [MTLDepthStencilDescriptor new];
        dd.depthCompareFunction = MTLCompareFunctionAlways;
        dd.depthWriteEnabled = (flags & kClearZ) != 0;
        if (flags & kClearStencil) {
            MTLStencilDescriptor* sd = [MTLStencilDescriptor new];
            sd.stencilCompareFunction = MTLCompareFunctionAlways;
            sd.stencilFailureOperation = sd.depthFailureOperation = sd.depthStencilPassOperation = MTLStencilOperationReplace;
            dd.frontFaceStencil = dd.backFaceStencil = sd;
        }
        dss = [dev_ newDepthStencilStateWithDescriptor:dd];
    }
    SetPassTarget(t.depth ? nil : t.res, t.depth ? t.res : nil, t.w, t.h);
    float cb[8] = {colour[0], colour[1], colour[2], colour[3], z, 0, 0, 0};
    for (const RECT& r : rs) {
        if (r.right <= r.left || r.bottom <= r.top) continue;
        [enc_ setScissorRect:MTLScissorRect{NSUInteger(r.left), NSUInteger(r.top), NSUInteger(r.right - r.left), NSUInteger(r.bottom - r.top)}];
        FullScreen(f, cb, 8, nullptr, 0, false, dss, stencil & 0xFF);
    }
    Unbind();
}

void MetalBackend::Clear(const State& s, uint32_t count, const uint8_t* rects, uint32_t flags, uint32_t color, float z, uint32_t stencil) {
    @autoreleasepool {
        Target* rt = (flags & kClearTarget) ? EnsureTarget(s.rt, false) : nullptr;
        Target* ds = (flags & (kClearZ | kClearStencil)) ? DepthTarget(s) : nullptr;
        std::vector<RECT> rs;
        float col[4];
        Color(color, col);
        for (Target* t : {rt, ds}) {
            if (!t) continue;
            rs.clear();
            if (count && rects) {
                for (uint32_t i = 0; i < count; i++) {
                    int32_t r[4];
                    std::memcpy(r, rects + size_t(i) * sizeof(r), sizeof(r));
                    rs.push_back(ScaledRect(*t, r[0], r[1], r[2], r[3]));
                }
            } else {
                const auto& v = s.viewport;
                rs.push_back(ScaledRect(*t, int(v.x), int(v.y), int(v.x + v.w), int(v.y + v.h)));
            }
            bool whole = rs.size() == 1 && rs[0].left == 0 && rs[0].top == 0 && rs[0].right == LONG(t->w) && rs[0].bottom == LONG(t->h);
            // pixels the CPU wrote before this clear go in first, a later upload would paint over the clear
            if (!t->depth && LatchCpuWrite(*t)) UploadTarget(*t, s.rt);
            if (whole) ClearWhole(*t, flags, col, z, stencil);
            else ClearRects(*t, rs, flags, col, std::clamp(z, 0.0f, 1.0f), stencil);
            t->drawn = true;
        }
    }
}

// ---- copies

// one Blit per rect: the targets' scales differ
void MetalBackend::CopyByDraw(Target& src_t, Target& dst_t, uint32_t count, uint32_t src_rects, uint32_t dst_points) {
    SetPassTarget(dst_t.res, nil, dst_t.w, dst_t.h);
    float src_scale = float(src_t.w) / src_t.gw;
    float dst_scale = float(dst_t.w) / dst_t.gw;
    float dst_scale_y = float(dst_t.h) / dst_t.gh;
    bool listed = count && src_rects;
    uint32_t n = listed ? count : 1;
    for (uint32_t i = 0; i < n; i++) {
        float whole_w = float(std::min(src_t.gw, dst_t.gw));
        float whole_h = float(std::min(src_t.gh, dst_t.gh));
        DrawnRect r{.x0 = 0, .y0 = 0, .x1 = whole_w, .y1 = whole_h};
        if (listed) r = ReadDrawnRect(src_rects, dst_points, i);
        if (r.x1 <= r.x0 || r.y1 <= r.y0) continue;
        MTLViewport vp{r.dx * dst_scale, r.dy * dst_scale_y, (r.x1 - r.x0) * dst_scale, (r.y1 - r.y0) * dst_scale_y, 0, 1};
        long left = std::max(0L, long(vp.originX)), top = std::max(0L, long(vp.originY));
        long right = std::min(long(vp.originX + vp.width), long(dst_t.w));
        long bottom = std::min(long(vp.originY + vp.height), long(dst_t.h));
        if (right <= left || bottom <= top) continue;
        [enc_ setViewport:vp];
        [enc_ setScissorRect:MTLScissorRect{NSUInteger(left), NSUInteger(top), NSUInteger(right - left), NSUInteger(bottom - top)}];
        float uv[4] = {r.x0 / src_t.gw, r.y0 / src_t.gh, (r.x1 - r.x0) / src_t.gw, (r.y1 - r.y0) / src_t.gh};
        uint32_t taps = uint32_t(std::ceil(std::max(src_scale / dst_scale, 1.0f) - kTapsSlack));
        Blit(src_t.res, uv, {.alpha = true, .taps = taps});
    }
    dst_t.drawn = true;
    Unbind();
}

// one texture copy per rect, clamped to both targets
void MetalBackend::CopyByRegion(Target& src_t, Target& dst_t, uint32_t count, uint32_t src_rects, uint32_t dst_points) {
    float dst_scale = float(dst_t.w) / dst_t.gw;
    float dst_scale_y = float(dst_t.h) / dst_t.gh;
    bool listed = count && src_rects;
    uint32_t n = listed ? count : 1;
    id<MTLBlitCommandEncoder> blit = Blitter();
    for (uint32_t i = 0; i < n; i++) {
        uint32_t left = 0, top = 0, right = std::min(src_t.w, dst_t.w), bottom = std::min(src_t.h, dst_t.h), dx = 0, dy = 0;
        if (listed) {
            GuestRect r = ReadGuestRect(src_rects, i);
            left = ScaleUnsigned(r.left, dst_scale);
            top = ScaleUnsigned(r.top, dst_scale_y);
            right = std::min(ScaleUnsigned(r.right, dst_scale), src_t.w);
            bottom = std::min(ScaleUnsigned(r.bottom, dst_scale_y), src_t.h);
            dx = left;
            dy = top;
            if (dst_points) {
                GuestPoint p = ReadGuestPoint(dst_points, i);
                dx = ScaleUnsigned(p.x, dst_scale);
                dy = ScaleUnsigned(p.y, dst_scale_y);
            }
        }
        if (right <= left || bottom <= top || dx >= dst_t.w || dy >= dst_t.h) continue;
        right = std::min(right, left + dst_t.w - dx);
        bottom = std::min(bottom, top + dst_t.h - dy);
        [blit copyFromTexture:src_t.res sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(left, top, 0)
                   sourceSize:MTLSizeMake(right - left, bottom - top, 1) toTexture:dst_t.res destinationSlice:0
             destinationLevel:0 destinationOrigin:MTLOriginMake(dx, dy, 0)];
    }
    [blit endEncoding];
    dst_t.drawn = true;
}

void MetalBackend::Copy(const Surface& src, const Surface& dst, uint32_t count, uint32_t src_rects, uint32_t dst_points) {
    @autoreleasepool {
        Target* src_t = FindTarget(src, false);
        // pixels the CPU made: the copy happens in memory, textures and targets pick it up from there
        if (!src_t || !src_t->drawn) {
            if (!CopySurfaceMemory(src, dst, count, src_rects, dst_points)) {
                LOG_FIRST(8, "metal: CopyRects %08X to %08X skipped", src.data, dst.data);
            } else if (Target* dst_t = FindTarget(dst, false)) {
                UploadTarget(*dst_t, dst);
            }
            return;
        }
        Target* dst_t = EnsureTarget(dst, false);
        if (!dst_t || src_t == dst_t) return;
        bool same_scale = src_t->w * dst_t->gw == dst_t->w * src_t->gw && src_t->h * dst_t->gh == dst_t->h * src_t->gh;
        if (!same_scale) CopyByDraw(*src_t, *dst_t, count, src_rects, dst_points);
        else CopyByRegion(*src_t, *dst_t, count, src_rects, dst_points);
    }
}

// ---- textures

id<MTLTexture> MetalBackend::BindTargetTexture(const State& s, int stage, const TextureDesc& d, Target& t, DrawTextures& stages) {
    TexInfo& info = stages.info[stage];
    if (&t == draw_ds_) return nil;
    stages.reads.rt[stage] = true;
    if (!t.depth && LatchCpuWrite(t)) {
        if (++reloads_[t.data] <= 3)
            LOGR("metal: sampled target %08X %ux%u reloaded from guest memory (frame %llu)", t.data, t.gw, t.gh,
                 (unsigned long long)presents_);
        Surface su;
        su.header = s.textures[size_t(stage)];
        UploadTarget(t, su);
    }
    info.kind = kTex2D;
    // a pass shrinking it averages each tap over a console texel, else upscaled taps skip texels and ghost
    Target* rt = FindTarget(s.rt, false);
    float k = float(t.w) / t.gw;
    if (!t.depth && k > kUpscaledMin && rt && rt != &t && (t.gw > rt->gw || t.gh > rt->gh)) {
        stages.reads.wide[stage] = k;
        info.flags |= kTexWide;
    }
    if (NoAlphaFormat(d.format)) info.flags |= kTexOpaque;  // also a depth buffer read as X8R8G8B8
    // the console reads the pixels it is about to overwrite: the draw gets a copy
    if (&t == draw_rt_) {
        if (!t.fb) {
            MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kBgra width:t.w height:t.h mipmapped:NO];
            td.storageMode = MTLStorageModePrivate;
            td.usage = MTLTextureUsageShaderRead;
            t.fb = [dev_ newTextureWithDescriptor:td];
            if (!t.fb) return nil;
        }
        id<MTLBlitCommandEncoder> b = Blitter();
        [b copyFromTexture:t.res toTexture:t.fb];
        [b endEncoding];
        return t.fb;
    }
    if (t.depth) DepthRead(d.format, info);
    return t.res;
}

bool MetalBackend::CreateTexture(Tex& tx, const TextureDesc& d, const FormatInfo& fi) {
    tx = Tex{};
    tx.d = d;
    tx.fmt = MetalFormat(HostFormat(fi.host));
    uint32_t w = d.width, h = d.height;
    tx.levels = d.levels;
    if (fi.compressed && ((w & 3) || (h & 3))) {
        w = (w + 3) & ~3u;
        h = (h + 3) & ~3u;
        tx.levels = 1;
    }
    MTLTextureDescriptor* td = [MTLTextureDescriptor new];
    td.pixelFormat = tx.fmt;
    td.width = w;
    td.height = h;
    td.mipmapLevelCount = std::max(1u, tx.levels);
    td.usage = MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    if (d.depth > 1) {
        td.textureType = MTLTextureType3D;
        td.depth = d.depth;
    } else if (d.cube) {
        td.textureType = MTLTextureTypeCube;
    }
    // Metal refuses more levels than the size has
    uint32_t most = 1;
    for (uint32_t m = std::max({w, h, d.depth > 1 ? d.depth : 1u}); m > 1; m >>= 1) most++;
    if (td.mipmapLevelCount > most) td.mipmapLevelCount = tx.levels = most;
    tx.res = [dev_ newTextureWithDescriptor:td];
    return tx.res != nil;
}

// a level or slice: at once on a texture no draw has used, else in order with the draws
void MetalBackend::UploadLevel(Tex& tx, uint32_t level, uint32_t slice, uint32_t lw, uint32_t lh, uint32_t depth, const uint8_t* p, uint32_t row, uint32_t image) {
    if (!tx.uploaded) {
        [tx.res replaceRegion:MTLRegionMake3D(0, 0, 0, lw, lh, depth) mipmapLevel:level slice:slice withBytes:p bytesPerRow:row bytesPerImage:depth > 1 || tx.d.cube ? image : 0];
        return;
    }
    Alloc a = Reserve(image * depth, 16);
    std::memcpy(a.p, p, size_t(image) * depth);
    id<MTLBlitCommandEncoder> b = Blitter();
    [b copyFromBuffer:a.buf sourceOffset:a.offset sourceBytesPerRow:row sourceBytesPerImage:image sourceSize:MTLSizeMake(lw, lh, depth)
            toTexture:tx.res destinationSlice:slice destinationLevel:level destinationOrigin:MTLOriginMake(0, 0, 0)];
    [b endEncoding];
}

// every face and level decoded and uploaded, rows padded or cut to what the host format needs
void MetalBackend::UploadTexture(Tex& tx, const TextureDesc& d, const uint8_t* p, const uint32_t* colours) {
    const auto start = std::chrono::steady_clock::now();
    uint32_t faces = d.cube ? 6 : 1;
    uint32_t block = BlockBytes(tx.fmt);
    bool volume = d.depth > 1;
    uint32_t w0 = d.width, h0 = d.height;
    if (block && tx.levels == 1) w0 = (w0 + 3) & ~3u, h0 = (h0 + 3) & ~3u;
    std::vector<uint8_t> px, rows;
    for (uint32_t f = 0; f < faces; f++) {
        for (uint32_t l = 0; l < tx.levels; l++) {
            uint32_t pitch = 0;
            DecodeLevel(d, l, f, p, colours, px, pitch);
            uint32_t lw = std::max(1u, w0 >> l), lh = std::max(1u, h0 >> l);
            uint32_t slices = volume ? std::max(1u, d.depth >> l) : 1;
            uint32_t row_bytes = block ? std::max(1u, (lw + 3) / 4) * block : lw * 4;
            uint32_t n_rows = block ? std::max(1u, (lh + 3) / 4) : lh;
            size_t src_rows = pitch ? px.size() / pitch / slices : 0;
            rows.assign(size_t(row_bytes) * n_rows * slices, 0);
            for (uint32_t z = 0; z < slices; z++) {
                for (uint32_t y = 0; y < n_rows && y < src_rows; y++)
                    std::memcpy(&rows[(size_t(z) * n_rows + y) * row_bytes], &px[(size_t(z) * src_rows + y) * pitch], std::min(pitch, row_bytes));
            }
            UploadLevel(tx, l, f, lw, lh, slices, rows.data(), row_bytes, row_bytes * n_rows);
        }
    }
    tx.uploaded = true;
    cost_.uploads++;
    cost_.uploads_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

// a P8 texture is decoded with the stage's palette: 256 colours, fewer by the size code in the top two bits
struct StagePalette {
    uint32_t colours[256] = {};
    uint32_t va = 0;
    uint64_t hash = 0;
};

void ReadStagePalette(const State& s, int stage, const TextureDesc& d, StagePalette& palette) {
    constexpr uint32_t kFmtP8 = 0x0B;
    constexpr uint32_t kPaletteCommon = 0, kPaletteData = 4, kPaletteSizeShift = 30;
    if (d.format != kFmtP8 || !s.palettes[size_t(stage)]) return;
    uint32_t header = s.palettes[size_t(stage)];
    palette.va = mem::VirtualFromPhysical(kernel::R32(header + kPaletteData));
    uint32_t size_code = kernel::R32(header + kPaletteCommon) >> kPaletteSizeShift;
    size_t bytes = (std::size(palette.colours) >> size_code) * sizeof(uint32_t);
    std::memcpy(palette.colours, kernel::HostPtr(palette.va), bytes);
    palette.hash = Hash(palette.colours, sizeof(palette.colours));
}

bool SameTexture(const Tex& tx, const TextureDesc& d) {
    return tx.res && tx.d.format == d.format && tx.d.width == d.width && tx.d.height == d.height && tx.d.depth == d.depth &&
           tx.d.levels == d.levels && tx.d.cube == d.cube && tx.d.pitch == d.pitch;
}

id<MTLTexture> MetalBackend::BindGuestTexture(const State& s, int stage, const TextureDesc& d, const FormatInfo& fi, DrawTextures& stages) {
    if (!d.cube && d.depth == 1 && !fi.depth) {
        auto r = FindReplacement(d.data);
        if (r && ((r->sharpened && !(r->world ? app::Get().world_textures : app::Get().sharp_2d)) || !ReplacementFits(*r, d))) r = nullptr;
        if (id<MTLTexture> t = r ? BindReplacement(*r, d, stages.levels[stage]) : nil) {
            stages.replaced[stage] = true;
            return t;
        }
    }
    if (draw_2d_) Sharpen2D(d);
    else SharpenWorld(d, s.tss[size_t(stage)][kTssAddressU] == 1, s.tss[size_t(stage)][kTssAddressV] == 1);
    StagePalette palette;
    ReadStagePalette(s, stage, d, palette);
    const uint64_t key = uint64_t(palette.va) << 32 | d.data;
    Tex& tx = textures_[key];
    bool same = SameTexture(tx, d);
    if (!same && !CreateTexture(tx, d, fi)) {
        textures_.erase(key);
        return nil;
    }
    // cheap change check once per frame: lock epochs plus a sample of the bytes
    if (tx.frame != presents_ + 1 || !same || tx.palette != palette.hash) {
        tx.frame = presents_ + 1;
        tx.palette = palette.hash;
        uint64_t sig = SurfaceSig(d.data, TotalSize(d)) ^ DirtyEpoch(d.data) ^ palette.hash;
        if (sig != tx.sig || !same) {
            tx.sig = sig;
            UploadTexture(tx, d, static_cast<const uint8_t*>(kernel::HostPtr(d.data)), palette.va ? palette.colours : nullptr);
        }
    }
    if (d.cube && !fi.depth) CopyCubeTargets(tx, d);
    stages.levels[stage] = tx.levels;
    if (fi.depth) stages.info[stage].shadow = ShadowDepthKind(d.format);
    return tx.res;
}

id<MTLTexture> MetalBackend::BindReplacement(const Replacement& r, const TextureDesc& d, uint32_t& levels) {
    constexpr uint32_t kFmtA8R8G8B8 = 0x06;
    const uint64_t key = uint64_t(0xFFFFFFFFu) << 32 | d.data;  // no palette is at the top of memory
    if (auto had = textures_.find(key); r.sharpened && (had == textures_.end() || !had->second.res || had->second.sig != r.version) &&
                                        !TakePictureBudget(presents_, r))
        return nil;
    Tex& tx = textures_[key];
    if (tx.frame != presents_ + 1 && !ReplacementHolds(r)) {
        textures_.erase(key);
        return nil;
    }
    tx.frame = presents_ + 1;
    if (!tx.res || tx.sig != r.version) {
        std::vector<std::vector<uint32_t>> pictures = ReplacementLevels(r);
        if (!tx.res || tx.d.width != r.width || tx.d.height != r.height || tx.levels != pictures.size()) {
            TextureDesc rd;
            rd.data = d.data;
            rd.format = kFmtA8R8G8B8;
            rd.width = r.width;
            rd.height = r.height;
            rd.levels = uint32_t(pictures.size());
            if (!CreateTexture(tx, rd, Format(kFmtA8R8G8B8))) {
                textures_.erase(key);
                return nil;
            }
            tx.frame = presents_ + 1;
        }
        uint32_t w = r.width, h = r.height;
        for (uint32_t l = 0; l < pictures.size() && l < tx.levels; l++) {
            UploadLevel(tx, l, 0, w, h, 1, reinterpret_cast<const uint8_t*>(pictures[l].data()), w * 4, w * 4 * h);
            w = std::max(w / 2, 1u);
            h = std::max(h / 2, 1u);
        }
        tx.uploaded = true;
        tx.sig = r.version;
    }
    levels = tx.levels;
    return tx.res;
}

// a cube's faces drawn on the GPU (an environment map rendered each frame) replace level 0 of the face, once a frame
void MetalBackend::CopyCubeTargets(Tex& tx, const TextureDesc& d) {
    if (tx.fmt != kBgra || tx.faces == presents_ + 1) return;
    tx.faces = presents_ + 1;
    for (uint32_t f = 0; f < 6; f++) {
        uint32_t face = d.data + f * FaceSize(d);
        auto it = targets_.find(TargetMapKey(face, false));
        if (it == targets_.end() || TargetStale(face)) continue;
        Target& t = it->second;
        if (t.depth || t.gw != d.width || t.gh != d.height) continue;
        id<MTLTexture> src = PrepareCopySource(t);
        if (!src) continue;
        id<MTLBlitCommandEncoder> b = Blitter();
        [b copyFromTexture:src sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(t.gw, t.gh, 1)
                 toTexture:tx.res destinationSlice:f destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
        [b endEncoding];
    }
}

id<MTLTexture> MetalBackend::BindTexture(const State& s, int stage, DrawTextures& stages) {
    stages.info[stage] = {};
    stages.reads.rt[stage] = false;
    stages.reads.k[stage] = 1;
    stages.reads.wide[stage] = 0;
    stages.levels[stage] = 1;
    stages.replaced[stage] = false;
    uint32_t header = s.textures[size_t(stage)];
    TextureDesc d;
    if (!header || !Describe(header, d)) return nil;
    const FormatInfo& fi = Format(d.format);
    stages.info[stage].kind = TextureKind(d);
    stages.info[stage].rect = d.linear;
    // rendered by the GPU: sample the target itself; DXT data loaded over a target's memory is not it
    auto tit = targets_.find(SizedTargetKey(d.data, fi.depth, d.width, d.height));
    if (tit == targets_.end() && !fi.depth && DepthBytesOrder(d.format)) tit = targets_.find(SizedTargetKey(d.data, true, d.width, d.height));
    if (tit == targets_.end() && fi.depth && d.linear) {
        auto big = targets_.find(TargetMapKey(d.data, true));
        if (big != targets_.end() && big->second.gw >= d.width && big->second.gh >= d.height) {
            tit = big;
            stages.reads.size[stage][0] = big->second.gw;
            stages.reads.size[stage][1] = big->second.gh;
        }
    }
    if (tit != targets_.end() && !fi.depth && !tit->second.depth && d.linear && (tit->second.gw > d.width || tit->second.gh > d.height) &&
        tit->second.gw >= d.width && tit->second.gh >= d.height) {
        stages.reads.size[stage][0] = tit->second.gw;
        stages.reads.size[stage][1] = tit->second.gh;
    }
    if (tit != targets_.end() && !d.cube && d.depth == 1 && !fi.compressed) return BindTargetTexture(s, stage, d, tit->second, stages);
    return BindGuestTexture(s, stage, d, fi, stages);
}

id<MTLTexture> MetalBackend::Dummy(const TexInfo& info) {
    if (info.shadow) return dummy_depth_;
    if (info.kind == kTex3D) return dummy_3d_;
    if (info.kind == kTexCube) return dummy_cube_;
    return dummy_2d_;
}

// ---- the draw

void MetalBackend::EnsureVertexShader(const State& s, const Assembled& a, DrawShaders& sh) {
    auto it = vs_libs_.find(sh.vkey);
    if (it == vs_libs_.end()) {
        TakeReady();
        it = vs_libs_.find(sh.vkey);
    }
    if (it == vs_libs_.end()) {
        std::string src = VertexHlsl(s, a, sh);
        std::string msl = VertexMsl(sh.quad ? QuadHlsl(std::move(src)) : src);
        id<MTLLibrary> lib = Library(msl);
        it = vs_libs_.emplace(sh.vkey, lib).first;
        if (lib) {
            RecordShader('V', sh.vkey, msl);
            fn_vs_[sh.vkey] = [lib newFunctionWithName:@"vmain"];
            fn_vp_[sh.vkey] = [lib newFunctionWithName:@"vpoint"];
        }
    }
    sh.vh = sh.vkey;
    sh.vs_code = it->second ? reinterpret_cast<ID3DBlob*>(1) : nullptr;  // made
}

void MetalBackend::EnsurePixelShader(const PixelKey& pk, DrawShaders& sh) {
    auto it = ps_libs_.find(sh.pkey);
    if (it == ps_libs_.end()) {
        TakeReady();
        it = ps_libs_.find(sh.pkey);
    }
    if (it == ps_libs_.end()) {
        std::string msl = PixelMsl(PixelHlsl(pk));
        id<MTLLibrary> lib = Library(msl);
        it = ps_libs_.emplace(sh.pkey, lib).first;
        if (lib) {
            RecordShader('P', sh.pkey, msl);
            fn_ps_[sh.pkey] = [lib newFunctionWithName:@"fmain"];
        }
    }
    sh.ph = sh.pkey;
    sh.ps_code = it->second ? reinterpret_cast<ID3DBlob*>(1) : nullptr;
}

id<MTLDepthStencilState> MetalBackend::DepthState(const PsoKey& k) {
    uint32_t f[] = {k.zen, k.zwrite, k.zfunc, k.sten, k.sfunc, k.smask, k.swmask, k.sfail, k.szfail, k.spass};
    uint64_t h = Hash(f, sizeof(f));
    auto it = depth_states_.find(h);
    if (it != depth_states_.end()) return it->second;
    MTLDepthStencilDescriptor* dd = [MTLDepthStencilDescriptor new];
    if (k.zen) {
        dd.depthCompareFunction = CompareOf(Cmp(k.zfunc));
        dd.depthWriteEnabled = k.zwrite != 0;
    }
    if (k.sten) {
        MTLStencilDescriptor* sd = [MTLStencilDescriptor new];
        sd.stencilCompareFunction = CompareOf(Cmp(k.sfunc));
        sd.stencilFailureOperation = StencilOf(StencilOp(k.sfail));
        sd.depthFailureOperation = StencilOf(StencilOp(k.szfail));
        sd.depthStencilPassOperation = StencilOf(StencilOp(k.spass));
        sd.readMask = k.smask & 0xFF;
        sd.writeMask = k.swmask & 0xFF;
        dd.frontFaceStencil = dd.backFaceStencil = sd;
    }
    return depth_states_[h] = [dev_ newDepthStencilStateWithDescriptor:dd];
}

MTLRenderPipelineDescriptor* MetalBackend::PipelineDescriptor(const PsoKey& k, const uint32_t strides[16], id<MTLFunction> vs, id<MTLFunction> fs) {
    const bool points = k.topo == uint32_t(Topology::Points);
    MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = vs;
    d.fragmentFunction = fs;
    if (cache_.archive) d.binaryArchives = @[cache_.archive];
    MTLVertexDescriptor* vd = [MTLVertexDescriptor vertexDescriptor];
    uint32_t used = 0;
    for (uint32_t r = 0; r < 16; r++) {
        if (!((k.regs >> r) & 1)) continue;
        uint32_t e = k.elements[r];
        uint32_t buffer = (e >> 8) & 0xFF;
        MTLVertexFormat fmt = VertexFormat(e & 0xFF);
        if (fmt == MTLVertexFormatInvalid) LOG_FIRST(8, "metal: vertex format %u has no Metal format", e & 0xFF);
        vd.attributes[r].format = fmt;
        vd.attributes[r].offset = e >> 16;
        vd.attributes[r].bufferIndex = kVertexBufferBase + buffer;
        used |= 1u << buffer;
    }
    for (uint32_t b = 0; b < 16; b++) {
        if (!((used >> b) & 1)) continue;
        auto* l = vd.layouts[kVertexBufferBase + b];
        if (strides[b]) {
            l.stride = strides[b];
            l.stepFunction = points ? MTLVertexStepFunctionPerInstance : MTLVertexStepFunctionPerVertex;
        } else {  // one vertex repeated
            l.stride = 16;
            l.stepFunction = MTLVertexStepFunctionConstant;
            l.stepRate = 0;
        }
    }
    if (used) d.vertexDescriptor = vd;
    if (k.has_rt) {
        auto* c = d.colorAttachments[0];
        c.pixelFormat = kBgra;
        c.blendingEnabled = k.blend != 0;
        c.sourceRGBBlendFactor = BlendFactorOf(Blend(k.src, false));
        c.destinationRGBBlendFactor = BlendFactorOf(Blend(k.dst, false));
        c.rgbBlendOperation = BlendOpOf(BlendOp(k.op));
        c.sourceAlphaBlendFactor = BlendFactorOf(Blend(k.src, true));
        c.destinationAlphaBlendFactor = BlendFactorOf(Blend(k.dst, true));
        c.alphaBlendOperation = BlendOpOf(BlendOp(k.op));
        // D3D's write mask has red lowest, Metal's alpha lowest
        MTLColorWriteMask mask = MTLColorWriteMaskNone;
        if (k.wmask & 1) mask |= MTLColorWriteMaskRed;
        if (k.wmask & 2) mask |= MTLColorWriteMaskGreen;
        if (k.wmask & 4) mask |= MTLColorWriteMaskBlue;
        if (k.wmask & 8) mask |= MTLColorWriteMaskAlpha;
        c.writeMask = mask;
    }
    if (k.has_ds) d.depthAttachmentPixelFormat = d.stencilAttachmentPixelFormat = kDepth;
    return d;
}

Pipeline* MetalBackend::EnsurePipeline(const PsoKey& key, const Assembled& a, const DrawShaders& sh) {
    const PsoKey k = PipelineOnly(key);
    uint32_t strides[16] = {};
    for (uint32_t i = 0; i < a.buffer_count && i < 16; i++) strides[i] = a.buffers[i].stride;
    uint64_t h = Hash(strides, sizeof(strides), Hash(&k, sizeof(k)));
    auto it = pipelines_.find(h);
    if (it != pipelines_.end()) return &it->second;
    TakeReady();
    it = pipelines_.find(h);
    if (it != pipelines_.end()) return &it->second;
    const auto start = std::chrono::steady_clock::now();
    Pipeline p;
    p.points = k.topo == uint32_t(Topology::Points);
    MTLRenderPipelineDescriptor* d = PipelineDescriptor(k, strides, p.points ? fn_vp_[sh.vkey] : fn_vs_[sh.vkey], fn_ps_[sh.pkey]);
    NSError* err = nil;
    p.state = [dev_ newRenderPipelineStateWithDescriptor:d error:&err];
    if (!p.state) {
        LOG_FIRST(16, "metal: pipeline failed: %s", err.localizedDescription.UTF8String);
    } else {
        RecordPipeline(k, strides);
        if (cache_.archive) {
            CacheFiles* files = &cache_;
            dispatch_async(cache_.archive_queue, ^{
                [files->archive addRenderPipelineFunctionsWithDescriptor:d error:nil];
                files->dirty = true;
            });
        }
    }
    cost_.pipelines++;
    cost_.pipelines_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return &(pipelines_[h] = std::move(p));
}

id<MTLSamplerState> MetalBackend::Sampler(const D3D12_SAMPLER_DESC& sd) {
    uint64_t h = Hash(&sd, sizeof(sd));
    auto it = samplers_.find(h);
    if (it != samplers_.end()) return it->second;
    MTLSamplerDescriptor* m = [MTLSamplerDescriptor new];
    uint32_t f = uint32_t(sd.Filter);
    bool aniso = f == D3D12_FILTER_ANISOTROPIC;
    m.minFilter = aniso || ((f >> 4) & 3) ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    m.magFilter = aniso || ((f >> 2) & 3) ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    m.mipFilter = aniso || (f & 3) ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
    if (aniso) m.maxAnisotropy = std::clamp<UINT>(sd.MaxAnisotropy, 1, 16);
    m.sAddressMode = AddressOf(sd.AddressU);
    m.tAddressMode = AddressOf(sd.AddressV);
    m.rAddressMode = AddressOf(sd.AddressW);
    // three border colours to choose from: the one nearest
    float a = sd.BorderColor[3], l = (sd.BorderColor[0] + sd.BorderColor[1] + sd.BorderColor[2]) / 3;
    m.borderColor = a < 0.5f ? MTLSamplerBorderColorTransparentBlack : l < 0.5f ? MTLSamplerBorderColorOpaqueBlack : MTLSamplerBorderColorOpaqueWhite;
    // White with no alpha, which PGR2's ground shadows use, is not one of Metal's three. They multiply the road by its colour, so
    // white it is: black drew a line along the mesh's edge, and repeating the texture's edge drew rectangles out past the car.
    // A comparison reads the border as a depth, white being the far plane, as on D3D.
    static const bool edge_test = std::getenv("XBE_BORDER_EDGE") != nullptr;  // testing: the old way
    if (l >= 0.5f && !(f & 0x80)) m.borderColor = MTLSamplerBorderColorOpaqueWhite;
    if (edge_test && ((f & 0x80) || (a < 0.5f && l >= 0.5f))) {
        auto edge = [](MTLSamplerAddressMode m) { return m == MTLSamplerAddressModeClampToBorderColor ? MTLSamplerAddressModeClampToEdge : m; };
        m.sAddressMode = edge(m.sAddressMode);
        m.tAddressMode = edge(m.tAddressMode);
        m.rAddressMode = edge(m.rAddressMode);
    }
    if (sd.AddressU == D3D12_TEXTURE_ADDRESS_MODE_BORDER || sd.AddressV == D3D12_TEXTURE_ADDRESS_MODE_BORDER)
        LOG_FIRST(16, "metal: border sampler, colour %.2f %.2f %.2f %.2f, comparison %d, address %d %d", sd.BorderColor[0], sd.BorderColor[1],
                  sd.BorderColor[2], sd.BorderColor[3], (f & 0x80) != 0, int(sd.AddressU), int(sd.AddressV));
    m.lodMinClamp = sd.MinLOD;
    m.lodMaxClamp = std::max(sd.MinLOD, sd.MaxLOD);
    if (f & 0x80) m.compareFunction = CompareOf(sd.ComparisonFunc);
    if (sd.MipLODBias != 0) LOG_FIRST(1, "metal: a sampler's LOD bias is not applied (%g)", sd.MipLODBias);
    return samplers_[h] = [dev_ newSamplerStateWithDescriptor:m];
}

void MetalBackend::UploadConstants(const State& s, const DrawShaders& sh, const PsoKey& k, const DrawTextures& stages, GuestExtent size, DrawConstants& c) {
    FillSurfaceConstants(s, size, c);
    if (k.topo == uint32_t(Topology::Points)) {
        const Target* t = bound_rt_ ? bound_rt_ : bound_ds_;
        c.surf[3] = t && t->gh ? float(t->h) * s.aa_y / float(t->gh) : 1.0f;
    }
    FillFogConstants(s, sh.prog, c);
    FillCombinerConstants(s, c);
    FillTextureConstants(s, stages.reads, c);
    if (sh.fixed_transform) {
        FillTransformConstants(s, size, c);
        FillLightConstants(s, sh.fk, c);
    }
    // VC then DC, one block: a new one only when something in it changed
    const size_t total = sizeof(s.vsc) + sizeof(c);
    const uint8_t* vsc = reinterpret_cast<const uint8_t*>(s.vsc.data());
    bool same = const_last_.size() == total && !std::memcmp(const_last_.data(), vsc, sizeof(s.vsc)) &&
                !std::memcmp(const_last_.data() + sizeof(s.vsc), &c, sizeof(c)) && const_at_.buf == ring_;
    if (!same) {
        const_last_.resize(total);
        std::memcpy(const_last_.data(), vsc, sizeof(s.vsc));
        std::memcpy(const_last_.data() + sizeof(s.vsc), &c, sizeof(c));
        const_at_ = Reserve(uint32_t(total), 256);
        std::memcpy(const_at_.p, const_last_.data(), total);
        const_bound_ = false;
    }
    if (!const_bound_) {
        [enc_ setVertexBuffer:const_at_.buf offset:const_at_.offset atIndex:0];
        [enc_ setFragmentBuffer:const_at_.buf offset:const_at_.offset atIndex:0];
        const_bound_ = true;
    }
}

void MetalBackend::DrawGeometry(const Assembled& a, const Pipeline& p) {
    for (uint32_t i = 0; i < a.buffer_count; i++) {
        const VertexBuffer& b = a.buffers[i];
        if (!b.bytes) continue;
        Alloc at = Reserve(b.bytes, 16);
        std::memcpy(at.p, b.src, b.bytes);
        [enc_ setVertexBuffer:at.buf offset:at.offset atIndex:kVertexBufferBase + i];
    }
    if (p.points) {
        // a quad per point: the vertex's inputs step per instance
        if (a.index_count) {
            for (uint32_t i = 0; i < a.index_count; i++) {
                uint32_t v = a.index16 ? static_cast<const uint16_t*>(a.index_src)[i] : static_cast<const uint32_t*>(a.index_src)[i];
                [enc_ drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4 instanceCount:1 baseInstance:v + a.base_vertex];
            }
        } else {
            [enc_ drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4 instanceCount:a.draw_count baseInstance:0];
        }
        return;
    }
    MTLPrimitiveType type = MTLPrimitiveTypeTriangle;
    switch (PrimitiveTopology(a)) {
        case D3D_PRIMITIVE_TOPOLOGY_LINELIST: type = MTLPrimitiveTypeLine; break;
        case D3D_PRIMITIVE_TOPOLOGY_LINESTRIP: type = MTLPrimitiveTypeLineStrip; break;
        case D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP: type = MTLPrimitiveTypeTriangleStrip; break;
        case D3D_PRIMITIVE_TOPOLOGY_POINTLIST: type = MTLPrimitiveTypePoint; break;
        default: break;
    }
    if (a.index_count) {
        uint32_t ibytes = a.index_count * (a.index16 ? 2 : 4);
        Alloc at = Reserve(ibytes, 16);
        std::memcpy(at.p, a.index_src, ibytes);
        [enc_ drawIndexedPrimitives:type indexCount:a.index_count indexType:a.index16 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                        indexBuffer:at.buf indexBufferOffset:at.offset instanceCount:1 baseVertex:a.base_vertex baseInstance:0];
    } else {
        [enc_ drawPrimitives:type vertexStart:0 vertexCount:a.draw_count];
    }
}

void MetalBackend::Draw(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices) {
    if (const State* sprites = PointSpriteStages(s, prim)) return Draw(*sprites, prim, start, count, indices);
    @autoreleasepool {
        if (s.blank_format && s.textures[0] && (::kernel::R32(s.textures[0] + 12) >> 8 & 0xFF) == s.blank_format) {
            Clear(s, 0, nullptr, kClearTarget, 0xFF000000u, 0, 0);  // a black clear in its place (State::blank_format)
            return;
        }
        if (DepthCopy(s)) return;
        if (diag::SkippedDraw(s.vs_handle)) return;
        DrawAdjust adjust;
        if (g_draw_filter) g_draw_filter(s, adjust);
        if (adjust.skip) return;
        Assembled& a = asm_;
        if (!Assemble(s, prim, start, count, indices, a)) return;
        if (!BindTargets(s)) return;

        DrawShaders sh = ClassifyShaders(s, a, count);
        if (sh.prog) EnsureProgramHash(s);
        sh.vkey = VertexKey(s, a, sh);
        EnsureVertexShader(s, a, sh);
        // stage textures, which may draw copies, then the targets again
        DrawTextures stages;
        draw_rt_ = bound_rt_;
        draw_ds_ = bound_ds_;
        draw_2d_ = a.pretransformed || ScreenDraw(s);
        for (int i = 0; i < 4; i++) stages.tex[i] = BindTexture(s, i, stages);
        if (!BindTargets(s)) return;
        PixelKey pk = MakePixelKey(s, stages.info);
        pk.quad = sh.quad;
        // D3DZB_USEW draws write w as their depth, as the console does; XBE_WBUFFER=0 goes back to z / w
        static const bool wbuffer = [] {
            const char* e = std::getenv("XBE_WBUFFER");
            return !(e && *e == '0');
        }();
        pk.wbuf = wbuffer && s.rs[kRsZEnable] == 2 && bound_ds_ ? 1 : 0;
        sh.pkey = Hash(&pk, sizeof(pk));
        EnsurePixelShader(pk, sh);
        if (!sh.vs_code || !sh.ps_code) return;

        PsoKey k;
        PipelineTargets pt{.rt = bound_rt_ != nullptr, .rtfmt = bound_rt_ ? uint32_t(DXGI_FORMAT_B8G8R8A8_UNORM) : 0u, .ds = bound_ds_ != nullptr,
                           .ds_scale = bound_ds_ ? float(bound_ds_->h) / bound_ds_->gh : 1.0f, .samples = 1};
        BuildPipelineKey(s, a, sh.vh, sh.ph, pt, k);
        if (!adjust.depth_test) k.zen = k.zwrite = 0;
        k.zbias += adjust.depth_bias;
        // W-buffered: the pixel shader adds the polygon offset to w (d3d12.cpp Draw)
        float wdepth[4] = {float(k.zbias) * s.zscale / kDepth24Max, k.zslope, 1.0f / (s.zscale + 1.0f), sh.prog ? 1.0f : s.zscale * s.inv_wfar};
        if (pk.wbuf) k.zbias = 0, k.zslope = 0;
        Pipeline* p = EnsurePipeline(k, a, sh);
        if (!p->state) return;

        OpenPass(bound_rt_ ? bound_rt_->res : nil, bound_ds_ ? bound_ds_->res : nil);
        if (enc_pipeline_ != p->state) [enc_ setRenderPipelineState:(enc_pipeline_ = p->state)];
        id<MTLDepthStencilState> depth_state = DepthState(k);
        if (enc_depth_state_ != depth_state) [enc_ setDepthStencilState:(enc_depth_state_ = depth_state)];
        [enc_ setStencilReferenceValue:SimpleState(s, kMStencilRef) & 0xFF];
        float bf[4];
        BlendFactor(s, bf);
        [enc_ setBlendColorRed:bf[0] green:bf[1] blue:bf[2] alpha:bf[3]];
        [enc_ setCullMode:k.cull && !p->points ? MTLCullModeBack : MTLCullModeNone];
        [enc_ setFrontFacingWinding:k.ccw && !p->points ? MTLWindingCounterClockwise : MTLWindingClockwise];
        [enc_ setTriangleFillMode:k.fill == kFillWireframe && !p->points ? MTLTriangleFillModeLines : MTLTriangleFillModeFill];
        [enc_ setDepthBias:float(k.zbias) slopeScale:k.zslope clamp:0];

        GuestExtent size;
        SurfaceSize(bound_rt_ ? s.rt : s.zs, size.w, size.h);
        size = DrawnExtent(s, size);
        DrawConstants c{};
        std::memcpy(c.wdepth, wdepth, sizeof(wdepth));
        float stretch = bound_rt_ ? XStretch(aspect_, guest_h_, bound_rt_->gw, bound_rt_->gh) : 0;
        c.hud[0] = HudSqueeze(s, a, aspect_, stretch);
        c.hud[2] = DepthTested(s) ? 1.0f : 0.0f;
        c.hud[3] = pk.wbuf ? 1.0f : 0.0f;
        if (bound_rt_) NoteSqueeze(s, a, bound_rt_->data, bound_rt_->gw, bound_rt_->gh, bound_rt_->w, aspect_, stretch, c.hud[0]);
        c.hud[1] = (bound_rt_ ? bound_rt_->gw : guest_w_) * 0.5f / s.aa_x;
        if (c.hud[0] != 1 && !s.squeeze_beyond) {
            Span span = DrawSpan(s, a, c.hud[1] * 2);
            if (span == kSpanNone) c.hud[0] = -c.hud[0];  // squeezed whole
            if (span == kSpanViewport) c.hud[0] = 1;      // a pass filling its own viewport
        }
        SqueezeBeyond(s, c.hud);
        UploadConstants(s, sh, k, stages, size, c);

        // textures and samplers
        bool clamp = UnitTexcoords(a);
        uint32_t aniso = app::Get().aniso;
        for (int i = 0; i < 4; i++) {
            id<MTLTexture> t = stages.tex[i];
            // a shadow stage reads a depth format, whatever is bound
            if (!t || (stages.info[i].shadow && t.pixelFormat != kDepth)) t = Dummy(stages.info[i]);
            [enc_ setFragmentTexture:t atIndex:i];
            [enc_ setFragmentSamplerState:Sampler(StageSampler(s, i, stages.levels[i], stages.info[i].shadow, clamp, aniso, stages.replaced[i])) atIndex:i];
        }

        // the viewport covers the target, the guest viewport is in the shader; the scissor clips to it
        const Target* b = bound_rt_ ? bound_rt_ : bound_ds_;
        float W = float(b->w), H = float(b->h);
        if (!enc_has_viewport_) {
            [enc_ setViewport:MTLViewport{0, 0, W, H, 0, 1}];
            enc_has_viewport_ = true;
        }
        float sx = W / size.w, sy = b->gh ? H * s.aa_y / b->gh : sx;
        long x0 = long(SqueezedX(c.hud, s.viewport.x) * sx);
        long y0 = long(s.viewport.y * sy);
        long x1 = long(SqueezedX(c.hud, s.viewport.x + s.viewport.w) * sx);
        long y1 = long((s.viewport.y + s.viewport.h) * sy);
        if (s.scissor_count) {
            x0 = std::max(x0, long(SqueezedX(c.hud, float(s.scissors[0])) * sx));
            y0 = std::max(y0, long(s.scissors[1] * sy));
            x1 = std::min(x1, long(SqueezedX(c.hud, float(s.scissors[2])) * sx));
            y1 = std::min(y1, long(s.scissors[3] * sy));
        }
        if (s.squeeze_margin) {
            const long edge = long(SqueezedX(c.hud, s.squeeze_margin < 0 ? 0.0f : c.hud[1] * 2) * sx);
            x0 = s.squeeze_margin < 0 ? 0 : edge;
            x1 = s.squeeze_margin < 0 ? edge : long(W);
        }
        x0 = std::clamp(x0, 0L, long(W));
        y0 = std::clamp(y0, 0L, long(H));
        x1 = std::clamp(x1, 0L, long(W));
        y1 = std::clamp(y1, 0L, long(H));
        if (x1 <= x0 || y1 <= y0) return;  // Metal takes no empty scissor; D3D draws nothing
        MTLScissorRect sc{NSUInteger(x0), NSUInteger(y0), NSUInteger(x1 - x0), NSUInteger(y1 - y0)};
        if (!enc_has_scissor_ || std::memcmp(&enc_scissor_, &sc, sizeof(sc))) [enc_ setScissorRect:sc];
        enc_scissor_ = sc;
        enc_has_scissor_ = true;

        // geometry, counted while the title runs a visibility test
        const bool counting = s.simple[kMCountEnable >> 2] != 0;
        if (counting) {
            uint32_t slot = vis_next_++ % kVisSlots;
            static_cast<uint64_t*>(vis_.contents)[slot] = 0;
            [enc_ setVisibilityResultMode:MTLVisibilityResultModeCounting offset:slot * 8];
            counted_.push_back({slot, 0, SamplesPerPixel()});
        }
        DrawGeometry(a, *p);
        if (counting) [enc_ setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
    }
}

// ---- visibility tests

float MetalBackend::SamplesPerPixel() const {
    const Target* t = bound_rt_ ? bound_rt_ : bound_ds_;
    if (!t || !t->gw || !t->gh) return 1;
    return float(t->w) * float(t->h) / (float(t->gw) * float(t->gh));
}

// reports whose draws the GPU has finished, written as the console's GPU writes them: time, count, then 0 for done
void MetalBackend::DeliverReports() {
    constexpr uint32_t kTime = 0, kTimeHigh = 4, kCount = 8, kStatus = 12;
    const uint64_t done = completed_serial_.load();
    while (!reports_.empty()) {
        Report& r = reports_.front();
        double pixels = 0;
        for (const Query& q : r.queries) {
            if (!q.serial || q.serial > done) return;  // not finished: this and every later report wait
            pixels += double(static_cast<const uint64_t*>(vis_.contents)[q.slot]) / q.samples_per_pixel;
        }
        kernel::W32(r.va + kTime, uint32_t(presents_));
        kernel::W32(r.va + kTimeHigh, 0);
        kernel::W32(r.va + kCount, uint32_t(std::lround(pixels)));
        std::atomic_thread_fence(std::memory_order_release);
        kernel::W32(r.va + kStatus, 0);
        reports_.pop_front();
    }
}

// ---- present

void MetalBackend::UploadGammaRamp() {
    constexpr int kRed = 0, kGreen = 256, kBlue = 512;
    ramp_dirty_ = false;
    uint8_t px[256 * 4];
    bool identity = true;
    for (int i = 0; i < 256; i++) {
        px[i * 4] = ramp_[kBlue + i];
        px[i * 4 + 1] = ramp_[kGreen + i];
        px[i * 4 + 2] = ramp_[kRed + i];
        px[i * 4 + 3] = 255;
        identity &= ramp_[kRed + i] == i && ramp_[kGreen + i] == i && ramp_[kBlue + i] == i;
    }
    // a new texture rather than a change under draws still in flight
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kBgra width:256 height:1 mipmapped:NO];
    lut_ = [dev_ newTextureWithDescriptor:td];
    [lut_ replaceRegion:MTLRegionMake2D(0, 0, 256, 1) mipmapLevel:0 withBytes:px bytesPerRow:sizeof(px)];
    gamma_ = !identity;
}

void MetalBackend::RenderUi(id<MTLRenderCommandEncoder> enc, MTLRenderPassDescriptor* desc) {
    if (!ui_) return;
    std::lock_guard lock(app::UiLock());
    app::FeedPad();
    ImGui_ImplMetal_NewFrame(desc);
    ImGuiIO& io = ImGui::GetIO();
    const app::Mode px = app::DrawablePixels();
    io.DisplaySize = ImVec2(float(px.width), float(px.height));
    static auto last = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    io.DeltaTime = std::max(1e-4f, std::chrono::duration<float>(now - last).count());
    last = now;
    ImGui::NewFrame();
    app::DrawHud(fps_);
    app::DrawOverlay();
    if (app::kGame.ui) app::kGame.ui(fps_);
    ImGui::Render();
    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), Cb(), enc);
}

void MetalBackend::ReleaseOldTextures() {
    constexpr uint64_t kSweepEvery = 60;
    constexpr size_t kTexturesKept = 2048;
    constexpr uint64_t kIdleFrames = 600;
    if (presents_ % kSweepEvery != 0 || textures_.size() <= kTexturesKept) return;
    for (auto it = textures_.begin(); it != textures_.end();) {
        if (it->second.frame + kIdleFrames < presents_) it = textures_.erase(it);
        else ++it;
    }
}

void MetalBackend::CountFps() {
    fps_n_++;
    auto now = std::chrono::steady_clock::now();
    double el = std::chrono::duration<double>(now - fps_t_).count();
    if (el >= 0.5) {
        fps_ = fps_n_ / el;
        fps_n_ = 0;
        fps_t_ = now;
    }
}

void MetalBackend::Present(const Surface& front) {
    @autoreleasepool {
        Target* t = EnsureTarget(front, false);
        if (t && LatchCpuWrite(*t)) UploadTarget(*t, front);
        if (ramp_dirty_) UploadGammaRamp();
        const app::Mode px = app::DrawablePixels();
        if (uint32_t(layer_.drawableSize.width) != px.width || uint32_t(layer_.drawableSize.height) != px.height)
            layer_.drawableSize = CGSizeMake(px.width, px.height);
        layer_.displaySyncEnabled = app::Get().vsync;
        const auto wait_start = std::chrono::steady_clock::now();
        id<CAMetalDrawable> drawable = [layer_ nextDrawable];
        cost_.drawable_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wait_start).count();
        id<MTLTexture> frame = t ? t->res : nil;
        EndEncoder();
        if (drawable) {
            MTLRenderPassDescriptor* d = [MTLRenderPassDescriptor renderPassDescriptor];
            d.colorAttachments[0].texture = drawable.texture;
            d.colorAttachments[0].loadAction = MTLLoadActionClear;
            d.colorAttachments[0].storeAction = MTLStoreActionStore;
            d.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
            enc_ = [Cb() renderCommandEncoderWithDescriptor:d];
            enc_colour_ = drawable.texture;
            enc_depth_ = nil;
            const float w = float(drawable.texture.width), h = float(drawable.texture.height);
            if (frame) {
                // the frame at its shape (16:9 or 4:3, ultrawide the screen's), centred on black
                float aspect = app::FrameAspect();
                float vw = std::min(w, h * aspect), vh = vw / aspect;
                [enc_ setViewport:MTLViewport{(w - vw) / 2, (h - vh) / 2, vw, vh, 0, 1}];
                [enc_ setScissorRect:MTLScissorRect{0, 0, NSUInteger(w), NSUInteger(h)}];
                float uv[4] = {0, 0, 1, 1};
                Blit(frame, uv, {.gamma = gamma_, .final = true});
            }
            // F12 (or every XBE_SHOT_EVERY frames, testing): the frame as shown, before the interface
            static const uint64_t shot_every = std::getenv("XBE_SHOT_EVERY") ? std::strtoull(std::getenv("XBE_SHOT_EVERY"), nullptr, 10) : 0;
            if (app::TakeScreenshotRequest()) {
                std::vector<uint8_t> shot;
                if (ReadBack(drawable.texture, uint32_t(w), uint32_t(h), shot)) app::SaveScreenshot(std::move(shot), uint32_t(w), uint32_t(h));
                d.colorAttachments[0].loadAction = MTLLoadActionLoad;
                enc_ = [Cb() renderCommandEncoderWithDescriptor:d];
                enc_colour_ = drawable.texture;
            }
            [enc_ setViewport:MTLViewport{0, 0, w, h, 0, 1}];
            [enc_ setScissorRect:MTLScissorRect{0, 0, NSUInteger(w), NSUInteger(h)}];
            RenderUi(enc_, d);
            EndEncoder();
            if (shot_every && presents_ % shot_every == shot_every - 1) {  // testing: with the interface
                std::vector<uint8_t> shot;
                if (ReadBack(drawable.texture, uint32_t(w), uint32_t(h), shot)) app::SaveScreenshot(std::move(shot), uint32_t(w), uint32_t(h));
            }
            [Cb() presentDrawable:drawable];
        }
        // the frame's ring stays untouched until the GPU is done with it
        dispatch_semaphore_t frames = frames_;
        [Cb() addCompletedHandler:^(id<MTLCommandBuffer>) { dispatch_semaphore_signal(frames); }];
        Commit();
        Unbind();
        dispatch_semaphore_wait(frames_, DISPATCH_TIME_FOREVER);
        frame_slot_ = (frame_slot_ + 1) % kFramesInFlight;
        ring_ = rings_[frame_slot_];
        ring_used_ = 0;
        const_last_.clear();
        DeliverReports();
        presents_++;
        presents_seen_.store(presents_);
        // a hitch, and where the frame's time went: what is left over is the title's own code, or waiting on the GPU
        const auto now = std::chrono::steady_clock::now();
        const double frame_ms = std::chrono::duration<double, std::milli>(now - last_present_).count();
        last_present_ = now;
        if (frame_ms > 50 && presents_ > 60 && hitch_lines_ < 500) {
            hitch_lines_++;
            LOGR("hitch: %.0f ms: shaders %.1f (%u), pipelines %.1f (%u), texture uploads %.1f (%u), read-backs %.1f (%u), drawable wait %.1f",
                 frame_ms, cost_.shaders_ms, cost_.shaders, cost_.pipelines_ms, cost_.pipelines, cost_.uploads_ms, cost_.uploads, cost_.readback_ms,
                 cost_.readbacks, cost_.drawable_ms);
        }
        cost_ = {};
        TakeReady();
        if (cache_.archive) {  // the pipelines added lately, at most every half minute
            CacheFiles* files = &cache_;
            dispatch_async(cache_.archive_queue, ^{
                if (files->dirty && std::chrono::steady_clock::now() - files->saved > std::chrono::seconds(30)) SaveArchiveNow(*files);
            });
        }
        ReleaseOldTextures();
        CountFps();
        ApplySettings();
    }
}

void MetalBackend::ApplySettings() {
    if (app::Get().msaa) LOG_FIRST(1, "metal: MSAA is not implemented yet, drawn without");
    if (app::Get().smaa) LOG_FIRST(1, "metal: SMAA is not implemented yet, drawn without");
    float aspect = app::FrameAspect();
    if (std::fabs(aspect - aspect_) > 1e-4f) {
        aspect_ = aspect;
        RescaleTargets();
    }
    // a new scale waits until it has held for half a second: dragging would remake every target each frame
    float want = app::RenderScale(guest_h_);
    static float pending = 0;
    static std::chrono::steady_clock::time_point since;
    auto now = std::chrono::steady_clock::now();
    if (std::fabs(want - scale_) <= 1e-4f) return;
    if (std::fabs(want - pending) > 1e-4f) {
        pending = want;
        since = now;
        return;
    }
    if (now - since < std::chrono::milliseconds(500)) return;
    scale_ = want;
    RescaleTargets();
}

// a new render scale or shape: colour targets keep their pictures, scaled into new ones (a title may draw a target once
// and sample it for long after: PGR2's mini map), depth targets start again
void MetalBackend::RescaleTargets() {
    Unbind();
    for (auto it = targets_.begin(); it != targets_.end();) {
        Target& old = it->second;
        if (old.depth) {
            it = targets_.erase(it);
            continue;
        }
        Target t;
        t.gw = old.gw;
        t.gh = old.gh;
        if (!CreateTarget(t)) {
            it = targets_.erase(it);
            continue;
        }
        SetPassTarget(t.res, nil, t.w, t.h);
        float uv[4] = {0, 0, 1, 1};
        Blit(old.res, uv, {.alpha = true});
        t.data = old.data;
        t.bytes = old.bytes;
        t.epoch = old.epoch;
        t.sig = old.sig;
        t.drawn = old.drawn;
        t.linear32 = old.linear32;
        t.swizzled32 = old.swizzled32;
        t.read = std::move(old.read);
        old = std::move(t);
        ++it;
    }
    EndEncoder();
    std::lock_guard lock(late_lock_);
    late_.clear();
    LOGR("metal: render scale %.2f, %zu targets kept", scale_, targets_.size());
}

}  // namespace

std::unique_ptr<Backend> CreateBackend() { return std::make_unique<MetalBackend>(); }
std::unique_ptr<Backend> CreateFallbackBackend() { return nullptr; }

// the window closed: what the archive has not saved yet
void SaveMetalCache() {
    if (!g_cache_files || !g_cache_files->archive) return;
    CacheFiles* files = g_cache_files;
    dispatch_sync(files->archive_queue, ^{ SaveArchiveNow(*files); });
}

}  // namespace gpu
