// The title's Direct3D 8 calls drawn with Metal, on macOS. The console path of d3d11.cpp, what does not depend on the
// API shared through d3d_common, the generated HLSL translated by gpu/msl.cpp. Not here yet: SMAA, a sampler's LOD
// bias, renderer extensions, F3 captures.
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include <objc/runtime.h>
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
#include "gpu/baked_ao.h"
#include "gpu/d3d_common.h"
#include "app/perf.h"
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

// ---- render passes encoded on worker threads
// A recorder stands in for the render and blit encoders while a run of passes to the game's targets is made (XBE_PARALLEL_SMALL=1:
// only the small square ones, the reflection cubes' faces): the drawing thread does all the work of a draw as ever, and the Metal calls it would have made go in a list. A worker
// replays the list into a command buffer of its own, enqueued when the run began, so the GPU sees everything in order. The
// workers touch nothing but Metal. Each run ends when anything else needs the GPU's order to be settled. XBE_NO_PARALLEL=1: off.
namespace {
struct RecOp {
    enum Kind : uint8_t { BeginRender, EndRender, BeginBlit, EndBlit, BlitCall, Viewport, Scissor, Pipeline, Depth, StencilRef, Cull, Fill, Winding, DepthBias, Blend,
                          VBytes, FBytes, VBuffer, FBuffer, FTexture, FSampler, Draw, DrawInst, DrawIndexed, VisMode, Accel } kind;
    id a = nil, b = nil;
    uint64_t n[6] = {};
    double d[6] = {};
};
struct EncRun {
    std::vector<RecOp> ops;
    std::vector<uint8_t> bytes;
    id<MTLCommandBuffer> cb;
    id rec, blit;
};
}  // namespace

@interface RecEnc : NSObject {
@public
    EncRun* run;
}
@end
@implementation RecEnc
- (RecOp&)op:(RecOp::Kind)k { run->ops.emplace_back(); RecOp& o = run->ops.back(); o.kind = k; return o; }
- (void)endEncoding { [self op:RecOp::EndRender]; }
- (void)setViewport:(MTLViewport)v { RecOp& o = [self op:RecOp::Viewport]; o.d[0] = v.originX, o.d[1] = v.originY, o.d[2] = v.width, o.d[3] = v.height, o.d[4] = v.znear, o.d[5] = v.zfar; }
- (void)setScissorRect:(MTLScissorRect)r { RecOp& o = [self op:RecOp::Scissor]; o.n[0] = r.x, o.n[1] = r.y, o.n[2] = r.width, o.n[3] = r.height; }
- (void)setRenderPipelineState:(id)p { [self op:RecOp::Pipeline].a = p; }
- (void)setDepthStencilState:(id)p { [self op:RecOp::Depth].a = p; }
- (void)setStencilReferenceValue:(uint32_t)v { [self op:RecOp::StencilRef].n[0] = v; }
- (void)setCullMode:(MTLCullMode)v { [self op:RecOp::Cull].n[0] = uint64_t(v); }
- (void)setTriangleFillMode:(MTLTriangleFillMode)v { [self op:RecOp::Fill].n[0] = uint64_t(v); }
- (void)setFrontFacingWinding:(MTLWinding)v { [self op:RecOp::Winding].n[0] = uint64_t(v); }
- (void)setDepthBias:(float)b slopeScale:(float)s clamp:(float)c { RecOp& o = [self op:RecOp::DepthBias]; o.d[0] = b, o.d[1] = s, o.d[2] = c; }
- (void)setBlendColorRed:(float)r green:(float)g blue:(float)b alpha:(float)a { RecOp& o = [self op:RecOp::Blend]; o.d[0] = r, o.d[1] = g, o.d[2] = b, o.d[3] = a; }
- (void)keepBytes:(const void*)p length:(NSUInteger)n index:(NSUInteger)i kind:(RecOp::Kind)k {
    const size_t at = run->bytes.size();
    run->bytes.resize(at + n);
    std::memcpy(run->bytes.data() + at, p, n);
    RecOp& o = [self op:k];
    o.n[0] = at, o.n[1] = n, o.n[2] = i;
}
- (void)setVertexBytes:(const void*)p length:(NSUInteger)n atIndex:(NSUInteger)i { [self keepBytes:p length:n index:i kind:RecOp::VBytes]; }
- (void)setFragmentBytes:(const void*)p length:(NSUInteger)n atIndex:(NSUInteger)i { [self keepBytes:p length:n index:i kind:RecOp::FBytes]; }
- (void)setVertexBuffer:(id)b offset:(NSUInteger)f atIndex:(NSUInteger)i { RecOp& o = [self op:RecOp::VBuffer]; o.a = b, o.n[0] = f, o.n[1] = i; }
- (void)setFragmentBuffer:(id)b offset:(NSUInteger)f atIndex:(NSUInteger)i { RecOp& o = [self op:RecOp::FBuffer]; o.a = b, o.n[0] = f, o.n[1] = i; }
- (void)setFragmentTexture:(id)t atIndex:(NSUInteger)i { RecOp& o = [self op:RecOp::FTexture]; o.a = t, o.n[0] = i; }
- (void)setFragmentSamplerState:(id)t atIndex:(NSUInteger)i { RecOp& o = [self op:RecOp::FSampler]; o.a = t, o.n[0] = i; }
- (void)setFragmentAccelerationStructure:(id)t atBufferIndex:(NSUInteger)i { RecOp& o = [self op:RecOp::Accel]; o.a = t, o.n[0] = i; }
- (void)setVisibilityResultMode:(MTLVisibilityResultMode)m offset:(NSUInteger)f { RecOp& o = [self op:RecOp::VisMode]; o.n[0] = uint64_t(m), o.n[1] = f; }
- (void)drawPrimitives:(MTLPrimitiveType)t vertexStart:(NSUInteger)s vertexCount:(NSUInteger)c { RecOp& o = [self op:RecOp::Draw]; o.n[0] = uint64_t(t), o.n[1] = s, o.n[2] = c; }
- (void)drawPrimitives:(MTLPrimitiveType)t vertexStart:(NSUInteger)s vertexCount:(NSUInteger)c instanceCount:(NSUInteger)ic baseInstance:(NSUInteger)bi {
    RecOp& o = [self op:RecOp::DrawInst]; o.n[0] = uint64_t(t), o.n[1] = s, o.n[2] = c, o.n[3] = ic, o.n[4] = bi;
}
- (void)drawIndexedPrimitives:(MTLPrimitiveType)t indexCount:(NSUInteger)c indexType:(MTLIndexType)it indexBuffer:(id)b indexBufferOffset:(NSUInteger)f instanceCount:(NSUInteger)ic baseVertex:(NSInteger)bv baseInstance:(NSUInteger)bi {
    RecOp& o = [self op:RecOp::DrawIndexed]; o.a = b, o.n[0] = uint64_t(t), o.n[1] = c, o.n[2] = uint64_t(it), o.n[3] = f, o.n[4] = ic, o.n[5] = uint64_t(bv); o.d[0] = double(bi);
}
@end

// blits are few: whatever is called on it is kept as an invocation
@interface RecBlit : NSObject {
@public
    EncRun* run;
}
@end
@implementation RecBlit
- (void)endEncoding { run->ops.emplace_back(); run->ops.back().kind = RecOp::EndBlit; }
- (NSMethodSignature*)methodSignatureForSelector:(SEL)sel {
    struct objc_method_description m = protocol_getMethodDescription(@protocol(MTLBlitCommandEncoder), sel, YES, YES);
    return m.types ? [NSMethodSignature signatureWithObjCTypes:m.types] : [super methodSignatureForSelector:sel];
}
- (void)forwardInvocation:(NSInvocation*)inv {
    [inv retainArguments];
    run->ops.emplace_back();
    RecOp& o = run->ops.back();
    o.kind = RecOp::BlitCall;
    o.a = inv;
}
@end

namespace {
void ReplayRun(EncRun& r) {
    id<MTLRenderCommandEncoder> e = nil;
    id<MTLBlitCommandEncoder> b = nil;
    for (RecOp& o : r.ops) {
        switch (o.kind) {
            case RecOp::BeginRender: e = [r.cb renderCommandEncoderWithDescriptor:(MTLRenderPassDescriptor*)o.a]; break;
            case RecOp::EndRender: [e endEncoding]; e = nil; break;
            case RecOp::BeginBlit: b = [r.cb blitCommandEncoderWithDescriptor:(MTLBlitPassDescriptor*)o.a]; break;
            case RecOp::EndBlit: [b endEncoding]; b = nil; break;
            case RecOp::BlitCall: [(NSInvocation*)o.a invokeWithTarget:b]; break;
            case RecOp::Viewport: [e setViewport:MTLViewport{o.d[0], o.d[1], o.d[2], o.d[3], o.d[4], o.d[5]}]; break;
            case RecOp::Scissor: [e setScissorRect:MTLScissorRect{o.n[0], o.n[1], o.n[2], o.n[3]}]; break;
            case RecOp::Pipeline: [e setRenderPipelineState:o.a]; break;
            case RecOp::Depth: [e setDepthStencilState:o.a]; break;
            case RecOp::StencilRef: [e setStencilReferenceValue:uint32_t(o.n[0])]; break;
            case RecOp::Cull: [e setCullMode:MTLCullMode(o.n[0])]; break;
            case RecOp::Fill: [e setTriangleFillMode:MTLTriangleFillMode(o.n[0])]; break;
            case RecOp::Winding: [e setFrontFacingWinding:MTLWinding(o.n[0])]; break;
            case RecOp::DepthBias: [e setDepthBias:float(o.d[0]) slopeScale:float(o.d[1]) clamp:float(o.d[2])]; break;
            case RecOp::Blend: [e setBlendColorRed:float(o.d[0]) green:float(o.d[1]) blue:float(o.d[2]) alpha:float(o.d[3])]; break;
            case RecOp::VBytes: [e setVertexBytes:r.bytes.data() + o.n[0] length:o.n[1] atIndex:o.n[2]]; break;
            case RecOp::FBytes: [e setFragmentBytes:r.bytes.data() + o.n[0] length:o.n[1] atIndex:o.n[2]]; break;
            case RecOp::VBuffer: [e setVertexBuffer:o.a offset:o.n[0] atIndex:o.n[1]]; break;
            case RecOp::FBuffer: [e setFragmentBuffer:o.a offset:o.n[0] atIndex:o.n[1]]; break;
            case RecOp::FTexture: [e setFragmentTexture:o.a atIndex:o.n[0]]; break;
            case RecOp::FSampler: [e setFragmentSamplerState:o.a atIndex:o.n[0]]; break;
            case RecOp::Accel: [e setFragmentAccelerationStructure:o.a atBufferIndex:o.n[0]]; break;
            case RecOp::VisMode: [e setVisibilityResultMode:MTLVisibilityResultMode(o.n[0]) offset:o.n[1]]; break;
            case RecOp::Draw: [e drawPrimitives:MTLPrimitiveType(o.n[0]) vertexStart:o.n[1] vertexCount:o.n[2]]; break;
            case RecOp::DrawInst: [e drawPrimitives:MTLPrimitiveType(o.n[0]) vertexStart:o.n[1] vertexCount:o.n[2] instanceCount:o.n[3] baseInstance:o.n[4]]; break;
            case RecOp::DrawIndexed:
                [e drawIndexedPrimitives:MTLPrimitiveType(o.n[0]) indexCount:o.n[1] indexType:MTLIndexType(o.n[2]) indexBuffer:o.a indexBufferOffset:o.n[3] instanceCount:o.n[4] baseVertex:NSInteger(o.n[5]) baseInstance:NSUInteger(o.d[0])];
                break;
        }
    }
}
}  // namespace


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
constexpr int kFramesInFlight = 4;  // rings: the frame being written and up to three the GPU has yet to finish
constexpr int kLateSlots = 3;              // copies in flight per surface locked without a wait
constexpr uint32_t kVisSlots = 1u << 16;   // visibility counters, 8 bytes each
constexpr uint32_t kVertexBufferBase = 1;  // buffer 0 holds the constants

// a console surface drawn into
struct Target {
    id<MTLTexture> res;
    id<MTLTexture> fb;  // copy read by draws that sample the target they render to
    // MSAA: a multisampled twin that draws go to. Each pass on it resolves into res as it ends, so res is always current for readers
    id<MTLTexture> ms;
    bool ms_stale = false;  // res was written without the twin: the twin takes res's pixels before its next draw
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
            // what a report waits for may have gone in an earlier commit: a title polling for it in a loop without presenting
            // has no later flush to deliver it (PGR2 froze at a race's end, waiting on every light test in 0x187DC0). The
            // queue finishes in order, and a completed handler can still be to come after this wait, so the serial moves here.
            Commit();
            if (ReportsPending() && last_sent_) {
                const auto wait_start = std::chrono::steady_clock::now();
                [last_sent_ waitUntilCompleted];
                perf::AddReportWait(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wait_start).count());
                const uint64_t serial = next_serial_ - 1;
                uint64_t seen = completed_serial_.load();
                while (seen < serial && !completed_serial_.compare_exchange_weak(seen, serial)) {}
            }
            DeliverReports();
        }
    }
    // a report asked for and the title not waiting: what is recorded goes to the GPU and this thread goes on. Each command
    // buffer's completion delivers the reports it finishes (Commit), so a title polling without presenting still gets them.
    // XBE_REPORTS_BLOCK=1: this thread waits for the GPU here, as Flush does
    void Kick() override {
        static const bool block = std::getenv("XBE_REPORTS_BLOCK") && *std::getenv("XBE_REPORTS_BLOCK") == '1';
        if (block || g_reports_wait.load(std::memory_order_relaxed)) return Flush();
        @autoreleasepool {
            Commit();
            DeliverReports();
        }
    }
    void SetGamma(const uint8_t* ramp) override {
        std::memcpy(ramp_, ramp, sizeof(ramp_));
        ramp_dirty_ = true;
    }
    void ClearReport() override { counted_.clear(); }
    void GetReport(uint32_t report) override {
        std::lock_guard lock(reports_lock_);
        reports_.push_back({report, counted_});
    }
    bool ReportsPending() {
        std::lock_guard lock(reports_lock_);
        return !reports_.empty();
    }

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
    void OpenPass(id<MTLTexture> colour, id<MTLTexture> depth, id<MTLTexture> resolve_colour = nil, id<MTLTexture> resolve_depth = nil);
    id<MTLBlitCommandEncoder> Blitter();  // ends the render pass; end it after use
    // the run of reflection passes being recorded for a worker, null none
    std::shared_ptr<EncRun> run_;
    bool draw_counting_ = false;  // the draw being made is under a visibility test: never in a run
    bool RunWanted(id<MTLTexture> colour) const;
    void StartRun();
    void EndRun();
    void BeginRender(MTLRenderPassDescriptor* d, id<MTLTexture> colour);
    void ClearRender(MTLRenderPassDescriptor* d);
    Alloc Reserve(uint32_t bytes, uint32_t align = 16);
    // targets
    uint64_t SizedTargetKey(uint32_t data, bool depth, uint32_t gw, uint32_t gh) const;
    Target* FindTarget(const Surface& s, bool depth);
    Target* EnsureTarget(const Surface& s, bool depth, uint32_t gw = 0, uint32_t gh = 0);
    Target* DepthTarget(const State& s);
    bool CreateTarget(Target& t);
    bool MsEligible(const Target& t) const;
    bool EnsureMs(Target& t);
    void RefreshMs(Target& t);
    uint32_t msaa_ = 1;
    bool bound_ms_ = false;  // the bound targets draw into their twins
    float small_factor_ = 1.0f;  // g_small_target_factor as the targets were made with it
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
    id<MTLRenderPipelineState> PassPipeline(const PassFn& f, MTLPixelFormat colour, MTLPixelFormat depth, bool over, NSUInteger samples);
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
    id<MTLCommandBuffer> last_sent_;  // the newest committed
    id<MTLRenderCommandEncoder> enc_;
    id<MTLTexture> enc_colour_, enc_depth_;
    id<MTLBuffer> rings_[kFramesInFlight];
    id<MTLBuffer> ring_;  // this frame's, or a bigger one when it filled
    uint32_t ring_used_ = 0;
    int frame_slot_ = 0;
    dispatch_semaphore_t frames_ = nullptr;
    std::atomic<uint64_t> completed_serial_{0};
    // perf: each pass's GPU time, from timestamps at its stage boundaries; one buffer a frame in flight
    static constexpr uint32_t kTimestampSamples = 4096;
    id<MTLCounterSampleBuffer> ts_[kFramesInFlight] = {};
    bool ts_tried_ = false;
    uint32_t ts_used_ = 0;
    // each encoder sampled (four samples each), for GPU time per pass and encoders.csv
    enum EncoderKind : uint8_t { kEncDraw, kEncClear, kEncPresent, kEncBlit };
    struct EncoderInfo {
        EncoderKind kind = kEncDraw;
        int pass = 0;
        uint32_t w = 0, h = 0, format = 0, zw = 0, zh = 0;
        uint8_t cload = 0, cstore = 0, zload = 0, zstore = 0;
        uint32_t draws = 0, vertices = 0;
        uint64_t vs = 0, ps = 0;
        uint32_t tex_w = 0, tex_h = 0, tex_format = 0;
        bool flat = false, blend = false;
    };
    std::vector<EncoderInfo> ts_pass_;
    int ts_cur_ = -1;           // the open encoder's, -1 none
    id<MTLTexture> enc_tex_[4] = {};        // what the open encoder has bound, to skip binds of the same thing
    id<MTLSamplerState> enc_smp_[4] = {};
    bool ts_probe_ = false;     // this frame: each screen-space draw an encoder of its own, to time it (1 frame in 120)
    void SampleRender(MTLRenderPassDescriptor* d, EncoderKind kind);
    MTLBlitPassDescriptor* SampleBlit();
    MTLTimestamp ts_cpu_ = 0, ts_gpu_ = 0;
    double ts_ns_per_tick_ = 1.0;
    std::atomic<double> gpu_last_end_{0};  // the end of the last command buffer: frames in the queue overlap
    bool EnsureTimestamps();
    uint64_t next_serial_ = 1;
    id<MTLBuffer> vis_;
    uint32_t vis_next_ = 0;
    // constants: the block in the ring holding what the last draw used
    Alloc const_at_;
    uint8_t const_last_[sizeof(State::vsc) + sizeof(DrawConstants)] = {};  // what const_at_ holds
    bool const_whole_ = false;  // const_last_ has the fixed function constants
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
    std::mutex reports_lock_;                 // reports_: the drawing thread, and command buffers completing
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
    // ray traced reflections (DrawAdjust::trace_*): the frame's scene as its draws give it, built at the present into the
    // slot of the frame drawn next, which traces against it
    struct TraceSlot {
        id<MTLBuffer> verts, colours, scratch;
        id<MTLAccelerationStructure> as;
        std::vector<id<MTLTexture>> tex;  // scene_tex_ as it was built
        std::vector<uint32_t> object;     // each draw's object (by DrawAdjust::trace_draw)
    };
    TraceSlot trace_[kFramesInFlight];
    std::vector<float> scene_xyz_;         // three x y z per triangle, in the world
    // per triangle kTraceWords: three texture coordinates (half2), normals (snorm4x8), lit colours (RGBA8, alpha the
    // cube's share), then the material: its texture's index in scene_tex_ (255 none) | 256 glass | its object << 16
    static constexpr size_t kTraceWords = 10, kSceneTextures = 16;
    std::vector<uint32_t> scene_colour_;
    std::vector<id<MTLTexture>> scene_tex_;
    struct SceneDraw {  // by DrawAdjust::trace_draw
        float lo[3] = {}, hi[3] = {};
        size_t first = 0, end = 0;  // its triangles
    };
    std::vector<SceneDraw> scene_draws_;
    struct SceneMesh {  // a mesh's triangles as its draws read them (DrawAdjust::trace_mesh), x y z each
        std::vector<float> xyz, normals, uvs;
        uint64_t used = 0;  // presents_
    };
    std::unordered_map<uint64_t, SceneMesh> scene_meshes_;
    int trace_live_ = -1;                  // the slot this frame traces against, -1 none
    bool trace_ok_ = false;                // the device traces rays from fragment functions
    const DrawAdjust* carried_ = nullptr;  // the filter's answer for a draw drawn again with another stage 3
    void AddToScene(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices, const DrawAdjust& a);
    void BuildScene();
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
        // one permit per ring but the one the next frame writes: a frame waits after it is committed, so with a permit per
        // ring the fourth frame wrote the first one's ring while the GPU still drew it (under GPU load: the start of the
        // frame, the race's mirror, drawn from the next frame's vertices and constants: coloured blocks, a flickering HUD)
        // XBE_FRAMES_AHEAD=1..3 (testing): fewer frames queued
        static const long ahead = std::getenv("XBE_FRAMES_AHEAD") ? std::clamp(std::atol(std::getenv("XBE_FRAMES_AHEAD")), 1L, long(kFramesInFlight - 1)) : kFramesInFlight - 1;
        frames_ = dispatch_semaphore_create(ahead);
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
        trace_ok_ = dev_.supportsRaytracing && dev_.supportsRaytracingFromRender;
        OpenCache();
        LOGR("metal: drawing with %s, %ux%u guest, render scale %.3f%s", dev_.name.UTF8String, width, height, scale_,
             trace_ok_ ? ", rays traced from fragment shaders" : ", no ray tracing from fragment shaders");
        return true;
    }
}

// ---- recording

bool MetalBackend::RunWanted(id<MTLTexture> colour) const {
    static const bool off = std::getenv("XBE_NO_PARALLEL") && *std::getenv("XBE_NO_PARALLEL") == '1';
    static const bool small_only = std::getenv("XBE_PARALLEL_SMALL") && *std::getenv("XBE_PARALLEL_SMALL") == '1';  // the reflection cubes alone
    return !off && colour && !draw_counting_ && (!small_only || (colour.width == colour.height && colour.width <= 1400));
}

void MetalBackend::StartRun() {
    Commit();  // what came before goes first
    run_ = std::make_shared<EncRun>();
    run_->ops.reserve(1u << 16);  // a world pass is tens of thousands of calls: no regrowth on the drawing thread
    run_->cb = [queue_ commandBuffer];
    [run_->cb enqueue];  // its place in the GPU's order, whenever it is committed
    RecEnc* rec = [RecEnc new];
    rec->run = run_.get();
    run_->rec = rec;
    RecBlit* blit = [RecBlit new];
    blit->run = run_.get();
    run_->blit = blit;
    perf::Add(perf::kNCommandBuffers);
    const uint64_t frame = presents_;
    auto* last_end = &gpu_last_end_;
    [run_->cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
        const double from = std::max(b.GPUStartTime, last_end->load()), to = b.GPUEndTime;
        if (to > last_end->load()) last_end->store(to);
        if (perf::On() && to > from) perf::Gpu(frame, (to - from) * 1000.0);
    }];
}

void MetalBackend::EndRun() {
    if (!run_) return;
    if (enc_) EndEncoder();
    std::shared_ptr<EncRun> r = std::move(run_);
    run_ = nullptr;
    last_sent_ = r->cb;
    static dispatch_queue_t workers = dispatch_queue_create("reflection encoding", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_CONCURRENT, QOS_CLASS_USER_INTERACTIVE, 0));
    dispatch_async(workers, ^{
        @autoreleasepool {
            ReplayRun(*r);
            [r->cb commit];
        }
    });
}

void MetalBackend::BeginRender(MTLRenderPassDescriptor* d, id<MTLTexture> colour) {
    if (RunWanted(colour)) {
        if (!run_) StartRun();
        run_->ops.emplace_back();
        run_->ops.back().kind = RecOp::BeginRender;
        run_->ops.back().a = d;
        enc_ = (id<MTLRenderCommandEncoder>)run_->rec;
        return;
    }
    EndRun();
    enc_ = [Cb() renderCommandEncoderWithDescriptor:d];
}

// a pass that only clears
void MetalBackend::ClearRender(MTLRenderPassDescriptor* d) {
    if (RunWanted(d.colorAttachments[0].texture) || (run_ && !d.colorAttachments[0].texture && d.depthAttachment.texture && d.depthAttachment.texture.width <= 1400)) {
        if (!run_) StartRun();
        run_->ops.emplace_back();
        run_->ops.back().kind = RecOp::BeginRender;
        run_->ops.back().a = d;
        run_->ops.emplace_back();
        run_->ops.back().kind = RecOp::EndRender;
        return;
    }
    EndRun();
    [[Cb() renderCommandEncoderWithDescriptor:d] endEncoding];
}

id<MTLCommandBuffer> MetalBackend::Cb() {
    if (run_) EndRun();  // anything else's order is settled by the run going first
    if (!cb_) cb_ = [queue_ commandBuffer];
    return cb_;
}

void MetalBackend::EndEncoder() {
    if (!enc_) return;
    [enc_ endEncoding];
    ts_cur_ = -1;
    for (auto& t : enc_tex_) t = nil;
    for (auto& t : enc_smp_) t = nil;
    enc_ = nil;
    enc_colour_ = enc_depth_ = nil;
    enc_pipeline_ = nil;
    enc_depth_state_ = nil;
    enc_has_scissor_ = enc_has_viewport_ = const_bound_ = false;
}

id<MTLCommandBuffer> MetalBackend::Commit() {
    EndRun();
    EndEncoder();
    id<MTLCommandBuffer> sent = cb_;
    if (!sent) return nil;
    uint64_t serial = next_serial_++;
    perf::Add(perf::kNCommandBuffers);
    for (Query& q : counted_)
        if (!q.serial) q.serial = serial;
    {
        std::lock_guard lock(reports_lock_);
        for (Report& r : reports_)
            for (Query& q : r.queries)
                if (!q.serial) q.serial = serial;
    }
    auto* done = &completed_serial_;
    const uint64_t frame = presents_;  // perf: the frame it was sent in, for its GPU time
    auto* last_end = &gpu_last_end_;
    [sent addCompletedHandler:^(id<MTLCommandBuffer> b) {
        uint64_t seen = done->load();
        while (seen < serial && !done->compare_exchange_weak(seen, serial)) {}
        DeliverReports();
        // busy time, not each buffer's span: buffers queued behind one another overlap in their start and end times
        const double from = std::max(b.GPUStartTime, last_end->load()), to = b.GPUEndTime;
        if (to > last_end->load()) last_end->store(to);
        if (perf::On() && to > from) perf::Gpu(frame, (to - from) * 1000.0);
    }];
    [sent commit];
    cb_ = nil;
    last_sent_ = sent;
    return sent;
}

// each pass on a twin ends by resolving into the target: the first sample for depth and stencil, as d3d11.cpp's resolve
static void ResolveInto(MTLRenderPassColorAttachmentDescriptor* a, id<MTLTexture> to) {
    a.resolveTexture = to;
    a.storeAction = MTLStoreActionStoreAndMultisampleResolve;
}
static void ResolveDepthInto(MTLRenderPassDescriptor* d, id<MTLTexture> to) {
    d.depthAttachment.resolveTexture = to;
    d.depthAttachment.depthResolveFilter = MTLMultisampleDepthResolveFilterSample0;
    d.depthAttachment.storeAction = MTLStoreActionStoreAndMultisampleResolve;
    d.stencilAttachment.resolveTexture = to;
    d.stencilAttachment.stencilResolveFilter = MTLMultisampleStencilResolveFilterSample0;
    d.stencilAttachment.storeAction = MTLStoreActionStoreAndMultisampleResolve;
}

void MetalBackend::OpenPass(id<MTLTexture> colour, id<MTLTexture> depth, id<MTLTexture> resolve_colour, id<MTLTexture> resolve_depth) {
    if (enc_ && enc_colour_ == colour && enc_depth_ == depth) return;
    EndEncoder();
    MTLRenderPassDescriptor* d = [MTLRenderPassDescriptor renderPassDescriptor];
    if (colour) {
        d.colorAttachments[0].texture = colour;
        d.colorAttachments[0].loadAction = MTLLoadActionLoad;
        d.colorAttachments[0].storeAction = MTLStoreActionStore;
        if (resolve_colour) ResolveInto(d.colorAttachments[0], resolve_colour);
    }
    if (depth) {
        d.depthAttachment.texture = depth;
        d.depthAttachment.loadAction = MTLLoadActionLoad;
        d.depthAttachment.storeAction = MTLStoreActionStore;
        d.stencilAttachment.texture = depth;
        d.stencilAttachment.loadAction = MTLLoadActionLoad;
        d.stencilAttachment.storeAction = MTLStoreActionStore;
        if (resolve_depth) ResolveDepthInto(d, resolve_depth);
    }
    d.visibilityResultBuffer = vis_;
    perf::Add(perf::kNEncoders);
    SampleRender(d, kEncDraw);
    BeginRender(d, colour);
    enc_colour_ = colour;
    enc_depth_ = depth;
}

// perf: the encoder's vertex and fragment time, and what it draws to
void MetalBackend::SampleRender(MTLRenderPassDescriptor* d, EncoderKind kind) {
    if (!perf::On() || !EnsureTimestamps() || ts_used_ + 4 > kTimestampSamples) return;
    MTLRenderPassSampleBufferAttachmentDescriptor* a = d.sampleBufferAttachments[0];
    a.sampleBuffer = ts_[frame_slot_];
    a.startOfVertexSampleIndex = ts_used_;
    a.endOfVertexSampleIndex = ts_used_ + 1;
    a.startOfFragmentSampleIndex = ts_used_ + 2;
    a.endOfFragmentSampleIndex = ts_used_ + 3;
    ts_used_ += 4;
    EncoderInfo e;
    e.kind = kind;
    e.pass = perf::CurrentPass();
    if (id<MTLTexture> c = d.colorAttachments[0].texture) e.w = uint32_t(c.width), e.h = uint32_t(c.height), e.format = uint32_t(c.pixelFormat);
    if (id<MTLTexture> z = d.depthAttachment.texture) e.zw = uint32_t(z.width), e.zh = uint32_t(z.height);
    e.cload = uint8_t(d.colorAttachments[0].loadAction), e.cstore = uint8_t(d.colorAttachments[0].storeAction);
    e.zload = uint8_t(d.depthAttachment.loadAction), e.zstore = uint8_t(d.depthAttachment.storeAction);
    ts_pass_.push_back(e);
    ts_cur_ = kind == kEncDraw ? int(ts_pass_.size()) - 1 : -1;
}

// a blit encoder timed from start to end (the first two of its four samples)
MTLBlitPassDescriptor* MetalBackend::SampleBlit() {
    MTLBlitPassDescriptor* d = [MTLBlitPassDescriptor blitPassDescriptor];
    if (!perf::On() || !EnsureTimestamps() || ts_used_ + 4 > kTimestampSamples) return d;
    MTLBlitPassSampleBufferAttachmentDescriptor* a = d.sampleBufferAttachments[0];
    a.sampleBuffer = ts_[frame_slot_];
    a.startOfEncoderSampleIndex = ts_used_;
    a.endOfEncoderSampleIndex = ts_used_ + 1;
    ts_used_ += 4;
    EncoderInfo e;
    e.kind = kEncBlit;
    e.pass = perf::CurrentPass();
    ts_pass_.push_back(e);
    return d;
}

bool MetalBackend::EnsureTimestamps() {
    if (ts_tried_) return ts_[0] != nil;
    ts_tried_ = true;
    if (![dev_ supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
        LOGR("perf: this GPU cannot time its stages: no GPU time per pass");
        return false;
    }
    id<MTLCounterSet> set = nil;
    for (id<MTLCounterSet> s in dev_.counterSets)
        if ([s.name isEqualToString:MTLCommonCounterSetTimestamp]) set = s;
    if (!set) return false;
    MTLCounterSampleBufferDescriptor* d = [MTLCounterSampleBufferDescriptor new];
    d.counterSet = set;
    d.storageMode = MTLStorageModeShared;
    d.sampleCount = kTimestampSamples;
    for (auto& t : ts_) {
        NSError* err = nil;
        t = [dev_ newCounterSampleBufferWithDescriptor:d error:&err];
        if (!t) {
            LOGR("perf: no timestamp buffer: %s", err.localizedDescription.UTF8String);
            ts_[0] = nil;
            return false;
        }
    }
    return true;
}

id<MTLBlitCommandEncoder> MetalBackend::Blitter() {
    EndEncoder();
    perf::Add(perf::kNBlits);
    if (run_) {
        run_->ops.emplace_back();
        run_->ops.back().kind = RecOp::BeginBlit;
        run_->ops.back().a = SampleBlit();
        return (id<MTLBlitCommandEncoder>)run_->blit;
    }
    return [Cb() blitCommandEncoderWithDescriptor:SampleBlit()];
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

// the scene's targets, as big as the console's screen, and the depth drawn with them. Not the water's reflection (g_small_target)
bool MetalBackend::MsEligible(const Target& t) const {
    if (msaa_ <= 1 || (t.data && (t.data == g_small_target[0].load() || t.data == g_small_target[1].load()))) return false;
    return t.depth || (t.gw == guest_w_ && t.gh == guest_h_);
}

bool MetalBackend::EnsureMs(Target& t) {
    if (t.ms) return true;
    if (![dev_ supportsTextureSampleCount:msaa_]) return false;
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:t.depth ? kDepth : kBgra width:t.w height:t.h mipmapped:NO];
    td.textureType = MTLTextureType2DMultisample;
    td.sampleCount = msaa_;
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageRenderTarget;
    t.ms = [dev_ newTextureWithDescriptor:td];
    if (!t.ms) {
        LOG_FIRST(4, "metal: MSAA twin of %08X %ux%u could not be made", t.data, t.w, t.h);
        return false;
    }
    t.ms_stale = true;
    LOG_FIRST(8, "metal: MSAA %ux twin for %s %08X %ux%u", msaa_, t.depth ? "depth" : "target", t.data, t.w, t.h);
    return true;
}

// res's pixels into the twin, after something other than the twin wrote them
void MetalBackend::RefreshMs(Target& t) {
    if (!t.ms || !t.ms_stale) return;
    t.ms_stale = false;
    if (!t.depth) {
        SetPassTarget(t.ms, nil, t.w, t.h);
        float uv[4] = {0, 0, 1, 1};
        Blit(t.res, uv, {.alpha = true});
        return;
    }
    PassFn f = PassFunctions(kDepthHlsl, "vs", "psmain");
    if (!f.vs || !f.fs) return;
    SetPassTarget(nil, t.ms, t.w, t.h);
    id<MTLTexture> tex = t.res;
    FullScreen(f, nullptr, 0, &tex, 1, false, depth_always_);
}

bool MetalBackend::CreateTarget(Target& t) {
    const float small = t.data && (t.data == g_small_target[0].load() || t.data == g_small_target[1].load()) ? small_factor_ : 1.0f;
    t.w = std::max(1u, uint32_t(std::lround(t.gw * scale_ * small * XStretch(aspect_, guest_h_, t.gw, t.gh))));
    t.h = std::max(1u, uint32_t(std::lround(t.gh * scale_ * small)));
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
    SampleRender(d, kEncClear);
    ClearRender(d);
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
    t.data = data;  // CreateTarget's size may depend on it
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
    const bool ms = rt && MsEligible(*rt) && EnsureMs(*rt) && (!ds || EnsureMs(*ds));
    for (Target* t : {rt, ds}) {
        if (!t) continue;
        if (ms) RefreshMs(*t);
        else if (t->ms) t->ms_stale = true;
        t->drawn = true;
    }
    bound_rt_ = rt;
    bound_ds_ = ds;
    bound_ms_ = ms;
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
    if (t.ms) t.ms_stale = true;
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
    EndRun();  // the copy below is waited for: it goes in a command buffer of this thread's
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

id<MTLRenderPipelineState> MetalBackend::PassPipeline(const PassFn& f, MTLPixelFormat colour, MTLPixelFormat depth, bool over, NSUInteger samples) {
    if (!f.vs || !f.fs) return nil;
    uint64_t key = uint64_t((__bridge void*)f.vs) * 31 + uint64_t((__bridge void*)f.fs);
    (void)key;
    std::string name = std::string(f.vs.name.UTF8String) + "/" + f.fs.name.UTF8String;
    key = Hash(name.data(), name.size(), uint64_t(colour) * 1000 + uint64_t(depth) * 3 + (over ? 1 : 0) + uint64_t(samples) * 1000000);
    auto it = pass_pipelines_.find(key);
    if (it != pass_pipelines_.end()) return it->second;
    MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = f.vs;
    d.fragmentFunction = f.fs;
    d.rasterSampleCount = samples;
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
    const NSUInteger samples = enc_colour_ ? enc_colour_.sampleCount : enc_depth_ ? enc_depth_.sampleCount : 1;
    id<MTLRenderPipelineState> state = PassPipeline(f, cf, df, over, samples);
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
    for (uint32_t i = 0; i < 4; i++) enc_tex_[i] = nil, enc_smp_[i] = nil;  // the pass's own, not the draws'
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
    if (dst.ms) dst.ms_stale = true;
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
    id<MTLTexture> into = t.ms ? t.ms : t.res;  // a twin is cleared and resolved
    if (t.ms && (!t.depth || ((flags & kClearZ) && (flags & kClearStencil)))) t.ms_stale = false;
    if (t.depth) {
        d.depthAttachment.texture = into;
        d.depthAttachment.loadAction = (flags & kClearZ) ? MTLLoadActionClear : MTLLoadActionLoad;
        d.depthAttachment.storeAction = MTLStoreActionStore;
        d.depthAttachment.clearDepth = std::clamp(z, 0.0f, 1.0f);
        d.stencilAttachment.texture = into;
        d.stencilAttachment.loadAction = (flags & kClearStencil) ? MTLLoadActionClear : MTLLoadActionLoad;
        d.stencilAttachment.storeAction = MTLStoreActionStore;
        d.stencilAttachment.clearStencil = stencil & 0xFF;
        if (t.ms) ResolveDepthInto(d, t.res);
    } else {
        d.colorAttachments[0].texture = into;
        d.colorAttachments[0].loadAction = MTLLoadActionClear;
        d.colorAttachments[0].storeAction = MTLStoreActionStore;
        d.colorAttachments[0].clearColor = MTLClearColorMake(colour[0], colour[1], colour[2], colour[3]);
        if (t.ms) ResolveInto(d.colorAttachments[0], t.res);
    }
    SampleRender(d, kEncClear);
    ClearRender(d);
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
    if (t.ms) t.ms_stale = true;
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
    if (dst_t.ms) dst_t.ms_stale = true;
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
        if (dst_t.ms) dst_t.ms_stale = true;
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
        if (r && ((r->sharpened && !app::Get().sharp_2d) || !ReplacementFits(*r, d))) r = nullptr;
        if (id<MTLTexture> t = r ? BindReplacement(*r, d, stages.levels[stage]) : nil) {
            stages.replaced[stage] = true;
            return t;
        }
    }
    if (draw_2d_) Sharpen2D(d);
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

// ---- ray traced reflections

uint64_t g_trace_presents = 0;  // MetalBackend::presents_, for TraceLogFrame

// XBE_TRACE_LOG=frame: that frame's scene and traced draws in the log
bool TraceLogFrame() {
    static const uint64_t frame = std::getenv("XBE_TRACE_LOG") ? std::strtoull(std::getenv("XBE_TRACE_LOG"), nullptr, 10) : 0;
    return frame && g_trace_presents == frame;
}

// a draw's triangles into the frame's scene, in the world, with what shades a hit on each (DrawAdjust::trace_scene)
void MetalBackend::AddToScene(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices, const DrawAdjust& a) {
    constexpr size_t kMaxTriangles = 1u << 20;
    if (!trace_ok_ || scene_colour_.size() >= kMaxTriangles * kTraceWords) return;
    const size_t from = scene_xyz_.size();
    // the mesh's vertices read once (a car's meshes stay as they are), kept while drawn
    thread_local SceneMesh fresh;
    SceneMesh* mesh = nullptr;
    if (a.trace_mesh) {
        auto it = scene_meshes_.find(a.trace_mesh);
        if (it != scene_meshes_.end()) mesh = &it->second;
    }
    if (!mesh) {
        fresh.xyz.clear(), fresh.normals.clear(), fresh.uvs.clear();
        if (!DrawTriangles(s, prim, start, count, indices, 0, fresh.xyz)) return;
        if (!DrawTriangles(s, prim, start, count, indices, 1, fresh.normals) || fresh.normals.size() != fresh.xyz.size()) fresh.normals.clear();
        if (!DrawTriangles(s, prim, start, count, indices, 2, fresh.uvs) || fresh.uvs.size() != fresh.xyz.size()) fresh.uvs.clear();
        if (scene_meshes_.size() > 4096) scene_meshes_.clear();
        mesh = a.trace_mesh ? &(scene_meshes_[a.trace_mesh] = fresh) : &fresh;
    }
    mesh->used = presents_;
    scene_xyz_.insert(scene_xyz_.end(), mesh->xyz.begin(), mesh->xyz.end());
    const std::vector<float>& normals = mesh->normals;
    const std::vector<float>& uvs = mesh->uvs;
    const size_t n_verts = mesh->xyz.size() / 3;
    const bool have_n = !normals.empty();
    const bool have_uv = a.trace_uv != 0 && !uvs.empty();
    const float* m = a.trace_world;
    for (size_t i = from; i + 2 < scene_xyz_.size(); i += 3) {
        const float x = scene_xyz_[i], y = scene_xyz_[i + 1], z = scene_xyz_[i + 2];
        for (int r = 0; r < 3; r++) scene_xyz_[i + size_t(r)] = m[r * 4] * x + m[r * 4 + 1] * y + m[r * 4 + 2] * z + m[r * 4 + 3];
    }
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    for (size_t i = from; i + 2 < scene_xyz_.size(); i += 3)
        for (int k = 0; k < 3; k++) lo[k] = std::min(lo[k], scene_xyz_[i + size_t(k)]), hi[k] = std::max(hi[k], scene_xyz_[i + size_t(k)]);
    if (a.trace_draw < 4096) {
        if (scene_draws_.size() <= a.trace_draw) scene_draws_.resize(a.trace_draw + 1);
        SceneDraw& sd = scene_draws_[a.trace_draw];
        std::memcpy(sd.lo, lo, sizeof lo);
        std::memcpy(sd.hi, hi, sizeof hi);
        sd.first = scene_colour_.size() / kTraceWords;
        sd.end = sd.first + n_verts / 3;
    }
    if (TraceLogFrame()) {
        LOGR("trace scene: vs %08X rt %08X %zu triangles, x %.2f %.2f y %.2f %.2f z %.2f %.2f, normals %d uvs %d", s.vs_handle, s.rt.data,
             n_verts / 3, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], int(have_n), int(have_uv));
    }
    // the material: stage 0's texture, one of the frame's few
    uint32_t material = 255;
    if (have_uv && !a.trace_glass) {
        DrawTextures stages{};
        id<MTLTexture> tex = BindTexture(s, 0, stages);
        if (tex && tex.textureType == MTLTextureType2D) {
            auto it = std::find(scene_tex_.begin(), scene_tex_.end(), tex);
            if (it != scene_tex_.end()) material = uint32_t(it - scene_tex_.begin());
            else if (scene_tex_.size() < kSceneTextures) material = uint32_t(scene_tex_.size()), scene_tex_.push_back(tex);
        }
    }
    if (a.trace_glass) material |= 256;
    auto byte = [](float v) { return uint32_t(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f); };
    auto snorm = [](float v) { return uint32_t(int32_t(std::lround(std::clamp(v, -1.0f, 1.0f) * 127.0f)) & 0xFF); };
    auto half = [](float v) {
        const _Float16 h = _Float16(std::clamp(v, -60000.0f, 60000.0f));
        uint16_t b;
        std::memcpy(&b, &h, 2);
        return uint32_t(b);
    };
    const float* l = a.trace_light;
    const float* c = a.trace_colour;
    for (size_t v = 0; v + 2 < n_verts; v += 3) {
        const float* p = &scene_xyz_[from + v * 3];
        const float e1[3] = {p[3] - p[0], p[4] - p[1], p[5] - p[2]}, e2[3] = {p[6] - p[0], p[7] - p[1], p[8] - p[2]};
        const float face[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
        uint32_t w[kTraceWords];
        for (int k = 0; k < 3; k++) {
            float n[3];
            if (have_n) {
                const float* o = &normals[(v + size_t(k)) * 3];
                for (int r = 0; r < 3; r++) n[r] = m[r * 4] * o[0] + m[r * 4 + 1] * o[1] + m[r * 4 + 2] * o[2];
            } else std::memcpy(n, face, sizeof n);
            const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            if (len > 0) n[0] /= len, n[1] /= len, n[2] /= len;
            const float nl = n[0] * l[0] + n[1] * l[1] + n[2] * l[2];
            const float lit = l[3] + (1 - l[3]) * (have_n ? std::max(nl, 0.0f) : std::fabs(nl));
            w[k] = have_uv ? half(uvs[(v + size_t(k)) * 3] * a.trace_uv) | half(uvs[(v + size_t(k)) * 3 + 1] * a.trace_uv) << 16 : 0;
            w[3 + k] = snorm(n[0]) | snorm(n[1]) << 8 | snorm(n[2]) << 16;
            w[6 + k] = byte(c[0] * lit) | byte(c[1] * lit) << 8 | byte(c[2] * lit) << 16 | byte(c[3]) << 24;
        }
        w[9] = material;  // the object: BuildScene
        scene_colour_.insert(scene_colour_.end(), w, w + kTraceWords);
    }
}

// the frame's scene into the slot the next frame draws with (its last use done: Present waited for it)
void MetalBackend::BuildScene() {
    trace_live_ = -1;
    const uint32_t tris = uint32_t(scene_colour_.size() / kTraceWords);
    if (trace_ok_ && tris) {
        TraceSlot& t = trace_[frame_slot_];
        // the draws grouped into objects (a car's body, wheels, lights and glass): the largest first, each other joining
        // the first whose bounds (30 cm wider) hold its centre
        std::vector<uint32_t> order(scene_draws_.size());
        for (uint32_t i = 0; i < order.size(); i++) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return scene_draws_[a].end - scene_draws_[a].first > scene_draws_[b].end - scene_draws_[b].first;
        });
        t.object.assign(scene_draws_.size(), 0);
        std::vector<uint32_t> heads;
        for (uint32_t i : order) {
            const SceneDraw& d = scene_draws_[i];
            if (d.end == d.first) continue;
            const float c[3] = {(d.lo[0] + d.hi[0]) / 2, (d.lo[1] + d.hi[1]) / 2, (d.lo[2] + d.hi[2]) / 2};
            for (uint32_t h : heads) {
                const SceneDraw& b = scene_draws_[h];
                bool in = true;
                for (int k = 0; k < 3; k++) in &= c[k] >= b.lo[k] - 0.3f && c[k] <= b.hi[k] + 0.3f;
                if (in) {
                    t.object[i] = t.object[h];
                    break;
                }
            }
            if (!t.object[i]) heads.push_back(i), t.object[i] = uint32_t(std::min<size_t>(heads.size(), 0xFFFF));
            for (size_t k = d.first; k < d.end; k++) scene_colour_[k * kTraceWords + 9] |= t.object[i] << 16;
        }
        const size_t vbytes = size_t(tris) * 36, cbytes = size_t(tris) * kTraceWords * 4;
        if (!t.verts || t.verts.length < vbytes) t.verts = [dev_ newBufferWithLength:vbytes * 3 / 2 options:MTLResourceStorageModeShared];
        if (!t.colours || t.colours.length < cbytes) t.colours = [dev_ newBufferWithLength:cbytes * 3 / 2 options:MTLResourceStorageModeShared];
        std::memcpy(t.verts.contents, scene_xyz_.data(), vbytes);
        std::memcpy(t.colours.contents, scene_colour_.data(), cbytes);
        MTLAccelerationStructureTriangleGeometryDescriptor* g = [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
        g.vertexBuffer = t.verts;
        g.vertexStride = 12;
        g.triangleCount = tris;
        g.opaque = YES;
        MTLPrimitiveAccelerationStructureDescriptor* d = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        d.geometryDescriptors = @[ g ];
        d.usage = MTLAccelerationStructureUsagePreferFastBuild;
        const MTLAccelerationStructureSizes sz = [dev_ accelerationStructureSizesWithDescriptor:d];
        if (!t.as || t.as.size < sz.accelerationStructureSize) t.as = [dev_ newAccelerationStructureWithSize:sz.accelerationStructureSize * 3 / 2];
        if (!t.scratch || t.scratch.length < sz.buildScratchBufferSize)
            t.scratch = [dev_ newBufferWithLength:std::max<NSUInteger>(sz.buildScratchBufferSize * 3 / 2, 256) options:MTLResourceStorageModePrivate];
        if (t.as && t.scratch) {
            EndEncoder();
            id<MTLAccelerationStructureCommandEncoder> e = [Cb() accelerationStructureCommandEncoder];
            [e buildAccelerationStructure:t.as descriptor:d scratchBuffer:t.scratch scratchBufferOffset:0];
            [e endEncoding];
            trace_live_ = frame_slot_;
            t.tex = scene_tex_;
        }
        static bool told = false;
        if (!told) {
            told = true;
            LOGR("metal: ray traced reflections: the first scene has %u triangles", tris);
        }
    }
    scene_xyz_.clear();
    scene_colour_.clear();
    scene_tex_.clear();
    scene_draws_.clear();
    // meshes not drawn for ten seconds dropped
    if (presents_ % 600 == 0)
        for (auto it = scene_meshes_.begin(); it != scene_meshes_.end();)
            it = presents_ - it->second.used > 600 ? scene_meshes_.erase(it) : std::next(it);
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
    d.rasterSampleCount = std::max<uint32_t>(k.samples, 1);
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
    // the title's bias (PGR2's city textures mostly -1), as D3D applies it; before macOS 26 a sampler has none
    if (sd.MipLODBias != 0) {
        if (@available(macOS 26.0, *)) m.lodBias = std::clamp(sd.MipLODBias, -16.0f, 15.999f);
        else LOG_FIRST(1, "metal: a sampler's LOD bias is not applied before macOS 26 (%g)", sd.MipLODBias);
    }
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
    // VC then DC, one block in the ring: a new one only when something read from it changed. Only fixed function vertex shaders read
    // wv to lt, so other draws neither compare nor write them (their bytes are stale). The comparing is against a copy in cached
    // memory: the ring is write-combined, and reading it costs more than the copy saves
    constexpr size_t kVc = sizeof(s.vsc), kDc = sizeof(c), kHole = offsetof(DrawConstants, wv), kHoleEnd = offsetof(DrawConstants, wdepth);
    const uint8_t* vsc = reinterpret_cast<const uint8_t*>(s.vsc.data());
    const uint8_t* dc = reinterpret_cast<const uint8_t*>(&c);
    uint8_t* last_dc = const_last_ + kVc;
    const bool whole = sh.fixed_transform;
    const size_t head = whole ? kDc : kHole, tail = whole ? 0 : kDc - kHoleEnd;
    const bool vc_same = !std::memcmp(const_last_, vsc, kVc);
    const bool dc_same = const_whole_ == whole && !std::memcmp(last_dc, dc, head) && !std::memcmp(last_dc + kHoleEnd, dc + kHoleEnd, tail);
    if (!(vc_same && dc_same && const_at_.p && const_at_.buf == ring_)) {
        if (!vc_same) std::memcpy(const_last_, vsc, kVc);
        if (!dc_same) {
            std::memcpy(last_dc, dc, head);
            std::memcpy(last_dc + kHoleEnd, dc + kHoleEnd, tail);
            const_whole_ = whole;
        }
        const_at_ = Reserve(uint32_t(kVc + kDc), 256);
        std::memcpy(const_at_.p, vsc, kVc);
        std::memcpy(const_at_.p + kVc, dc, head);
        std::memcpy(const_at_.p + kVc + kHoleEnd, dc + kHoleEnd, tail);
        perf::Add(perf::kNConstantUploads);
        perf::Add(perf::kNConstantBytes, kVc + head + tail);
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
        perf::Add(perf::kNVertexBytes, b.bytes);
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
        perf::Add(perf::kNIndexBytes, ibytes);
        [enc_ drawIndexedPrimitives:type indexCount:a.index_count indexType:a.index16 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                        indexBuffer:at.buf indexBufferOffset:at.offset instanceCount:1 baseVertex:a.base_vertex baseInstance:0];
    } else {
        [enc_ drawPrimitives:type vertexStart:0 vertexCount:a.draw_count];
    }
}

void MetalBackend::Draw(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices) {
    if (const State* sprites = PointSpriteStages(s, prim)) return Draw(*sprites, prim, start, count, indices);
    // perf: the CPU time of a draw, the outermost call only (one drawn again with another stage 3 is inside it)
    struct Timed {
        bool on;
        std::chrono::steady_clock::time_point start;
        ~Timed() {
            if (on) perf::AddDrawCpu(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        }
    } timed{perf::On() && !g_draw_filtered, perf::On() && !g_draw_filtered ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}};
    @autoreleasepool {
        if (s.blank_format && s.textures[0] && (::kernel::R32(s.textures[0] + 12) >> 8 & 0xFF) == s.blank_format) {
            Clear(s, 0, nullptr, kClearTarget, 0xFF000000u, 0, 0);  // a black clear in its place (State::blank_format)
            return;
        }
        if (DepthCopy(s)) return;
        if (diag::SkippedDraw(s.vs_handle)) return;
        DrawAdjust adjust;
        adjust.start = start, adjust.count = count, adjust.indices = indices;
        adjust.frame = presents_;
        if (g_draw_filtered && carried_ && carried_->trace_reflect) {  // drawn again with another stage 3: still traced
            adjust.trace_reflect = true;
            std::memcpy(adjust.trace_from, carried_->trace_from, sizeof(adjust.trace_from));
            std::memcpy(adjust.trace_turn, carried_->trace_turn, sizeof(adjust.trace_turn));
            std::memcpy(adjust.trace_range, carried_->trace_range, sizeof(adjust.trace_range));
            adjust.trace_self = carried_->trace_self;
        }
        if (g_draw_filter && !g_draw_filtered) g_draw_filter(s, adjust);
        if (adjust.skip) return;
        if (!g_draw_filtered) perf::CountDraw(prim, count);
        if (adjust.trace_scene) AddToScene(s, prim, start, count, indices, adjust);
        if (adjust.stage3) {
            g_draw_filtered = true;
            carried_ = &adjust;
            Draw(*WithStage3(s, adjust.stage3), prim, start, count, indices);
            carried_ = nullptr;
            g_draw_filtered = false;
            return;
        }
        perf::Lap lap;
        Assembled& a = asm_;
        if (!Assemble(s, prim, start, count, indices, a)) return;
        lap.Mark(perf::kStAssemble);
        if (!BindTargets(s)) return;
        lap.Mark(perf::kStTargets);

        DrawShaders sh = ClassifyShaders(s, a, count);
        if (sh.prog) EnsureProgramHash(s);
        sh.vkey = VertexKey(s, a, sh);
        EnsureVertexShader(s, a, sh);
        // stage textures, which may draw copies, then the targets again
        DrawTextures stages;
        draw_rt_ = bound_rt_;
        draw_ds_ = bound_ds_;
        draw_2d_ = a.pretransformed || ScreenDraw(s);
        lap.Mark(perf::kStShaders);
        for (int i = 0; i < 4; i++) stages.tex[i] = BindTexture(s, i, stages);
        lap.Mark(perf::kStTextures);
        if (!BindTargets(s)) return;
        lap.Mark(perf::kStTargets);
        PixelKey pk = MakePixelKey(s, stages.info);
        pk.quad = sh.quad;
        // baked occlusion for a draw of the static world (gpu/baked_ao.h)
        const float ao_strength = ao::g_strength.load(std::memory_order_relaxed);
        const bool baked = ao::g_draw && ao_strength > 0 && !draw_2d_ && ao::Triangles();
        if (baked) pk.ao = 1;
        const bool traced = adjust.trace_reflect && trace_live_ >= 0 && pk.programmable && pk.tex[3].kind == kTexCube && !pk.tex[3].shadow;
        if (TraceLogFrame() && adjust.trace_reflect)
            LOGR("trace draw: vs %08X rt %08X %s (scene %d, self %u of %zu -> object %u, programmable %u, stage 3 kind %u) from %.4g %.4g %.4g %.4g / %.4g %.4g %.4g %.4g / %.4g %.4g %.4g %.4g",
                 s.vs_handle, s.rt.data, traced ? "traced" : "not traced", trace_live_, adjust.trace_self, trace_live_ >= 0 ? trace_[trace_live_].object.size() : 0,
                 trace_live_ >= 0 && adjust.trace_self && adjust.trace_self <= trace_[trace_live_].object.size() ? trace_[trace_live_].object[adjust.trace_self - 1] : 0, pk.programmable, pk.tex[3].kind, adjust.trace_from[0],
                 adjust.trace_from[1], adjust.trace_from[2], adjust.trace_from[3], adjust.trace_from[4], adjust.trace_from[5], adjust.trace_from[6],
                 adjust.trace_from[7], adjust.trace_from[8], adjust.trace_from[9], adjust.trace_from[10], adjust.trace_from[11]);
        if (traced) pk.tex[3].flags |= kTexTraced;
        // D3DZB_USEW draws write w as their depth, as the console does; XBE_WBUFFER=0 goes back to z / w
        static const bool wbuffer = [] {
            const char* e = std::getenv("XBE_WBUFFER");
            return !(e && *e == '0');
        }();
        pk.wbuf = wbuffer && s.rs[kRsZEnable] == 2 && bound_ds_ ? 1 : 0;
        // alpha tested, not blended, onto twins: Settings::alpha_coverage (not quads, whose shader is rewritten)
        if (app::Get().alpha_coverage && bound_ms_ && !sh.quad && SimpleState(s, kMAlphaTestEnable) && !SimpleState(s, kMBlendEnable)) pk.coverage = msaa_;
        sh.pkey = Hash(&pk, sizeof(pk));
        EnsurePixelShader(pk, sh);
        if (!sh.vs_code || !sh.ps_code) return;

        PsoKey k;
        PipelineTargets pt{.rt = bound_rt_ != nullptr, .rtfmt = bound_rt_ ? uint32_t(DXGI_FORMAT_B8G8R8A8_UNORM) : 0u, .ds = bound_ds_ != nullptr,
                           .ds_scale = bound_ds_ ? float(bound_ds_->h) / bound_ds_->gh : 1.0f, .samples = bound_ms_ ? msaa_ : 1};
        BuildPipelineKey(s, a, sh.vh, sh.ph, pt, k);
        if (!adjust.depth_test) k.zen = k.zwrite = 0;
        k.zbias += adjust.depth_bias;
        // W-buffered: the pixel shader adds the polygon offset to w (d3d12.cpp Draw)
        float wdepth[4] = {float(k.zbias) * s.zscale / kDepth24Max, k.zslope, 1.0f / (s.zscale + 1.0f), sh.prog ? 1.0f : s.zscale * s.inv_wfar};
        if (pk.wbuf) k.zbias = 0, k.zslope = 0;
        Pipeline* p = EnsurePipeline(k, a, sh);
        if (!p->state) return;

        lap.Mark(perf::kStPipeline);
        draw_counting_ = s.simple[kMCountEnable >> 2] != 0;
        if (draw_counting_ && run_) EndRun();
        if (ts_probe_ && perf::On() && enc_ && draw_2d_) EndEncoder();  // this draw alone in its encoder
        if (bound_ms_)
            OpenPass(bound_rt_ ? bound_rt_->ms : nil, bound_ds_ ? bound_ds_->ms : nil, bound_rt_ ? bound_rt_->res : nil, bound_ds_ ? bound_ds_->res : nil);
        else
            OpenPass(bound_rt_ ? bound_rt_->res : nil, bound_ds_ ? bound_ds_->res : nil);
        lap.Mark(perf::kStOpenPass);
        if (ts_cur_ >= 0) {
            EncoderInfo& e = ts_pass_[size_t(ts_cur_)];
            if (!e.draws) {
                e.vs = sh.vkey, e.ps = sh.pkey, e.flat = draw_2d_, e.blend = s.simple[kMBlendEnable >> 2] != 0;
                if (stages.tex[0]) e.tex_w = uint32_t(stages.tex[0].width), e.tex_h = uint32_t(stages.tex[0].height), e.tex_format = uint32_t(stages.tex[0].pixelFormat);
            }
            e.draws++;
            e.vertices += count;
        }
        if (enc_pipeline_ != p->state) {
            [enc_ setRenderPipelineState:(enc_pipeline_ = p->state)];
            perf::Add(perf::kNPipelineBinds);
        }
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
        if (baked) {
            // a strip drawn as one: Apple GPUs give its odd triangles' barycentrics in the order the bake's list
            // (gpu/vertex.cpp ListIndices) has them; XBE_AO_FLIP=1 swaps the first two (diagonal seams on walls when wrong)
            static const bool flip = std::getenv("XBE_AO_FLIP") && *std::getenv("XBE_AO_FLIP") == '1';
            static const float debug = std::getenv("XBE_AO_DEBUG") && *std::getenv("XBE_AO_DEBUG") == '1' ? 1.0f : 0.0f;
            c.aoc[0] = float(ao::g_draw - 1), c.aoc[1] = std::min(ao_strength, 1.0f), c.aoc[2] = a.strip && flip ? 1.0f : 0.0f, c.aoc[3] = debug;
        }
        if (traced) {
            // the filter's place in the scene from console pixels, here from the target's own (as the scissor below scales)
            const Target* tb = bound_rt_ ? bound_rt_ : bound_ds_;
            const float px = float(tb->w) / size.w, py = float(tb->h) / size.h;  // as the vertex shader's surf maps them
            for (int r = 0; r < 3; r++) {
                const float* f = adjust.trace_from + r * 4;
                c.trace[r][0] = f[0] / px, c.trace[r][1] = f[1] / py, c.trace[r][2] = f[2], c.trace[r][3] = f[3];
                std::memcpy(c.trace[4 + r], adjust.trace_turn + r * 4, 16);
            }
            c.trace[3][0] = 1, c.trace[3][1] = adjust.trace_range[0], c.trace[3][2] = adjust.trace_range[1];
            static const float debug = std::getenv("XBE_TRACE_DEBUG") ? float(std::atoi(std::getenv("XBE_TRACE_DEBUG"))) : 0.0f;
            c.trace[3][3] = debug;
            // the triangles' faces: anticlockwise in front, as PGR2's cars are wound (XBE_TRACE_CULL=0 none, 1 clockwise)
            static const float cull = std::getenv("XBE_TRACE_CULL") ? float(std::atoi(std::getenv("XBE_TRACE_CULL"))) : 2.0f;
            c.trace[7][0] = cull;
            const auto& object = trace_[trace_live_].object;
            c.trace[7][1] = adjust.trace_self && adjust.trace_self <= object.size() ? float(object[adjust.trace_self - 1]) : 0.0f;
        }
        lap.Mark(perf::kStEncoderState);
        UploadConstants(s, sh, k, stages, size, c);
        lap.Mark(perf::kStConstants);
        if (baked) {
            [enc_ setFragmentBuffer:ao::Triangles() offset:0 atIndex:3];
            [enc_ setFragmentBuffer:ao::Values() offset:0 atIndex:4];
        }
        if (traced) {
            [enc_ setFragmentAccelerationStructure:trace_[trace_live_].as atBufferIndex:1];
            [enc_ setFragmentBuffer:trace_[trace_live_].colours offset:0 atIndex:2];
            const auto& tex = trace_[trace_live_].tex;
            for (size_t i = 0; i < kSceneTextures; i++) [enc_ setFragmentTexture:i < tex.size() ? tex[i] : dummy_2d_ atIndex:8 + i];
        }

        // textures and samplers
        bool clamp = UnitTexcoords(a);
        uint32_t aniso = app::Get().aniso;
        for (int i = 0; i < 4; i++) {
            id<MTLTexture> t = stages.tex[i];
            // a shadow stage reads a depth format, whatever is bound
            if (!t || (stages.info[i].shadow && t.pixelFormat != kDepth)) t = Dummy(stages.info[i]);
            static const bool no_skip = std::getenv("XBE_NO_BINDSKIP") && *std::getenv("XBE_NO_BINDSKIP") == '1';  // testing: every bind made
            if (no_skip || enc_tex_[i] != t) {
                [enc_ setFragmentTexture:t atIndex:i];
                enc_tex_[i] = t;
                perf::Add(perf::kNTextureBinds);
            }
            id<MTLSamplerState> sampler = Sampler(StageSampler(s, i, stages.levels[i], stages.info[i].shadow, clamp, aniso, stages.replaced[i]));
            if (no_skip || enc_smp_[i] != sampler) {
                [enc_ setFragmentSamplerState:sampler atIndex:i];
                enc_smp_[i] = sampler;
            }
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
        lap.Mark(perf::kStTextureSet);
        DrawGeometry(a, *p);
        lap.Mark(perf::kStGeometry);
        if (counting) [enc_ setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
    }
}

// ---- visibility tests

float MetalBackend::SamplesPerPixel() const {
    const Target* t = bound_rt_ ? bound_rt_ : bound_ds_;
    if (!t || !t->gw || !t->gh) return 1;
    return float(t->w) * float(t->h) / (float(t->gw) * float(t->gh)) * float(bound_ms_ ? msaa_ : 1);
}

// reports whose draws the GPU has finished, written as the console's GPU writes them: time, count, then 0 for done
void MetalBackend::DeliverReports() {
    constexpr uint32_t kTime = 0, kTimeHigh = 4, kCount = 8, kStatus = 12;
    std::lock_guard lock(reports_lock_);
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
            SampleRender(d, kEncPresent);
            enc_ = [Cb() renderCommandEncoderWithDescriptor:d];
            enc_colour_ = drawable.texture;
            enc_depth_ = nil;
            const float w = float(drawable.texture.width), h = float(drawable.texture.height);
            if (frame) {
                // the frame at its own shape (16:9 or 4:3, ultrawide the screen's), centred on black: its target's, so a frame
                // drawn before a change of shape reached the targets is not shown stretched
                float aspect = frame.height ? float(frame.width) / float(frame.height) : aspect_;
                float vw = std::min(w, h * aspect), vh = vw / aspect;
                [enc_ setViewport:MTLViewport{(w - vw) / 2, (h - vh) / 2, vw, vh, 0, 1}];
                // each size the picture passes through, when one changes: the window's, the layer's, the drawable's, the
                // frame's and where it is put (a picture scaled by the system or between sizes shows here)
                static uint64_t sizes = 0;
                static int told = 0;
                const CGSize bounds = layer_.bounds.size;
                const uint64_t key = uint64_t(w) << 48 ^ uint64_t(h) << 32 ^ uint64_t(frame.width) << 16 ^ frame.height ^
                                     uint64_t(bounds.width) << 40 ^ uint64_t(layer_.contentsScale * 8) << 56;
                if (key != sizes && told++ < 40) {
                    sizes = key;
                    LOGR("metal: sizes: layer %.0fx%.0f points at %.2f, drawable %.0fx%.0f (wanted %ux%u), frame %lux%lu (%.3f:1, "
                         "targets %.3f:1, scale %.2f), shown %.0fx%.0f at %.0f,%.0f",
                         bounds.width, bounds.height, layer_.contentsScale, w, h, px.width, px.height,
                         (unsigned long)frame.width, (unsigned long)frame.height, aspect, aspect_, scale_, vw, vh, (w - vw) / 2,
                         (h - vh) / 2);
                }
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
        // perf: the frame's pass times, read when its last command buffer is done (before its timestamp buffer can be reused)
        if (ts_used_) {
            if (presents_ % 60 == 0) {
                MTLTimestamp cpu = 0, gpu = 0;
                [dev_ sampleTimestamps:&cpu gpuTimestamp:&gpu];
                if (ts_gpu_ && gpu > ts_gpu_ && cpu > ts_cpu_) ts_ns_per_tick_ = double(cpu - ts_cpu_) / double(gpu - ts_gpu_);
                ts_cpu_ = cpu, ts_gpu_ = gpu;
            }
            id<MTLCounterSampleBuffer> buffer = ts_[frame_slot_];
            const uint32_t used = ts_used_;
            auto passes = std::make_shared<std::vector<EncoderInfo>>(std::move(ts_pass_));
            const uint64_t frame_number = presents_;
            const double ms_per_tick = ts_ns_per_tick_ / 1e6;
            const bool write_rows = presents_ % 10 == 0, probe = ts_probe_;
            [Cb() addCompletedHandler:^(id<MTLCommandBuffer>) {
                NSData* data = [buffer resolveCounterRange:NSMakeRange(0, used)];
                if (!data || data.length < used * sizeof(MTLCounterResultTimestamp)) return;
                auto* t = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
                auto span = [&](uint32_t a, uint32_t b) {
                    return t[a].timestamp != MTLCounterErrorValue && t[b].timestamp != MTLCounterErrorValue && t[b].timestamp > t[a].timestamp
                               ? double(t[b].timestamp - t[a].timestamp) * ms_per_tick : 0.0;
                };
                std::string rows;
                for (uint32_t i = 0; i < passes->size(); i++) {
                    const EncoderInfo& e = (*passes)[i];
                    const double vertex = span(i * 4, i * 4 + 1), fragment = e.kind == kEncBlit ? 0.0 : span(i * 4 + 2, i * 4 + 3);
                    if (e.kind == kEncDraw) perf::GpuPass(frame_number, e.pass, vertex, fragment);
                    else perf::GpuOther(frame_number, e.kind == kEncClear ? perf::kGpuClear : e.kind == kEncPresent ? perf::kGpuPresent : perf::kGpuBlit, vertex + fragment);
                    if (write_rows) {
                        char line[320];
                        std::snprintf(line, sizeof line, "%llu,%d,%u,%d,%d,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%llx,%llx,%u,%u,%u,%d,%d,%.3f,%.3f\n", (unsigned long long)frame_number, probe ? 1 : 0, i, int(e.kind), e.pass,
                                      e.w, e.h, e.format, e.zw, e.zh, e.cload, e.cstore, e.zload, e.zstore, e.draws, e.vertices, (unsigned long long)e.vs, (unsigned long long)e.ps, e.tex_w, e.tex_h, e.tex_format, e.flat ? 1 : 0, e.blend ? 1 : 0, vertex, fragment);
                        rows += line;
                    }
                }
                if (write_rows) perf::EncoderRows(rows);
            }];
            ts_used_ = 0;
            ts_pass_.clear();
        }
        // the frame's ring stays untouched until the GPU is done with it
        dispatch_semaphore_t frames = frames_;
        [Cb() addCompletedHandler:^(id<MTLCommandBuffer>) { dispatch_semaphore_signal(frames); }];
        Commit();
        Unbind();
        const auto inflight_start = std::chrono::steady_clock::now();
        dispatch_semaphore_wait(frames_, DISPATCH_TIME_FOREVER);
        const double inflight_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - inflight_start).count();
        frame_slot_ = (frame_slot_ + 1) % kFramesInFlight;
        ring_ = rings_[frame_slot_];
        const uint32_t ring_kb = ring_used_ / 1024;  // perf: what the finished frame used
        ring_used_ = 0;
        const_at_ = {};
        BuildScene();
        DeliverReports();
        static double gpu_mem_mb = 0;
        static uint32_t thermal = 0;
        if (perf::On() && presents_ % 30 == 0) {
            gpu_mem_mb = double(dev_.currentAllocatedSize) / 1048576.0;
            thermal = uint32_t(NSProcessInfo.processInfo.thermalState);
        }
        perf::Frame(presents_, {.shaders_ms = cost_.shaders_ms, .pipelines_ms = cost_.pipelines_ms, .uploads_ms = cost_.uploads_ms,
                                .readback_ms = cost_.readback_ms, .drawable_ms = cost_.drawable_ms, .inflight_ms = inflight_ms,
                                .title_wait_ms = {double(g_perf_wait_kind_ns[0].exchange(0)) / 1e6, double(g_perf_wait_kind_ns[1].exchange(0)) / 1e6,
                                                  double(g_perf_wait_kind_ns[2].exchange(0)) / 1e6, double(g_perf_wait_kind_ns[3].exchange(0)) / 1e6,
                                                  double(g_perf_wait_kind_ns[4].exchange(0)) / 1e6, double(g_perf_wait_kind_ns[5].exchange(0)) / 1e6},
                                .shaders = cost_.shaders, .pipelines = cost_.pipelines, .uploads = cost_.uploads, .readbacks = cost_.readbacks,
                                .game_capture_ms = double(g_perf_capture_ns.exchange(0)) / 1e6, .gpu_mem_mb = gpu_mem_mb, .textures = uint32_t(textures_.size()),
                                .targets = uint32_t(targets_.size()), .ring_kb = ring_kb, .thermal = thermal});
        presents_++;
        ts_probe_ = perf::On() && presents_ % 120 == 0;
        g_trace_presents = presents_;
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
    // 1 2 3: MSAA 2x, 4x, 8x; a change drops every twin (res is current), the next draws make new ones
    static const uint32_t kSamples[4] = {1, 2, 4, 8};
    const uint32_t want_ms = kSamples[std::min<uint32_t>(app::Get().msaa, 3)];
    if (want_ms != msaa_) {
        EndEncoder();
        msaa_ = want_ms;
        for (auto& [key, t] : targets_) t.ms = nil, t.ms_stale = false;
        bound_ms_ = false;
        bound_rt_ = bound_ds_ = nullptr;
        LOGR("metal: MSAA %ux", msaa_);
    }
    if (app::Get().smaa) LOG_FIRST(1, "metal: SMAA is not implemented yet, drawn without");
    float aspect = app::FrameAspect();
    if (std::fabs(aspect - aspect_) > 1e-4f) {
        aspect_ = aspect;
        RescaleTargets();
    }
    if (const float f = g_small_target_factor.load(std::memory_order_relaxed); f != small_factor_) {
        small_factor_ = f;
        RescaleTargets();
        return;
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
