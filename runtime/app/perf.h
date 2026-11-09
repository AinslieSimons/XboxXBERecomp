// Performance recording, for finding what holds a frame back after the fact. F9 (or XBE_PERF=1 from the start) starts
// and stops a recording; F10 drops a marker in it. Each recording is a folder perf/<date-time> beside the executable:
//   frames.csv   one row a frame: its interval and target, where the frame waited (pacing, the drawable, frames in
//                flight), the GPU's time for its command buffers, the CPU time spent in draws, the backend's one-off costs
//                (shaders, pipelines, texture uploads, read-backs), and draws and triangles in all and by pass
//   threads.csv  each host thread's CPU time each frame (frame, thread id, name, ms)
//   events.csv   markers and anything else worth a line (start, stop, what the title registered)
//   info.txt     the machine, the thread that presents, the pass names; and a copy of the settings file
// tools/perf/analyze.py turns a folder into report.html. Cheap enough to leave on: counters, a clock read per draw
// while recording, and one look at the threads a frame.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace perf {

constexpr int kMaxPasses = 12, kMaxKinds = 8, kMaxZones = 10;

extern std::atomic<bool> g_on;
inline bool On() { return g_on.load(std::memory_order_relaxed); }

// a cheap clock: the CPU's counter on ARM (a few ns to read, 24 MHz), summed over thousands of short stages a frame
inline uint64_t Ticks() {
#if defined(__aarch64__)
    uint64_t v;
    asm volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
#else
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
#endif
}

// where a draw's CPU time goes on the drawing thread, and the game thread's state hand-over
enum Stage { kStAssemble, kStTargets, kStShaders, kStTextures, kStPipeline, kStOpenPass, kStEncoderState, kStConstants, kStTextureSet, kStGeometry, kStSendState, kStageCount };
// things counted between presents
enum Count { kNEncoders, kNBlits, kNCommandBuffers, kNPipelineBinds, kNTextureBinds, kNVertexBytes, kNIndexBytes, kNConstantUploads, kNConstantBytes,
             kNStatesSent, kNDeltas, kNDeltaBytes, kNMerged, kCountCount };
extern std::atomic<uint64_t> g_stage[kStageCount], g_count[kCountCount];
inline void Add(Count c, uint64_t n = 1) {
    if (On()) g_count[c].fetch_add(n, std::memory_order_relaxed);
}
// the time since the last mark goes to a stage
struct Lap {
    bool on = On();
    uint64_t last = on ? Ticks() : 0;
    void Mark(Stage s) {
        if (!on) return;
        const uint64_t now = Ticks();
        g_stage[s].fetch_add(now - last, std::memory_order_relaxed);
        last = now;
    }
};
// the title's own functions, timed around their calls (inclusive: a zone may hold another)
void SetZoneNames(std::vector<std::string> names);
void AddZone(int zone, uint64_t ticks);
struct Zone {
    int zone;
    bool on = On();
    uint64_t start = on ? Ticks() : 0;
    explicit Zone(int z) : zone(z) {}
    ~Zone() {
        if (on) AddZone(zone, Ticks() - start);
    }
};
// the core the calling thread is on now, for frames.csv (0 the drawing thread, 1 the game's)
void NoteCore(int which);

// this thread's name in threads.csv (the runtime names its threads through here too: kernel::NameHostThread)
void NameThread(const char* name);
void Toggle();  // F9
void Marker();  // F10
void Event(const std::string& text);  // a line in events.csv while recording, any thread

// the title's names for the parts of a frame its draws belong to (pass 0 is "other"), set once at the start
void SetPassNames(std::vector<std::string> names);
// the draw being made belongs to this pass (the title's draw filter, on the drawing thread)
void SetPass(int pass);
int CurrentPass();
// and what kinds of thing draws are ("points" counts every point list, whatever the filter said)
void SetKindNames(std::vector<std::string> names);
void SetKind(int kind);
// a recording's extra file, objects.csv: the title's header (once, at the start) and whole lines (its own thread), and the
// recording's clock for them
void SetObjectsHeader(std::string header);
void ObjectRows(const std::string& rows);
double ElapsedMs();
// the title's own columns, read once a frame on the presenting thread (its city, mode and so on)
void SetState(std::vector<std::string> names, std::function<std::vector<int64_t>()> read);
// a draw made (after the filter): D3D primitive type and vertex count
void CountDraw(uint32_t prim, uint32_t vertices);
// CPU time in the backend's draw (the drawing thread)
void AddDrawCpu(double ms);
// the frame pacer's wait and the frame time it paces to
void PaceWait(double ms, double target_ms);

struct FrameCosts {
    double shaders_ms = 0, pipelines_ms = 0, uploads_ms = 0, readback_ms = 0, drawable_ms = 0, inflight_ms = 0;
    double title_wait_ms[6] = {};  // the game's thread waiting for the drawing thread: read-back, resource, fence, flush, copy, next frame
    uint32_t shaders = 0, pipelines = 0, uploads = 0, readbacks = 0;
    double game_capture_ms = 0;  // the game's thread recording commands for the drawing thread
    // sampled now and then: the GPU's allocations, live textures and targets, this frame's ring use, macOS's thermal state (0 nominal to 3 critical)
    double gpu_mem_mb = 0;
    uint32_t textures = 0, targets = 0, ring_kb = 0, thermal = 0;
};
// a frame presented (its number, counted by the backend), after its work is sent: closes the row
void Frame(uint64_t frame, const FrameCosts& c);
// a command buffer's GPU time, when it completes (any thread), for the frame it was sent in
void Gpu(uint64_t frame, double ms);
// one pass's vertex and fragment time from the GPU's timestamps (any thread)
void GpuPass(uint64_t frame, int pass, double vertex_ms, double fragment_ms);
// the GPU's time in work outside the draw passes (any thread): clears, the frame's last pass to the screen, copies
enum GpuWork { kGpuClear, kGpuPresent, kGpuBlit, kGpuWorkCount };
void GpuOther(uint64_t frame, GpuWork work, double ms);
// lines for encoders.csv (each encoder of a sampled frame), whole lines (any thread)
void EncoderRows(const std::string& rows);

// time the presenting thread spent stopped: waiting for the GPU to deliver reports, or in the kernel's waits and delays
void AddReportWait(double ms);
struct KernelWait {  // around a kernel wait; counts only on the presenting thread
    KernelWait();
    ~KernelWait();
    int64_t start_ns;
};

}  // namespace perf
