// Performance recording (perf.h).
#include "app/perf.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <unordered_map>

#include "app/app.h"
#include "core/log.h"

#ifdef __APPLE__
#include <mach/mach.h>
#include <pthread.h>
#include <sys/sysctl.h>
#endif

namespace perf {

std::atomic<bool> g_on{false};
std::atomic<uint64_t> g_stage[kStageCount], g_count[kCountCount];

namespace {

using clock = std::chrono::steady_clock;

const char* const kStageNames[kStageCount] = {"assemble", "targets", "shaders", "textures", "pipeline", "open_pass", "encoder_state", "constants", "texture_set", "geometry", "send_state"};
const char* const kCountNames[kCountCount] = {"encoders", "blits", "command_buffers", "pipeline_binds", "texture_binds", "vertex_bytes", "index_bytes", "constant_uploads", "constant_bytes", "states_sent", "deltas", "delta_bytes", "merged_draws"};
std::vector<std::string> g_zone_names;
std::atomic<uint64_t> g_zone_ticks[kMaxZones], g_zone_calls[kMaxZones];
std::atomic<int> g_core[2] = {-1, -1};

double TickMs() {
#if defined(__aarch64__)
    static const double ms = [] {
        uint64_t f;
        asm volatile("mrs %0, cntfrq_el0" : "=r"(f));
        return 1000.0 / double(f);
    }();
    return ms;
#else
    return 1e-6;
#endif
}
std::atomic<bool> g_toggle{false}, g_marker{false};
std::mutex g_files;  // the files, opened and closed on the presenting thread; events from any
FILE* g_frames = nullptr;
FILE* g_threads = nullptr;
FILE* g_events = nullptr;
FILE* g_encoders = nullptr;
FILE* g_objects = nullptr;
std::string g_objects_header;
std::filesystem::path g_dir;
clock::time_point g_start, g_last_present;
bool g_have_last = false;

std::vector<std::string> g_pass_names = {"other"}, g_kind_names = {"draws"}, g_state_names;
std::function<std::vector<int64_t>()> g_state_read;
int g_points_kind = -1;
thread_local int t_pass = 0, t_kind = 0;
thread_local bool t_presenting = false;

// counted between presents (the drawing thread, the pacer)
struct Counts {
    uint64_t draws[kMaxPasses] = {}, triangles[kMaxPasses] = {};
    uint64_t kinds[kMaxKinds] = {};
    double draw_cpu_ms = 0, pace_ms = 0, target_ms = 0, report_wait_ms = 0, kernel_wait_ms = 0;
};
Counts g_counts;
std::mutex g_counts_lock;  // the pacer may be on another thread than the draws

// a frame's row, held until its GPU time is in (the semaphore at a present keeps the GPU a few frames behind at most)
constexpr int kRing = 64, kLag = 8;
struct Row {
    uint64_t frame = ~0ull;
    double t_ms = 0, interval_ms = 0;
    FrameCosts c;
    Counts n;
    bool marker = false;
    std::vector<int64_t> state;
    double stage_ms[kStageCount] = {}, zone_ms[kMaxZones] = {};
    uint64_t counts[kCountCount] = {}, zone_calls[kMaxZones] = {};
    int core[2] = {-1, -1};
    double mem_mb = 0;
};
std::array<Row, kRing> g_rows;
std::array<std::atomic<uint64_t>, kRing> g_gpu_tag{}, g_gpu_us{};
std::atomic<uint64_t> g_gpu_pass_us[kRing][kMaxPasses][2];
std::atomic<uint64_t> g_gpu_other_us[kRing][kGpuWorkCount];

double Ms(clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

uint64_t Triangles(uint32_t prim, uint32_t n) {
    switch (prim) {  // D3DPRIMITIVETYPE on the Xbox
    case 5: return n / 3;                   // triangle list
    case 6: case 7: case 9: case 10: return n > 2 ? n - 2 : 0;  // strip, fan, quad strip, polygon
    case 8: return n / 2;                   // quad list: two triangles a quad
    default: return 0;                      // points and lines
    }
}

#ifdef __APPLE__
struct ThreadSeen {
    uint64_t ns = 0;
};
std::unordered_map<uint64_t, ThreadSeen> g_seen;

// the names the runtime gives its threads (NameThread), by system thread id; the system's own name, else the id
std::mutex g_names_lock;
std::unordered_map<uint64_t, std::string> g_names;
std::string NameOf(uint64_t id, const char* system) {
    {
        std::lock_guard lock(g_names_lock);
        if (auto it = g_names.find(id); it != g_names.end()) return it->second;
    }
    if (system && system[0]) return system;
    return "thread " + std::to_string(id);
}
uint64_t ThisThreadId() {
    uint64_t id = 0;
    pthread_threadid_np(nullptr, &id);
    return id;
}
std::string ThisThreadName() {
    char name[64] = {};
    pthread_getname_np(pthread_self(), name, sizeof name);
    return NameOf(ThisThreadId(), name);
}

// each thread's CPU time since the last look
void SampleThreads(uint64_t frame) {
    thread_act_array_t list = nullptr;
    mach_msg_type_number_t n = 0;
    if (task_threads(mach_task_self(), &list, &n) != KERN_SUCCESS) return;
    for (mach_msg_type_number_t i = 0; i < n; i++) {
        thread_extended_info_data_t ext;
        mach_msg_type_number_t count = THREAD_EXTENDED_INFO_COUNT;
        thread_identifier_info_data_t ident;
        mach_msg_type_number_t icount = THREAD_IDENTIFIER_INFO_COUNT;
        if (thread_info(list[i], THREAD_EXTENDED_INFO, reinterpret_cast<thread_info_t>(&ext), &count) == KERN_SUCCESS &&
            thread_info(list[i], THREAD_IDENTIFIER_INFO, reinterpret_cast<thread_info_t>(&ident), &icount) == KERN_SUCCESS) {
            const uint64_t ns = ext.pth_user_time + ext.pth_system_time;
            auto [it, fresh] = g_seen.try_emplace(ident.thread_id);
            const uint64_t was = it->second.ns;
            it->second.ns = ns;
            if (!fresh && ns > was && g_threads) {
                const double ms = double(ns - was) / 1e6;
                if (ms >= 0.01) std::fprintf(g_threads, "%llu,%llu,\"%s\",%.3f\n", (unsigned long long)frame, (unsigned long long)ident.thread_id,
                                             NameOf(ident.thread_id, ext.pth_name).c_str(), ms);
            }
        }
        mach_port_deallocate(mach_task_self(), list[i]);
    }
    vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(list), n * sizeof(thread_act_t));
}

std::string Machine() {
    char chip[128] = {};
    size_t n = sizeof(chip);
    sysctlbyname("machdep.cpu.brand_string", chip, &n, nullptr, 0);
    int cores = 0, perf_cores = 0;
    n = sizeof(cores);
    sysctlbyname("hw.ncpu", &cores, &n, nullptr, 0);
    n = sizeof(perf_cores);
    sysctlbyname("hw.perflevel0.physicalcpu", &perf_cores, &n, nullptr, 0);
    int mem_gb = 0;
    uint64_t mem = 0;
    n = sizeof(mem);
    sysctlbyname("hw.memsize", &mem, &n, nullptr, 0);
    mem_gb = int(mem >> 30);
    return std::string(chip) + ", " + std::to_string(cores) + " cores (" + std::to_string(perf_cores) + " performance), " + std::to_string(mem_gb) + " GB";
}
#else
std::string ThisThreadName() { return "(this thread)"; }
uint64_t ThisThreadId() { return 0; }
std::mutex g_names_lock;
std::unordered_map<uint64_t, std::string> g_names;
void SampleThreads(uint64_t) {}
std::string Machine() { return "(not recorded on this system)"; }
#endif

void WriteRow(const Row& r) {
    const size_t slot = r.frame % kRing;
    const bool tagged = g_gpu_tag[slot].load() == r.frame;
    const uint64_t gpu_us = tagged ? g_gpu_us[slot].load() : 0;
    uint64_t draws = 0, tris = 0;
    for (int p = 0; p < kMaxPasses; p++) draws += r.n.draws[p], tris += r.n.triangles[p];
    // work: the frame's time without the waits for its turn on the display, so what headroom a frame at the target had
    const double work_ms = std::max(0.0, r.interval_ms - r.n.pace_ms - r.c.drawable_ms - r.c.inflight_ms);
    std::fprintf(g_frames, "%llu,%.3f,%.3f,%.3f,%.1f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%u,%.3f,%u,%.3f,%u,%.3f,%u,%.3f,%llu,%llu",
                 (unsigned long long)r.frame, r.t_ms, r.interval_ms, work_ms, r.interval_ms > 0 ? 1000.0 / r.interval_ms : 0.0, r.n.target_ms, r.n.pace_ms, r.c.drawable_ms, r.c.inflight_ms,
                 double(gpu_us) / 1000.0, r.n.draw_cpu_ms, r.c.shaders, r.c.shaders_ms, r.c.pipelines, r.c.pipelines_ms, r.c.uploads,
                 r.c.uploads_ms, r.c.readbacks, r.c.readback_ms, (unsigned long long)draws, (unsigned long long)tris);
    for (size_t p = 0; p < g_pass_names.size(); p++) std::fprintf(g_frames, ",%llu", (unsigned long long)r.n.draws[p]);
    for (size_t p = 0; p < g_pass_names.size(); p++) std::fprintf(g_frames, ",%llu", (unsigned long long)r.n.triangles[p]);
    std::fprintf(g_frames, ",%.3f,%.3f", r.n.report_wait_ms, r.n.kernel_wait_ms);
    for (auto& w : g_gpu_other_us[slot]) std::fprintf(g_frames, ",%.3f", tagged ? double(w.load()) / 1000.0 : 0.0);
    for (double ms : r.c.title_wait_ms) std::fprintf(g_frames, ",%.3f", ms);
    std::fprintf(g_frames, ",%.3f", r.c.game_capture_ms);
    for (double ms : r.stage_ms) std::fprintf(g_frames, ",%.3f", ms);
    for (uint64_t n : r.counts) std::fprintf(g_frames, ",%llu", (unsigned long long)n);
    for (size_t z = 0; z < g_zone_names.size(); z++) std::fprintf(g_frames, ",%.3f,%llu", r.zone_ms[z], (unsigned long long)r.zone_calls[z]);
    std::fprintf(g_frames, ",%d,%d,%.1f,%.1f,%u,%u,%u,%u", r.core[0], r.core[1], r.mem_mb, r.c.gpu_mem_mb, r.c.textures, r.c.targets, r.c.ring_kb, r.c.thermal);
    for (size_t p = 0; p < g_pass_names.size(); p++)
        for (int stage = 0; stage < 2; stage++) std::fprintf(g_frames, ",%.3f", tagged ? double(g_gpu_pass_us[slot][p][stage].load()) / 1000.0 : 0.0);
    for (size_t k = 0; k < g_kind_names.size(); k++) std::fprintf(g_frames, ",%llu", (unsigned long long)r.n.kinds[k]);
    for (size_t i = 0; i < g_state_names.size(); i++) std::fprintf(g_frames, ",%lld", i < r.state.size() ? (long long)r.state[i] : -1ll);
    std::fprintf(g_frames, ",%d\n", r.marker ? 1 : 0);
}

void Stop();

void Start() {
    char stamp[32];
    const std::time_t now = std::time(nullptr);
    std::strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", std::localtime(&now));
    std::error_code ec;
    g_dir = app::ExeDir() / "perf" / stamp;
    std::filesystem::create_directories(g_dir, ec);
    std::lock_guard lock(g_files);
    g_frames = std::fopen((g_dir / "frames.csv").string().c_str(), "w");
    g_threads = std::fopen((g_dir / "threads.csv").string().c_str(), "w");
    g_events = std::fopen((g_dir / "events.csv").string().c_str(), "w");
    g_encoders = std::fopen((g_dir / "encoders.csv").string().c_str(), "w");
    if (!g_objects_header.empty()) {
        g_objects = std::fopen((g_dir / "objects.csv").string().c_str(), "w");
        if (g_objects) std::fprintf(g_objects, "%s\n", g_objects_header.c_str());
    }
    if (g_encoders) std::fprintf(g_encoders, "frame,probe,index,kind,pass,width,height,format,depth_width,depth_height,colour_load,colour_store,depth_load,depth_store,draws,vertices,first_vs,first_ps,tex_width,tex_height,tex_format,flat,blend,vertex_ms,fragment_ms\n");
    if (!g_frames || !g_threads || !g_events) {
        LOGR("perf: could not open a recording in %s", g_dir.string().c_str());
        for (FILE** f : {&g_frames, &g_threads, &g_events, &g_encoders, &g_objects})
            if (*f) std::fclose(*f), *f = nullptr;
        return;
    }
    std::fprintf(g_frames, "frame,t_ms,interval_ms,work_ms,fps,target_ms,pace_wait_ms,drawable_wait_ms,inflight_wait_ms,gpu_ms,draw_cpu_ms,shaders,"
                           "shaders_ms,pipelines,pipelines_ms,uploads,uploads_ms,readbacks,readback_ms,draws,triangles");
    for (const std::string& p : g_pass_names) std::fprintf(g_frames, ",draws_%s", p.c_str());
    for (const std::string& p : g_pass_names) std::fprintf(g_frames, ",tris_%s", p.c_str());
    std::fprintf(g_frames, ",report_wait_ms,kernel_wait_ms,gpu_clear_ms,gpu_present_ms,gpu_blit_ms,game_wait_readback_ms,game_wait_resource_ms,game_wait_fence_ms,game_wait_flush_ms,game_wait_copy_ms,game_wait_next_frame_ms,game_capture_ms");
    for (const char* n : kStageNames) std::fprintf(g_frames, ",stage_%s_ms", n);
    for (const char* n : kCountNames) std::fprintf(g_frames, ",n_%s", n);
    for (const std::string& z : g_zone_names) std::fprintf(g_frames, ",zone_%s_ms,zone_%s_calls", z.c_str(), z.c_str());
    std::fprintf(g_frames, ",core_drawing,core_game,mem_mb,gpu_mem_mb,textures,targets,ring_kb,thermal");
    for (const std::string& p : g_pass_names) std::fprintf(g_frames, ",gpuv_%s,gpuf_%s", p.c_str(), p.c_str());
    for (const std::string& k : g_kind_names) std::fprintf(g_frames, ",kind_%s", k.c_str());
    for (const std::string& n : g_state_names) std::fprintf(g_frames, ",state_%s", n.c_str());
    std::fprintf(g_frames, ",marker\n");
    std::fprintf(g_threads, "frame,thread,name,cpu_ms\n");
    std::fprintf(g_events, "t_ms,frame,event\n");
    {  // the presenting thread is the one that draws (on the Mac a thread the runtime does not name)
        std::lock_guard names(g_names_lock);
        g_names.try_emplace(ThisThreadId(), "drawing");
    }
    if (FILE* info = std::fopen((g_dir / "info.txt").string().c_str(), "w")) {
        std::fprintf(info, "game: %s\nmachine: %s\npresenting thread: %s\npasses:", app::kGame.settings, Machine().c_str(), ThisThreadName().c_str());
        for (const std::string& p : g_pass_names) std::fprintf(info, " %s;", p.c_str());
        std::fprintf(info, "\nstarted: %s\n", stamp);
        std::fclose(info);
    }
    std::filesystem::copy_file(app::ExeDir() / app::kGame.settings, g_dir / app::kGame.settings, std::filesystem::copy_options::overwrite_existing, ec);
    g_start = clock::now();
    g_have_last = false;
    for (Row& r : g_rows) r = Row{};
#ifdef __APPLE__
    g_seen.clear();
#endif
    {
        std::lock_guard c(g_counts_lock);
        g_counts = Counts{};
    }
    g_on = true;
    std::fprintf(g_events, "0,0,recording started\n");
    static bool hooked = false;
    if (!hooked) hooked = true, std::atexit([] {
        if (On()) Stop();  // a quit while recording keeps what was recorded
    });
    LOGR("perf: recording to %s (F9 stops, F10 marks)", g_dir.string().c_str());
}

void Stop() {
    g_on = false;
    std::lock_guard lock(g_files);
    for (const Row& r : g_rows)
        if (r.frame != ~0ull && g_frames) WriteRow(r);  // what was still waiting for its GPU time (written without it if not in)
    if (g_events) std::fprintf(g_events, "%.1f,0,recording stopped\n", Ms(clock::now() - g_start));
    for (FILE** f : {&g_frames, &g_threads, &g_events, &g_encoders, &g_objects})
        if (*f) std::fclose(*f), *f = nullptr;
    LOGR("perf: recording saved in %s (tools/perf/analyze.py makes its report)", g_dir.string().c_str());
}

}  // namespace

void NameThread(const char* name) {
    if (!name) return;
    std::lock_guard lock(g_names_lock);
    g_names[ThisThreadId()] = name;
}
void Toggle() { g_toggle = true; }
void Marker() { g_marker = true; }

void Event(const std::string& text) {
    if (!On()) return;
    std::lock_guard lock(g_files);
    if (g_events) std::fprintf(g_events, "%.1f,,\"%s\"\n", Ms(clock::now() - g_start), text.c_str());
}

void SetPassNames(std::vector<std::string> names) {
    if (names.empty() || names.size() > size_t(kMaxPasses)) return;
    g_pass_names = std::move(names);
}
void SetPass(int pass) { t_pass = pass >= 0 && pass < int(g_pass_names.size()) ? pass : 0; }
int CurrentPass() { return t_pass; }
void SetKindNames(std::vector<std::string> names) {
    if (names.empty() || names.size() > size_t(kMaxKinds)) return;
    g_kind_names = std::move(names);
    g_points_kind = -1;
    for (size_t i = 0; i < g_kind_names.size(); i++)
        if (g_kind_names[i] == "points") g_points_kind = int(i);
}
void SetKind(int kind) { t_kind = kind >= 0 && kind < int(g_kind_names.size()) ? kind : 0; }
void SetState(std::vector<std::string> names, std::function<std::vector<int64_t>()> read) {
    g_state_names = std::move(names);
    g_state_read = std::move(read);
}

void CountDraw(uint32_t prim, uint32_t vertices) {
    if (!On()) return;
    std::lock_guard c(g_counts_lock);
    g_counts.draws[t_pass]++;
    g_counts.triangles[t_pass] += Triangles(prim, vertices);
    g_counts.kinds[prim == 1 && g_points_kind >= 0 ? g_points_kind : t_kind]++;
}
void AddDrawCpu(double ms) {
    if (!On()) return;
    std::lock_guard c(g_counts_lock);
    g_counts.draw_cpu_ms += ms;
}
void PaceWait(double ms, double target_ms) {
    if (!On()) return;
    std::lock_guard c(g_counts_lock);
    g_counts.pace_ms += ms;
    g_counts.target_ms = target_ms;
}

void Frame(uint64_t frame, const FrameCosts& costs) {
    static const bool from_start = [] {
        const char* e = std::getenv("XBE_PERF");
        return e && *e == '1';
    }();
    static bool started_once = false;
    if (from_start && !started_once) started_once = true, g_toggle = true;
    if (g_toggle.exchange(false)) {
        if (On()) Stop();
        else Start();
    }
    // the next frame's GPU slot, before any of its command buffers can complete
    g_gpu_tag[(frame + 1) % kRing] = frame + 1;
    g_gpu_us[(frame + 1) % kRing] = 0;
    for (auto& w : g_gpu_other_us[(frame + 1) % kRing]) w = 0;
    for (auto& pass : g_gpu_pass_us[(frame + 1) % kRing])
        for (auto& stage : pass) stage = 0;
    t_presenting = true;
    if (!On()) return;
    const auto now = clock::now();
    Row& r = g_rows[frame % kRing];
    std::lock_guard lock(g_files);
    if (r.frame != ~0ull && g_frames) WriteRow(r);  // kRing frames old: long since done
    r = Row{};
    r.frame = frame;
    r.t_ms = Ms(now - g_start);
    r.interval_ms = g_have_last ? Ms(now - g_last_present) : 0;
    g_last_present = now, g_have_last = true;
    r.c = costs;
    {
        std::lock_guard c(g_counts_lock);
        r.n = g_counts;
        const double target = g_counts.target_ms;
        g_counts = Counts{};
        g_counts.target_ms = target;
    }
    r.marker = g_marker.exchange(false);
    NoteCore(0);
    for (int i = 0; i < kStageCount; i++) r.stage_ms[i] = double(g_stage[i].exchange(0, std::memory_order_relaxed)) * TickMs();
    for (int i = 0; i < kCountCount; i++) r.counts[i] = g_count[i].exchange(0, std::memory_order_relaxed);
    for (size_t z = 0; z < g_zone_names.size(); z++) {
        r.zone_ms[z] = double(g_zone_ticks[z].exchange(0, std::memory_order_relaxed)) * TickMs();
        r.zone_calls[z] = g_zone_calls[z].exchange(0, std::memory_order_relaxed);
    }
    for (int i = 0; i < 2; i++) r.core[i] = g_core[i].load(std::memory_order_relaxed);
    static double mem_mb = 0;
#ifdef __APPLE__
    if (frame % 30 == 0) {
        task_vm_info_data_t vm;
        mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
        if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&vm), &n) == KERN_SUCCESS) mem_mb = double(vm.phys_footprint) / 1048576.0;
    }
#endif
    r.mem_mb = mem_mb;
    if (g_state_read) r.state = g_state_read();
    if (r.marker && g_events) std::fprintf(g_events, "%.1f,%llu,marker\n", r.t_ms, (unsigned long long)frame);
    SampleThreads(frame);
    if (frame % 60 == 0)  // a crash or a kill loses a second at most
        for (FILE* f : {g_frames, g_threads, g_events, g_encoders, g_objects})
            if (f) std::fflush(f);
    // rows old enough for their GPU time, oldest first
    if (frame >= uint64_t(kLag)) {
        Row& old = g_rows[(frame - kLag) % kRing];
        if (old.frame == frame - kLag && g_frames) {
            WriteRow(old);
            old.frame = ~0ull;
        }
    }
}

void GpuPass(uint64_t frame, int pass, double vertex_ms, double fragment_ms) {
    const size_t slot = frame % kRing;
    if (pass < 0 || pass >= kMaxPasses || g_gpu_tag[slot].load() != frame) return;
    if (vertex_ms > 0 && vertex_ms < 10000) g_gpu_pass_us[slot][pass][0].fetch_add(uint64_t(vertex_ms * 1000.0));
    if (fragment_ms > 0 && fragment_ms < 10000) g_gpu_pass_us[slot][pass][1].fetch_add(uint64_t(fragment_ms * 1000.0));
}

void GpuOther(uint64_t frame, GpuWork work, double ms) {
    const size_t slot = frame % kRing;
    if (g_gpu_tag[slot].load() == frame && ms > 0 && ms < 10000) g_gpu_other_us[slot][work].fetch_add(uint64_t(ms * 1000.0));
}
void EncoderRows(const std::string& rows) {
    if (!On()) return;
    std::lock_guard lock(g_files);
    if (g_encoders) std::fwrite(rows.data(), 1, rows.size(), g_encoders);
}

void SetObjectsHeader(std::string header) { g_objects_header = std::move(header); }
void ObjectRows(const std::string& rows) {
    if (!On()) return;
    std::lock_guard lock(g_files);
    if (g_objects) std::fwrite(rows.data(), 1, rows.size(), g_objects);
}
double ElapsedMs() { return Ms(clock::now() - g_start); }

void AddReportWait(double ms) {
    if (!On()) return;
    std::lock_guard c(g_counts_lock);
    g_counts.report_wait_ms += ms;
}

static int64_t NowNs() { return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count(); }
KernelWait::KernelWait() : start_ns(t_presenting && On() ? NowNs() : 0) {}
KernelWait::~KernelWait() {
    if (!start_ns || !On()) return;
    const double ms = double(NowNs() - start_ns) / 1e6;
    std::lock_guard c(g_counts_lock);
    g_counts.kernel_wait_ms += ms;
}

void SetZoneNames(std::vector<std::string> names) {
    if (names.size() <= size_t(kMaxZones)) g_zone_names = std::move(names);
}
void AddZone(int zone, uint64_t ticks) {
    if (zone < 0 || zone >= kMaxZones) return;
    g_zone_ticks[zone].fetch_add(ticks, std::memory_order_relaxed);
    g_zone_calls[zone].fetch_add(1, std::memory_order_relaxed);
}
void NoteCore(int which) {
#ifdef __APPLE__
    size_t cpu = 0;
    if (On() && pthread_cpu_number_np(&cpu) == 0) g_core[which & 1].store(int(cpu), std::memory_order_relaxed);
#else
    (void)which;
#endif
}

void Gpu(uint64_t frame, double ms) {
    const int slot = int(frame % kRing);
    if (g_gpu_tag[slot].load() != frame || ms <= 0 || ms > 10000) return;
    g_gpu_us[slot].fetch_add(uint64_t(ms * 1000.0));
}

}  // namespace perf
