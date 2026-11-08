// XAPI functions that talk to hardware: controller ports and the hard disk cache partition
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

#include "app/app.h"
#include "core/input.h"
#include "core/log.h"
#include "xdk/hle.h"
#include "xdk/layout.h"

using namespace kernel;

// configs written before the SDK knew the name
#ifndef SYM_XapiVerifyMediaInDrive
#define SYM_XapiVerifyMediaInDrive FFFFFC00
#define HAS_XapiVerifyMediaInDrive 0
#endif
#ifndef SYM_XInputGetCapabilities
#define SYM_XInputGetCapabilities FFFFFC04
#define HAS_XInputGetCapabilities 0
#endif

namespace {

// XPP device type records: CurrentConnected, ChangeConnected, PreviousConnected
constexpr uint32_t kGamepadType = SYM(XPP_GamepadType);
constexpr uint32_t kMemoryUnitType = SYM(XPP_MemoryUnitType);

std::mutex g_ports_lock;
bool g_poller = false;
uint32_t g_wheels = 0;  // ports a steering wheel steers, as XInputGetCapabilities last told

void UpdatePorts() {
    std::lock_guard<std::mutex> lock(g_ports_lock);
    uint32_t cur = 0, wheels = 0;
    for (int p = 0; p < 4; p++) {
        input::Pad pad = input::Poll(p);
        if (pad.connected) cur |= 1u << p;
        if (pad.connected && pad.wheel) wheels |= 1u << p;
    }
    uint32_t old = R32(kGamepadType + XDK_XPP_Current);
    // a port that became or stopped being a wheel is unplugged and plugged again (XGetDeviceChanges reports both),
    // so the title opens it again and reads its capabilities
    uint32_t replug = (wheels ^ g_wheels) & cur & old;
    if (cur != old || replug) {
        W32(kGamepadType + XDK_XPP_Changed, R32(kGamepadType + XDK_XPP_Changed) | (cur ^ old) | replug);
        W32(kGamepadType + XDK_XPP_Current, cur);
        LOGI("xapi: controller ports now %X, wheels %X", cur, wheels);
    }
    g_wheels = wheels;
}

struct Handle {
    uint32_t port;
    uint32_t packet;
    input::Pad last;
};
std::unordered_map<uint32_t, Handle> g_handles;

// Controller 1 recorded and played back, for repeatable runs. Keyed to the title's own reads of that controller
// (XInputGetState on port 0, once a frame in PGR2), not the clock, so a playback presses the same buttons on the same
// frames however long loading takes. XBE_PAD_RECORD=file writes each change: "read buttons a b x y black white lt rt
// lx ly rx ry" (buttons in hex, the rest decimal), with the clock in a comment; XBE_PAD_PLAY=file plays one back in
// place of the controller, and then the controller again once it ends (XBE_PAD_PLAY_HOLD=1: the last state held).
// Relative paths are beside the executable.
struct PadTrack {
    uint64_t read;
    input::Pad pad;
};
std::filesystem::path PadPath(const char* e) {
    std::filesystem::path p = e;
    return p.is_relative() ? app::ExeDir() / p : p;
}
FILE* PadRecordFile() {
    static FILE* f = [] {
        const char* e = std::getenv("XBE_PAD_RECORD");
        if (!e || !*e) return (FILE*)nullptr;
        FILE* f = std::fopen(PadPath(e).string().c_str(), "w");
        if (f) {
            std::fprintf(f, "# controller 1 by the title's reads of it (XInputGetState, port 0): read buttons a b x y black white lt rt lx ly rx ry\n");
            LOGR("input: recording controller 1 to %s", PadPath(e).string().c_str());
        }
        return f;
    }();
    return f;
}
const std::vector<PadTrack>& PadPlayback() {
    static const std::vector<PadTrack> track = [] {
        std::vector<PadTrack> t;
        const char* e = std::getenv("XBE_PAD_PLAY");
        if (!e || !*e) return t;
        FILE* f = std::fopen(PadPath(e).string().c_str(), "r");
        if (!f) {
            LOGE("input: no pad recording at %s", PadPath(e).string().c_str());
            return t;
        }
        char line[256];
        while (std::fgets(line, sizeof line, f)) {
            if (line[0] == '#') continue;
            unsigned long long read;
            unsigned buttons, a[8];
            int lx, ly, rx, ry;
            if (std::sscanf(line, "%llu %x %u %u %u %u %u %u %u %u %d %d %d %d", &read, &buttons, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6],
                            &a[7], &lx, &ly, &rx, &ry) != 14)
                continue;
            PadTrack k{read, {}};
            k.pad.connected = true;
            k.pad.buttons = uint16_t(buttons);
            for (int i = 0; i < 8; i++) k.pad.analog[i] = uint8_t(a[i]);
            k.pad.lx = int16_t(lx), k.pad.ly = int16_t(ly), k.pad.rx = int16_t(rx), k.pad.ry = int16_t(ry);
            t.push_back(k);
        }
        std::fclose(f);
        LOGR("input: playing controller 1 from %s, %zu changes to read %llu", PadPath(e).string().c_str(), t.size(),
             t.empty() ? 0ull : (unsigned long long)t.back().read);
        return t;
    }();
    return track;
}
uint64_t g_pad_reads = 0;  // the title's reads of controller 1

}  // namespace

HLE_STDCALL(XMountUtilityDrive, 1) {
    CreateSymbolicLink("\\??\\Z:", "\\Device\\Harddisk0\\Partition5");
    return 1;
}

HLE_STDCALL(XInitDevices, 2) {
    XDK_REQUIRE("xapi", XDK_KEY(XPP_Current), XDK_KEY(XPP_Changed), XDK_KEY(XPP_Previous), XDK_DATA(XPP_GamepadType));
    W32(kGamepadType + XDK_XPP_Current, 0);
    W32(kGamepadType + XDK_XPP_Changed, 0);
    W32(kGamepadType + XDK_XPP_Previous, 0);
    if constexpr (HAS_XPP_MemoryUnitType != 0) {  // titles without memory unit support have no record
        W32(kMemoryUnitType + XDK_XPP_Current, 0);
        W32(kMemoryUnitType + XDK_XPP_Changed, 0);
        W32(kMemoryUnitType + XDK_XPP_Previous, 0);
    }
    UpdatePorts();
    if (!g_poller) {
        g_poller = true;
        std::thread([] {
            for (;;) {
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                UpdatePorts();
            }
        }).detach();
    }
    return 0;
}

HLE_STDCALL(XInputOpen, 4) {
    uint32_t type = a[0], port = a[1];
    if (type != kGamepadType || port > 3) return 0;
    uint32_t h = mem::AllocSystem(16);
    std::lock_guard<std::mutex> lock(g_ports_lock);
    g_handles[h] = {port, 0, {}};
    return h;
}

HLE_STDCALL(XInputClose, 1) {
    std::lock_guard<std::mutex> lock(g_ports_lock);
    g_handles.erase(a[0]);
    mem::FreeSystem(a[0]);
    return 0;
}

HLE_STDCALL(XInputGetState, 2) {
    uint32_t port;
    {
        std::lock_guard<std::mutex> lock(g_ports_lock);
        auto it = g_handles.find(a[0]);
        if (it == g_handles.end()) return 6;  // ERROR_INVALID_HANDLE
        port = it->second.port;
    }
    input::Pad p = input::Poll(int(port));
    if (port == 0) {
        const uint64_t read = g_pad_reads++;
        const auto& track = PadPlayback();
        static const bool hold = [] {
            const char* e = std::getenv("XBE_PAD_PLAY_HOLD");
            return e && *e == '1';
        }();
        if (!track.empty() && (read <= track.back().read || hold)) {
            // the last change at or before this read
            auto it = std::upper_bound(track.begin(), track.end(), read, [](uint64_t r, const PadTrack& k) { return r < k.read; });
            if (it != track.begin()) p = std::prev(it)->pad;
            else p = input::Pad{true};
            if (read == track.back().read) LOGR("input: the pad recording ends at read %llu", (unsigned long long)read);
        }
        // XBE_SHOT_READS="r1 r2 ..." (testing): a screenshot at each of those reads of controller 1, the same frames in
        // every playback of one recording
        static const std::vector<uint64_t> shots = [] {
            std::vector<uint64_t> v;
            if (const char* e = std::getenv("XBE_SHOT_READS"))
                for (char* end = nullptr; *e; e = end) {
                    const uint64_t r = std::strtoull(e, &end, 10);
                    if (end == e) break;
                    v.push_back(r);
                }
            return v;
        }();
        if (std::find(shots.begin(), shots.end(), read) != shots.end()) {
            app::RequestScreenshot();
            LOGR("input: screenshot at read %llu", (unsigned long long)read);
        }
        if (FILE* f = PadRecordFile()) {
            static input::Pad last{};
            static const auto start = std::chrono::steady_clock::now();
            static uint64_t reads_then = 0;
            static auto then = start;
            const bool same = p.buttons == last.buttons && !std::memcmp(p.analog, last.analog, sizeof p.analog) && p.lx == last.lx &&
                              p.ly == last.ly && p.rx == last.rx && p.ry == last.ry;
            if (read == 0 || !same) {
                const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                std::fprintf(f, "%llu %X %u %u %u %u %u %u %u %u %d %d %d %d # %.2f s\n", (unsigned long long)read, p.buttons, p.analog[0],
                             p.analog[1], p.analog[2], p.analog[3], p.analog[4], p.analog[5], p.analog[6], p.analog[7], p.lx, p.ly, p.rx, p.ry, t);
                std::fflush(f);
                last = p;
            }
            // how often the title reads it, to tell a frame's read from a timer's
            const auto now = std::chrono::steady_clock::now();
            if (now - then >= std::chrono::seconds(10)) {
                LOGR("input: controller 1 read %llu times in the last 10 s", (unsigned long long)(read - reads_then));
                reads_then = read, then = now;
            }
        }
    }
    if (!p.connected) return 1167;  // ERROR_DEVICE_NOT_CONNECTED
    std::lock_guard<std::mutex> lock(g_ports_lock);
    Handle& h = g_handles[a[0]];
    if (std::memcmp(&p, &h.last, sizeof(p))) h.packet++;
    h.last = p;
    uint32_t s = a[1];
    W32(s, h.packet);  // XINPUT_STATE: packet, buttons, 8 analog buttons, thumbs
    W16(s + 4, p.buttons);
    std::memcpy(HostPtr(s + 6), p.analog, 8);
    W16(s + 14, uint16_t(p.lx));
    W16(s + 16, uint16_t(p.ly));
    W16(s + 18, uint16_t(p.rx));
    W16(s + 20, uint16_t(p.ry));
    return 0;
}

// XINPUT_CAPABILITIES: SubType, Reserved, In (an XINPUT_GAMEPAD of what it has), Out (XINPUT_RUMBLE). A title may
// treat a wheel (XINPUT_DEVSUBTYPE_GC_WHEEL) differently, PGR2 without its gamepad dead zone and stick shaping
HLE_STDCALL(XInputGetCapabilities, 2) {
    uint32_t port;
    {
        std::lock_guard<std::mutex> lock(g_ports_lock);
        auto it = g_handles.find(a[0]);
        if (it == g_handles.end()) return 6;
        port = it->second.port;
    }
    input::Pad p = input::Poll(int(port));
    if (!p.connected) return 1167;
    uint32_t c = a[1];
    std::memset(HostPtr(c), 0, 24);
    G<uint8_t>(c) = p.wheel ? 0x10 : 0x01;  // XINPUT_DEVSUBTYPE_GC_WHEEL, XINPUT_DEVSUBTYPE_GC_GAMEPAD
    W16(c + 2, 0xFF);
    std::memset(HostPtr(c + 4), 0xFF, 8);
    for (int i = 0; i < 4; i++) W16(c + 12 + i * 2, 0xFFFF);
    W16(c + 20, 0xFFFF);
    W16(c + 22, 0xFFFF);
    {
        std::lock_guard<std::mutex> lock(g_ports_lock);
        if (p.wheel) g_wheels |= 1u << port;
        else g_wheels &= ~(1u << port);
    }
    LOGI("xapi: controller %u is a %s", port + 1, p.wheel ? "steering wheel" : "gamepad");
    return 0;
}

HLE_STDCALL(XInputSetState, 2) {
    uint32_t port;
    {
        std::lock_guard<std::mutex> lock(g_ports_lock);
        auto it = g_handles.find(a[0]);
        if (it == g_handles.end()) return 6;
        port = it->second.port;
    }
    uint32_t fb = a[1];
    // XINPUT_FEEDBACK_HEADER is dwStatus, hEvent and 58 reserved bytes, the rumble follows
    input::Rumble(int(port), R16(fb + 66), R16(fb + 68));
    W32(fb, 0);
    if (uint32_t ev = R32(fb + 4))
        if (ObjectRef o = FromHandle(ev)) SetEvent(*o);
    return 997;  // ERROR_IO_PENDING, already complete
}

// A recompiled image launches through the XAPI, then a process restart into its executable; any other
// (the dashboard, updates) is refused and the title stays in its menu.
X86_EXTERN_FUNC(GUEST_ORIG(XLaunchNewImageA));
X86_FUNC(GUEST(XLaunchNewImageA)) {
    std::string path = ReadCString(R32(ctx.esp + 4));
    if (ImageExe(path)) {
        GUEST_ORIG(XLaunchNewImageA)(ctx, base);
        return;
    }
    LOGW("xapi: launch of %s refused, it is not recompiled", path.c_str());
    Return(ctx, 2, 2);  // ERROR_FILE_NOT_FOUND
}

// XAPI's startup check of the disc in the drive (the name is ours, the XDK's is not known): an authentication mode
// page, then for titles whose certificate asks for it the signed security sector. The game runs from the player's
// own files, so there is no drive or disc to authenticate; the check passes as it does with the disc inserted.
HLE_STDCALL(XapiVerifyMediaInDrive, 0) { return 0; }
