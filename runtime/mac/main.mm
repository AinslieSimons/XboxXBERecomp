// The host side on macOS: the SDL window and its event loop on the main thread, the display settings, keys, sound output.
// In place of the Win32 app/main.cpp, app/splash.cpp, app/handover.cpp, app/system_info.cpp, audio/xaudio2.cpp, core/wheel.cpp.
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <QuartzCore/CAMetalLayer.h>
#import <CoreServices/CoreServices.h>
#include <mach-o/dyld.h>
#include <sys/sysctl.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_metal.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <codecvt>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <imgui.h>

#include "app/app.h"
#include "audio/audio.h"
#include "core/input.h"
#include "core/log.h"
#include "gpu/backend.h"
#include "image_symbols.h"
#include "kernel/kernel.h"

#include <windows.h>

namespace gpu {
void SaveMetalCache();  // gpu/metal.mm
}

namespace input {
void HandleSdlEvent(const SDL_Event& e);  // core/input.cpp: controllers plugged in and out
}

namespace app {

namespace {
SDL_Window* g_window = nullptr;
SDL_MetalView g_view = nullptr;
void* g_layer = nullptr;
std::atomic<bool> g_focused{true};
std::atomic<uint32_t> g_px_w{1280}, g_px_h{720};
Uint32 g_apply_event = 0;

std::string Narrow(const wchar_t* w) {
    std::string out;
    for (; w && *w; w++) {  // the texts here are ASCII and the odd symbol
        wchar_t c = *w;
        if (c < 0x80) out += char(c);
        else if (c < 0x800) out += char(0xC0 | (c >> 6)), out += char(0x80 | (c & 63));
        else out += char(0xE0 | (c >> 12)), out += char(0x80 | ((c >> 6) & 63)), out += char(0x80 | (c & 63));
    }
    return out;
}

void UpdatePixels() {
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(g_window, &w, &h);
    if (w > 0 && h > 0) g_px_w = uint32_t(w), g_px_h = uint32_t(h);
}

// the display settings: windowed at the size asked (in pixels), else filling the screen
void ApplyNow() {
    const Settings& s = Get();
    if (s.display_mode == DisplayMode::Windowed) {
        SDL_SetWindowFullscreen(g_window, false);
        float density = SDL_GetWindowPixelDensity(g_window);
        if (density <= 0) density = 1;
        if (s.width && s.height) SDL_SetWindowSize(g_window, int(s.width / density), int(s.height / density));
    } else {
        SDL_SetWindowFullscreen(g_window, true);
    }
    SDL_SyncWindow(g_window);
    UpdatePixels();
}

// Windows virtual keys to where SDL has them
SDL_Scancode ScancodeOf(int vk) {
    if (vk >= 'A' && vk <= 'Z') return SDL_Scancode(SDL_SCANCODE_A + (vk - 'A'));
    if (vk >= '1' && vk <= '9') return SDL_Scancode(SDL_SCANCODE_1 + (vk - '1'));
    if (vk == '0') return SDL_SCANCODE_0;
    if (vk >= VK_F1 && vk <= VK_F12) return SDL_Scancode(SDL_SCANCODE_F1 + (vk - VK_F1));
    if (vk >= VK_NUMPAD1 && vk <= VK_NUMPAD9) return SDL_Scancode(SDL_SCANCODE_KP_1 + (vk - VK_NUMPAD1));
    switch (vk) {
        case VK_NUMPAD0: return SDL_SCANCODE_KP_0;
        case VK_BACK: return SDL_SCANCODE_BACKSPACE;
        case VK_TAB: return SDL_SCANCODE_TAB;
        case VK_RETURN: return SDL_SCANCODE_RETURN;
        case VK_LSHIFT: case VK_SHIFT: return SDL_SCANCODE_LSHIFT;
        case VK_RSHIFT: return SDL_SCANCODE_RSHIFT;
        case VK_LCONTROL: case VK_CONTROL: return SDL_SCANCODE_LCTRL;
        case VK_RCONTROL: return SDL_SCANCODE_RCTRL;
        case VK_LMENU: case VK_MENU: return SDL_SCANCODE_LALT;
        case VK_RMENU: return SDL_SCANCODE_RALT;
        case VK_ESCAPE: return SDL_SCANCODE_ESCAPE;
        case VK_SPACE: return SDL_SCANCODE_SPACE;
        case VK_PRIOR: return SDL_SCANCODE_PAGEUP;
        case VK_NEXT: return SDL_SCANCODE_PAGEDOWN;
        case VK_END: return SDL_SCANCODE_END;
        case VK_HOME: return SDL_SCANCODE_HOME;
        case VK_LEFT: return SDL_SCANCODE_LEFT;
        case VK_UP: return SDL_SCANCODE_UP;
        case VK_RIGHT: return SDL_SCANCODE_RIGHT;
        case VK_DOWN: return SDL_SCANCODE_DOWN;
        case VK_INSERT: return SDL_SCANCODE_INSERT;
        case VK_DELETE: return SDL_SCANCODE_DELETE;
        case VK_OEM_1: return SDL_SCANCODE_SEMICOLON;
        case VK_OEM_PLUS: return SDL_SCANCODE_EQUALS;
        case VK_OEM_COMMA: return SDL_SCANCODE_COMMA;
        case VK_OEM_MINUS: return SDL_SCANCODE_MINUS;
        case VK_OEM_PERIOD: return SDL_SCANCODE_PERIOD;
        case VK_OEM_2: return SDL_SCANCODE_SLASH;
        case VK_OEM_3: return SDL_SCANCODE_GRAVE;
        case VK_OEM_4: return SDL_SCANCODE_LEFTBRACKET;
        case VK_OEM_5: return SDL_SCANCODE_BACKSLASH;
        case VK_OEM_6: return SDL_SCANCODE_RIGHTBRACKET;
        case VK_OEM_7: return SDL_SCANCODE_APOSTROPHE;
        default: return SDL_SCANCODE_UNKNOWN;
    }
}

uint16_t VirtualKey(SDL_Scancode sc) {
    for (int vk = 1; vk < 256; vk++)
        if (vk != VK_SHIFT && vk != VK_CONTROL && vk != VK_MENU && ScancodeOf(vk) == sc) return uint16_t(vk);
    return 0;
}

ImGuiKey ImGuiKeyOf(SDL_Scancode sc) {
    if (sc >= SDL_SCANCODE_A && sc <= SDL_SCANCODE_Z) return ImGuiKey(ImGuiKey_A + (sc - SDL_SCANCODE_A));
    switch (sc) {
        case SDL_SCANCODE_LEFT: return ImGuiKey_LeftArrow;
        case SDL_SCANCODE_RIGHT: return ImGuiKey_RightArrow;
        case SDL_SCANCODE_UP: return ImGuiKey_UpArrow;
        case SDL_SCANCODE_DOWN: return ImGuiKey_DownArrow;
        case SDL_SCANCODE_RETURN: return ImGuiKey_Enter;
        case SDL_SCANCODE_ESCAPE: return ImGuiKey_Escape;
        case SDL_SCANCODE_SPACE: return ImGuiKey_Space;
        case SDL_SCANCODE_TAB: return ImGuiKey_Tab;
        case SDL_SCANCODE_BACKSPACE: return ImGuiKey_Backspace;
        case SDL_SCANCODE_DELETE: return ImGuiKey_Delete;
        case SDL_SCANCODE_LSHIFT: return ImGuiKey_LeftShift;
        case SDL_SCANCODE_RSHIFT: return ImGuiKey_RightShift;
        case SDL_SCANCODE_LCTRL: return ImGuiKey_LeftCtrl;
        case SDL_SCANCODE_RCTRL: return ImGuiKey_RightCtrl;
        default: return ImGuiKey_None;
    }
}

// the window's events for the settings window, in the drawable's pixels as the renderer gives ImGui its display
void FeedImGui(const SDL_Event& e) {
    std::lock_guard lock(UiLock());
    ImGuiIO& io = ImGui::GetIO();
    const float density = SDL_GetWindowPixelDensity(g_window);
    switch (e.type) {
        case SDL_EVENT_MOUSE_MOTION: io.AddMousePosEvent(e.motion.x * density, e.motion.y * density); break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
        case SDL_EVENT_MOUSE_BUTTON_UP: {
            int b = e.button.button == SDL_BUTTON_LEFT ? 0 : e.button.button == SDL_BUTTON_RIGHT ? 1 : e.button.button == SDL_BUTTON_MIDDLE ? 2 : -1;
            if (b >= 0) io.AddMouseButtonEvent(b, e.type == SDL_EVENT_MOUSE_BUTTON_DOWN);
            break;
        }
        case SDL_EVENT_MOUSE_WHEEL: io.AddMouseWheelEvent(e.wheel.x, e.wheel.y); break;
        case SDL_EVENT_KEY_DOWN:
        case SDL_EVENT_KEY_UP:
            if (ImGuiKey k = ImGuiKeyOf(e.key.scancode); k != ImGuiKey_None) io.AddKeyEvent(k, e.type == SDL_EVENT_KEY_DOWN);
            break;
        case SDL_EVENT_WINDOW_FOCUS_GAINED: io.AddFocusEvent(true); break;
        case SDL_EVENT_WINDOW_FOCUS_LOST: io.AddFocusEvent(false); break;
        default: break;
    }
}

std::atomic<bool> g_screenshot{false};
std::atomic<int64_t> g_shot_at{0};
}  // namespace

void* MetalLayer() { return g_layer; }
Mode DrawablePixels() { return {g_px_w.load(), g_px_h.load()}; }

std::filesystem::path ExeDir() {
    static const std::filesystem::path dir = [] {
        char p[4096];
        uint32_t n = sizeof(p);
        if (_NSGetExecutablePath(p, &n)) return std::filesystem::current_path();
        std::error_code ec;
        auto real = std::filesystem::canonical(p, ec);
        return ec ? std::filesystem::path(p).parent_path() : real.parent_path();
    }();
    return dir;
}

int Instance() {
    static const int n = [] {
        const char* e = std::getenv("INSTANCE");
        return e ? std::max(0, std::min(8, std::atoi(e))) : 0;
    }();
    return n;
}

void* Window() { return g_window; }

std::recursive_mutex& UiLock() {
    static std::recursive_mutex m;
    return m;
}

void ApplyDisplay() {
    SDL_Event e{};
    e.type = g_apply_event;
    SDL_PushEvent(&e);
}

void Quit() {
    LOGR("quit asked for");
    SDL_Event e{};
    e.type = SDL_EVENT_QUIT;
    SDL_PushEvent(&e);
}

bool Paused() { return Get().pause_unfocused && !g_focused; }

// display settings (the Win32 parts of app/settings.cpp)
void* Monitor() { return nullptr; }
std::vector<std::string> Monitors() {
    int n = 0;
    SDL_DisplayID* ids = SDL_GetDisplays(&n);
    std::vector<std::string> names;
    SDL_DisplayID primary = SDL_GetPrimaryDisplay();
    for (int i = 0; i < n; i++) {
        const SDL_DisplayMode* m = SDL_GetDesktopDisplayMode(ids[i]);
        const char* name = SDL_GetDisplayName(ids[i]);
        int w = m ? int(m->w * m->pixel_density) : 0, h = m ? int(m->h * m->pixel_density) : 0;
        names.push_back(std::to_string(i + 1) + ": " + (name ? name : "Display") + " (" + std::to_string(w) + " x " + std::to_string(h) +
                        (ids[i] == primary ? ", primary)" : ")"));
    }
    SDL_free(ids);
    return names;
}
Mode Desktop() {
    const SDL_DisplayMode* m = SDL_GetDesktopDisplayMode(g_window ? SDL_GetDisplayForWindow(g_window) : SDL_GetPrimaryDisplay());
    if (!m) return {1920, 1080};
    return {uint32_t(m->w * m->pixel_density), uint32_t(m->h * m->pixel_density)};
}
std::vector<Mode> MacModes() {
    // Windows' display modes stand for render sizes here: the screen is left as it is
    Mode d = Desktop();
    std::vector<Mode> modes{d};
    for (Mode m : {Mode{3840, 2160}, Mode{2560, 1440}, Mode{1920, 1080}, Mode{1600, 900}, Mode{1280, 720}, Mode{1024, 768}, Mode{640, 480}})
        if (m.width < d.width && m.height < d.height) modes.push_back(m);
    return modes;
}

// the window itself, when no size is set: what is on screen is what is drawn
Mode OutputDefault() { return g_window ? Mode{g_px_w.load(), g_px_h.load()} : Desktop(); }

bool OpenSplash() { return false; }
void SplashFirstFrame() {}
void CloseSplash() {}
void OpenGameWindow(bool) {}
bool StartHandover() { return false; }
bool HandingOver() { return false; }
void FinishHandover() {}

void Pace(uint32_t vblanks) {
    using clock = std::chrono::steady_clock;
    static clock::time_point next = clock::now();
    if (Paused()) {
        LOGR("paused: the window is not in front");
        while (Paused()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        LOGR("resumed");
        next = clock::now();
    }
    uint32_t cap = PaceRate(Get().fps_cap, vblanks);
    auto now = clock::now();
    if (!cap) {
        next = now;
        return;
    }
    auto period = std::chrono::nanoseconds(1000000000ull / cap);
    static clock::time_point last = now, report = now + std::chrono::seconds(10);
    static uint32_t frames = 0, slow = 0;
    static double longest = 0;
    double took = std::chrono::duration<double, std::milli>(now - last).count();
    last = now;
    frames++;
    if (took > 1.5 * std::chrono::duration<double, std::milli>(period).count()) slow++, longest = std::max(longest, took);
    if (now >= report) {
        uint32_t compiled = gpu::g_shaders_compiled.exchange(0);
        if (slow) LOGR("frames: %u of %u in the last 10 s over %.1f ms (%u a second paced), the longest %.1f ms; %u shaders compiled", slow, frames,
                       1.5 * std::chrono::duration<double, std::milli>(period).count(), cap, longest, compiled);
        report = now + std::chrono::seconds(10);
        frames = slow = 0;
        longest = 0;
    }
    next += period;
    if (next < now - period) next = now;
    while (clock::now() + std::chrono::milliseconds(2) < next) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    while (clock::now() < next) std::this_thread::yield();
}

// the machine, at a run's start
void LogSystem() {
    char chip[128] = {};
    size_t n = sizeof(chip);
    sysctlbyname("machdep.cpu.brand_string", chip, &n, nullptr, 0);
    uint64_t mem = 0;
    n = sizeof(mem);
    sysctlbyname("hw.memsize", &mem, &n, nullptr, 0);
    NSOperatingSystemVersion v = NSProcessInfo.processInfo.operatingSystemVersion;
    LOGR("System: macOS %ld.%ld.%ld, %s, %llu GB", long(v.majorVersion), long(v.minorVersion), long(v.patchVersion), chip, (unsigned long long)(mem >> 30));
    LogSettings();
}

// F12: a PNG beside the executable
void RequestScreenshot() { g_screenshot = true; }
bool TakeScreenshotRequest() { return g_screenshot.exchange(false); }
bool ScreenshotJustSaved() {
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return g_shot_at && now - g_shot_at < 3'000'000'000LL;
}
void SaveScreenshot(std::vector<uint8_t> bgra, uint32_t w, uint32_t h) {
    @autoreleasepool {
        std::filesystem::path dir = ExeDir() / "screenshots";
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        std::string name = std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + ".png";
        CFURLRef url = CFURLCreateFromFileSystemRepresentation(nullptr, reinterpret_cast<const UInt8*>((dir / name).c_str()), CFIndex((dir / name).string().size()), false);
        for (size_t i = 3; i < bgra.size(); i += 4) bgra[i] = 255;
        CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
        CGContextRef ctx = CGBitmapContextCreate(bgra.data(), w, h, 8, w * 4, cs, kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little);
        CGImageRef img = CGBitmapContextCreateImage(ctx);
        CGImageDestinationRef dst = CGImageDestinationCreateWithURL(url, kUTTypePNG, 1, nullptr);
        if (dst) {
            CGImageDestinationAddImage(dst, img, nullptr);
            CGImageDestinationFinalize(dst);
            CFRelease(dst);
        }
        CGImageRelease(img);
        CGContextRelease(ctx);
        CGColorSpaceRelease(cs);
        CFRelease(url);
        g_shot_at = std::chrono::steady_clock::now().time_since_epoch().count();
    }
}

}  // namespace app

// testing: XBE_KEYS="seconds:vk:held seconds:vk:held ..." presses keys by the clock, with no one at the keyboard
static bool ScriptedKey(int vk) {
    struct Press { double at, held; int vk; };
    static const std::vector<Press> script = [] {
        std::vector<Press> out;
        const char* e = std::getenv("XBE_KEYS");
        for (std::string rest = e ? e : ""; !rest.empty();) {
            size_t sp = rest.find(' ');
            Press p{};
            if (std::sscanf(rest.substr(0, sp).c_str(), "%lf:%i:%lf", &p.at, &p.vk, &p.held) == 3) out.push_back(p);
            rest = sp == std::string::npos ? "" : rest.substr(sp + 1);
        }
        return out;
    }();
    if (script.empty()) return false;
    static const auto start = std::chrono::steady_clock::now();
    double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    for (const Press& p : script)
        if (p.vk == vk && t >= p.at && t < p.at + p.held) return true;
    return false;
}

short GetAsyncKeyState(int vk) {
    if (ScriptedKey(vk)) return short(0x8000);
    if (vk == VK_LBUTTON || vk == VK_RBUTTON || vk == VK_MBUTTON) {
        SDL_MouseButtonFlags b = SDL_GetMouseState(nullptr, nullptr);
        Uint32 m = vk == VK_LBUTTON ? SDL_BUTTON_LMASK : vk == VK_RBUTTON ? SDL_BUTTON_RMASK : SDL_BUTTON_MMASK;
        return (b & m) ? short(0x8000) : 0;
    }
    SDL_Scancode sc = app::ScancodeOf(vk);
    if (sc == SDL_SCANCODE_UNKNOWN) return 0;
    return SDL_GetKeyboardState(nullptr)[sc] ? short(0x8000) : 0;
}

HWND GetForegroundWindow() { return app::g_focused || std::getenv("XBE_KEYS") ? app::g_window : nullptr; }

int MessageBoxW(HWND, const wchar_t* text, const wchar_t* title, unsigned type) {
    std::string t = app::Narrow(text), c = app::Narrow(title);
    LOGW("%s: %s", c.c_str(), t.c_str());
    if (type & MB_YESNO) return IDYES;  // no question stops a start on macOS yet: the log has what it asked
    SDL_ShowSimpleMessageBox((type & MB_ICONERROR) ? SDL_MESSAGEBOX_ERROR : SDL_MESSAGEBOX_WARNING, c.c_str(), t.c_str(), nullptr);
    return IDOK;
}

// ---- sound

namespace audio {

void StartOutput(std::function<void(float*, uint32_t)> render) {
    static auto* pull = new std::function<void(float*, uint32_t)>(std::move(render));
    SDL_AudioSpec spec{SDL_AUDIO_F32, 2, int(kRate)};
    auto cb = [](void*, SDL_AudioStream* stream, int additional, int) {
        static std::vector<float> buf;
        // paused (app::Paused): the mixer is not pulled, so the title's sound and its sample clock stand still
        if (additional <= 0 || app::Paused()) return;
        int frames = additional / 8;
        buf.resize(size_t(frames) * 2);
        (*pull)(buf.data(), uint32_t(frames));
        SDL_PutAudioStreamData(stream, buf.data(), frames * 8);
        // XBE_AUDIO_WAV=file (testing): what the mixer hands the output, as 16 bit stereo at kRate, beside the executable
        // unless a path; its header is kept current, so the file plays even if the game is killed
        static FILE* wav = [] {
            const char* e = std::getenv("XBE_AUDIO_WAV");
            if (!e || !*e) return (FILE*)nullptr;
            std::filesystem::path p = e;
            if (p.is_relative()) p = app::ExeDir() / p;
            FILE* f = std::fopen(p.string().c_str(), "wb");
            if (f) LOGR("audio: recording the output to %s", p.string().c_str());
            return f;
        }();
        if (wav) {
            static uint32_t bytes = 0;
            static std::vector<int16_t> pcm;
            pcm.resize(buf.size());
            for (size_t i = 0; i < buf.size(); i++) pcm[i] = int16_t(std::lround(std::clamp(buf[i], -1.0f, 1.0f) * 32767));
            std::fseek(wav, 44 + long(bytes), SEEK_SET);
            std::fwrite(pcm.data(), 2, pcm.size(), wav);
            bytes += uint32_t(pcm.size() * 2);
            auto u32 = [](uint8_t* p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = uint8_t(v >> (8 * i)); };
            uint8_t h[44] = {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ', 16, 0, 0, 0, 1, 0, 2, 0};
            u32(h + 4, 36 + bytes);
            u32(h + 24, kRate);
            u32(h + 28, kRate * 4);
            h[32] = 4, h[34] = 16;
            std::memcpy(h + 36, "data", 4);
            u32(h + 40, bytes);
            std::fseek(wav, 0, SEEK_SET);
            std::fwrite(h, 1, 44, wav);
            std::fflush(wav);
        }
    };
    SDL_AudioStream* s = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, cb, nullptr);
    if (!s) {
        LOGE("audio: no output: %s", SDL_GetError());
        return;
    }
    SDL_ResumeAudioStreamDevice(s);
    LOGR("audio: SDL output at %u Hz", kRate);
}

}  // namespace audio

// ---- steering wheels: core/wheel.cpp speaks DirectInput; on macOS none yet, a controller or the keyboard plays

namespace input {
void WheelInit(void*) {}
void WheelSetup() {}
bool WheelRead(float*, bool*) { return false; }
void WheelFeedback(const WheelFeel&, uint16_t, uint16_t) {}
WheelInfo Wheel() { return {}; }
void WheelTestForce(float) {}
}  // namespace input

// ---- start

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    std::string log = app::kGame.log;
    if (app::Instance() > 1) log += "-" + std::to_string(app::Instance());
    logx::Init((app::ExeDir() / "logs").string(), log);
    SDL_SetHint(SDL_HINT_MAC_CTRL_CLICK_EMULATE_RIGHT_CLICK, "0");
    SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_GAMEPAD)) {
        LOGE("SDL failed to start: %s", SDL_GetError());
        return 1;
    }
    // first run on a Mac: a window; the settings file then decides
    app::Get().display_mode = app::DisplayMode::Windowed;
    app::Load();
    if (app::kGame.init) app::kGame.init();
    app::LogSystem();

    app::GameFiles game;
    if (!app::FindGame(game)) return 1;

    std::string title = app::Narrow(app::kGame.title);
    app::g_window = SDL_CreateWindow(title.c_str(), 1280, 720, SDL_WINDOW_METAL | SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_RESIZABLE);
    if (!app::g_window) {
        LOGE("no window: %s", SDL_GetError());
        return 1;
    }
    app::g_view = SDL_Metal_CreateView(app::g_window);
    app::g_layer = SDL_Metal_GetLayer(app::g_view);
    app::g_apply_event = SDL_RegisterEvents(1);
    app::ApplyNow();
    SDL_ShowWindow(app::g_window);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
    app::OverlayInit();

    input::Init();
    if (app::kGame.window) app::kGame.window(app::g_window);
    std::string hdd = app::Instance() > 1 ? "hdd" + std::to_string(app::Instance()) : "hdd";
    if (!kernel::Boot(std::move(game.xbe), IMAGE_XBE, game.dir.string(), (app::ExeDir() / hdd).string())) {
        MessageBoxW(nullptr, L"The game failed to start, see the log in the logs folder.", app::kGame.short_title, MB_ICONERROR);
        return 1;
    }

    if (std::getenv("XBE_OPEN_SETTINGS")) app::ToggleOverlay();  // testing
    bool running = true;
    while (running) {
        SDL_Event e;
        if (!SDL_WaitEventTimeout(&e, 4)) continue;
        do {
            app::FeedImGui(e);
            input::HandleSdlEvent(e);
            switch (e.type) {
                case SDL_EVENT_QUIT:
                case SDL_EVENT_WINDOW_CLOSE_REQUESTED: running = false; break;
                case SDL_EVENT_WINDOW_FOCUS_GAINED: app::g_focused = true; break;
                case SDL_EVENT_WINDOW_FOCUS_LOST: app::g_focused = false; break;
                case SDL_EVENT_WINDOW_RESIZED:
                case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED: app::UpdatePixels(); break;
                case SDL_EVENT_KEY_DOWN:
                    if (e.key.repeat) break;
                    {
                        std::lock_guard lock(app::UiLock());
                        if (app::OverlayOpen() && app::OverlayKey(app::VirtualKey(e.key.scancode))) break;
                        if (e.key.scancode == SDL_SCANCODE_F1) app::ToggleOverlay();
                        else if (e.key.scancode == SDL_SCANCODE_F12) app::RequestScreenshot();
                    }
                    break;
                default:
                    if (e.type == app::g_apply_event) app::ApplyNow();
                    break;
            }
        } while (SDL_PollEvent(&e));
    }
    LOGR("the window closed");
    gpu::SaveMetalCache();
    logx::Flush();
    kernel::Shutdown(0);
    return 0;
}
