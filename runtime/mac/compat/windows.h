#pragma once
// macOS: the Win32 names the portable runtime and game sources touch. Keys are Windows virtual key codes throughout
// (the settings file names them); GetAsyncKeyState reads SDL's keyboard state through them (mac/main.mm).
#include <cstdint>
#include <cwchar>
#include <string>

typedef long HRESULT;
typedef long LONG;
typedef unsigned long DWORD;
typedef void* HANDLE;
typedef void* HMODULE;
typedef void* HWND;
typedef void* HCURSOR;
typedef const wchar_t* PCWSTR;
typedef unsigned int UINT;
typedef unsigned char UINT8;
typedef unsigned long long UINT64;
#ifndef __OBJC__
typedef int BOOL;  // Objective-C has its own
#endif
#define WINAPI
#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif
#define MAX_PATH 1024
#define SUCCEEDED(hr) (HRESULT(hr) >= 0)
#define FAILED(hr) (HRESULT(hr) < 0)

struct RECT { LONG left, top, right, bottom; };
struct POINT { LONG x, y; };

inline HMODULE GetModuleHandleW(const wchar_t*) { return nullptr; }
inline void* GetProcAddress(HMODULE, const char*) { return nullptr; }
inline HANDLE GetCurrentThread() { return nullptr; }

enum : uint32_t {
    WM_ACTIVATE = 0x0006, WM_SETCURSOR = 0x0020, WM_INPUT = 0x00FF, WM_KEYDOWN = 0x0100, WM_SYSKEYDOWN = 0x0104,
    WM_LBUTTONUP = 0x0202,
};
enum { WA_INACTIVE = 0, HTCLIENT = 1 };
enum {
    VK_LBUTTON = 0x01, VK_RBUTTON = 0x02, VK_MBUTTON = 0x04, VK_XBUTTON1 = 0x05, VK_XBUTTON2 = 0x06, VK_BACK = 0x08,
    VK_TAB = 0x09, VK_RETURN = 0x0D, VK_SHIFT = 0x10, VK_CONTROL = 0x11, VK_MENU = 0x12, VK_ESCAPE = 0x1B,
    VK_SPACE = 0x20, VK_PRIOR = 0x21, VK_NEXT = 0x22, VK_END = 0x23, VK_HOME = 0x24, VK_LEFT = 0x25, VK_UP = 0x26,
    VK_RIGHT = 0x27, VK_DOWN = 0x28, VK_INSERT = 0x2D, VK_DELETE = 0x2E, VK_NUMPAD0 = 0x60, VK_NUMPAD1, VK_NUMPAD2,
    VK_NUMPAD3, VK_NUMPAD4, VK_NUMPAD5, VK_NUMPAD6, VK_NUMPAD7, VK_NUMPAD8, VK_NUMPAD9, VK_F1 = 0x70, VK_F2, VK_F3,
    VK_F4, VK_F5, VK_F6, VK_F7, VK_F8, VK_F9, VK_F10, VK_F11, VK_F12, VK_LSHIFT = 0xA0, VK_RSHIFT, VK_LCONTROL,
    VK_RCONTROL, VK_LMENU, VK_RMENU, VK_OEM_1 = 0xBA, VK_OEM_PLUS, VK_OEM_COMMA, VK_OEM_MINUS, VK_OEM_PERIOD, VK_OEM_2,
    VK_OEM_3, VK_OEM_4 = 0xDB, VK_OEM_5, VK_OEM_6, VK_OEM_7,
};
#define LOWORD(v) ((uint16_t)((uintptr_t)(v) & 0xFFFF))
#define MAKEINTRESOURCEW(i) ((const wchar_t*)(uintptr_t)(uint16_t)(i))
inline HCURSOR LoadCursorW(void*, const wchar_t*) { return nullptr; }
inline HCURSOR SetCursor(HCURSOR) { return nullptr; }

short GetAsyncKeyState(int vk);  // mac/main.mm
HWND GetForegroundWindow();      // the game window while it has the keyboard, else null

enum { MB_ICONERROR = 0x10, MB_ICONWARNING = 0x30, MB_YESNO = 4, MB_DEFBUTTON2 = 0x100, IDYES = 6, IDNO = 7, IDOK = 1 };
int MessageBoxW(HWND, const wchar_t* text, const wchar_t* title, unsigned type);  // mac/main.mm; a question answers yes
