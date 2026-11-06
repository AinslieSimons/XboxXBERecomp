#pragma once
#include <cstdlib>
#include <windows.h>
enum D3D_PRIMITIVE_TOPOLOGY { D3D_PRIMITIVE_TOPOLOGY_POINTLIST = 1, D3D_PRIMITIVE_TOPOLOGY_LINELIST, D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP };
// the shared shader cache keeps D3D blobs; on macOS nothing fills it (Metal keeps compiled shaders itself)
struct ID3DBlob {
    void* p = nullptr;
    size_t n = 0;
    void* GetBufferPointer() { return p; }
    size_t GetBufferSize() const { return n; }
};
inline HRESULT D3DCreateBlob(size_t, ID3DBlob**) { return -1; }
