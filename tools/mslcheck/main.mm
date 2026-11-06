// mslcheck: every .hlsl in a folder (the harness's AUDIT dump) translated and compiled by Metal; prints what fails
#import <Metal/Metal.h>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include "gpu/msl.h"
namespace gpu { extern const char* kBlitHlsl; extern const char* kDepthHlsl; extern const char* kDepthResolveHlsl; }
int main(int argc, char** argv) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    MTLCompileOptions* opt = [MTLCompileOptions new];
    opt.mathMode = MTLMathModeSafe;
    opt.preserveInvariance = YES;
    int ok = 0, bad = 0, shown = 0;
    auto check = [&](const std::string& name, const std::string& src) {
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:opt error:&err];
        if (lib) { ok++; return; }
        bad++;
        if (shown++ < 4) {
            printf("=== %s\n%s\n", name.c_str(), err.localizedDescription.UTF8String);
            if (argc > 2) std::ofstream(argv[2] + ("/" + name + ".metal")) << src;
        }
    };
    for (auto& e : std::filesystem::directory_iterator(argv[1])) {
        std::ifstream f(e.path());
        std::string s((std::istreambuf_iterator<char>(f)), {});
        std::string n = e.path().filename().string();
        check(n, n.rfind("vs", 0) == 0 ? gpu::VertexMsl(s) : gpu::PixelMsl(s));
    }
    printf("generated: %d ok, %d failed\n", ok, bad);
    return bad != 0;
}
