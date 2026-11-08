// HLSL to MSL by text: the generators' HLSL is a small, known subset. MSL is C++, so the source stays nearly as it is
// inside one struct: the constant buffers become its fields, helpers and entry points its member functions (seeing the
// constants as HLSL's globals do), and textures are passed to the entry point as arguments.
#include "gpu/msl.h"

#include <cstring>
#include <regex>

namespace gpu {

namespace {

const char* kPrelude = R"(#include <metal_stdlib>
using namespace metal;
#define HV(N) vec<float, N>
#define MIXED(F) \
    inline float F(float a, int b) { return F(a, float(b)); } inline float F(int a, float b) { return F(float(a), b); } \
    template <int N> inline HV(N) F(HV(N) a, float b) { return F(a, HV(N)(b)); } \
    template <int N> inline HV(N) F(float a, HV(N) b) { return F(HV(N)(a), b); } \
    template <int N> inline HV(N) F(HV(N) a, int b) { return F(a, HV(N)(float(b))); } \
    template <int N> inline HV(N) F(int a, HV(N) b) { return F(HV(N)(float(a)), b); }
MIXED(min) MIXED(max)
template <typename T, typename A, typename B> inline T hclamp(T v, A a, B b) { return clamp(v, T(a), T(b)); }
inline float lerp(float a, float b, float t) { return mix(a, b, t); }
template <int N> inline HV(N) lerp(HV(N) a, HV(N) b, float t) { return mix(a, b, HV(N)(t)); }
template <int N> inline HV(N) lerp(HV(N) a, HV(N) b, HV(N) t) { return mix(a, b, t); }
inline float4 mul(float4 v, float4x4 m) { return v * m; }
inline float4 mul(float4x4 m, float4 v) { return m * v; }
inline float3 mul(float3 v, float3x3 m) { return v * m; }
inline float3 mul(float3x3 m, float3 v) { return m * v; }
inline float4x4 mul(float4x4 a, float4x4 b) { return a * b; }
inline float frac(float v) { return fract(v); }
template <int N> inline HV(N) frac(HV(N) v) { return fract(v); }
inline float rcp(float v) { return 1.0 / v; }
struct Tex2D {
    texture2d<float> t;
    float4 Sample(sampler s, float2 uv) const { return t.sample(s, uv); }
    float4 SampleLevel(sampler s, float2 uv, float l) const { return t.sample(s, uv, level(l)); }
    float4 SampleBias(sampler s, float2 uv, float b) const { return t.sample(s, uv, bias(b)); }
    float2 Dim() const { return float2(t.get_width(), t.get_height()); }
};
struct Tex3D {
    texture3d<float> t;
    float4 Sample(sampler s, float3 uv) const { return t.sample(s, uv); }
    float4 SampleLevel(sampler s, float3 uv, float l) const { return t.sample(s, uv, level(l)); }
};
struct TexCube {
    texturecube<float> t;
    float4 Sample(sampler s, float3 uv) const { return t.sample(s, uv); }
    float4 SampleLevel(sampler s, float3 uv, float l) const { return t.sample(s, uv, level(l)); }
};
struct TexDepth {
    depth2d<float> t;
    float SampleCmpLevelZero(sampler s, float2 uv, float z) const { return t.sample_compare(s, uv, z, level(0)); }
    float SampleLevel(sampler s, float2 uv, float l) const { return t.sample(s, uv, level(l)); }
};
)";

void ReplaceAll(std::string& s, const std::string& from, const std::string& to) {
    for (size_t p = 0; (p = s.find(from, p)) != std::string::npos; p += to.size()) s.replace(p, from.size(), to);
}

// dot(a, b).xxxx: MSL's scalars have no swizzles
void FixScalarSwizzles(std::string& s) {
    for (size_t p = 0; (p = s.find(").xxxx", p)) != std::string::npos;) {
        int depth = 0;
        size_t open = p;
        for (;; open--) {
            if (s[open] == ')') depth++;
            if (s[open] == '(' && --depth == 0) break;
            if (open == 0) break;
        }
        if (open < 3 || s.compare(open - 3, 3, "dot") != 0) {
            p += 6;
            continue;
        }
        s.replace(p, 6, "))");
        s.insert(open - 3, "float4(");
        p += 7;
    }
}

// what MSL takes differently, as the generators write it
void FixKnown(std::string& s) {
    FixScalarSwizzles(s);
    static const std::pair<const char*, const char*> kFixes[] = {
        {"return (a == 0 || b == 0) ? (float4)0 : a * b;", "return select(a * b, float4(0), a == 0 || b == 0);"},
        {"return (1.0 / v).xxxx;", "return float4(1.0 / v);"},
        {"return magclamp(1.0 / v).xxxx;", "return float4(magclamp(1.0 / v));"},
        {"return v == 0 ? (1.#INF).xxxx : (rsqrt(abs(v))).xxxx;", "return float4(v == 0 ? INFINITY : rsqrt(abs(v)));"},
        {"1.#INF", "INFINITY"},
        {"return (b - (b >= 128 ? hi : lo)) / d;", "return (b - select(float3(lo), float3(hi), b >= 128)) / d;"},
        {"return t + (d.x ? d.x / g.x * ddx(t) : 0) + (d.y ? d.y / g.y * ddy(t) : 0);",
         "return t + (d.x != 0 ? d.x / g.x * ddx(t) : float4(0)) + (d.y != 0 ? d.y / g.y * ddy(t) : float4(0));"},
        {"t.SampleCmpLevelZero(s, uv, clamp(z, 0, mx) / mx).xxxx", "float4(t.SampleCmpLevelZero(s, uv, clamp(z, 0.0, mx) / mx))"},
        {"img.GetDimensions(size.x, size.y);", "size = img.Dim();"},
        {"Texture2D<float>", "TexDepth"},
        {"Texture2DMS<float>", "TexDepthMs"},
        {"Texture2D", "Tex2D"},
        {"TextureCube", "TexCube"},
        {"Texture3D", "Tex3D"},
        {"SamplerComparisonState", "sampler"},
        {"SamplerState", "sampler"},
        {"struct O { float4 p : SV_Position; float2 uv : TEXCOORD0; };", "struct O { float4 p [[position]]; float2 uv; };"},
        {"struct O { float4 p : SV_Position; };", "struct O { float4 p [[position]]; };"},
        {"precise ", ""},
        {"static const ", "const "},
        {"noperspective float4 e", "float4 e"},
    };
    for (auto& [from, to] : kFixes) ReplaceAll(s, from, to);
    static const std::regex kWords[][1] = {{std::regex(R"(\bddx\()")}, {std::regex(R"(\bddy\()")},
                                           {std::regex(R"(\bdiscard\b)")}, {std::regex(R"(\bclamp\()")}};
    static const char* kTo[] = {"dfdx(", "dfdy(", "discard_fragment()", "hclamp("};
    for (size_t i = 0; i < 4; i++) s = std::regex_replace(s, kWords[i][0], kTo[i]);
}

struct Global {
    std::string type, name;
    int slot;
};

struct Param {
    std::string type, name, semantic;
    bool out = false;
};

// " : SEMANTIC" out of a struct field: [[position]], [[attribute(n)]], or nothing (matched by name between stages)
std::string FieldAttr(const std::string& field, const std::string& sem) {
    if (sem == "SV_Position") return "[[position, invariant]]";
    if (sem.rfind("ATTR", 0) == 0) return "[[attribute(" + sem.substr(4) + ")]]";
    if (field.find(" e") != std::string::npos && sem == "TEXCOORD5") return "[[center_no_perspective]]";
    return "";
}

// the source inside the struct: cbuffers opened into fields, register'd globals taken out, semantics on fields made
// attributes, top level functions marked constant (when the struct lives in a constant buffer)
std::string Body(const std::string& hlsl, std::vector<Global>& globals, std::string& fields, bool constant_this) {
    static const std::regex kGlobal(R"(^\s*(Tex\w+|sampler)\s+(\w+)\s*:\s*register\([ts](\d+)\);)");
    static const std::regex kCbufferOne(R"(^\s*cbuffer\s+\w+\s*:\s*register\(b\d+\)\s*\{(.*)\};\s*$)");
    static const std::regex kCbufferOpen(R"(^\s*cbuffer\s+\w+\s*:\s*register\(b\d+\)\s*\{\s*$)");
    static const std::regex kField(R"(^(\s*[\w ]+\s+\w+(?:\[\d+\])?)\s*:\s*(\w+)\s*;(.*)$)");
    static const std::regex kFunc(R"(^([A-Za-z_][\w:]*)\s+(\w+)\((.*)\)\s*(:\s*\w+\s*)?\{(.*)$)");
    std::string out;
    size_t at = 0;
    int depth = 0;
    bool in_cbuffer = false;
    while (at < hlsl.size()) {
        size_t nl = hlsl.find('\n', at);
        std::string line = hlsl.substr(at, nl == std::string::npos ? std::string::npos : nl - at);
        at = nl == std::string::npos ? hlsl.size() : nl + 1;
        std::smatch m;
        if (depth == 0 && !in_cbuffer) {
            if (std::regex_search(line, m, kGlobal)) {
                globals.push_back({m[1], m[2], std::stoi(m[3])});
                continue;
            }
            if (std::regex_match(line, m, kCbufferOne)) {
                fields += std::string(m[1]) + "\n";
                continue;
            }
            if (std::regex_match(line, kCbufferOpen)) {
                in_cbuffer = true;
                continue;
            }
        }
        if (in_cbuffer) {
            if (line.rfind("};", 0) == 0) in_cbuffer = false;
            else fields += line + "\n";
            continue;
        }
        if (depth == 0 && std::regex_match(line, m, kFunc)) {
            // semantics in parameters and after the list are read off by the entry point code; helpers have none
            std::string head = std::string(m[1]) + " " + std::string(m[2]) + "(" + std::string(m[3]) + ")";
            if (m[4].matched) head += " " + std::string(m[4]);
            head += constant_this ? " constant {" : " {";
            line = head + std::string(m[5]);
        } else if (depth > 0 && std::regex_match(line, m, kField) && line.find('?') == std::string::npos) {
            std::string attr = FieldAttr(m[1], m[2]);
            line = std::string(m[1]) + (attr.empty() ? "" : " " + attr) + ";" + std::string(m[3]);
        }
        for (size_t i = 0; i < line.size(); i++) {
            if (line[i] == '/' && i + 1 < line.size() && line[i + 1] == '/') break;
            if (line[i] == '{') depth++;
            if (line[i] == '}') depth--;
        }
        out += line + "\n";
    }
    return out;
}

std::vector<Param> SplitParams(const std::string& list) {
    std::vector<Param> params;
    size_t at = 0;
    while (at < list.size()) {
        size_t comma = list.find(',', at);
        std::string one = list.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        at = comma == std::string::npos ? list.size() : comma + 1;
        Param p;
        if (size_t c = one.find(':'); c != std::string::npos) {
            p.semantic = one.substr(c + 1);
            p.semantic.erase(0, p.semantic.find_first_not_of(' '));
            p.semantic.erase(p.semantic.find_last_not_of(' ') + 1);
            one.resize(c);
        }
        one.erase(one.find_last_not_of(' ') + 1);
        one.erase(0, one.find_first_not_of(' '));
        if (one.rfind("out ", 0) == 0) {
            p.out = true;
            one.erase(0, 4);
        }
        size_t sp = one.find_last_of(' ');
        if (sp == std::string::npos) continue;
        p.type = one.substr(0, sp);
        p.name = one.substr(sp + 1);
        params.push_back(p);
    }
    return params;
}

struct EntrySig {
    std::string ret, ret_semantic;
    std::vector<Param> params;
};

// finds the entry's definition and rewrites its header: semantics gone, outs references, textures as arguments
// (extra_args), and reads its signature
bool RewriteEntry(std::string& body, const std::string& entry, const std::string& extra_args, EntrySig& sig) {
    const std::regex re("(^|\\n)([A-Za-z_]\\w*) " + entry + "\\(([^\\n]*?)\\)\\s*(:\\s*(\\w+)\\s*)?(constant )?\\{");
    std::smatch m;
    if (!std::regex_search(body, m, re)) return false;
    sig.ret = m[2];
    sig.ret_semantic = m[5];
    sig.params = SplitParams(m[3]);
    std::string list;
    for (const Param& p : sig.params) {
        if (!list.empty()) list += ", ";
        list += (p.out ? "thread " + p.type + "& " : p.type + " ") + p.name;
    }
    if (!extra_args.empty()) list += (list.empty() ? "" : ", ") + extra_args;
    std::string head = std::string(m[1]) + sig.ret + " " + entry + "(" + list + ") " + std::string(m[6]) + "{";
    body.replace(size_t(m.position(0)), size_t(m.length(0)), head);
    return true;
}

std::string HandleType(const std::string& t) {
    if (t == "Tex2D") return "texture2d<float>";
    if (t == "Tex3D") return "texture3d<float>";
    if (t == "TexCube") return "texturecube<float>";
    if (t == "TexDepth") return "depth2d<float>";
    if (t == "TexDepthMs") return "depth2d_ms<float>";
    return t;
}

// a point drawn as a quad, from the geometry shader of gpu/shaders.cpp PointHlsl
const char* kExpandPoint = R"(
VOut expand_point(VOut v, int i) constant {
    float2 half_pixel = v.pos.w / surf.xy;
    float2 back = (0.5 - 1e-3) * 2 * half_pixel / surf.w;
    VOut o = v;
    if (pts.x > 0) {
        float size = hclamp(v.psize > 0 ? v.psize : pts.w, pts.y, max(pts.y, pts.z));
        float2 half_size = size * v.pos.w / surf.xy;
        o.pos.x += (i & 1) ? half_size.x : -half_size.x;
        o.pos.y += (i & 2) ? -half_size.y : half_size.y;
        float4 tc = float4(i & 1, (i & 2) ? 1 : 0, 0, 1);
        o.tc0 = tc;
        o.tc1 = tc;
        o.tc2 = tc;
        o.tc3 = tc;
        return o;
    }
    o.pos.x += (i & 1) ? half_pixel.x : -half_pixel.x - back.x;
    o.pos.y += (i & 2) ? -half_pixel.y : half_pixel.y + back.y;
    return o;
}
)";

// the input struct has any attributes
bool HasAttributes(const std::string& body, const std::string& type) {
    size_t p = body.find("struct " + type + " {");
    if (p == std::string::npos) return false;
    size_t end = body.find("};", p);
    return body.substr(p, end - p).find("[[attribute(") != std::string::npos;
}

}  // namespace

std::string VertexMsl(const std::string& hlsl) {
    std::string src = hlsl;
    FixKnown(src);
    std::vector<Global> globals;
    std::string fields;
    std::string body = Body(src, globals, fields, true);
    EntrySig sig;
    RewriteEntry(body, "main", "", sig);
    const std::string in = sig.params.empty() ? "VIn" : sig.params[0].type;
    const bool attrs = HasAttributes(body, in);
    // no inputs at all: the vertex id, the one field of an input struct without attributes
    const bool has_id = !attrs && body.find("uint id;") != std::string::npos;
    std::string out = kPrelude;
    out += "struct Shader {\n" + fields + body + kExpandPoint + "};\n";
    std::string args = "constant Shader& S [[buffer(0)]]";
    std::string arg_in = attrs ? "Shader::" + in + " i [[stage_in]], " : "";
    std::string make = attrs ? "" : "Shader::" + in + " i;" + std::string(has_id ? " i.id = vid;" : "");
    out += "vertex Shader::VOut vmain(" + arg_in + "uint vid [[vertex_id]], " + args + ") { " + make + " return S.main(i); }\n";
    out += "vertex Shader::VOut vpoint(" + arg_in + "uint vid [[vertex_id]], " + args + ") { " + make +
           " return S.expand_point(S.main(i), int(vid)); }\n";
    return out;
}

std::string PixelMsl(const std::string& hlsl) {
    std::string src = hlsl;
    FixKnown(src);
    std::vector<Global> globals;
    std::string fields;
    std::string body = Body(src, globals, fields, true);
    std::string extra, handles, pass;
    bool traced = false;
    for (const Global& g : globals) {
        if (g.type == "TexTrace") {  // kTexTraced: the scene, its triangles' shading and their textures (from slot 8)
            traced = true;
            const std::string as = "raytracing::primitive_acceleration_structure " + g.name, colour = "const device uint* " + g.name + "_colour",
                              tex = "array<texture2d<float>, 16> " + g.name + "_tex";
            extra += (extra.empty() ? "" : ", ") + as + ", " + colour + ", " + tex;
            handles += ", " + as + " [[buffer(1)]], " + colour + " [[buffer(2)]], " + tex + " [[texture(8)]]";
            pass += ", " + g.name + ", " + g.name + "_colour, " + g.name + "_tex";
            continue;
        }
        const bool smp = g.type == "sampler";
        extra += (extra.empty() ? "" : ", ") + g.type + " " + g.name;
        handles += ", " + HandleType(g.type) + " " + g.name + (smp ? " [[sampler(" : " [[texture(") + std::to_string(g.slot) + ")]]";
        pass += ", " + (smp ? g.name : g.type + "{" + g.name + "}");
    }
    EntrySig sig;
    RewriteEntry(body, "main", extra, sig);
    std::string out = std::string(traced ? "#include <metal_raytracing>\n" : "") + kPrelude;
    out += "struct Shader {\n" + fields + body + "};\n";
    std::string fout = "struct FOut { float4 colour [[color(0)]];";
    std::string call = "S.main(i, front";
    std::string decl = "fragment FOut fmain(Shader::VOut i [[stage_in]], bool front [[front_facing]], constant Shader& S [[buffer(0)]]" + handles + ") { FOut o; o.colour = ";
    // the entry's parameters after the first two, in order: outs land in the result. Textures follow them
    std::string outs;
    for (size_t k = 2; k < sig.params.size(); k++) {
        const Param& p = sig.params[k];
        if (p.semantic == "SV_Depth") fout += " float depth [[depth(any)]];", outs += ", o.depth";
        if (p.semantic == "SV_Coverage") fout += " uint cov [[sample_mask]];", outs += ", o.cov";
    }
    fout += " };\n";
    out += fout + decl + call + outs + pass + "); return o; }\n";
    return out;
}

std::string PassMsl(const std::string& hlsl, const std::vector<std::pair<std::string, bool>>& entries) {
    std::string src = hlsl;
    FixKnown(src);
    std::vector<Global> globals;
    std::string fields;
    std::string body = Body(src, globals, fields, false);
    std::string members;
    for (const Global& g : globals) members += g.type + " " + g.name + ";\n";
    // constants are separate from the members with handles: a value of the struct is made in each function
    std::string out = kPrelude;
    if (!fields.empty()) out += "struct Consts {\n" + fields + "};\n";
    out += "struct TexDepthMs { depth2d_ms<float> t; float Load(int2 p, int s) const { return t.read(uint2(p), s); } };\n";
    std::string field_copy;
    {
        static const std::regex kName(R"(float4\s+(\w+))");
        for (auto it = std::sregex_iterator(fields.begin(), fields.end(), kName); it != std::sregex_iterator(); ++it)
            field_copy += " S." + std::string((*it)[1]) + " = K." + std::string((*it)[1]) + ";";
    }
    std::vector<EntrySig> sigs;
    for (auto& [entry, vertex] : entries) {
        sigs.emplace_back();
        RewriteEntry(body, entry, "", sigs.back());
    }
    // other entry points of the source, not asked for: plain functions
    static const std::regex kSemantic(R"(\s*:\s*SV_\w+)");
    body = std::regex_replace(body, kSemantic, "");
    out += "struct Shader {\n" + fields + members + body + "};\n";
    size_t entry_index = 0;
    for (auto& [entry, vertex] : entries) {
        const EntrySig& sig = sigs[entry_index++];
        std::string constants = fields.empty() ? "" : "constant Consts& K [[buffer(0)]]";
        if (vertex) {
            std::string list = constants, call;
            for (const Param& p : sig.params) {
                list += (list.empty() ? "" : ", ") + p.type + " " + p.name + " [[vertex_id]]";
                call += (call.empty() ? "" : ", ") + p.name;
            }
            std::string ret = sig.ret == "float4" ? "float4" : "Shader::" + sig.ret;
            std::string attr = sig.ret == "float4" ? " [[position]]" : "";
            if (!attr.empty()) out += "struct VP_" + entry + " { float4 p [[position]]; };\n", ret = "VP_" + entry;
            out += "vertex " + ret + " " + entry + "(" + list + ") { Shader S;" + (fields.empty() ? "" : field_copy) +
                   (attr.empty() ? " return S." + entry + "(" + call + "); }\n"
                                 : " " + ret + " o; o.p = S." + entry + "(" + call + "); return o; }\n");
        } else {
            std::string list = constants;
            std::string init;
            for (const Global& g : globals) {
                const bool smp = g.type == "sampler";
                list += (list.empty() ? "" : ", ") + HandleType(g.type) + " " + g.name + (smp ? " [[sampler(" : " [[texture(") + std::to_string(g.slot) + ")]]";
                init += " S." + g.name + " = " + (smp ? g.name : g.type + "{" + g.name + "}") + ";";
            }
            std::string call;
            for (const Param& p : sig.params) {
                const bool pos = p.semantic == "SV_Position";
                std::string t = p.type == "float4" ? "float4" : "Shader::" + p.type;
                list += (list.empty() ? "" : ", ") + t + " " + p.name + (pos ? " [[position]]" : " [[stage_in]]");
                call += (call.empty() ? "" : ", ") + p.name;
            }
            bool depth = sig.ret_semantic == "SV_Depth";
            std::string res = entry + "_out";
            out += "struct " + res + " { " + sig.ret + " v " + (depth ? "[[depth(any)]]" : "[[color(0)]]") + "; };\n";
            out += "fragment " + res + " " + entry + "(" + list + ") { Shader S;" + (fields.empty() ? "" : field_copy) + init + " " + res +
                   " o; o.v = S." + entry + "(" + call + "); return o; }\n";
        }
    }
    return out;
}

}  // namespace gpu
