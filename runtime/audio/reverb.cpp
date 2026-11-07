// I3DL2 reverb on the host (reverb.h): what the APU's DSP image does with mix bin 10, following I3DL2's parameters
// rather than its code (the image's DSP code is not read).
#include "audio/reverb.h"

#include <algorithm>
#include <cmath>

#include "audio/audio.h"

namespace audio {

namespace {

constexpr float kPi = 3.14159265f;
float Db(int32_t mb) { return mb <= -10000 ? 0.0f : std::pow(10.0f, float(mb) / 2000.0f); }

// the pole of a one pole low pass that is 1 at DC and r (below 1) at hz
float LowpassPole(float r, float hz) {
    if (r >= 0.999f) return 0;
    r = std::max(r, 0.001f);
    const float c = std::cos(2 * kPi * std::min(hz, float(kRate) * 0.45f) / float(kRate)), r2 = r * r;
    const float b = 1 - r2 * c, q = 1 - r2;
    return std::clamp((b - std::sqrt(std::max(0.0f, b * b - q * q))) / q, 0.0f, 0.999f);
}

// the late reverb's lines at full density (frames at 48 kHz, 30 to 58 ms, no common factors), and the input diffusers
constexpr uint32_t kLineLen[8] = {1447, 1621, 1789, 1931, 2131, 2297, 2539, 2791};
constexpr uint32_t kDiffLen[4] = {142, 107, 379, 277};
// early reflections: after reflections_delay, ms apart, alternating left and right
constexpr float kTapMs[6] = {0, 4.3f, 8.9f, 13.7f, 19.1f, 25.3f};
constexpr float kTapGain[6] = {1, 0.85f, 0.74f, 0.64f, 0.55f, 0.46f};

}  // namespace

Reverb::Reverb() {
    pre_.assign(size_t(kRate) * 45 / 100, 0.0f);  // reflections_delay (0.3 s at most) and reverb_delay (0.1 s) and the taps
    for (int i = 0; i < kLines; i++) line_[i].assign(kLineLen[i] + 1, 0.0f);
    for (int i = 0; i < kDiffusers; i++) diff_[i].assign(kDiffLen[i], 0.0f);
    Set(I3dl2{});
}

void Reverb::Set(const I3dl2& p) {
    const float decay = std::clamp(p.decay, 0.1f, 20.0f), hf_ratio = std::clamp(p.decay_hf, 0.1f, 2.0f);
    const float hf = std::clamp(p.hf_reference, 20.0f, 20000.0f);
    room_ = Db(std::min(p.room, 0));
    room_damp_ = LowpassPole(Db(std::min(p.room_hf, 0)), hf);
    refl_gain_ = Db(std::min(p.reflections, 1000));
    refl_delay_ = uint32_t(std::clamp(p.reflections_delay, 0.0f, 0.3f) * kRate);
    late_delay_ = refl_delay_ + uint32_t(std::clamp(p.reverb_delay, 0.0f, 0.1f) * kRate);
    diff_coef_ = 0.7f * std::clamp(p.diffusion, 0.0f, 100.0f) / 100.0f;
    // density shortens the lines (closer echoes), 60% of their length at 0
    const float scale = 0.6f + 0.4f * std::clamp(p.density, 0.0f, 100.0f) / 100.0f;
    for (int i = 0; i < kLines; i++) {
        line_len_[i] = std::max<uint32_t>(64, uint32_t(float(kLineLen[i]) * scale));
        // 60 dB down in decay seconds, and in decay x hf_ratio seconds at hf_reference
        const float t = float(line_len_[i]) / float(kRate);
        line_gain_[i] = std::pow(10.0f, -3 * t / decay);
        const float hf_gain = std::pow(10.0f, -3 * t / (decay * hf_ratio));
        line_damp_[i] = LowpassPole(std::min(1.0f, hf_gain / line_gain_[i]), hf);
    }
    // The network (orthonormal mixing, input energy 1) holds 1 / (1 - g(w)^2) of its input's energy at each frequency,
    // g(w) a line's gain with its damping; each side hears a quarter of four lines. Averaged over the band (white
    // input), this makes each side's late tail carry the reverb level's energy.
    double held = 0;
    constexpr int kBands = 64;
    for (int b = 0; b < kBands; b++) {
        const double w = kPi * (b + 0.5) / kBands;
        for (int i = 0; i < kLines; i++) {
            const double d = line_damp_[i], h2 = (1 - d) * (1 - d) / (1 - 2 * d * std::cos(w) + d * d);
            held += 1 / (1 - double(line_gain_[i]) * line_gain_[i] * h2);
        }
    }
    held /= kBands * kLines;
    late_gain_ = Db(std::min(p.reverb, 2000)) * float(std::sqrt(8 / held));
}

void Reverb::Process(const float* in, float* out, uint32_t frames) {
    const uint32_t n = uint32_t(pre_.size());
    uint32_t taps[kTaps];
    for (int t = 0; t < kTaps; t++) taps[t] = refl_delay_ + uint32_t(kTapMs[t] * float(kRate) / 1000.0f);
    float tap_norm = 0;
    for (float g : kTapGain) tap_norm += g * g;
    tap_norm = 1 / std::sqrt(tap_norm / 2);  // each side's reflections at the reflections level
    for (uint32_t i = 0; i < frames; i++) {
        room_now_ += (room_ - room_now_) * 0.0005f;
        refl_now_ += (refl_gain_ - refl_now_) * 0.0005f;
        late_now_ += (late_gain_ - late_now_) * 0.0005f;
        // the room level and its high cut
        room_lp_ = in[i] * room_now_ * (1 - room_damp_) + room_lp_ * room_damp_;
        pre_[pre_at_] = room_lp_;
        auto past = [&](uint32_t d) { return pre_[(pre_at_ + n - std::min(d, n - 1)) % n]; };
        float l = 0, r = 0;
        for (int t = 0; t < kTaps; t++) (t & 1 ? r : l) += past(taps[t]) * kTapGain[t];
        l *= refl_now_ * tap_norm;
        r *= refl_now_ * tap_norm;
        // the late reverb: its input diffused, then the network
        float x = past(late_delay_);
        for (int d = 0; d < kDiffusers; d++) {
            float& z = diff_[d][diff_at_[d]];
            const float y = z - diff_coef_ * x;
            z = x + diff_coef_ * y;
            x = y;
            diff_at_[d] = (diff_at_[d] + 1) % kDiffLen[d];
        }
        float o[kLines], v[kLines];
        for (int k = 0; k < kLines; k++) {
            o[k] = line_[k][(line_at_[k] + line_[k].size() - line_len_[k]) % line_[k].size()];
            line_lp_[k] = o[k] * (1 - line_damp_[k]) + line_lp_[k] * line_damp_[k];
            v[k] = line_lp_[k] * line_gain_[k];
        }
        // an 8 point Hadamard mix, energy kept
        for (int h = 1; h < kLines; h <<= 1)
            for (int a = 0; a < kLines; a += h * 2)
                for (int b = a; b < a + h; b++) {
                    const float p = v[b], q = v[b + h];
                    v[b] = p + q;
                    v[b + h] = p - q;
                }
        for (int k = 0; k < kLines; k++) {
            line_[k][line_at_[k]] = v[k] * 0.35355339f + x * ((k & 1) ? -0.35355339f : 0.35355339f);
            line_at_[k] = (line_at_[k] + 1) % uint32_t(line_[k].size());
        }
        l += (o[0] - o[2] + o[4] - o[6]) * late_now_ * 0.5f;
        r += (o[1] - o[3] + o[5] - o[7]) * late_now_ * 0.5f;
        out[i * 2] += l;
        out[i * 2 + 1] += r;
        pre_at_ = (pre_at_ + 1) % n;
    }
}

}  // namespace audio
