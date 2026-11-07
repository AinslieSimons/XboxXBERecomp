#pragma once
#include <cstdint>
#include <vector>

namespace audio {

// DSI3DL2LISTENER: an I3DL2 environment, as a title gives IDirectSound_SetI3DL2Listener (levels in mB, times in s)
struct I3dl2 {
    int32_t room = -10000, room_hf = 0;
    float rolloff = 0, decay = 1.49f, decay_hf = 0.83f;
    int32_t reflections = -2602;
    float reflections_delay = 0.007f;
    int32_t reverb = 200;
    float reverb_delay = 0.011f, diffusion = 100, density = 100, hf_reference = 5000;
};

// The APU's I3DL2 reverb (an effect in the title's DSP image, fed by mix bin 10) done on the host: the room level and
// its high cut, early reflections after reflections_delay, and late reverb after reverb_delay more from a feedback delay
// network whose lines decay in decay seconds (decay x decay_hf at hf_reference). Mono in, stereo out, at kRate.
class Reverb {
public:
    Reverb();
    void Set(const I3dl2& p);
    // adds the reverb of in (frames samples) to out (interleaved stereo)
    void Process(const float* in, float* out, uint32_t frames);

private:
    static constexpr int kLines = 8, kTaps = 6, kDiffusers = 4;
    std::vector<float> pre_;  // the input, for the reflections' and the late reverb's delays
    uint32_t pre_at_ = 0;
    std::vector<float> line_[kLines];
    uint32_t line_at_[kLines] = {}, line_len_[kLines] = {};
    float line_damp_[kLines] = {}, line_lp_[kLines] = {}, line_gain_[kLines] = {};
    std::vector<float> diff_[kDiffusers];
    uint32_t diff_at_[kDiffusers] = {};
    float diff_coef_ = 0.6f;
    float room_ = 0, room_lp_ = 0, room_damp_ = 0;
    float refl_gain_ = 0, late_gain_ = 0;
    uint32_t refl_delay_ = 0, late_delay_ = 0;
    // the levels glide to new settings, as the title blends between presets
    float room_now_ = 0, refl_now_ = 0, late_now_ = 0;
};

}  // namespace audio
