// Thin C ABI around minacalc-sys v515 (SSR @ score_goal).
#include "API.h"

#include <cstdint>
#include <exception>
#include <vector>

extern "C" {

int audio2map_minacalc_version(void) { return calc_version(); }

int audio2map_minacalc_ssr(
    const std::uint32_t* masks,
    const float* times,
    std::size_t n,
    float music_rate,
    float score_goal,
    unsigned int keycount,
    float* out) {
    if (out == nullptr) {
        return 1;
    }
    if (n <= 1) {
        for (int i = 0; i < 8; ++i) {
            out[i] = 0.0f;
        }
        return 0;
    }
    if (masks == nullptr || times == nullptr) {
        return 2;
    }
    CalcHandle* calc = nullptr;
    try {
        calc = create_calc();
        if (calc == nullptr) {
            return 3;
        }
        std::vector<NoteInfo> rows(n);
        for (std::size_t i = 0; i < n; ++i) {
            rows[i].notes = masks[i];
            rows[i].rowTime = times[i];
        }
        Ssr s = calc_at_rate(
            calc, rows.data(), n, music_rate, score_goal, keycount, CalcMode::SSR);
        destroy_calc(calc);
        calc = nullptr;
        out[0] = s.overall;
        out[1] = s.stream;
        out[2] = s.jumpstream;
        out[3] = s.handstream;
        out[4] = s.stamina;
        out[5] = s.jackspeed;
        out[6] = s.chordjack;
        out[7] = s.technical;
        return 0;
    } catch (...) {
        if (calc != nullptr) {
            destroy_calc(calc);
        }
        return 4;
    }
}

}
