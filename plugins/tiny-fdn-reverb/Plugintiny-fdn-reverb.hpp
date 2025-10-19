/*
 * tiny-fdn-reverb audio effect based on DISTRHO Plugin Framework (DPF)
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#ifndef PLUGIN_TINY_FDN_REVERB_HPP
#define PLUGIN_TINY_FDN_REVERB_HPP

#include "DistrhoPlugin.hpp"
#include "CParamSmooth.hpp"
#include <array>
#include <memory>
#include <vector>

START_NAMESPACE_DISTRHO

class PluginTinyFdnReverb : public Plugin {
public:
    // ---- parameters ----
    enum Parameters {
        paramRt60 = 0,   // seconds
        paramMix,        // 0..1
        paramCount
    };

    PluginTinyFdnReverb();
    ~PluginTinyFdnReverb() override {}

protected:
    // ---- required metadata ----
    const char* getLabel()   const override;
    const char* getMaker()   const override;
    const char* getLicense() const override;
    uint32_t    getVersion() const override;
    int64_t     getUniqueId() const override;

    // ---- params ----
    void  initParameter(uint32_t index, Parameter& parameter) override;
    float getParameterValue(uint32_t index) const override;
    void  setParameterValue(uint32_t index, float value) override;

    // ---- programs (optional) ----
    void initProgramName(uint32_t index, String& programName) override;
    void loadProgram(uint32_t index) override;

    // ---- lifecycle / processing ----
    void activate() override; // clear buffers
    void run(const float** inputs, float** outputs, uint32_t frames) override;

private:
    // ====== FDN-4 core (fixed for Day-1) ======
    static constexpr int kN = 4;
    static constexpr int kMaxDelay = 96000; // 2s @ 48k (ample headroom)

    // base delay lengths at 48kHz, in samples (≈31, 49, 67, 92 ms)
    static constexpr std::array<int, kN> kBase48 = {1499, 2377, 3217, 4421};

    // state
    double fLastSR;
    float  fRt60;                 // seconds
    float  fMix;                  // 0..1
    CParamSmooth fMixSmoothL;     // 10 ms smoothing
    CParamSmooth fMixSmoothR;

    // delay lines
    std::array<std::vector<float>, kN> mBuf;  // each size = kMaxDelay (allocated once)
    std::array<int, kN> mLen;                 // actual lengths in samples (scaled by SR)
    std::array<int, kN> mIdx;                 // write indices
    std::array<float, kN> mGain;              // per-line loss from RT60

    // helpers
    void updateDelaysForSR(double sr) noexcept;
    void updateLineGainsFromRt60(double sr) noexcept;

    DISTRHO_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginTinyFdnReverb)
};

END_NAMESPACE_DISTRHO

#endif // PLUGIN_TINY_FDN_REVERB_HPP