/*
 * tiny-fdn-reverb audio effect based on DISTRHO Plugin Framework (DPF)
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */
#ifndef PLUGIN_TINY_FDN_REVERB_HPP
#define PLUGIN_TINY_FDN_REVERB_HPP

#include "DistrhoPlugin.hpp"
#include "CParamSmooth.hpp"
#include <array>
#include <vector>
#include <memory>

START_NAMESPACE_DISTRHO

class PluginTinyFdnReverb : public Plugin {
public:
    enum Parameters {
        paramRt60 = 0,   // seconds
        paramMix,        // 0..1
        paramMatrixType, // 0=Hadamard, 1=Householder
        paramDelaySet,   // 0=Prime,    1=Spread
        paramSize,       // 0.5..2.0 scalar on delays (optional but handy)
        paramCount
    };

    PluginTinyFdnReverb();
    ~PluginTinyFdnReverb() override {}

protected:
    // metadata
    const char* getLabel()   const override;
    const char* getMaker()   const override;
    const char* getLicense() const override;
    uint32_t    getVersion() const override;
    int64_t     getUniqueId() const override;

    // params
    void  initParameter(uint32_t index, Parameter& parameter) override;
    float getParameterValue(uint32_t index) const override;
    void  setParameterValue(uint32_t index, float value) override;

    // programs
    void initProgramName(uint32_t index, String& programName) override;
    void loadProgram(uint32_t index) override;

    // lifecycle / audio
    void activate() override;
    void run(const float** inputs, float** outputs, uint32_t frames) override;

private:
    // ===== core =====
    static constexpr int kN = 4;
    static constexpr int kMaxDelay = 96000; // 2s @ 48k

    // two base delay sets at 48k (samples)
    static constexpr std::array<int,kN> kBasePrime48  = {1499, 2377, 3217, 4421};
    static constexpr std::array<int,kN> kBaseSpread48 = {1103, 1733, 2549, 3917};

    // state
    double fLastSR;
    float  fRt60;   // seconds
    float  fMix;    // 0..1
    int    fMatrixType; // 0/1
    int    fDelaySet;   // 0/1
    float  fSize;       // 0.5..2.0

    // smoothers (10 ms)
    CParamSmooth fMixSmoothL;
    CParamSmooth fMixSmoothR;
    CParamSmooth fRt60Smooth;

    // delay lines
    std::array<std::vector<float>, kN> mBuf;
    std::array<int, kN> mLen;
    std::array<int, kN> mIdx;
    std::array<float, kN> mGain;

    // helpers
    void selectBaseAndUpdateDelays(double sr) noexcept;
    void updateLineGainsFromRt60(double sr) noexcept;
    inline void hadamardMix4(const float in[kN], float out[kN]) const noexcept;
    inline void householderMix4(const float in[kN], float out[kN]) const noexcept;

    DISTRHO_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginTinyFdnReverb)
};

END_NAMESPACE_DISTRHO
#endif
