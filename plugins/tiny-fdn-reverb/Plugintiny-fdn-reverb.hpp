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
        paramDampHz,       // 1500..12000 Hz low-pass in the loop
        paramMatrixMorph,  // 0..1: 0 Hadamard → 1 Householder (smooth blend)
        paramPing,         // momentary trigger for one-sample impulse
        paramEDTms,          // Early Decay Time (ms)
        paramRT60est,        // Estimated RT60 (s) from measured EDC
        paramDensity100ms,   // events/ms @100 ms
        paramDensity300ms,   // events/ms @300 ms
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
    // Near-commensurate 'Spread' to create obvious metallic modes
    static constexpr std::array<int,kN> kBaseSpread48 = {1200, 1800, 2400, 3000};
    // state
    double fLastSR;
    float  fRt60;   // seconds
    float  fMix;    // 0..1
    int    fMatrixType; // 0/1
    int    fDelaySet;   // 0/1
    float  fSize;       // 0.5..2.0
    float fDampHz      = 6000.0f;  // Hz
    float fMatrixMorph = 0.0f;     // 0..1
    int   fPing        = 0;        // momentary
    float mLP[kN] = {0,0,0,0};
    float fAppliedSize = 1.0f;
    int   fAppliedDelaySet = 0;
    int   fAppliedMatrixType = 0;

    // short mute after topology changes (samples)
    int   mMuteSamples = 0;

    // helper to clear state without clicks
    void  resetStateForTopologyChange() noexcept;

    // --- IR capture for metrics (2 s @ 48k) ---
    static constexpr int kIRMax = 96000;
    std::array<float, kIRMax> mIR{};
    int   irWrite = 0;
    bool  irCapturing = false;
    bool  irReady     = false;

    // metrics
    float mEDTms = 0.f;      // ms
    float mRT60s = 0.f;      // s
    float mDen100 = 0.f;     // events/ms
    float mDen300 = 0.f;     // events/ms
    // throttle UI updates to ~20 Hz
    uint32_t mSamplesSincePush = 0;

    // smoothers (10 ms)
    CParamSmooth fMixSmoothL;
    CParamSmooth fMixSmoothR;
    CParamSmooth fRt60Smooth;
    CParamSmooth fDampSmooth   {10.0f, 48000.0}; // updated on SR change
    CParamSmooth fMorphSmooth  {10.0f, 48000.0};

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
