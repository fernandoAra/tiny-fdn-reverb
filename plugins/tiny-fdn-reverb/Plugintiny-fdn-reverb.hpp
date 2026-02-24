/*
 * tiny-fdn-reverb audio effect based on DISTRHO Plugin Framework (DPF)
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */
#ifndef PLUGIN_TINY_FDN_REVERB_HPP
#define PLUGIN_TINY_FDN_REVERB_HPP

#include "DistrhoPlugin.hpp"
#include "CParamSmooth.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <vector>
#include <memory>

// [BOILERPLATE: DPF namespace macros]
START_NAMESPACE_DISTRHO

class PluginTinyFdnReverb : public Plugin {
public:
    enum Parameters {
        paramRt60 = 0,   // seconds
        paramMix,        // 0..1
        paramMatrixType, // 0=Hadamard, 1=Householder
        paramDelaySet,   // 0=Prime,    1=Spread
        paramSize,       // 0.5..2.0 scalar on delays
        paramDampHz,       // 1500..12000 Hz low-pass in the loop
        paramMatrixMorph,  // 0..1: 0 Hadamard → 1 Householder (smooth blend)
        paramPing,         // momentary trigger for one-sample impulse

        // --- NEW interactive params ---
        paramModDepth,     // 0..1 modulation amount (independent of matrix)
        paramDetune,       // 0..1 static AP detune amount
        paramMetalBoost,   // boolean: force worst-case metallic
        paramExciteNoise,  // boolean: 20 ms noise burst

        // --- Outputs to UI ---
        paramEDTms,          // Early Decay Time (ms)
        paramRT60est,        // Estimated RT60 (s) from measured EDC
        paramDensity100ms,   // events/ms @100 ms
        paramDensity300ms,   // events/ms @300 ms
        paramRinginess,      // 0..1 periodicity index (higher = more metallic)
        paramWetEnv,         // 0..1 RMS of wet output (for UI trace fallback)
        paramCount
    };

    static constexpr uint32_t kEnvTraceSize = 256u;

    PluginTinyFdnReverb();
    ~PluginTinyFdnReverb() override {}

    uint32_t getEnvTraceWriteIndex() const noexcept;
    float getEnvTraceValue(uint32_t sequenceIndex) const noexcept;

protected:
    // === BOILERPLATE BEGIN: DPF plugin interface hooks (metadata/params/programs) ===
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
    // === BOILERPLATE END ===

    // lifecycle / audio
    void activate() override;
    void run(const float** inputs, float** outputs, uint32_t frames) override;

private:
    // === Dal Santo-style homogeneous tiny FDN core ======================
    // - 4 delay lines (kN = 4)
    // - Feedback matrix A = U * Gamma, where U is unilossless (Hadamard/Householder)
    // - Homogeneous decay: per-line gains mGain[i] = gamma^{mLen[i]},
    //   with gamma derived from RT60 and sample rate (Dal Santo 2023/2025).
    static constexpr int kN        = 4;
    static constexpr int kMaxDelay = 96000; // 2s @ 48k

    // two base delay sets at 48k (samples)
    static constexpr std::array<int,kN> kBasePrime48  = {1499, 2377, 3217, 4421};
    // Near-commensurate 'Spread' to create obvious metallic modes
    static constexpr std::array<int,kN> kBaseSpread48 = {1200, 1800, 2400, 3000};

    // Core controls (smoothed)
    float  fLastSR      = 48000.f;
    float  fRt60        = 1.5f;
    float  fMix         = 0.25f;
    int    fMatrixType  = 0;
    int    fDelaySet    = 0;
    float  fSize        = 1.0f;
    float  fDampHz      = 6000.f;
    float  fMatrixMorph = 0.0f;
    int    fPing        = 0;

    // NEW interactive controls
    float  fModDepth    = 0.0f;     // 0..1
    float  fDetune      = 0.0f;     // 0..1
    int    fMetalBoost  = 0;        // boolean
    int    fExciteNoise = 0;        // boolean (momentary)

    // per-line state
    float  mLP[kN]   = {0,0,0,0};
    float  mAPx1[kN] = {0,0,0,0};   // x[n-1] for LFO AP
    float  mAPy1[kN] = {0,0,0,0};   // y[n-1] for LFO AP
    // NEW detune AP state (static)
    float  mAPd_x1[kN] = {0,0,0,0};
    float  mAPd_y1[kN] = {0,0,0,0};

    double mLFOph[kN] = {0,0,0,0};  // LFO phase per line

    // applied/snapshotted values (to detect changes)
    float  fAppliedSize = 1.0f;
    int    fAppliedDelaySet = 0;
    int    fAppliedMatrixType = 0;
    int    fAppliedMetalBoost = 0;

    // short mute after topology changes (samples)
    int    mMuteSamples = 0;

    // helper to clear state without clicks
    void   resetStateForTopologyChange() noexcept;

    // --- IR capture for metrics (2 s @ 48k) ---
    static constexpr int kIRMax = 96000;
    std::array<float, kIRMax> mIR{};
    int   irWrite = 0;
    bool  irCapturing = false;
    bool  irReady     = false;
    bool  irAnalyzed  = false; // avoid recomputing metrics

    // metrics
    float mEDTms = 0.f;      // ms
    float mRT60s = 0.f;      // s
    float mDen100 = 0.f;     // events/ms
    float mDen300 = 0.f;     // events/ms
    float mRinginess = 0.f;  // 0..1
    float mWetEnv = 0.f;     // 0..1 block RMS of wet output

    // throttle UI updates to ~20 Hz
    uint32_t mSamplesSincePush = 0;

    // Audio-thread writer, UI-thread reader (direct access path).
    std::array<std::atomic<uint32_t>, kEnvTraceSize> mEnvTraceBits{};
    std::atomic<uint32_t> mEnvTraceWrite{0};

    // smoothers (init with a default SR; reconfigured on SR change)
    CParamSmooth fMixSmoothL  {10.0f, 48000.0};
    CParamSmooth fMixSmoothR  {10.0f, 48000.0};
    CParamSmooth fRt60Smooth  {10.0f, 48000.0};
    CParamSmooth fDampSmooth  {10.0f, 48000.0};
    CParamSmooth fMorphSmooth {10.0f, 48000.0};
    CParamSmooth fModSmooth   {10.0f, 48000.0};
    CParamSmooth fDetuneSmooth{10.0f, 48000.0};

    // delay lines
    std::array<std::vector<float>, kN> mBuf;
    std::array<int, kN>   mLen;
    std::array<int, kN>   mIdx;
    std::array<float, kN> mGain;

    // --- ringiness meter buffer (for autocorr) ---
    static constexpr int kMeterMax = 4096;
    std::array<float, kMeterMax> mMeterBuf{};
    int   mMeterIdx = 0;
    bool  mMeterFilled = false;

    // noise burst generator
    int   mNoiseBurstLeft = 0;
    uint32_t mNoiseSeed = 222222u;
    inline float nextNoise() {
        // simple LCG; returns ~[-1,1]
        mNoiseSeed = 1664525u * mNoiseSeed + 1013904223u;
        return float(int32_t(mNoiseSeed)) / 2147483648.0f;
    }

    // helpers
    void selectBaseAndUpdateDelays(double sr) noexcept;
    void updateLineGainsFromRt60(double sr) noexcept;
    inline void hadamardMix4(const float in[kN], float out[kN]) const noexcept;
    inline void householderMix4(const float in[kN], float out[kN]) const noexcept;
    void computeRinginess(double sr) noexcept;
    static uint32_t floatToBits(float value) noexcept;
    static float bitsToFloat(uint32_t bits) noexcept;

    DISTRHO_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginTinyFdnReverb)
};

END_NAMESPACE_DISTRHO
#endif
