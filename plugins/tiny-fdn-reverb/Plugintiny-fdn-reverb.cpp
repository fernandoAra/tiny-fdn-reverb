#include "Plugintiny-fdn-reverb.hpp"

#include <cmath>
#include <algorithm>
#include <array>
#include <numeric>
#include <cstdio>
#include <cstring>
#include <mutex>

// Boilerplate: very lightweight logger to /tmp/tfdn.log
// (generic helper, not core DSP logic)

// Uncomment to enable heavy logging (NOT RT-safe in production!)
#define TFDN_ENABLE_LOG 1

#if defined(TFDN_ENABLE_LOG)
static FILE* gTFDNLog = nullptr;
static std::once_flag gTFDNOnce;

static void tfdn_open_log() {
    gTFDNLog = std::fopen("/tmp/tfdn.log", "a");
}

#define DBG(...) do { \
    std::call_once(gTFDNOnce, tfdn_open_log); \
    if (gTFDNLog) { \
        std::fprintf(gTFDNLog, __VA_ARGS__); \
        std::fprintf(gTFDNLog, "\n"); \
        std::fflush(gTFDNLog); \
    } \
} while(0)
#else
#define DBG(...) do {} while(0)
#endif
// --------------------------------------------------------------

// [BOILERPLATE: DPF namespace macros]
START_NAMESPACE_DISTRHO

// Avoid relying on M_PI
static constexpr double kPI = 3.14159265358979323846;

struct Preset {
    const char* name;
    float params[PluginTinyFdnReverb::paramCount];
};

static const Preset kPresets[] = {
    //    RT60  Mix  Mat  Delay  Size  Damp  Morph  Ping | Mod  Det  Metal Burst | (outputs auto)
    { "Default",
      { 2.80f, 1.00f, 0.0f, 0.0f, 1.00f, 6000.0f, 0.0f, 0.0f,
        0.0f,  0.00f, 0.0f, 0.0f } },

    { "House+Spread",
      { 1.80f, 0.70f, 1.0f, 1.0f, 1.10f, 5000.0f, 0.0f, 0.0f,
        0.20f, 0.10f, 0.0f, 0.0f } },
};

static const uint32_t kPresetCount = sizeof(kPresets)/sizeof(kPresets[0]);

static inline void dump_state_lengths(const char* tag,
                                      int matrixType, int delaySet,
                                      const std::array<int, 4>& L)
{
    DBG("[%s] L=[%d %d %d %d] matrix=%s delay=%s",
        tag, L[0], L[1], L[2], L[3],
        matrixType ? "Householder" : "Hadamard",
        delaySet   ? "Spread"      : "Prime");
#if !defined(TFDN_ENABLE_LOG)
    (void)tag; (void)matrixType; (void)delaySet; (void)L;
#endif
}

uint32_t PluginTinyFdnReverb::floatToBits(const float value) noexcept
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(float));
    return bits;
}

PluginTinyFdnReverb::PluginTinyFdnReverb()
    : Plugin(paramCount, kPresetCount, 0)
    , fLastSR(0.0)
    , fRt60(2.80f)
    , fMix(1.00f)
    , fMatrixType(0)
    , fDelaySet(0)
    , fSize(1.00f)
{
    for (int i=0; i<kN; ++i) {
        mBuf[i].assign(kMaxDelay, 0.0f);
        mIdx[i] = 0;
        mLen[i] = 64;
        mGain[i] = 0.0f;
    }

    for (auto& slot : mEnvTraceBits)
        slot.store(0u, std::memory_order_relaxed);
    mEnvTraceWrite.store(0u, std::memory_order_relaxed);
    mEDTmsBits.store(0u, std::memory_order_relaxed);
    mRT60sBits.store(0u, std::memory_order_relaxed);
    mDen100Bits.store(0u, std::memory_order_relaxed);
    mDen300Bits.store(0u, std::memory_order_relaxed);
    mRinginessBits.store(0u, std::memory_order_relaxed);
    mWetEnvBits.store(0u, std::memory_order_relaxed);
}

const char* PluginTinyFdnReverb::getLabel()   const { return "tiny-fdn-reverb_1.19"; }
const char* PluginTinyFdnReverb::getMaker()   const { return "Fernando Ara"; }
const char* PluginTinyFdnReverb::getLicense() const { return "MIT"; }
uint32_t    PluginTinyFdnReverb::getVersion() const { return d_version(1,19,0); }
int64_t     PluginTinyFdnReverb::getUniqueId() const { return d_cconst('t','f','d','n'); }

void PluginTinyFdnReverb::initParameter(uint32_t i, Parameter& p) {
    p.hints = kParameterIsAutomable;
    switch (i) {
    case paramRt60:
        p.name = "RT60 (s)"; p.symbol = "rt60";
        p.ranges = { 0.20f, 8.00f, fRt60 };
        break;
    case paramMix:
        p.name = "Mix"; p.symbol = "mix";
        p.ranges = { 0.00f, 1.00f, fMix };
        break;
    case paramMatrixType:
        p.name = "MatrixType"; p.symbol = "matrix";
        p.hints = kParameterIsOutput | kParameterIsInteger;
        p.ranges = { 0.0f, 1.0f, float(fMatrixType) };
        break;
    case paramDelaySet:
        p.name = "DelaySet"; p.symbol = "delayset";
        p.hints |= kParameterIsInteger;
        p.ranges = { 0.0f, 1.0f, float(fDelaySet) };
        break;
    case paramSize:
        p.name = "Size"; p.symbol = "size";
        p.ranges = { 0.50f, 2.00f, fSize };
        break;
    case paramDampHz:
        p.hints  = kParameterIsAutomable;
        p.name   = "Damp (Hz)";  p.symbol = "damp";  p.unit = "Hz";
        p.ranges = { 1500.0f, 12000.0f, fDampHz };
        break;
    case paramMatrixMorph:
        p.hints  = kParameterIsAutomable;
        p.name   = "Matrix Morph"; p.symbol = "mmorph";
        p.ranges = { 0.0f, 1.0f, fMatrixMorph };
        break;
    case paramPing:
        p.hints  = kParameterIsAutomable | kParameterIsBoolean;
        p.name   = "Ping"; p.symbol = "ping";
        p.ranges = { 0.0f, 1.0f, 0.0f };
        break;

    // interactive params
    case paramModDepth:
        p.hints = kParameterIsAutomable;
        p.name  = "Mod Depth"; p.symbol = "moddepth";
        p.ranges= { 0.0f, 1.0f, fModDepth };
        break;
    case paramDetune:
        p.hints = kParameterIsAutomable;
        p.name  = "Detune"; p.symbol = "detune";
        p.ranges= { 0.0f, 1.0f, fDetune };
        break;
    case paramMetalBoost:
        p.hints = kParameterIsAutomable | kParameterIsBoolean;
        p.name  = "Over-spread delays"; p.symbol = "metal";
        p.ranges= { 0.0f, 1.0f, float(fMetalBoost) };
        break;
    case paramExciteNoise:
        p.hints = kParameterIsAutomable | kParameterIsBoolean;
        p.name  = "Noise Burst"; p.symbol = "burst";
        p.ranges= { 0.0f, 1.0f, 0.0f };
        break;

    // Output params (read-only to host/UI)
    case paramEDTms:
        p.hints = kParameterIsOutput;
        p.name="EDT"; p.symbol="edt"; p.unit="ms";
        p.ranges = {0.f, 4000.f, 0.f}; break;
    case paramRT60est:
        p.hints = kParameterIsOutput;
        p.name="RT60(est)"; p.symbol="rt60est"; p.unit="s";
        p.ranges = {0.f, 10.f, 0.f}; break;
    case paramDensity100ms:
        p.hints = kParameterIsOutput;
        p.name="Density@100ms"; p.symbol="dens100"; p.unit="ev/ms";
        p.ranges = {0.f, 50.f, 0.f}; break;
    case paramDensity300ms:
        p.hints = kParameterIsOutput;
        p.name="Density@300ms"; p.symbol="dens300"; p.unit="ev/ms";
        p.ranges = {0.f, 50.f, 0.f}; break;
    case paramRinginess:
        p.hints = kParameterIsOutput;
        p.name  = "Ringiness"; p.symbol = "ring";
        p.ranges= {0.f, 1.f, 0.f}; break;
    case paramWetEnv:
        p.hints = kParameterIsOutput;
        p.name  = "Wet Env"; p.symbol = "wetenv";
        p.ranges= {0.f, 1.f, 0.f}; break;

    default: break;
    }
}

float PluginTinyFdnReverb::getParameterValue(uint32_t i) const {
    switch (i) {
    case paramRt60:         return fRt60;
    case paramMix:          return fMix;
    case paramMatrixType:   return (fMatrixMorph < 0.5f) ? 0.0f : 1.0f;
    case paramDelaySet:     return float(fDelaySet);
    case paramSize:         return fSize;
    case paramDampHz:       return fDampHz;
    case paramMatrixMorph:  return fMatrixMorph;
    case paramPing:         return 0.0f; // momentary

    case paramModDepth:     return fModDepth;
    case paramDetune:       return fDetune;
    case paramMetalBoost:   return float(fMetalBoost);
    case paramExciteNoise:  return 0.0f;

    case paramEDTms:        return bitsToFloat(mEDTmsBits.load(std::memory_order_relaxed));
    case paramRT60est:      return bitsToFloat(mRT60sBits.load(std::memory_order_relaxed));
    case paramDensity100ms: return bitsToFloat(mDen100Bits.load(std::memory_order_relaxed));
    case paramDensity300ms: return bitsToFloat(mDen300Bits.load(std::memory_order_relaxed));
    case paramRinginess:    return bitsToFloat(mRinginessBits.load(std::memory_order_relaxed));
    case paramWetEnv:       return bitsToFloat(mWetEnvBits.load(std::memory_order_relaxed));

    default:                return 0.0f;
    }
}

void PluginTinyFdnReverb::setParameterValue(uint32_t i, float v) {
    switch (i) {
    case paramRt60:
        fRt60 = v; DBG("[PARAM] RT60=%.3f requested", double(fRt60)); break;
    case paramMix:
        fMix = v;  /*no log*/ break;
    case paramMatrixType: {
        (void)v;
        DBG("[DSP] type set attempt ignored");
        break;
    }
    case paramDelaySet: {
        const int old = fDelaySet;
        fDelaySet = int(std::round(v));
        DBG("[PARAM] DelaySet=%d (%s) requested", fDelaySet, fDelaySet? "Spread":"Prime");
        if (fDelaySet != old) {
            // handled in run()
        }
        break;
    }
    case paramSize:
        fSize = v; DBG("[PARAM] Size=%.3f requested", double(fSize)); break;
    case paramDampHz:
        fDampHz = v; /*no log*/ break;
    case paramMatrixMorph: {
        const float vm = std::max(0.0f, std::min(1.0f, v));
        fMatrixMorph = vm;
        fMatrixType = (vm < 0.5f) ? 0 : 1;
        DBG("[DSP] setParameterValue(MatrixMorph)=%.3f", double(v));
        break;
    }
    case paramPing:
        fPing = (v >= 0.5f); /*no log*/ break;

    case paramModDepth:
        fModDepth = v; DBG("[PARAM] ModDepth=%.3f", double(fModDepth)); break;
    case paramDetune:
        fDetune = v;   DBG("[PARAM] Detune=%.3f",   double(fDetune));  break;
    case paramMetalBoost:
        fMetalBoost = (v >= 0.5f); DBG("[PARAM] MetalBoost=%d", fMetalBoost); break;
    case paramExciteNoise:
        fExciteNoise = (v >= 0.5f); DBG("[PARAM] NoiseBurst=%d", fExciteNoise); break;

    default: break;
    }
}

void PluginTinyFdnReverb::initProgramName(uint32_t i, String& name) {
    if (i < kPresetCount) name = kPresets[i].name;
}

void PluginTinyFdnReverb::loadProgram(uint32_t i) {
    DBG("[DSP] loadProgram(%u) called", i);
    if (i < kPresetCount) {
        setParameterValue(paramRt60,         kPresets[i].params[paramRt60]);
        setParameterValue(paramMix,          kPresets[i].params[paramMix]);
        setParameterValue(paramDelaySet,     kPresets[i].params[paramDelaySet]);
        setParameterValue(paramSize,         kPresets[i].params[paramSize]);
        setParameterValue(paramDampHz,       kPresets[i].params[paramDampHz]);
        setParameterValue(paramModDepth,     kPresets[i].params[paramModDepth]);
        setParameterValue(paramDetune,       kPresets[i].params[paramDetune]);
        setParameterValue(paramMetalBoost,   kPresets[i].params[paramMetalBoost]);
        setParameterValue(paramExciteNoise,  kPresets[i].params[paramExciteNoise]);
    }
}

void PluginTinyFdnReverb::activate() {
    fAppliedSize        = fSize;
    fAppliedDelaySet    = fDelaySet;
    fAppliedMatrixType  = (fMatrixMorph < 0.5f) ? 0 : 1;
    fAppliedMetalBoost  = fMetalBoost;

    mMuteSamples      = 0;
    irWrite = 0; irCapturing = false; irReady = false; irAnalyzed = false;
    mEDTms = mRT60s = mDen100 = mDen300 = 0.f;
    mRinginess = 0.f;
    mWetEnv = 0.f;

    for (int k=0; k<kN; ++k) {
        std::fill(mBuf[k].begin(), mBuf[k].end(), 0.0f);
        mIdx[k]  = 0;
        mLP[k]   = 0.0f;
        mAPx1[k] = 0.f;
        mAPy1[k] = 0.f;
        mAPd_x1[k] = 0.f;
        mAPd_y1[k] = 0.f;
        mLFOph[k] = 0.7 * k; // staggered phases
    }
    mNoiseSeed = 222222u;
    mNoiseBurstLeft = 0;

    // ringiness meter buffer
    mMeterIdx = 0;
    mMeterFilled = false;
    mMeterBuf.fill(0.f);

    for (auto& slot : mEnvTraceBits)
        slot.store(0u, std::memory_order_relaxed);
    mEnvTraceWrite.store(0u, std::memory_order_relaxed);
    mEDTmsBits.store(0u, std::memory_order_relaxed);
    mRT60sBits.store(0u, std::memory_order_relaxed);
    mDen100Bits.store(0u, std::memory_order_relaxed);
    mDen300Bits.store(0u, std::memory_order_relaxed);
    mRinginessBits.store(0u, std::memory_order_relaxed);
    mWetEnvBits.store(0u, std::memory_order_relaxed);

    fLastSR = 0.0;
    DBG("[ACT] init done");
}

void PluginTinyFdnReverb::resetStateForTopologyChange() noexcept {
    for (int k=0; k<kN; ++k) {
        std::fill(mBuf[k].begin(), mBuf[k].end(), 0.0f);
        mIdx[k]  = 0;
        mLP[k]   = 0.0f;
        mAPx1[k] = 0.0f;
        mAPy1[k] = 0.0f;
        mAPd_x1[k] = 0.0f;
        mAPd_y1[k] = 0.0f;
        mLFOph[k] = 0.7 * k;
    }
    const double sr = (fLastSR > 0.0 ? fLastSR : 48000.0);
    mMuteSamples = int(0.050 * sr); // 50 ms mute to avoid clicks & “stickiness”
    irWrite = 0; irCapturing = false; irReady = false; irAnalyzed = false;
    DBG("[RESET] flush state, mute=%d samples", mMuteSamples);
}

void PluginTinyFdnReverb::selectBaseAndUpdateDelays(double sr) noexcept {
    const double scale = (sr > 0.0 ? sr/48000.0 : 1.0) * std::max(0.5, std::min(2.0, double(fSize)));
    const std::array<int,kN>& base = (fDelaySet == 0) ? kBasePrime48 : kBaseSpread48;
    for (int k=0; k<kN; ++k) {
        const int L = std::max(1, std::min(kMaxDelay, int(std::lround(base[k] * scale))));
        mLen[k] = L;
        if (mIdx[k] >= L) mIdx[k] = 0;
    }
    DBG("[DELAY] base=%s sr=%.0f size=%.3f -> L=[%d %d %d %d]",
        fDelaySet? "Spread":"Prime", sr, double(fSize), mLen[0], mLen[1], mLen[2], mLen[3]);
}

void PluginTinyFdnReverb::updateLineGainsFromRt60(double sr) noexcept {
    const float rt60Sec = std::max(0.20f, fRt60); // 0.2s min
    const double T60 = std::max(0.05, double(rt60Sec));

    for (int k = 0; k < kN; ++k) {
        const double Li = double(mLen[k]);
        const double gi = std::pow(10.0, -3.0 * (Li / (T60 * sr)));
        mGain[k] = float(gi);
    }

    DBG("[RT60 DEBUG] fRt60=%.3f sr=%.1f  T60=%.3f  gains={%.6f, %.6f, %.6f, %.6f}",
        double(fRt60), sr, T60,
        double(mGain[0]), double(mGain[1]),
        double(mGain[2]), double(mGain[3]));
}

inline void PluginTinyFdnReverb::hadamardMix4(const float in[kN], float out[kN]) const noexcept {
    const float a=in[0], b=in[1], c=in[2], d=in[3];
    out[0] = 0.5f*(+a + b + c + d);
    out[1] = 0.5f*(+a - b + c - d);
    out[2] = 0.5f*(+a + b - c - d);
    out[3] = 0.5f*(+a - b - c + d);
}

inline void PluginTinyFdnReverb::householderMix4(const float in[kN], float out[kN]) const noexcept {
    const float s = in[0]+in[1]+in[2]+in[3];
    const float halfS = 0.5f * s;
    out[0] = in[0] - halfS;
    out[1] = in[1] - halfS;
    out[2] = in[2] - halfS;
    out[3] = in[3] - halfS;
}

// Compute ringiness using short-lag autocorr on a window that
// excludes the newest ~20 ms (to avoid burst/ping bias).
void PluginTinyFdnReverb::computeRinginess(double sr) noexcept {
    const int N = mMeterFilled ? kMeterMax : mMeterIdx;
    if (N < 1024) { mRinginess = 0.f; return; }

    // window length ~2048 samples (≈43 ms @48k), end offset ~20 ms
    const int W = std::min(N, 2048);
    const int offset = std::min(N - W, std::max(0, int(0.020 * sr))); // ~20 ms
    // effective "end" index is (mMeterIdx - 1 - offset)
    int end = (mMeterIdx - 1 - offset);
    while (end < 0) end += kMeterMax;
    end %= kMeterMax;

    auto at = [&](int idx)->float {
        // idx in [0..W-1], map to circular buffer ending at 'end'
        int off = (end - (W-1 - idx));
        while (off < 0) off += kMeterMax;
        off %= kMeterMax;
        return mMeterBuf[off];
    };

    // energy at lag 0
    double E0 = 0.0;
    for (int i=0;i<W;++i) {
        const double s = at(i);
        E0 += s*s;
    }
    if (E0 < 1e-12) { mRinginess = 0.f; return; }

    // search lags ~ 1.5–8 ms (comb-ish)
    const int lagMin = int(0.0015 * sr);
    const int lagMax = int(0.0080 * sr);
    double best = 0.0;
    for (int L = lagMin; L <= lagMax; L += 4) {
        double C = 0.0;
        for (int i=L; i<W; ++i)
            C += double(at(i)) * double(at(i-L));
        if (C > best) best = C;
    }
    const double ring = std::max(0.0, std::min(1.0, best / (E0 + 1e-12)));
    mRinginess = float(ring);
}

void PluginTinyFdnReverb::run(const float** inputs, float** outputs, uint32_t frames) {
    const float* inL  = inputs[0];
    const float* inR  = inputs[1];
    float*       outL = outputs[0];
    float*       outR = outputs[1];

    const double sr = this->getSampleRate();
    if (sr != fLastSR) {
        fMixSmoothL   = CParamSmooth(10.0f, sr);
        fMixSmoothR   = CParamSmooth(10.0f, sr);
        fRt60Smooth   = CParamSmooth(10.0f, sr);
        fDampSmooth   = CParamSmooth(10.0f, sr);
        fMorphSmooth  = CParamSmooth(10.0f, sr);
        fModSmooth    = CParamSmooth(10.0f, sr);
        fDetuneSmooth = CParamSmooth(10.0f, sr);
        selectBaseAndUpdateDelays(sr);
        fLastSR = sr;
        DBG("[RUN] SR change -> %.0f", sr);
    }

    // Apply size change (integer lengths)
    const int currentMatrixType = (fMatrixMorph < 0.5f) ? 0 : 1;
    fMatrixType = currentMatrixType;

    if (std::fabs(fSize - fAppliedSize) > 1e-4f) {
        fAppliedSize = fSize;
        selectBaseAndUpdateDelays(sr);
        resetStateForTopologyChange();
        dump_state_lengths("RESET", currentMatrixType, fDelaySet, mLen);
    }

    // Metallic Boost toggled?
    if (fMetalBoost != fAppliedMetalBoost) {
        fAppliedMetalBoost = fMetalBoost;
        if (fAppliedMetalBoost) {
            // Force Spread and reinit state for clarity
            DBG("[RUN] MetalBoost ON -> force Spread, mod=0, light damping");
            fDelaySet = 1; // Spread
        } else {
            DBG("[RUN] MetalBoost OFF -> return to current delay set");
        }
        selectBaseAndUpdateDelays(sr);
        resetStateForTopologyChange();
        dump_state_lengths("RESET", currentMatrixType, fDelaySet, mLen);
    }

    // Delay set change?
    if (fDelaySet != fAppliedDelaySet) {
        DBG("[RUN] DelaySet changed: %s -> %s",
            fAppliedDelaySet? "Spread":"Prime", fDelaySet? "Spread":"Prime");
        fAppliedDelaySet = fDelaySet;
        selectBaseAndUpdateDelays(sr);
        resetStateForTopologyChange();
        dump_state_lengths("RESET", currentMatrixType, fDelaySet, mLen);
    }

    // Matrix change?
    if (currentMatrixType != fAppliedMatrixType) {
        DBG("[RUN] MatrixType changed: %s -> %s",
            fAppliedMatrixType? "Householder":"Hadamard",
            currentMatrixType? "Householder":"Hadamard");
        fAppliedMatrixType = currentMatrixType;
        resetStateForTopologyChange();
        dump_state_lengths("RESET", currentMatrixType, fDelaySet, mLen);
    }

    updateLineGainsFromRt60(sr);

    // Block-edge triggers
    const bool pingAtBlock  = (fPing != 0);
    const bool burstAtBlock = (fExciteNoise != 0);

    if (pingAtBlock) {
        fPing = 0;
        irWrite = 0; irCapturing = true; irReady = false; irAnalyzed = false;
        DBG("[RUN] Ping");
    }
    if (burstAtBlock) {
        fExciteNoise = 0;
        mNoiseBurstLeft = std::max(1, int(0.020 * sr)); // 20 ms
        irWrite = 0; irCapturing = true; irReady = false; irAnalyzed = false;
        DBG("[RUN] Noise burst start (%d samples) + IR capture", mNoiseBurstLeft);
    }

    double wetEnergy = 0.0;

    for (uint32_t i=0; i<frames; ++i) {
        const float mixL   = fMixSmoothL.process(fMix);
        const float mixR   = fMixSmoothR.process(fMix);
        const float dampHz = fDampSmooth.process(fDampHz);
        const float modDepth= fModSmooth.process(fModDepth);
        const float detu   = fDetuneSmooth.process(fDetune);

        // Smooth morph parameter 0..1 (Hadamard → Householder)
        const float morphTarget = fMorphSmooth.process(fMatrixMorph);

        // Spread is our “worst-case” metallic set; we still allow modulation unless MetalBoost forces off.
        float modAmt = modDepth;
        float dampHzLocal = dampHz;
        if (fAppliedMetalBoost) {
            modAmt = 0.0f;                               // remove LFO smear
            dampHzLocal = std::max(dampHzLocal, 11000.f);// lighter damping
        }

        // Read taps
        float x[kN];
        for (int k=0; k<kN; ++k) x[k] = mBuf[k][ mIdx[k] ];

        // Per-line loss from RT60
        float g[kN];
        for (int k=0; k<kN; ++k) g[k] = mGain[k] * x[k];

        // Feedback mix (Hada vs House)
        float yH[kN], yHo[kN], y[kN];
        hadamardMix4   (g, yH);
        householderMix4(g, yHo);
        for (int k=0; k<kN; ++k) y[k] = (1.0f - morphTarget)*yH[k] + morphTarget*yHo[k];

        // --- STATIC detune AP (small, alternating sign) ---
        const float a_det_base = 0.25f * detu;
        const float sign[4] = {+1.f, -1.f, +1.f, -1.f};
        float yDet[kN];
        for (int k=0; k<kN; ++k) {
            const float a = sign[k] * a_det_base;
            const float xin  = y[k];
            const float yout = -a * xin + mAPd_x1[k] + a * mAPd_y1[k];
            mAPd_x1[k] = xin;
            mAPd_y1[k] = yout;
            yDet[k]    = yout;
        }

        // --- LFO AP (scaled by modAmt) ---
        const float depth = 0.35f;
        const float rates[kN] = {0.21f, 0.27f, 0.33f, 0.41f};
        const double twopi = 2.0 * kPI;

        float yMod[kN];
        for (int k=0; k<kN; ++k) {
            mLFOph[k] += twopi * (rates[k] / float(sr));
            if (mLFOph[k] > twopi) mLFOph[k] -= twopi;
            const float aap = modAmt * depth * std::sin(float(mLFOph[k]));
            const float xin  = yDet[k];
            const float yout = -aap * xin + mAPx1[k] + aap * mAPy1[k];
            mAPx1[k] = xin;
            mAPy1[k] = yout;
            yMod[k]  = yout;
        }

        // --- damping (one-pole LP in the loop) ---
        const float a = std::exp(-2.0f * float(kPI) * dampHzLocal / float(sr));
        const float b = 1.0f - a;
        float yDamped[kN];
        for (int k=0; k<kN; ++k) {
            mLP[k] = a * mLP[k] + b * yMod[k];
            yDamped[k] = mLP[k];
        }

        // --- injection (ping or noise burst or input mono) ---
        float inj = 0.25f * (0.5f*(inL[i] + inR[i]));
        if (mNoiseBurstLeft > 0) {
            inj = nextNoise(); --mNoiseBurstLeft;
        } else if (pingAtBlock && i == 0) {
            inj = 1.0f;
        }

        // write and advance
        for (int k=0; k<kN; ++k) {
            mBuf[k][ mIdx[k] ] = yDamped[k] + inj;
            int idx = mIdx[k] + 1; if (idx >= mLen[k]) idx = 0;
            mIdx[k] = idx;
        }

        float wetL = 0.5f*(+yMod[0] - yMod[1] + yMod[2] - yMod[3]);
        float wetR = 0.5f*(+yMod[0] + yMod[1] - yMod[2] - yMod[3]);

        float wetGain = 1.0f;
        if (mMuteSamples > 0) { wetGain = 0.0f; --mMuteSamples; }
        wetL *= wetGain; wetR *= wetGain;

        const float wetMono = 0.5f * (wetL + wetR);
        wetEnergy += double(wetMono) * double(wetMono);

        // IR capture (mono)
        if (irCapturing) {
            if (irWrite < kIRMax) {
                mIR[irWrite++] = wetMono;
                const int maxN = std::min(kIRMax, int(2.0 * sr * std::max(0.5, double(fRt60))));
                if (irWrite >= maxN) { irCapturing = false; irReady = true; }
            } else {
                irCapturing = false; irReady = true;
            }
        }

        // Feed ringiness buffer
        mMeterBuf[mMeterIdx] = wetMono;
        mMeterIdx = (mMeterIdx + 1) % kMeterMax;
        if (mMeterIdx == 0) mMeterFilled = true;

        // Mix to outputs
        auto clamp15 = [](float v){ return (v>1.5f?1.5f:(v<-1.5f?-1.5f:v)); };
        outL[i] = (1.0f - mixL)*inL[i] + mixL*clamp15(wetL);
        outR[i] = (1.0f - mixR)*inR[i] + mixR*clamp15(wetR);

        if (std::fabs(outL[i]) < 1e-30f) outL[i] = 0.0f;
        if (std::fabs(outR[i]) < 1e-30f) outR[i] = 0.0f;
    } // per-sample

    mWetEnv = (frames > 0u) ? std::sqrt(float(wetEnergy / double(frames))) : 0.f;
    const uint32_t writeIndex = mEnvTraceWrite.load(std::memory_order_relaxed);
    mEnvTraceBits[writeIndex & (kEnvTraceSize - 1u)].store(floatToBits(mWetEnv), std::memory_order_relaxed);
    mEnvTraceWrite.store(writeIndex + 1u, std::memory_order_release);

    // If an IR just finished, analyze it once and push metrics
    if (irReady && !irAnalyzed) {
        irAnalyzed = true;

        const int N = std::max(0, std::min(int(irWrite), int(kIRMax)));
        if (N > 16) {
            // Normalize
            float maxAbs = 0.f;
            for (int i=0;i<N;++i) maxAbs = std::max(maxAbs, std::fabs(mIR[i]));
            const float invMax = (maxAbs > 1e-9f ? 1.0f/maxAbs : 1.0f);

            // EDC (reverse cumulative energy)
            std::vector<double> edc(N);
            double acc = 0.0;
            for (int i=N-1; i>=0; --i) {
                const double s = double(mIR[i]) * invMax;
                acc += s*s;
                edc[i] = acc;
            }
            // dB and offset
            const double edc0 = std::max(1e-20, edc[0]);
            std::vector<double> edcDb(N);
            for (int i=0;i<N;++i) edcDb[i] = 10.0*std::log10(std::max(1e-20, edc[i]/edc0));

            auto linreg = [&](double dbLo, double dbHi, double& slope, double& intercept){
                const double dt = 1.0 / fLastSR;
                int i0 = 0, i1 = N-1;
                // find window where edcDb is within [dbHi, dbLo] (note db values are <=0)
                for (; i0<N; ++i0) if (edcDb[i0] <= dbLo) break;
                for (; i1>i0; --i1) if (edcDb[i1] >= dbHi) break;
                if (i1 <= i0) { slope = -1.0; intercept = 0.0; return; }
                const int M = (i1 - i0 + 1);
                double Sx=0, Sy=0, Sxx=0, Sxy=0;
                for (int i=i0; i<=i1; ++i) {
                    const double t = i*dt;
                    const double y = edcDb[i];
                    Sx += t; Sy += y; Sxx += t*t; Sxy += t*y;
                }
                const double den = (M*Sxx - Sx*Sx);
                if (std::fabs(den) < 1e-12) { slope=-1.0; intercept=0.0; return; }
                slope = (M*Sxy - Sx*Sy) / den;
                intercept = (Sy - slope*Sx) / M;
            };

            double m, b;
            linreg(-5.0, -35.0, m, b);
            mRT60s = (m < 0.0 ? float(-60.0 / m) : 0.0f);

            linreg( 0.0, -10.0, m, b);
            // time where line hits -10 dB: -10 = m*t + b -> t = (-10 - b)/m
            mEDTms = (m < -1e-9 ? float(((-10.0 - b)/m) * 1000.0) : 0.0f);

            auto peakDensityAt = [&](double center_s)->float{
                const int win  = std::max(4, int(0.010 * fLastSR)); // 10 ms
                const int half = win/2;
                const int c    = std::max(half, std::min(int(center_s * fLastSR), N-half-1));
                const int a    = c - half;
                const int z    = c + half;
                int peaks = 0;
                const float thr = 0.02f; // relative to normalization
                for (int i=a+1; i<z-1; ++i) {
                    const float s0 = float(mIR[i-1]) * invMax;
                    const float s1 = float(mIR[i])   * invMax;
                    const float s2 = float(mIR[i+1]) * invMax;
                    if (std::fabs(s1) > thr && s1 >= s0 && s1 > s2) ++peaks;
                }
                return float(double(peaks) / (0.010 * 1000.0)); // events/ms
            };
            mDen100 = peakDensityAt(0.100);
            mDen300 = peakDensityAt(0.300);

            DBG("[DSP] IR analyzed: N=%d  RT60=%.3fs  EDT=%.1fms  dens100=%.2f  dens300=%.2f",
                N, mRT60s, mEDTms, mDen100, mDen300);
        }
    }

    // compute ringiness every block (cheap)
    computeRinginess(sr);

    // summary line every ~50 ms
    mSamplesSincePush += frames;
    if (sr > 0.0 && mSamplesSincePush >= (uint32_t)(0.05 * sr)) {
        // Publish meter snapshots for UI direct-access polling.
        mEDTmsBits.store(floatToBits(mEDTms), std::memory_order_relaxed);
        mRT60sBits.store(floatToBits(mRT60s), std::memory_order_relaxed);
        mDen100Bits.store(floatToBits(mDen100), std::memory_order_relaxed);
        mDen300Bits.store(floatToBits(mDen300), std::memory_order_relaxed);
        mRinginessBits.store(floatToBits(mRinginess), std::memory_order_relaxed);
        mWetEnvBits.store(floatToBits(mWetEnv), std::memory_order_relaxed);

        DBG("[DSP] meter publish idx=[%u,%u,%u,%u,%u,%u]",
            unsigned(paramEDTms),
            unsigned(paramRT60est),
            unsigned(paramDensity100ms),
            unsigned(paramDensity300ms),
            unsigned(paramRinginess),
            unsigned(paramWetEnv));

        DBG("[RUNDBG] rt60=%.3f matrix=%s delay=%s metal=%d modDepth=%.2f ring=%.2f L=[%d %d %d %d]",
            double(fRt60),
            fMatrixType ? "Householder" : "Hadamard",
            fDelaySet   ? "Spread"      : "Prime",
            fAppliedMetalBoost,
            double(fModDepth),
            double(mRinginess),
            mLen[0], mLen[1], mLen[2], mLen[3]);

        mSamplesSincePush = 0;
    }
}

// === BOILERPLATE BEGIN: DPF plugin entry point ===
Plugin* createPlugin()
{
    return new PluginTinyFdnReverb();
}
// === BOILERPLATE END ===

END_NAMESPACE_DISTRHO
