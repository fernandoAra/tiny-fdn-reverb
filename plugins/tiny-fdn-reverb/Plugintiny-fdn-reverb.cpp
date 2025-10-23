#include "Plugintiny-fdn-reverb.hpp"
#include <cmath>
#include <algorithm>
#include <cstdio>
#define DBG(...) do { std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } while(0)

START_NAMESPACE_DISTRHO

struct Preset {
    const char* name;
    float params[PluginTinyFdnReverb::paramCount];
};
static const Preset kPresets[] = {
    //   RT60   Mix   Mat  Delay Size  DampHz  Morph  Ping
    { "Default",
    { 2.80f, 1.00f, 0.0f, 0.0f, 1.00f, 6000.0f, 0.0f, 0.0f } },

    { "House+Spread",
    { 1.80f, 0.70f, 1.0f, 1.0f, 1.10f, 5000.0f, 0.0f, 0.0f } },
};
static const uint kPresetCount = sizeof(kPresets)/sizeof(kPresets[0]);

PluginTinyFdnReverb::PluginTinyFdnReverb()
    : Plugin(paramCount, kPresetCount, 0)
    , fLastSR(0.0)
    , fRt60(2.80f)
    , fMix(1.00f)
    , fMatrixType(0)
    , fDelaySet(0)
    , fSize(1.00f)
    , fMixSmoothL(10.0f, getSampleRate())
    , fMixSmoothR(10.0f, getSampleRate())
    , fRt60Smooth(10.0f, getSampleRate())
    , fDampSmooth (10.0f, getSampleRate())
    , fMorphSmooth(10.0f, getSampleRate())
{
    for (int i=0; i<kN; ++i) {
        mBuf[i].assign(kMaxDelay, 0.0f);
        mIdx[i] = 0;
        mLen[i] = 64;
        mGain[i] = 0.0f;
    }
}

const char* PluginTinyFdnReverb::getLabel()   const { return "tiny-fdn-reverb"; }
const char* PluginTinyFdnReverb::getMaker()   const { return "Fernando Ara"; }
const char* PluginTinyFdnReverb::getLicense() const { return "MIT"; }
uint32_t    PluginTinyFdnReverb::getVersion() const { return d_version(0,2,1); } // DEMO R2
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
        p.hints |= kParameterIsInteger;
        p.ranges = { 0.0f, 1.0f, float(fMatrixType) };
        // (Hosts will show 0/1; we can add enum labels in a later pass)
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
    default: break;
    }
}

float PluginTinyFdnReverb::getParameterValue(uint32_t i) const {
    switch (i) {
    case paramRt60:       return fRt60;
    case paramMix:        return fMix;
    case paramMatrixType: return float(fMatrixType);
    case paramDelaySet:   return float(fDelaySet);
    case paramSize:       return fSize;
    case paramDampHz:      return fDampHz;
    case paramMatrixMorph: return fMatrixMorph;
    case paramPing:        return 0.0f; // momentary
    case paramEDTms:         return mEDTms;
    case paramRT60est:       return mRT60s;
    case paramDensity100ms:  return mDen100;
    case paramDensity300ms:  return mDen300;
    default: return 0.0f;
    }
}

void PluginTinyFdnReverb::setParameterValue(uint32_t i, float v) {
    switch (i) {
    case paramRt60:       fRt60       = v; break;
    case paramMix:        fMix        = v; break;
    case paramMatrixType: fMatrixType = int(std::round(v)); break;
    case paramDelaySet:   fDelaySet   = int(std::round(v)); break;
    case paramSize:       fSize       = v; break;
    case paramDampHz:      fDampHz      = v; break;
    case paramMatrixMorph: fMatrixMorph = v; break;
    case paramPing:        fPing        = (v >= 0.5f); break;
    default: break;
    }
}

void PluginTinyFdnReverb::initProgramName(uint32_t i, String& name) {
    if (i < kPresetCount) name = kPresets[i].name;
}
void PluginTinyFdnReverb::loadProgram(uint32_t i) {
    if (i < kPresetCount) {
        setParameterValue(paramRt60,       kPresets[i].params[paramRt60]);
        setParameterValue(paramMix,        kPresets[i].params[paramMix]);
        setParameterValue(paramMatrixType, kPresets[i].params[paramMatrixType]);
        setParameterValue(paramDelaySet,   kPresets[i].params[paramDelaySet]);
        setParameterValue(paramSize,       kPresets[i].params[paramSize]);
    }
}

void PluginTinyFdnReverb::activate() {
    fAppliedSize = fSize;
    fAppliedDelaySet = fDelaySet;
    fAppliedMatrixType= fMatrixType;
    mMuteSamples      = 0;
    irWrite = 0; irCapturing = false; irReady = false;
    mEDTms = mRT60s = mDen100 = mDen300 = 0.f;
    for (int k=0; k<kN; ++k) {
        std::fill(mBuf[k].begin(), mBuf[k].end(), 0.0f);
        mIdx[k] = 0;
        mLP[k]  = 0.0f;
    }
    fLastSR = 0.0;
}

void PluginTinyFdnReverb::resetStateForTopologyChange() noexcept {
    // clear delay lines and filters, reset indices
    for (int k=0; k<kN; ++k) {
        std::fill(mBuf[k].begin(), mBuf[k].end(), 0.0f);
        mIdx[k] = 0;
        mLP[k]  = 0.0f;
    }
    // brief wet mute (5 ms) to avoid discontinuity clicks
    const double sr = (fLastSR > 0.0 ? fLastSR : 48000.0);
    mMuteSamples = int(0.005 * sr); // 5 ms
    // also stop any ongoing IR capture
    irWrite = 0; irCapturing = false; irReady = false;
}


void PluginTinyFdnReverb::selectBaseAndUpdateDelays(double sr) noexcept {
    const double scale = (sr > 0.0 ? sr/48000.0 : 1.0) * std::max(0.5, std::min(2.0, double(fSize)));
    const std::array<int,kN>& base = (fDelaySet == 0) ? kBasePrime48 : kBaseSpread48;
    for (int k=0; k<kN; ++k) {
        const int L = std::max(1, std::min(kMaxDelay, int(std::lround(base[k] * scale))));
        mLen[k] = L;
        if (mIdx[k] >= L) mIdx[k] = 0;
    }
}

void PluginTinyFdnReverb::updateLineGainsFromRt60(double sr) noexcept {
    // Smooth RT60 a bit to avoid zipper in the tail
    const float rt60Sm = fRt60Smooth.process(std::max(0.20f, fRt60));
    const double T60 = std::max(0.05, double(rt60Sm));
    for (int k=0; k<kN; ++k) {
        const double Li = double(mLen[k]);
        const double gi = std::pow(10.0, -3.0 * (Li / (T60 * sr)));
        mGain[k] = float(gi);
    }
}

inline void PluginTinyFdnReverb::hadamardMix4(const float in[kN], float out[kN]) const noexcept {
    const float a=in[0], b=in[1], c=in[2], d=in[3];
    out[0] = 0.5f*(+a + b + c + d);
    out[1] = 0.5f*(+a - b + c - d);
    out[2] = 0.5f*(+a + b - c - d);
    out[3] = 0.5f*(+a - b - c + d);
}

inline void PluginTinyFdnReverb::householderMix4(const float in[kN], float out[kN]) const noexcept {
    // Householder H = I - 2 v v^T with v = [1,1,1,1]/2 (unit norm)
    const float s = in[0]+in[1]+in[2]+in[3]; // v^T in * 2 (since v=1/2 each → v^T in = 0.5*s)
    // y = in - 2*v*(v^T in) = in - (s/2)*[1,1,1,1]
    const float halfS = 0.5f * s;
    out[0] = in[0] - halfS;
    out[1] = in[1] - halfS;
    out[2] = in[2] - halfS;
    out[3] = in[3] - halfS;
}

void PluginTinyFdnReverb::run(const float** inputs, float** outputs, uint32_t frames) {
    const float* inL  = inputs[0];
    const float* inR  = inputs[1];
    float*       outL = outputs[0];
    float*       outR = outputs[1];

    const double sr = getSampleRate();
    if (sr != fLastSR) {
        fMixSmoothL = CParamSmooth(10.0f, sr);
        fMixSmoothR = CParamSmooth(10.0f, sr);
        fRt60Smooth = CParamSmooth(10.0f, sr);
        fDampSmooth = CParamSmooth(10.0f, sr);
        fMorphSmooth= CParamSmooth(10.0f, sr);
        selectBaseAndUpdateDelays(sr);
        fLastSR = sr;
    }

    // If Size changed since last block, update delays (cheap, integer lengths)
    if (std::fabs(fSize - fAppliedSize) > 1e-4f) {
        selectBaseAndUpdateDelays(sr);
        fAppliedSize = fSize;
    }

    if (fDelaySet != fAppliedDelaySet) {
        selectBaseAndUpdateDelays(sr);
        fAppliedDelaySet = fDelaySet;
    }

    // if matrix changed (Hadamard ↔ House), reset state too
    if (fMatrixType != fAppliedMatrixType) {
        fAppliedMatrixType = fMatrixType;
        resetStateForTopologyChange();
    }

    updateLineGainsFromRt60(sr);

    for (uint32_t i=0; i<frames; ++i) {
        // --- smoothed params ---
        const float mixL   = fMixSmoothL.process(fMix);
        const float mixR   = fMixSmoothR.process(fMix);
        const float dampHz = fDampSmooth.process(fDampHz);
        // For the demo, MAKE THE SWITCH HARD: 0 = Hadamard, 1 = Householder
        const float morph  = float(fMatrixType);

        // --- Ping (one-sample Dirac at block start) ---
        const bool pingNow = (fPing != 0) && (i == 0);
        if (pingNow) { fPing = 0; irWrite = 0; irCapturing = true; irReady = false; }

        // --- read taps (pre-mix) ---
        float x[kN];
        for (int k=0; k<kN; ++k) x[k] = mBuf[k][ mIdx[k] ];

        // --- per-line loss from RT60 (mGain[] already updated) ---
        float g[kN];
        for (int k=0; k<kN; ++k) g[k] = mGain[k] * x[k];

        // --- feedback mix: both matrices, then morph (0..1) ---
        float yH[kN], yHo[kN], y[kN];
        hadamardMix4   (g, yH);
        householderMix4(g, yHo);
        for (int k=0; k<kN; ++k) y[k] = (1.0f - morph)*yH[k] + morph*yHo[k];

        // --- in-loop damping (one-pole LP) and injection ---
        const float a = std::exp(-2.0f * float(M_PI) * dampHz / float(sr));
        const float b = 1.0f - a;
        const float inj = 0.25f; // slightly lower to emphasize matrix timbre
        const float inMono = pingNow ? 1.0f : 0.5f*(inL[i] + inR[i]);

        for (int k=0; k<kN; ++k) {
            // low-pass the loop signal
            mLP[k] = a * mLP[k] + b * y[k];
            const float write = mLP[k] + inj*inMono;
            mBuf[k][ mIdx[k] ] = write;
            int idx = mIdx[k] + 1; if (idx >= mLen[k]) idx = 0;
            mIdx[k] = idx;
        }

        // --- wet output taken from post-mix y[...] (makes matrix differences obvious) ---
        // --- wet output from post-mix y[...] (matrix differences are obvious)
        float wetL = 0.5f*(+y[0] - y[1] + y[2] - y[3]);
        float wetR = 0.5f*(+y[0] + y[1] - y[2] - y[3]);

        // brief mute after topology change to avoid clicks / stale tail perception
        if (mMuteSamples > 0) {
            wetL = 0.0f; wetR = 0.0f;
            --mMuteSamples;
        }
        const float wetMono = 0.5f * (wetL + wetR);

        // --- IR capture on mixed mono ---
        if (irCapturing) {
            if (irWrite < kIRMax) {
                mIR[irWrite++] = wetMono;
                const int maxN = std::min(kIRMax, int(2.0 * sr * std::max(0.5, double(fRt60))));
                if (irWrite >= maxN) { irCapturing = false; irReady = true; }
            } else {
                irCapturing = false; irReady = true;
            }
        }

        // --- output mix + denormal guard ---
        auto clamp25 = [](float v){ return (v>2.5f?2.5f:(v<-2.5f?-2.5f:v)); };
        outL[i] = (1.0f - mixL)*inL[i] + mixL*clamp25(wetL);
        outR[i] = (1.0f - mixR)*inR[i] + mixR*clamp25(wetR);
        if (std::fabs(outL[i]) < 1e-30f) outL[i] = 0.0f;
        if (std::fabs(outR[i]) < 1e-30f) outR[i] = 0.0f;
    }

    // --- end of run(): push outputs to UI at ~20 Hz ---
    mSamplesSincePush += frames;
    if (sr > 0.0 && mSamplesSincePush >= (uint32_t)(0.05 * sr)) {
        setParameterValue(paramEDTms,        mEDTms);
        setParameterValue(paramRT60est,      mRT60s);
        setParameterValue(paramDensity100ms, mDen100);
        setParameterValue(paramDensity300ms, mDen300);
        DBG("[DSP] push EDT=%.0fms RT60=%.2fs dens100=%.2f dens300=%.2f",
            mEDTms, mRT60s, mDen100, mDen300);
        mSamplesSincePush = 0;
    }
}



Plugin* createPlugin() { return new PluginTinyFdnReverb(); }

END_NAMESPACE_DISTRHO
