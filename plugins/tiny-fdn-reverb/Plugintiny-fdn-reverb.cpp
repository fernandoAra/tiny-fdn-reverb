#include "Plugintiny-fdn-reverb.hpp"
#include <cmath>
#include <algorithm>

START_NAMESPACE_DISTRHO

struct Preset {
    const char* name;
    float params[PluginTinyFdnReverb::paramCount];
};
static const Preset kPresets[] = {
    //   RT60   Mix  Mat Delay Size
    { "Default", { 2.80f, 1.00f, 0.0f, 0.0f, 1.00f } }, // loud & obvious
    { "House+Spread", { 1.80f, 0.70f, 1.0f, 1.0f, 1.10f } },
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
uint32_t    PluginTinyFdnReverb::getVersion() const { return d_version(0,2,0); }
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
    for (int k=0; k<kN; ++k) {
        std::fill(mBuf[k].begin(), mBuf[k].end(), 0.0f);
        mIdx[k] = 0;
    }
    fLastSR = 0.0;
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
        selectBaseAndUpdateDelays(sr);
        fLastSR = sr;
    }
    updateLineGainsFromRt60(sr);

    for (uint32_t i=0; i<frames; ++i) {
        const float mixL = fMixSmoothL.process(fMix);
        const float mixR = fMixSmoothR.process(fMix);

        // read tap
        float x[kN];
        for (int k=0; k<kN; ++k) x[k] = mBuf[k][ mIdx[k] ];

        // apply per-line loss
        float g[kN];
        for (int k=0; k<kN; ++k) g[k] = mGain[k]*x[k];

        // feedback mix
        float y[kN];
        if (fMatrixType == 0) hadamardMix4(g, y);
        else                  householderMix4(g, y);

        // injection (mono)
        const float inMono = 0.5f*(inL[i] + inR[i]);
        const float inj = 0.40f;
        for (int k=0; k<kN; ++k) {
            const float write = y[k] + inj*inMono;
            mBuf[k][ mIdx[k] ] = write;
            int idx = mIdx[k] + 1; if (idx >= mLen[k]) idx = 0;
            mIdx[k] = idx;
        }

        // wet = mean of taps (soft clamp for safety)
        float wet = 0.25f*(x[0]+x[1]+x[2]+x[3]);
        if (wet >  1.5f) wet =  1.5f;
        if (wet < -1.5f) wet = -1.5f;

        outL[i] = (1.0f - mixL)*inL[i] + mixL*wet;
        outR[i] = (1.0f - mixR)*inR[i] + mixR*wet;

        // denormal guard
        if (std::fabs(outL[i]) < 1e-30f) outL[i] = 0.0f;
        if (std::fabs(outR[i]) < 1e-30f) outR[i] = 0.0f;
    }
}

Plugin* createPlugin() { return new PluginTinyFdnReverb(); }

END_NAMESPACE_DISTRHO
