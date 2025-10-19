/*
 * tiny-fdn-reverb audio effect based on DISTRHO Plugin Framework (DPF)
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#include "Plugintiny-fdn-reverb.hpp"
#include <cmath>
#include <algorithm>

START_NAMESPACE_DISTRHO

// Simple preset container
struct Preset {
    const char* name;
    float params[PluginTinyFdnReverb::paramCount];
};
static const Preset kPresets[] = {
    // { RT60(s), Mix }
    { "Default", { 1.80f, 0.50f } }
};
static const uint kPresetCount = sizeof(kPresets)/sizeof(kPresets[0]);

// -----------------------------------------------------------------------

PluginTinyFdnReverb::PluginTinyFdnReverb()
    : Plugin(paramCount, kPresetCount, 0)
    , fLastSR(0.0)
    , fRt60(2.80f)  
    , fMix(1.00f)    
    , fMixSmoothL(10.0f, getSampleRate())
    , fMixSmoothR(10.0f, getSampleRate())
{
    // allocate delay buffers once
    for (int i=0; i<kN; ++i) {
        mBuf[i].assign(kMaxDelay, 0.0f);
        mIdx[i] = 0;
        mLen[i] = 64; // temp; real values set in updateDelaysForSR()
        mGain[i] = 0.0f;
    }
}

const char* PluginTinyFdnReverb::getLabel()   const { return "tiny-fdn-reverb"; }
const char* PluginTinyFdnReverb::getMaker()   const { return "Fernando Ara"; }
const char* PluginTinyFdnReverb::getLicense() const { return "MIT"; }
uint32_t    PluginTinyFdnReverb::getVersion() const { return d_version(0,1,0); }
int64_t     PluginTinyFdnReverb::getUniqueId() const { return d_cconst('t','f','d','n'); }

// ---- parameters ----

void PluginTinyFdnReverb::initParameter(uint32_t index, Parameter& p) {
    p.hints = kParameterIsAutomable;
    switch (index) {
    case paramRt60:
        p.name = "RT60 (s)";
        p.symbol = "rt60";
        p.ranges = { 0.20f, 8.00f, 2.80f }; 
        break;
    case paramMix:
        p.name = "Mix";
        p.symbol = "mix";
        p.ranges = { 0.00f, 1.00f, 1.00f };
        break;
    default: break;
    }
}

float PluginTinyFdnReverb::getParameterValue(uint32_t index) const {
    switch (index) {
    case paramRt60: return fRt60;
    case paramMix:  return fMix;
    default: return 0.0f;
    }
}

void PluginTinyFdnReverb::setParameterValue(uint32_t index, float value) {
    switch (index) {
    case paramRt60: fRt60 = value; break;
    case paramMix:  fMix  = value; break;
    default: break;
    }
}

void PluginTinyFdnReverb::initProgramName(uint32_t index, String& programName) {
    if (index < kPresetCount) programName = kPresets[index].name;
}

void PluginTinyFdnReverb::loadProgram(uint32_t index) {
    if (index < kPresetCount) {
        setParameterValue(paramRt60, kPresets[index].params[paramRt60]);
        setParameterValue(paramMix,  kPresets[index].params[paramMix]);
    }
}

// ---- lifecycle ----

void PluginTinyFdnReverb::activate() {
    // clear all buffers and indices
    for (int i=0; i<kN; ++i) {
        std::fill(mBuf[i].begin(), mBuf[i].end(), 0.0f);
        mIdx[i] = 0;
    }
    fLastSR = 0.0; // force SR update on next run()
}

// scale the base delays to current SR and clamp to kMaxDelay
void PluginTinyFdnReverb::updateDelaysForSR(double sr) noexcept {
    const double scale = (sr <= 0.0 ? 1.0 : sr / 48000.0);
    for (int i=0; i<kN; ++i) {
        const int L = std::max(1, std::min(kMaxDelay, int(std::lround(kBase48[i] * scale))));
        mLen[i] = L;
        if (mIdx[i] >= L) mIdx[i] = 0; // keep index valid
    }
}

// from RT60 formula: gain per pass (per line) so that energy decays by 60 dB in RT60 secs
void PluginTinyFdnReverb::updateLineGainsFromRt60(double sr) noexcept {
    const double T60 = std::max(0.05, double(fRt60));
    for (int i=0; i<kN; ++i) {
        const double Li = double(mLen[i]);
        const double gi = std::pow(10.0, -3.0 * (Li / (T60 * sr))); // 60 dB → factor 10^-3 in amplitude
        mGain[i] = float(gi);
    }
}

// ---- processing ----

void PluginTinyFdnReverb::run(const float** inputs, float** outputs, uint32_t frames) {
    const float* inL  = inputs[0];
    const float* inR  = inputs[1];
    float*       outL = outputs[0];
    float*       outR = outputs[1];

    // update SR-dependent stuff once per block
    const double sr = getSampleRate();
    if (sr != fLastSR) {
        updateDelaysForSR(sr);
        fLastSR = sr;
    }
    updateLineGainsFromRt60(sr);

    // smoothed mix per sample
    for (uint32_t i = 0; i < frames; ++i) {
        const float mixTarget = fMix; // 0..1
        const float mixL = fMixSmoothL.process(mixTarget);
        const float mixR = fMixSmoothR.process(mixTarget);

        // --- read current outputs of the delay lines
        float x[kN];
        for (int k=0; k<kN; ++k)
            x[k] = mBuf[k][ mIdx[k] ];

        // --- wet output = average of line outputs
        float wet = 0.0f;
        for (int k=0; k<kN; ++k) wet += x[k];
        wet *= (1.0f / float(kN));

        // --- feedback: Hadamard(4)/2 (orthogonal)
        // H = 0.5 * [[+ + + +],
        //            [+ - + -],
        //            [+ + - -],
        //            [+ - - +]]
        float y[kN];
        const float g0 = mGain[0]*x[0], g1 = mGain[1]*x[1], g2 = mGain[2]*x[2], g3 = mGain[3]*x[3];
        y[0] = 0.5f*(+g0 + g1 + g2 + g3);
        y[1] = 0.5f*(+g0 - g1 + g2 - g3);
        y[2] = 0.5f*(+g0 + g1 - g2 - g3);
        y[3] = 0.5f*(+g0 - g1 - g2 + g3);

        // --- inject input equally (mono sum)
        const float inMono = 0.5f*(inL[i] + inR[i]);
        const float inj = 0.40f; // gentle excitation
        for (int k=0; k<kN; ++k) {
            const float write = y[k] + inj * inMono;
            // write into line then advance pointer
            mBuf[k][ mIdx[k] ] = write;
            int idx = mIdx[k] + 1; if (idx >= mLen[k]) idx = 0;
            mIdx[k] = idx;
        }

        // --- dry/wet mix per channel
        outL[i] = (1.0f - mixL)*inL[i] + mixL*wet;
        outR[i] = (1.0f - mixR)*inR[i] + mixR*wet;

        // very tiny denormal guard (optional; most compilers set DAZ/FTZ now)
        if (std::fabs(outL[i]) < 1e-30f) outL[i] = 0.0f;
        if (std::fabs(outR[i]) < 1e-30f) outR[i] = 0.0f;
    }
}

// Factory
Plugin* createPlugin() { return new PluginTinyFdnReverb(); }

END_NAMESPACE_DISTRHO
