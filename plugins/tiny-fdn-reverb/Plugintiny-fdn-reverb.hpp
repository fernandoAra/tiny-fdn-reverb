/*
 * tiny-fdn-reverb audio effect based on DISTRHO Plugin Framework (DPF)
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2025 Fernando de Souza Araujo <faraujo0080@gmail.com>
 */

#ifndef PLUGIN_TINY-FDN-REVERB_H
#define PLUGIN_TINY-FDN-REVERB_H

#include "DistrhoPlugin.hpp"
#include "CParamSmooth.hpp"

START_NAMESPACE_DISTRHO

#ifndef MIN
#define MIN(a,b) ( (a) < (b) ? (a) : (b) )
#endif

#ifndef MAX
#define MAX(a,b) ( (a) > (b) ? (a) : (b) )
#endif

#ifndef CLAMP
#define CLAMP(v, min, max) (MIN((max), MAX((min), (v))))
#endif

#ifndef DB_CO
#define DB_CO(g) ((g) > -90.0f ? powf(10.0f, (g) * 0.05f) : 0.0f)
#endif

// -----------------------------------------------------------------------

class Plugintiny-fdn-reverb : public Plugin {
public:
    enum Parameters {
        paramGain = 0,
        paramCount
    };

    Plugintiny-fdn-reverb();

    ~Plugintiny-fdn-reverb();

protected:
    // -------------------------------------------------------------------
    // Information

    const char* getLabel() const noexcept override {
        return "tiny-fdn-reverb";
    }

    const char* getDescription() const override {
        return "Small-order FDN reverb (N=4–8) that tames metallic ring via echo-density targeting and AM-safe unitary modulation.";
    }

    const char* getMaker() const noexcept override {
        return "N/A";
    }

    const char* getHomePage() const override {
        return "https://codeberg.org/fernandoAra/tiny-fdn-reverb#tiny-fdn-reverb";
    }

    const char* getLicense() const noexcept override {
        return "https://spdx.org/licenses/Apache-2.0";
    }

    uint32_t getVersion() const noexcept override {
        return d_version(0, 1, 0);
    }

    // Go to:
    //
    // http://service.steinberg.de/databases/plugin.nsf/plugIn
    //
    // Get a proper plugin UID and fill it in here!
    int64_t getUniqueId() const noexcept override {
        return d_cconst('a', 'b', 'c', 'd');
    }

    // -------------------------------------------------------------------
    // Init

    void initParameter(uint32_t index, Parameter& parameter) override;
    void initProgramName(uint32_t index, String& programName) override;

    // -------------------------------------------------------------------
    // Internal data

    float getParameterValue(uint32_t index) const override;
    void setParameterValue(uint32_t index, float value) override;
    void loadProgram(uint32_t index) override;

    // -------------------------------------------------------------------
    // Optional

    // Optional callback to inform the plugin about a sample rate change.
    void sampleRateChanged(double newSampleRate) override;

    // -------------------------------------------------------------------
    // Process

    void activate() override;

    void run(const float**, float** outputs, uint32_t frames) override;


    // -------------------------------------------------------------------

private:
    float           fParams[paramCount];
    double          fSampleRate;
    float           gain;
    CParamSmooth    *smooth_gain;

    DISTRHO_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Plugintiny-fdn-reverb)
};

struct Preset {
    const char* name;
    float params[Plugintiny-fdn-reverb::paramCount];
};

const Preset factoryPresets[] = {
    {
        "Unity Gain",
        {0.0f}
    }
    //,{
    //    "Another preset",  // preset name
    //    {-14.0f, ...}      // array of presetCount float param values
    //}
};

const uint presetCount = sizeof(factoryPresets) / sizeof(Preset);

// -----------------------------------------------------------------------

END_NAMESPACE_DISTRHO

#endif  // #ifndef PLUGIN_TINY-FDN-REVERB_H
