/*
 * tiny-fdn-reverb UI based on DPF
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#ifndef UI_TINY_FDN_REVERB_HPP
#define UI_TINY_FDN_REVERB_HPP

#include "DistrhoUI.hpp"

START_NAMESPACE_DISTRHO

class UITinyFdnReverb : public UI {
public:
    UITinyFdnReverb();

protected:
    void parameterChanged(uint32_t, float) override {}
    void programLoaded(uint32_t) override {}
    void sampleRateChanged(double) override {}
    void uiIdle() override {}
    void uiReshape(uint, uint) override {}
    void onNanoDisplay() override;

private:
    DISTRHO_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(UITinyFdnReverb)
};

END_NAMESPACE_DISTRHO

#endif // UI_TINY_FDN_REVERB_HPP
