/*
 * tiny-fdn-reverb audio effect based on DISTRHO Plugin Framework (DPF)
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2025 Fernando de Souza Araujo <faraujo0080@gmail.com>
 */

#ifndef UI_TINY-FDN-REVERB_H
#define UI_TINY-FDN-REVERB_H

#include "DistrhoUI.hpp"
#include "Plugintiny-fdn-reverb.hpp"

START_NAMESPACE_DISTRHO

class UItiny-fdn-reverb : public UI {
public:
    UItiny-fdn-reverb();
    ~UItiny-fdn-reverb();

protected:
    void parameterChanged(uint32_t, float value) override;
    void programLoaded(uint32_t index) override;
    void sampleRateChanged(double newSampleRate) override;

    void uiIdle() override;
    void uiReshape(uint width, uint height) override;

    void onNanoDisplay() override;

    bool onKeyboard(const KeyboardEvent& ev) override;
    bool onMouse(const MouseEvent& ev) override;
    bool onMotion(const MotionEvent& ev) override;
    bool onScroll(const ScrollEvent& ev) override;

private:
    DISTRHO_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(UItiny-fdn-reverb)
};

END_NAMESPACE_DISTRHO

#endif
