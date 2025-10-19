/*
 * tiny-fdn-reverb UI based on DPF
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#include "UItiny-fdn-reverb.hpp"

START_NAMESPACE_DISTRHO

UITinyFdnReverb::UITinyFdnReverb()
  : UI(420, 180) // initial window size
{
    // No widgets yet; drawing comes later
}

void UITinyFdnReverb::onNanoDisplay()
{
    // No-op for now (keeps UI compilation independent of NanoVG API differences)
}

// UI factory
UI* createUI() { return new UITinyFdnReverb(); }

END_NAMESPACE_DISTRHO
