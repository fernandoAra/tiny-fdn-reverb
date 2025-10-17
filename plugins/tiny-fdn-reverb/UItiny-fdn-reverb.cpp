/*
 * tiny-fdn-reverb audio effect based on DISTRHO Plugin Framework (DPF)
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2025 Fernando de Souza Araujo <faraujo0080@gmail.com>
 */

#include "UItiny-fdn-reverb.hpp"
#include "Window.hpp"

START_NAMESPACE_DISTRHO

// -----------------------------------------------------------------------
// Init / Deinit

UItiny-fdn-reverb::UItiny-fdn-reverb()
: UI(600, 400)  {

}

UItiny-fdn-reverb::~UItiny-fdn-reverb() {

}

// -----------------------------------------------------------------------
// DSP/Plugin callbacks

/**
  A parameter has changed on the plugin side.
  This is called by the host to inform the UI about parameter changes.
*/
void UItiny-fdn-reverb::parameterChanged(uint32_t index, float value) {
    switch (index) {
        case Plugintiny-fdn-reverb::paramGain:
            // do something when Gain param is set, such as update a widget
            break;
    }

    (void)value;
}

/**
  A program has been loaded on the plugin side.
  This is called by the host to inform the UI about program changes.
*/
void UItiny-fdn-reverb::programLoaded(uint32_t index) {
    if (index < presetCount) {
        for (int i=0; i < Plugintiny-fdn-reverb::paramCount; i++) {
            // set values for each parameter and update their widgets
            parameterChanged(i, factoryPresets[index].params[i]);
        }
    }
}

/**
  Optional callback to inform the UI about a sample rate change on the plugin side.
*/
void UItiny-fdn-reverb::sampleRateChanged(double newSampleRate) {
    (void)newSampleRate;
}

// -----------------------------------------------------------------------
// Optional UI callbacks

/**
  Idle callback.
  This function is called at regular intervals.
*/
void UItiny-fdn-reverb::uiIdle() {

}

/**
  Window reshape function, called when the parent window is resized.
*/
void UItiny-fdn-reverb::uiReshape(uint width, uint height) {
    (void)width;
    (void)height;
}

// -----------------------------------------------------------------------
// Widget callbacks


/**
  A function called to draw the view contents with NanoVG.
*/
void UItiny-fdn-reverb::onNanoDisplay() {

}


// -----------------------------------------------------------------------
// Optional widget callbacks; return true to stop event propagation, false otherwise.

/**
  A function called when a key is pressed or released.
*/
bool UItiny-fdn-reverb::onKeyboard(const KeyboardEvent& ev) {
    return false;
    (void)ev;
}

/**
  A function called when a mouse button is pressed or released.
*/
bool UItiny-fdn-reverb::onMouse(const MouseEvent& ev) {
    return false;
    (void)ev;
}

/**
  A function called when the mouse pointer moves.
*/
bool UItiny-fdn-reverb::onMotion(const MotionEvent& ev) {
    return false;
    (void)ev;
}

/**
  A function called on scrolling (e.g. mouse wheel or track pad).
*/
bool UItiny-fdn-reverb::onScroll(const ScrollEvent& ev) {
    return false;
    (void)ev;
}

// -----------------------------------------------------------------------

UI* createUI() {
    return new UItiny-fdn-reverb();
}

// -----------------------------------------------------------------------

END_NAMESPACE_DISTRHO
