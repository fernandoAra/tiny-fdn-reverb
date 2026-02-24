/*
 * Tiny FDN Reverb — minimal UI (NanoVG)
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
#ifndef UI_TINY_FDN_REVERB_HPP
#define UI_TINY_FDN_REVERB_HPP

#include "DistrhoUI.hpp"
#include "Plugintiny-fdn-reverb.hpp"
#include <array>
#include <chrono>

// [BOILERPLATE: DPF namespace macros]
START_NAMESPACE_DISTRHO

// [BOILERPLATE: DPF UI class inherits UI base; layout logic below is project-specific]
class UITinyFdnReverb : public UI {
public:
    UITinyFdnReverb();

protected:
    // plugin → UI updates
    void parameterChanged(uint32_t index, float value) override;
    void programLoaded(uint32_t) override {}
    void uiIdle() override;

    // drawing
    void onNanoDisplay() override;
    // mouse input
    bool onMouse(const MouseEvent& ev) override;
    bool onMotion(const MotionEvent& ev) override;

public:
    struct Rect { float x,y,w,h; };

private:
    // local copies of parameters
    float fRt60 = 2.8f;
    float fMix  = 1.0f;
    int   fMatrixType = 0; // 0 Hadamard, 1 Householder
    int   fHouseholderMode = 0; // 0 fixed, 1 diff
    int   fDelaySet   = 0; // 0 Prime,    1 Spread
    float fSize    = 1.0f;
    float fDampHz  = 6000.0f;
    float fMorph   = 0.0f;
    bool  fIsMorphing = false;
    

    // NEW controls
    float fModDepth = 0.0f;
    float fDetune   = 0.0f;
    int   fMetallic = 0;
    // Outputs
    float fEDTms = 0.f, fRT60est = 0.f, fDen100 = 0.f, fDen300 = 0.f;
    float fRinginess = 0.f;
    float fWetEnv = 0.f;

    // Layering: default view + advanced panel
    bool  fShowAdvanced = false;

    // dragging state
    enum DragTarget { DRAG_NONE, DRAG_RT60, DRAG_MIX, DRAG_SIZE, DRAG_DAMP, DRAG_MORPH, DRAG_MOD, DRAG_DETUNE } fDragging = DRAG_NONE;

    // layout rects
    Rect rLayerMatrix{}, rAdvancedBtn{};
    Rect rMatrix{}, rDelay{};
    Rect rRt60{}, rSize{}, rDamp{}, rMix{}, rDecay{};
    Rect rPreset{}, rMatH{}, rMatHo{}, rMorph{};
    Rect rPing{}, rBurst{}, rMetal{}, rMod{}, rDet{};
    Rect rRing{};

    // UI-side trace history used for visualization at UI refresh rate.
    static constexpr uint32_t kUiTraceSize = 256;
    std::array<float, kUiTraceSize> fUiTrace{};
    uint32_t fUiTraceWrite = 0;
    uint32_t fUiTraceCount = 0;
    std::chrono::steady_clock::time_point fLastUiTick{};
    bool fUiTickInit = false;
    uint32_t fTraceReadCursor = 0;
    bool fTraceReadInit = false;
#if DISTRHO_PLUGIN_WANT_DIRECT_ACCESS
    PluginTinyFdnReverb* fPluginInstance = nullptr;
#endif

    // layout + drawing helpers
    void layout();
    void pullTraceSamples() noexcept;
    void applyMatrixMorphFromUI(float value) noexcept;
    void applyHouseholderModeFromUI(int mode) noexcept;
    void pushTraceSample(float value) noexcept;
    void drawEnvelopeTrace(const Rect& r);
    void drawSlider(const Rect& r, const char* label, float v, float vmin, float vmax);
    void drawToggle(const Rect& r, const char* label, const char* a, const char* b, int v);
    void drawDecay(const Rect& r, float rt60);
    void drawRingMeter(const Rect& r, float v);
    static float clampf(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }

    // notify host when editing
    void beginEdit(uint32_t idx) { editParameter(idx, true); }
    void endEdit  (uint32_t idx) { editParameter(idx, false); }
    void setParam (uint32_t idx, float v) { setParameterValue(idx, v); }

    // [BOILERPLATE: DPF leak detector macro]
    DISTRHO_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(UITinyFdnReverb)
};

END_NAMESPACE_DISTRHO
#endif
