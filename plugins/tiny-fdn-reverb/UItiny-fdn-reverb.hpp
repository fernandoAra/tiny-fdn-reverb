/*
 * Tiny FDN Reverb — demo UI (NanoVG)
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
#ifndef UI_TINY_FDN_REVERB_HPP
#define UI_TINY_FDN_REVERB_HPP

#include "DistrhoUI.hpp"
#include "Plugintiny-fdn-reverb.hpp"
#include <array>
#include <chrono>
#include <cstdint>

START_NAMESPACE_DISTRHO

class UITinyFdnReverb : public UI {
public:
    UITinyFdnReverb();

protected:
    void parameterChanged(uint32_t index, float value) override;
    void programLoaded(uint32_t) override {}
    void uiIdle() override;
    void onNanoDisplay() override;
    bool onMouse(const MouseEvent& ev) override;
    bool onMotion(const MotionEvent& ev) override;
    void onResize(const ResizeEvent& ev) override;

public:
    struct Rect {
        float x = 0.f;
        float y = 0.f;
        float w = 0.f;
        float h = 0.f;

        Rect() = default;
        Rect(float xIn, float yIn, float wIn, float hIn)
            : x(xIn), y(yIn), w(wIn), h(hIn) {}
    };

private:
    struct UiMetrics {
        float padding = 16.f;
        float gap = 12.f;
        float rowHeight = 48.f;
        float compactRowHeight = 38.f;
        float panelRadius = 10.f;
        float titleSize = 21.f;
        float labelSize = 15.f;
        float valueSize = 15.f;
        float captionSize = 13.f;
        float sectionTitleSize = 14.f;
        float trackHeight = 8.f;
        float knobRadius = 8.f;
        float minControlWidth = 180.f;
        float minControlHeight = 44.f;
        float minimumWidth = 1040.f;
        float minimumHeight = 620.f;
    };

    using ValueFormatter = void (*)(char*, std::size_t, float);

    static constexpr uint32_t kNoParam = UINT32_MAX;

    struct SliderSpec {
        const char* name;
        const char* label;
        uint32_t paramIndex;
        const char* stateSource;
        Rect UITinyFdnReverb::* rect;
        float UITinyFdnReverb::* value;
        ValueFormatter formatter;
        float minValue;
        float maxValue;
    };

    struct ToggleSpec {
        const char* name;
        const char* label;
        uint32_t paramIndex;
        const char* stateSource;
        Rect UITinyFdnReverb::* rect;
        const char* optionA;
        const char* optionB;
        int UITinyFdnReverb::* state;
    };

    enum class DragTarget {
        None,
        Rt60,
        Size,
        Damp,
        Mix,
        Morph,
        Mod,
        Detune
    };

    float fRt60 = 2.8f;
    float fMix = 1.0f;
    int fMatrixType = 0;
    int fHouseholderMode = 0;
    int fDiffRoutingMode = 0;
    int fDelaySet = 0;
    float fSize = 1.0f;
    float fDampHz = 6000.0f;
    float fMorph = 0.0f;
    float fAppliedMorph = 0.0f;
    bool fIsMorphing = false;
    float fModDepth = 0.0f;
    float fDetune = 0.0f;
    int fMetallic = 0;
    float fEDTms = 0.f;
    float fRT60est = 0.f;
    float fDen100 = 0.f;
    float fDen300 = 0.f;
    float fRinginess = 0.f;
    float fWetEnv = 0.f;
    bool fShowAdvanced = false;
    int fPendingHouseholderMode = -1;

    PluginTinyFdnReverb::UiStateSnapshot fPluginState{};

    DragTarget fDragging = DragTarget::None;

    UiMetrics fMetrics{};
    Rect rHouseholderMode{};
    Rect rAdvancedBtn{};
    Rect rActiveStatePanel{};
    Rect rActiveStateMatrix{};
    Rect rHouseholderDeltaMatrix{};
    Rect rTracePanel{};
    Rect rTracePlot{};
    Rect rTraceMetrics{};
    Rect rAdvancedPanel{};
    Rect rPresetStrip{};
    Rect rDelayToggle{};
    Rect rMetalToggle{};
    Rect rRt60{};
    Rect rSize{};
    Rect rDamp{};
    Rect rMix{};
    Rect rMod{};
    Rect rDet{};
    Rect rMorph{};
    Rect rPing{};
    Rect rBurst{};

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

    void layout();
    void pullTraceSamples() noexcept;
    void applyMatrixMorphFromUI(float value) noexcept;
    void applyHouseholderModeFromUI(int mode) noexcept;
    void pushTraceSample(float value) noexcept;
    void drawSliderControl(const SliderSpec& spec);
    void drawToggleControl(const ToggleSpec& spec);
    void drawEnvelopeTrace(const Rect& r);
    void drawActiveStatePanel();
    void drawTracePanel();
    void drawSectionTitle(const Rect& r, const char* title);
    void drawValueRow(float x, float y, float w, const char* label, const char* value, bool emphasize = false);
    void drawSmallButton(const Rect& r, const char* label, const Color& fill);
    void drawMatrixHeatmap(const Rect& r, const float matrix[4][4], float range = 1.f);
    void drawStateSummaryLine(float x, float y, float w, const char* label, const char* value);
    void buildActiveMatrix(float out[4][4]) const noexcept;
    void buildHouseholderDeltaMatrix(float out[4][4]) const noexcept;
    const SliderSpec* getSliderSpec(DragTarget target) const noexcept;
    static const SliderSpec* getSliderSpecs(std::size_t& count) noexcept;
    static const ToggleSpec* getToggleSpecs(std::size_t& count) noexcept;
    void updateSliderDrag(const SliderSpec& spec, float x) noexcept;
    static float vectorDistance(const std::array<float, 4>& a, const std::array<float, 4>& b) noexcept;

    static float clampf(float v, float a, float b) noexcept
    {
        return v < a ? a : (v > b ? b : v);
    }

    static const char* modeLabel(int mode) noexcept;
    static const char* routingLabel(int mode) noexcept;

    void beginEdit(uint32_t idx) { editParameter(idx, true); }
    void endEdit(uint32_t idx) { editParameter(idx, false); }
    void setParam(uint32_t idx, float v) { setParameterValue(idx, v); }

    DISTRHO_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(UITinyFdnReverb)
};

END_NAMESPACE_DISTRHO
#endif
