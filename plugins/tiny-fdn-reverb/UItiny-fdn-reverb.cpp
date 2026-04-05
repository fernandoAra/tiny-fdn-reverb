/*
 * Tiny FDN Reverb — demo UI (NanoVG)
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
#include "UItiny-fdn-reverb.hpp"

#include "NanoVG.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

START_NAMESPACE_DISTRHO
using namespace DGL;

namespace {

constexpr const char* kPluginVersionText = "v1.33";

static void formatSeconds(char* buf, std::size_t size, float value) noexcept
{
    std::snprintf(buf, size, "%.2f s", value);
}

static void formatPercent(char* buf, std::size_t size, float value) noexcept
{
    std::snprintf(buf, size, "%.0f %%", value * 100.f);
}

static void formatHertz(char* buf, std::size_t size, float value) noexcept
{
    std::snprintf(buf, size, "%.0f Hz", value);
}

static void formatMorph(char* buf, std::size_t size, float value) noexcept
{
    std::snprintf(buf, size, "%.0f / 100", value * 100.f);
}

static void formatVector(char* buf, std::size_t size, const std::array<float, 4>& values) noexcept
{
    std::snprintf(buf,
                  size,
                  "[%+.2f %+.2f %+.2f %+.2f]",
                  values[0],
                  values[1],
                  values[2],
                  values[3]);
}

static void buildHouseholderFromU(const std::array<float, 4>& u, float out[4][4]) noexcept
{
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            const float delta = (row == col) ? 1.f : 0.f;
            out[row][col] = delta - 2.f * u[row] * u[col];
        }
    }
}

static bool pointIn(const UITinyFdnReverb::Rect& r, const float x, const float y) noexcept
{
    return r.w > 0.f && r.h > 0.f && x >= r.x && x <= (r.x + r.w) && y >= r.y && y <= (r.y + r.h);
}

static void drawPanel(UITinyFdnReverb* ui,
                      const UITinyFdnReverb::Rect& r,
                      const Color& fill,
                      const Color& stroke,
                      const float radius,
                      const float strokeWidth = 1.f)
{
    ui->beginPath();
    ui->roundedRect(r.x, r.y, r.w, r.h, radius);
    ui->fillColor(fill);
    ui->fill();

    ui->beginPath();
    ui->roundedRect(r.x + 0.5f, r.y + 0.5f, r.w - 1.f, r.h - 1.f, radius);
    ui->strokeColor(stroke);
    ui->strokeWidth(strokeWidth);
    ui->stroke();
}

} // namespace

UITinyFdnReverb::UITinyFdnReverb()
    : UI(1120, 640)
{
    int fontId = -1;

#if defined(__APPLE__)
    const char* candidates[] = {
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Menlo.ttc"
    };
#else
    const char* candidates[] = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    };
#endif

    for (unsigned i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        const int id = createFontFromFile("ui", candidates[i]);
        if (id >= 0) {
            fontId = id;
            break;
        }
    }

    if (fontId >= 0)
        fontFaceId(fontId);

    setGeometryConstraints(static_cast<uint>(fMetrics.minimumWidth),
                           static_cast<uint>(fMetrics.minimumHeight),
                           false);

    fUiTrace.fill(0.f);

#if DISTRHO_PLUGIN_WANT_DIRECT_ACCESS
    fPluginInstance = static_cast<PluginTinyFdnReverb*>(getPluginInstancePointer());
#endif

    layout();
}

const UITinyFdnReverb::SliderSpec* UITinyFdnReverb::getSliderSpecs(std::size_t& count) noexcept
{
    static const SliderSpec kSpecs[] = {
        {"matrix_morph", "Matrix Blend", kNoParam, "plugin.matrix_morph", &UITinyFdnReverb::rMorph, &UITinyFdnReverb::fMorph, formatMorph, 0.f, 1.f},
        {"rt60", "RT60", PluginTinyFdnReverb::paramRt60, "param.rt60", &UITinyFdnReverb::rRt60, &UITinyFdnReverb::fRt60, formatSeconds, 0.20f, 8.00f},
        {"size", "Size", PluginTinyFdnReverb::paramSize, "param.size", &UITinyFdnReverb::rSize, &UITinyFdnReverb::fSize, formatPercent, 0.50f, 2.00f},
        {"damp_hz", "Damp", PluginTinyFdnReverb::paramDampHz, "param.damp_hz", &UITinyFdnReverb::rDamp, &UITinyFdnReverb::fDampHz, formatHertz, 1500.f, 12000.f},
        {"mix", "Mix", PluginTinyFdnReverb::paramMix, "param.mix", &UITinyFdnReverb::rMix, &UITinyFdnReverb::fMix, formatPercent, 0.f, 1.f},
        {"mod_depth", "Mod Depth", PluginTinyFdnReverb::paramModDepth, "param.mod_depth", &UITinyFdnReverb::rMod, &UITinyFdnReverb::fModDepth, formatPercent, 0.f, 1.f},
        {"detune", "Detune", PluginTinyFdnReverb::paramDetune, "param.detune", &UITinyFdnReverb::rDet, &UITinyFdnReverb::fDetune, formatPercent, 0.f, 1.f},
    };

    count = sizeof(kSpecs) / sizeof(kSpecs[0]);
    return kSpecs;
}

const UITinyFdnReverb::ToggleSpec* UITinyFdnReverb::getToggleSpecs(std::size_t& count) noexcept
{
    static const ToggleSpec kSpecs[] = {
        {"householder_mode", "Active Householder", PluginTinyFdnReverb::paramHouseholderMode, "plugin.active_householder_mode", &UITinyFdnReverb::rHouseholderMode, "Fixed", "Diff", &UITinyFdnReverb::fHouseholderMode},
        {"delay_set", "Delay Set", PluginTinyFdnReverb::paramDelaySet, "param.delay_set", &UITinyFdnReverb::rDelayToggle, "Prime", "Spread", &UITinyFdnReverb::fDelaySet},
        {"metal_boost", "Metal Boost", PluginTinyFdnReverb::paramMetalBoost, "param.metal_boost", &UITinyFdnReverb::rMetalToggle, "Off", "On", &UITinyFdnReverb::fMetallic},
    };

    count = sizeof(kSpecs) / sizeof(kSpecs[0]);
    return kSpecs;
}

const UITinyFdnReverb::SliderSpec* UITinyFdnReverb::getSliderSpec(const DragTarget target) const noexcept
{
    std::size_t count = 0u;
    const SliderSpec* specs = getSliderSpecs(count);

    for (std::size_t i = 0; i < count; ++i) {
        if ((target == DragTarget::Morph && specs[i].rect == &UITinyFdnReverb::rMorph) ||
            (target == DragTarget::Rt60 && specs[i].rect == &UITinyFdnReverb::rRt60) ||
            (target == DragTarget::Size && specs[i].rect == &UITinyFdnReverb::rSize) ||
            (target == DragTarget::Damp && specs[i].rect == &UITinyFdnReverb::rDamp) ||
            (target == DragTarget::Mix && specs[i].rect == &UITinyFdnReverb::rMix) ||
            (target == DragTarget::Mod && specs[i].rect == &UITinyFdnReverb::rMod) ||
            (target == DragTarget::Detune && specs[i].rect == &UITinyFdnReverb::rDet)) {
            return &specs[i];
        }
    }

    return nullptr;
}

const char* UITinyFdnReverb::modeLabel(const int mode) noexcept
{
    return mode == 0 ? "Fixed" : "Diff";
}

const char* UITinyFdnReverb::routingLabel(const int mode) noexcept
{
    switch (mode) {
    case 0: return "fixed";
    case 1: return "u-only";
    default: return "full (u+b+c)";
    }
}

void UITinyFdnReverb::layout()
{
    const float W = getWidth();
    const float H = getHeight();
    const float scale = clampf(std::min(W / 1120.f, H / 640.f), 0.92f, 1.18f);

    fMetrics.padding = 16.f * scale;
    fMetrics.gap = 10.f * scale;
    fMetrics.rowHeight = std::max(46.f, 48.f * scale);
    fMetrics.compactRowHeight = std::max(36.f, 38.f * scale);
    fMetrics.panelRadius = 10.f * scale;
    fMetrics.titleSize = std::max(20.f, 21.f * scale);
    fMetrics.labelSize = std::max(14.f, 15.f * scale);
    fMetrics.valueSize = std::max(14.f, 15.f * scale);
    fMetrics.captionSize = std::max(12.f, 13.f * scale);
    fMetrics.sectionTitleSize = std::max(13.f, 14.f * scale);
    fMetrics.trackHeight = std::max(7.f, 8.f * scale);
    fMetrics.knobRadius = std::max(7.f, 8.f * scale);
    fMetrics.minControlWidth = 180.f * scale;
    fMetrics.minControlHeight = 44.f * scale;

    const float pad = fMetrics.padding;
    const float gap = fMetrics.gap;
    const float titleBandH = fMetrics.titleSize + 12.f;
    const float topY = pad + titleBandH;
    const float stateW = std::max(360.f * scale, W * 0.34f);
    const float leftW = W - (pad * 2.f) - gap - stateW;
    const float leftX = pad;
    const float rightX = leftX + leftW + gap;
    const float advancedW = 166.f * scale;
    float houseW = std::min(304.f * scale, leftW * 0.42f);
    float morphW = leftW - houseW - advancedW - (gap * 2.f);
    const float minMorphW = 180.f * scale;
    if (morphW < minMorphW) {
        houseW = std::max(240.f * scale, houseW - (minMorphW - morphW));
        morphW = leftW - houseW - advancedW - (gap * 2.f);
    }

    rHouseholderMode = {leftX, topY, houseW, fMetrics.rowHeight};
    rMorph = {rHouseholderMode.x + rHouseholderMode.w + gap, topY, morphW, fMetrics.rowHeight};
    rAdvancedBtn = {W - pad - advancedW, topY, advancedW, fMetrics.rowHeight};

    const float contentY = topY + fMetrics.rowHeight + gap;
    const float contentH = H - contentY - pad;

    rActiveStatePanel = {rightX, contentY, stateW, contentH};

    const float statePad = pad * 0.85f;
    const float summaryTop = rActiveStatePanel.y + statePad + fMetrics.sectionTitleSize + gap;
    const float summaryH = 194.f * scale;
    const float matrixY = summaryTop + summaryH + gap;
    const float matrixAreaH = rActiveStatePanel.h - (matrixY - rActiveStatePanel.y) - statePad;
    const float matrixH = std::max(112.f * scale, (matrixAreaH - gap) * 0.5f);
    rActiveStateMatrix = {rActiveStatePanel.x + statePad,
                          matrixY,
                          rActiveStatePanel.w - (statePad * 2.f),
                          matrixH};
    rHouseholderDeltaMatrix = {rActiveStateMatrix.x,
                               rActiveStateMatrix.y + rActiveStateMatrix.h + gap,
                               rActiveStateMatrix.w,
                               matrixH};

    rAdvancedPanel = {};
    rPresetStrip = {};
    rDelayToggle = {};
    rMetalToggle = {};
    rRt60 = {};
    rSize = {};
    rDamp = {};
    rMix = {};
    rMod = {};
    rDet = {};
    rPing = {};
    rBurst = {};

    float traceTop = contentY;

    if (fShowAdvanced) {
        const float panelPad = pad * 0.85f;
        const float innerW = leftW - (panelPad * 2.f);
        const float halfW = (innerW - gap) * 0.5f;
        const float buttonW = 70.f * scale;
        const float presetW = innerW - (buttonW * 2.f) - (gap * 2.f);
        const float titleH = fMetrics.sectionTitleSize + gap;
        const float rowY0 = contentY + panelPad + titleH;
        const float sliderH = fMetrics.rowHeight;
        const float compactH = fMetrics.compactRowHeight;
        const float advancedH = (panelPad * 2.f) + titleH + compactH + compactH + sliderH + sliderH + sliderH + (gap * 4.f);

        rAdvancedPanel = {leftX, contentY, leftW, advancedH};
        rPresetStrip = {leftX + panelPad, rowY0, presetW, compactH};
        rBurst = {rPresetStrip.x + rPresetStrip.w + gap, rowY0, buttonW, compactH};
        rPing = {rBurst.x + rBurst.w + gap, rowY0, buttonW, compactH};

        const float toggleY = rowY0 + compactH + gap;
        rDelayToggle = {leftX + panelPad, toggleY, halfW, compactH};
        rMetalToggle = {rDelayToggle.x + halfW + gap, toggleY, halfW, compactH};

        const float row1Y = toggleY + compactH + gap;
        rRt60 = {leftX + panelPad, row1Y, halfW, sliderH};
        rSize = {rRt60.x + halfW + gap, row1Y, halfW, sliderH};

        const float row2Y = row1Y + sliderH + gap;
        rDamp = {leftX + panelPad, row2Y, halfW, sliderH};
        rMix = {rDamp.x + halfW + gap, row2Y, halfW, sliderH};

        const float row3Y = row2Y + sliderH + gap;
        rMod = {leftX + panelPad, row3Y, halfW, sliderH};
        rDet = {rMod.x + halfW + gap, row3Y, halfW, sliderH};

        traceTop = rAdvancedPanel.y + rAdvancedPanel.h + gap;
    }

    rTracePanel = {leftX, traceTop, leftW, contentY + contentH - traceTop};
    const float tracePad = pad * 0.85f;
    rTraceMetrics = {rTracePanel.x + tracePad,
                     rTracePanel.y + tracePad + fMetrics.sectionTitleSize + gap,
                     rTracePanel.w - (tracePad * 2.f),
                     std::max(44.f, 52.f * scale)};
    rTracePlot = {rTracePanel.x + tracePad,
                  rTraceMetrics.y + rTraceMetrics.h + gap,
                  rTracePanel.w - (tracePad * 2.f),
                  std::max(96.f, rTracePanel.h - (rTraceMetrics.y + rTraceMetrics.h + gap - rTracePanel.y) - tracePad)};
}

void UITinyFdnReverb::onResize(const ResizeEvent& ev)
{
    UI::onResize(ev);
    layout();
}

void UITinyFdnReverb::uiIdle()
{
    using clock = std::chrono::steady_clock;
    const clock::time_point now = clock::now();

    if (!fUiTickInit) {
        fLastUiTick = now;
        fUiTickInit = true;
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - fLastUiTick).count();
    if (elapsed >= 33) {
        pullTraceSamples();
        fLastUiTick = now;
        repaint();
    }
}

void UITinyFdnReverb::applyMatrixMorphFromUI(const float value) noexcept
{
    const float v = clampf(value, 0.f, 1.f);
    fMorph = v;
    fMatrixType = (v < 0.5f) ? 0 : 1;
    fIsMorphing = (v > 0.01f && v < 0.99f);

#if DISTRHO_PLUGIN_WANT_DIRECT_ACCESS
    if (fPluginInstance == nullptr)
        fPluginInstance = static_cast<PluginTinyFdnReverb*>(getPluginInstancePointer());

    if (fPluginInstance != nullptr)
        fPluginInstance->setMatrixMorphFromUI(v);
#endif
}

void UITinyFdnReverb::applyHouseholderModeFromUI(const int mode) noexcept
{
    const int requested = mode != 0 ? 1 : 0;
    fPendingHouseholderMode = requested;

    beginEdit(PluginTinyFdnReverb::paramHouseholderMode);
    setParam(PluginTinyFdnReverb::paramHouseholderMode, static_cast<float>(requested));
    endEdit(PluginTinyFdnReverb::paramHouseholderMode);
}

void UITinyFdnReverb::pushTraceSample(const float value) noexcept
{
    fUiTrace[fUiTraceWrite & (kUiTraceSize - 1u)] = clampf(value, 0.f, 1.f);
    ++fUiTraceWrite;
    if (fUiTraceCount < kUiTraceSize)
        ++fUiTraceCount;
}

void UITinyFdnReverb::pullTraceSamples() noexcept
{
#if DISTRHO_PLUGIN_WANT_DIRECT_ACCESS
    if (fPluginInstance == nullptr)
        fPluginInstance = static_cast<PluginTinyFdnReverb*>(getPluginInstancePointer());

    if (fPluginInstance != nullptr) {
        fPluginInstance->copyUiStateSnapshot(fPluginState);
        fMorph = clampf(fPluginState.matrixMorphTarget, 0.f, 1.f);
        fAppliedMorph = clampf(fPluginState.matrixMorphApplied, 0.f, 1.f);
        fMatrixType = (fMorph < 0.5f) ? 0 : 1;
        fIsMorphing = (fAppliedMorph > 0.01f && fAppliedMorph < 0.99f);
        fHouseholderMode = fPluginState.householderMode;
        fDiffRoutingMode = fPluginState.diffRoutingMode;

        if (fPendingHouseholderMode == fHouseholderMode)
            fPendingHouseholderMode = -1;

        const uint32_t writeIndex = fPluginInstance->getEnvTraceWriteIndex();
        if (!fTraceReadInit) {
            fTraceReadCursor = (writeIndex > kUiTraceSize) ? (writeIndex - kUiTraceSize) : 0u;
            fTraceReadInit = true;
        }

        uint32_t unread = writeIndex - fTraceReadCursor;
        if (unread > kUiTraceSize) {
            fTraceReadCursor = writeIndex - kUiTraceSize;
            unread = kUiTraceSize;
        }

        for (uint32_t i = 0; i < unread; ++i) {
            pushTraceSample(fPluginInstance->getEnvTraceValue(fTraceReadCursor));
            ++fTraceReadCursor;
        }

        fEDTms = fPluginInstance->getMeterEDTms();
        fRT60est = fPluginInstance->getMeterRT60s();
        fDen100 = fPluginInstance->getMeterDensity100();
        fDen300 = fPluginInstance->getMeterDensity300();
        fRinginess = fPluginInstance->getMeterRinginess();
        fWetEnv = fPluginInstance->getMeterWetEnv();
        return;
    }
#endif

    pushTraceSample(fWetEnv);
}

void UITinyFdnReverb::parameterChanged(const uint32_t index, const float value)
{
    switch (index) {
    case PluginTinyFdnReverb::paramRt60:
        fRt60 = value;
        break;
    case PluginTinyFdnReverb::paramMix:
        fMix = value;
        break;
    case PluginTinyFdnReverb::paramMatrixType:
        fMatrixType = (int(std::lround(value)) >= 1) ? 1 : 0;
        break;
    case PluginTinyFdnReverb::paramDelaySet:
        fDelaySet = int(std::lround(value));
        break;
    case PluginTinyFdnReverb::paramSize:
        fSize = value;
        break;
    case PluginTinyFdnReverb::paramDampHz:
        fDampHz = value;
        break;
    case PluginTinyFdnReverb::paramMatrixMorph:
        fMorph = clampf(value, 0.f, 1.f);
        fAppliedMorph = fMorph;
        fMatrixType = (fMorph < 0.5f) ? 0 : 1;
        fIsMorphing = (fMorph > 0.01f && fMorph < 0.99f);
        break;
    case PluginTinyFdnReverb::paramModDepth:
        fModDepth = value;
        break;
    case PluginTinyFdnReverb::paramDetune:
        fDetune = value;
        break;
    case PluginTinyFdnReverb::paramMetalBoost:
        fMetallic = int(std::lround(value));
        break;
    case PluginTinyFdnReverb::paramEDTms:
        fEDTms = value;
        break;
    case PluginTinyFdnReverb::paramRT60est:
        fRT60est = value;
        break;
    case PluginTinyFdnReverb::paramDensity100ms:
        fDen100 = value;
        break;
    case PluginTinyFdnReverb::paramDensity300ms:
        fDen300 = value;
        break;
    case PluginTinyFdnReverb::paramRinginess:
        fRinginess = value;
        break;
    case PluginTinyFdnReverb::paramWetEnv:
        fWetEnv = value;
        break;
    case PluginTinyFdnReverb::paramHouseholderMode:
        fHouseholderMode = (int(std::lround(value)) >= 1) ? 1 : 0;
        if (fPendingHouseholderMode == fHouseholderMode)
            fPendingHouseholderMode = -1;
        break;
    default:
        break;
    }
}

void UITinyFdnReverb::drawSectionTitle(const Rect& r, const char* title)
{
    fontSize(fMetrics.sectionTitleSize);
    fillColor(Color(70, 74, 82));
    textAlign(ALIGN_LEFT | ALIGN_MIDDLE);
    text(r.x, r.y, title, nullptr);
}

void UITinyFdnReverb::drawValueRow(const float x,
                                   const float y,
                                   const float w,
                                   const char* label,
                                   const char* value,
                                   const bool emphasize)
{
    fontSize(fMetrics.captionSize);
    fillColor(Color(112, 117, 125));
    textAlign(ALIGN_LEFT | ALIGN_MIDDLE);
    text(x, y, label, nullptr);

    fontSize(emphasize ? fMetrics.valueSize + 1.f : fMetrics.valueSize);
    fillColor(Color(34, 38, 43));
    textAlign(ALIGN_RIGHT | ALIGN_MIDDLE);
    text(x + w, y, value, nullptr);
}

void UITinyFdnReverb::drawSmallButton(const Rect& r, const char* label, const Color& fill)
{
    drawPanel(this, r, fill, Color(0, 0, 0, 20), fMetrics.panelRadius * 0.75f);
    fontSize(fMetrics.labelSize - 1.f);
    fillColor(Color(255, 255, 255));
    textAlign(ALIGN_CENTER | ALIGN_MIDDLE);
    text(r.x + r.w * 0.5f, r.y + r.h * 0.5f, label, nullptr);
}

void UITinyFdnReverb::drawSliderControl(const SliderSpec& spec)
{
    const Rect& r = this->*spec.rect;
    if (r.w <= 0.f || r.h <= 0.f)
        return;

    drawPanel(this, r, Color(247, 248, 250), Color(214, 219, 224), fMetrics.panelRadius);

    const float value = this->*spec.value;
    char valueText[64];
    spec.formatter(valueText, sizeof(valueText), value);

    const float pad = 10.f;
    const float titleY = r.y + 14.f;
    fontSize(fMetrics.labelSize);
    fillColor(Color(58, 63, 71));
    textAlign(ALIGN_LEFT | ALIGN_MIDDLE);
    text(r.x + pad, titleY, spec.label, nullptr);

    fontSize(fMetrics.valueSize);
    fillColor(Color(33, 37, 41));
    textAlign(ALIGN_RIGHT | ALIGN_MIDDLE);
    text(r.x + r.w - pad, titleY, valueText, nullptr);

    const float t = clampf((value - spec.minValue) / (spec.maxValue - spec.minValue), 0.f, 1.f);
    const float trackX = r.x + pad;
    const float trackW = r.w - (pad * 2.f);
    const float trackY = r.y + r.h - 16.f;
    const float knobX = trackX + (trackW * t);

    beginPath();
    roundedRect(trackX, trackY, trackW, fMetrics.trackHeight, fMetrics.trackHeight * 0.5f);
    fillColor(Color(220, 224, 229));
    fill();

    beginPath();
    roundedRect(trackX, trackY, std::max(fMetrics.trackHeight, knobX - trackX), fMetrics.trackHeight, fMetrics.trackHeight * 0.5f);
    fillColor(Color(68, 131, 219));
    fill();

    beginPath();
    circle(knobX, trackY + (fMetrics.trackHeight * 0.5f), fMetrics.knobRadius);
    fillColor(Color(25, 36, 52));
    fill();
}

void UITinyFdnReverb::drawToggleControl(const ToggleSpec& spec)
{
    const Rect& r = this->*spec.rect;
    if (r.w <= 0.f || r.h <= 0.f)
        return;

    drawPanel(this, r, Color(247, 248, 250), Color(214, 219, 224), fMetrics.panelRadius);

    fontSize(fMetrics.labelSize);
    fillColor(Color(58, 63, 71));
    textAlign(ALIGN_LEFT | ALIGN_MIDDLE);
    text(r.x + 10.f, r.y + 14.f, spec.label, nullptr);

    const int state = this->*spec.state;
    const float bodyY = r.y + 22.f;
    const float bodyH = r.h - 28.f;
    const float halfW = r.w * 0.5f;
    const Rect left = {r.x + 6.f, bodyY, halfW - 8.f, bodyH};
    const Rect right = {r.x + halfW + 2.f, bodyY, halfW - 8.f, bodyH};

    drawPanel(this,
              left,
              state == 0 ? Color(68, 131, 219) : Color(227, 231, 236),
              state == 0 ? Color(53, 116, 203) : Color(214, 219, 224),
              fMetrics.panelRadius * 0.75f);
    drawPanel(this,
              right,
              state == 1 ? Color(68, 131, 219) : Color(227, 231, 236),
              state == 1 ? Color(53, 116, 203) : Color(214, 219, 224),
              fMetrics.panelRadius * 0.75f);

    const int pending = (spec.paramIndex == PluginTinyFdnReverb::paramHouseholderMode) ? fPendingHouseholderMode : -1;
    if (pending >= 0 && pending != state) {
        const Rect& target = pending == 0 ? left : right;
        beginPath();
        roundedRect(target.x + 1.5f, target.y + 1.5f, target.w - 3.f, target.h - 3.f, fMetrics.panelRadius * 0.65f);
        strokeColor(Color(218, 146, 45));
        strokeWidth(2.f);
        stroke();
    }

    fontSize(fMetrics.valueSize);
    textAlign(ALIGN_CENTER | ALIGN_MIDDLE);
    fillColor(state == 0 ? Color(255, 255, 255) : Color(58, 63, 71));
    text(left.x + left.w * 0.5f, left.y + left.h * 0.5f, spec.optionA, nullptr);
    fillColor(state == 1 ? Color(255, 255, 255) : Color(58, 63, 71));
    text(right.x + right.w * 0.5f, right.y + right.h * 0.5f, spec.optionB, nullptr);
}

void UITinyFdnReverb::drawEnvelopeTrace(const Rect& r)
{
    beginPath();
    roundedRect(r.x, r.y, r.w, r.h, fMetrics.panelRadius * 0.75f);
    fillColor(Color(252, 252, 253));
    fill();

    const float x0 = r.x + 8.f;
    const float y0 = r.y + 8.f;
    const float w = r.w - 16.f;
    const float h = r.h - 16.f;

    for (int i = 1; i < 4; ++i) {
        const float yy = y0 + (h * (float(i) / 4.f));
        beginPath();
        moveTo(x0, yy);
        lineTo(x0 + w, yy);
        strokeColor(Color(230, 234, 239));
        strokeWidth(1.f);
        stroke();
    }

    if (fUiTraceCount >= 2u) {
        const uint32_t count = fUiTraceCount;
        const uint32_t start = (fUiTraceWrite - count) & (kUiTraceSize - 1u);

        beginPath();
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t idx = (start + i) & (kUiTraceSize - 1u);
            const float v = std::sqrt(clampf(fUiTrace[idx] * 12.f, 0.f, 1.f));
            const float x = x0 + (count > 1u ? (w * float(i) / float(count - 1u)) : 0.f);
            const float y = y0 + h - (v * h);
            if (i == 0u)
                moveTo(x, y);
            else
                lineTo(x, y);
        }
        strokeColor(Color(52, 114, 204));
        strokeWidth(2.2f);
        stroke();
    }
}

void UITinyFdnReverb::buildActiveMatrix(float out[4][4]) const noexcept
{
    static const float kHadamard[4][4] = {
        {+0.5f, +0.5f, +0.5f, +0.5f},
        {+0.5f, -0.5f, +0.5f, -0.5f},
        {+0.5f, +0.5f, -0.5f, -0.5f},
        {+0.5f, -0.5f, -0.5f, +0.5f},
    };

    float householder[4][4];
    buildHouseholderFromU(fPluginState.activeU, householder);

    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            out[row][col] = (1.f - fAppliedMorph) * kHadamard[row][col]
                          + (fAppliedMorph * householder[row][col]);
        }
    }
}

void UITinyFdnReverb::buildHouseholderDeltaMatrix(float out[4][4]) const noexcept
{
    float fixedHouseholder[4][4];
    float activeHouseholder[4][4];
    buildHouseholderFromU(fPluginState.fixedU, fixedHouseholder);
    buildHouseholderFromU(fPluginState.activeU, activeHouseholder);

    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col)
            out[row][col] = activeHouseholder[row][col] - fixedHouseholder[row][col];
    }
}

void UITinyFdnReverb::drawMatrixHeatmap(const Rect& r, const float matrix[4][4], const float range)
{
    beginPath();
    roundedRect(r.x, r.y, r.w, r.h, fMetrics.panelRadius * 0.75f);
    fillColor(Color(250, 250, 251));
    fill();

    const float leftPad = 26.f;
    const float topPad = 18.f;
    const float cellW = (r.w - leftPad) / 4.f;
    const float cellH = (r.h - topPad) / 4.f;

    for (int i = 0; i < 4; ++i) {
        char axis[8];
        std::snprintf(axis, sizeof(axis), "L%d", i + 1);
        fontSize(fMetrics.captionSize);
        fillColor(Color(112, 117, 125));
        textAlign(ALIGN_CENTER | ALIGN_MIDDLE);
        text(r.x + leftPad + (cellW * (i + 0.5f)), r.y + 8.f, axis, nullptr);
        textAlign(ALIGN_RIGHT | ALIGN_MIDDLE);
        text(r.x + 20.f, r.y + topPad + (cellH * (i + 0.5f)), axis, nullptr);
    }

    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            const float value = matrix[row][col];
            const float norm = clampf(std::fabs(value) / std::max(range, 1e-6f), 0.f, 1.f);
            const int red = value < 0.f ? int(220.f * norm) : int(54.f + (20.f * (1.f - norm)));
            const int green = value < 0.f ? int(92.f + (70.f * (1.f - norm))) : int(130.f + (60.f * (1.f - norm)));
            const int blue = value > 0.f ? int(220.f * norm) : int(66.f + (40.f * (1.f - norm)));
            const Rect cell = {r.x + leftPad + (cellW * col) + 2.f,
                               r.y + topPad + (cellH * row) + 2.f,
                               cellW - 4.f,
                               cellH - 4.f};

            drawPanel(this, cell, Color(red, green, blue, 220), Color(210, 214, 220), fMetrics.panelRadius * 0.45f);

            char cellText[32];
            std::snprintf(cellText, sizeof(cellText), "%+.2f", value);
            fontSize(fMetrics.captionSize);
            fillColor(Color(255, 255, 255));
            textAlign(ALIGN_CENTER | ALIGN_MIDDLE);
            text(cell.x + cell.w * 0.5f, cell.y + cell.h * 0.5f, cellText, nullptr);
        }
    }
}

void UITinyFdnReverb::drawStateSummaryLine(const float x,
                                           const float y,
                                           const float w,
                                           const char* label,
                                           const char* value)
{
    fontSize(fMetrics.captionSize);
    fillColor(Color(90, 96, 105));
    textAlign(ALIGN_LEFT | ALIGN_MIDDLE);
    text(x, y, label, nullptr);
    fontSize(fMetrics.captionSize);
    fillColor(Color(34, 38, 43));
    textAlign(ALIGN_RIGHT | ALIGN_MIDDLE);
    text(x + w, y, value, nullptr);
}

float UITinyFdnReverb::vectorDistance(const std::array<float, 4>& a, const std::array<float, 4>& b) noexcept
{
    float sum = 0.f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

void UITinyFdnReverb::drawActiveStatePanel()
{
    drawPanel(this, rActiveStatePanel, Color(242, 245, 248), Color(210, 216, 222), fMetrics.panelRadius);

    const float pad = fMetrics.padding * 0.85f;
    drawSectionTitle({rActiveStatePanel.x + pad, rActiveStatePanel.y + pad, 0.f, 0.f}, "ACTIVE STATE");

    fontSize(fMetrics.captionSize);
    fillColor(Color(102, 108, 117));
    textAlign(ALIGN_LEFT | ALIGN_MIDDLE);
    text(rActiveStatePanel.x + pad,
         rActiveStatePanel.y + pad + fMetrics.sectionTitleSize + 12.f,
         "RT60 comes from per-line gains. Diff changes Householder u and, for full presets, b/c routing.",
         nullptr);

    const DiffPreset* preset = PluginTinyFdnReverb::getDiffPresetByIndex(fPluginState.diffPresetIndex);

    char presetText[160];
    std::snprintf(presetText, sizeof(presetText), "%s", preset != nullptr ? preset->configId : "none");

    char statusText[160];
    if (fPluginState.householderMode == 0) {
        std::snprintf(statusText,
                      sizeof(statusText),
                      "%s",
                      preset != nullptr
                          ? "Fixed baseline active; learned preset is loaded but not engaged."
                          : "Fixed baseline active; no learned Diff preset matched this context.");
    } else if (preset == nullptr) {
        std::snprintf(statusText,
                      sizeof(statusText),
                      "%s",
                      "Diff requested, but no learned preset matched; active structure stays fixed.");
    } else if (fPluginState.diffRoutingMode == 1) {
        std::snprintf(statusText,
                      sizeof(statusText),
                      "%s",
                      "Diff active: learned Householder u is engaged; b/c remain at the fixed baseline.");
    } else {
        std::snprintf(statusText,
                      sizeof(statusText),
                      "%s",
                      "Diff active: learned Householder u plus learned b/c routing are engaged.");
    }

    char blendText[96];
    std::snprintf(blendText,
                  sizeof(blendText),
                  "Hadamard %.0f%% / Householder %.0f%%",
                  double((1.f - fAppliedMorph) * 100.f),
                  double(fAppliedMorph * 100.f));

    char uVectorText[128];
    char bVectorText[128];
    char cLVectorText[128];
    char cRVectorText[128];
    formatVector(uVectorText, sizeof(uVectorText), fPluginState.activeU);
    formatVector(bVectorText, sizeof(bVectorText), fPluginState.activeB);
    formatVector(cLVectorText, sizeof(cLVectorText), fPluginState.activeCL);
    formatVector(cRVectorText, sizeof(cRVectorText), fPluginState.activeCR);

    char uDistanceText[64];
    char bDistanceText[64];
    char cLDistanceText[64];
    char cRDistanceText[64];
    std::snprintf(uDistanceText,
                  sizeof(uDistanceText),
                  "%.4f",
                  double(vectorDistance(fPluginState.activeU, fPluginState.fixedU)));
    std::snprintf(bDistanceText,
                  sizeof(bDistanceText),
                  "%.4f",
                  double(vectorDistance(fPluginState.activeB, fPluginState.fixedB)));
    std::snprintf(cLDistanceText,
                  sizeof(cLDistanceText),
                  "%.4f",
                  double(vectorDistance(fPluginState.activeCL, fPluginState.fixedCL)));
    std::snprintf(cRDistanceText,
                  sizeof(cRDistanceText),
                  "%.4f",
                  double(vectorDistance(fPluginState.activeCR, fPluginState.fixedCR)));

    const float summaryX = rActiveStatePanel.x + pad;
    const float summaryW = rActiveStatePanel.w - (pad * 2.f);
    const float summaryGap = 12.f;
    const float summaryColW = (summaryW - summaryGap) * 0.5f;
    const float rowStep = fMetrics.captionSize + 8.f;
    float y = rActiveStatePanel.y + pad + fMetrics.sectionTitleSize + fMetrics.gap + 24.f;
    drawValueRow(summaryX, y, summaryColW, "Mode", modeLabel(fPluginState.householderMode), true);
    drawValueRow(summaryX + summaryColW + summaryGap, y, summaryColW, "Routing", routingLabel(fPluginState.diffRoutingMode));
    y += rowStep;
    drawStateSummaryLine(summaryX, y, summaryW, "Status", statusText);
    y += rowStep;
    drawStateSummaryLine(summaryX, y, summaryW, "Config ID", presetText);
    y += rowStep;
    drawStateSummaryLine(summaryX, y, summaryW, "DSP blend", blendText);
    y += rowStep;
    drawStateSummaryLine(summaryX, y, summaryW, "u active", uVectorText);
    y += rowStep;
    drawStateSummaryLine(summaryX, y, summaryW, "b active", bVectorText);
    y += rowStep;
    drawStateSummaryLine(summaryX, y, summaryW, "cL active", cLVectorText);
    y += rowStep;
    drawStateSummaryLine(summaryX, y, summaryW, "cR active", cRVectorText);
    y += rowStep;
    drawStateSummaryLine(summaryX, y, summaryColW, "||u_active - u_fixed||", uDistanceText);
    drawStateSummaryLine(summaryX + summaryColW + summaryGap, y, summaryColW, "||b_active - b_fixed||", bDistanceText);
    y += rowStep;
    drawStateSummaryLine(summaryX, y, summaryColW, "||cL_active - cL_fixed||", cLDistanceText);
    drawStateSummaryLine(summaryX + summaryColW + summaryGap, y, summaryColW, "||cR_active - cR_fixed||", cRDistanceText);

    float matrix[4][4];
    buildActiveMatrix(matrix);
    drawMatrixHeatmap(rActiveStateMatrix, matrix, 1.f);
    fontSize(fMetrics.captionSize);
    fillColor(Color(102, 108, 117));
    textAlign(ALIGN_LEFT | ALIGN_MIDDLE);
    text(rActiveStateMatrix.x,
         rActiveStateMatrix.y - 28.f,
         "Panel A: Active blended matrix used by DSP",
         nullptr);
    text(rActiveStateMatrix.x,
         rActiveStateMatrix.y - 14.f,
         "Includes current Hadamard/Householder blend, so morph can visually dominate Diff here.",
         nullptr);

    float deltaMatrix[4][4];
    buildHouseholderDeltaMatrix(deltaMatrix);
    float deltaRange = 0.f;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col)
            deltaRange = std::max(deltaRange, std::fabs(deltaMatrix[row][col]));
    }
    drawMatrixHeatmap(rHouseholderDeltaMatrix, deltaMatrix, std::max(deltaRange, 0.25f));
    text(rHouseholderDeltaMatrix.x,
         rHouseholderDeltaMatrix.y - 28.f,
         "Panel B: Householder delta vs fixed",
         nullptr);
    text(rHouseholderDeltaMatrix.x,
         rHouseholderDeltaMatrix.y - 14.f,
         "Pure structural change from active u; independent of RT60 and scaled to show the delta clearly.",
         nullptr);
}

void UITinyFdnReverb::drawTracePanel()
{
    drawPanel(this, rTracePanel, Color(244, 245, 247), Color(210, 216, 222), fMetrics.panelRadius);

    const float pad = fMetrics.padding * 0.85f;
    drawSectionTitle({rTracePanel.x + pad, rTracePanel.y + pad, 0.f, 0.f}, "ENVELOPE + METRICS");

    const struct MetricCard {
        const char* label;
        char value[32];
    } cards[] = {
        {"EDT", ""},
        {"RT60(est)", ""},
        {"Density 100", ""},
        {"Ringiness", ""},
    };

    MetricCard metricCards[4];
    std::memcpy(metricCards, cards, sizeof(cards));
    std::snprintf(metricCards[0].value, sizeof(metricCards[0].value), fEDTms > 0.f ? "%.0f ms" : "n/a", fEDTms);
    std::snprintf(metricCards[1].value, sizeof(metricCards[1].value), fRT60est > 0.f ? "%.2f s" : "n/a", fRT60est);
    std::snprintf(metricCards[2].value, sizeof(metricCards[2].value), "%.2f ev/ms", fDen100);
    std::snprintf(metricCards[3].value, sizeof(metricCards[3].value), "%.2f", fRinginess);

    const float cardGap = 8.f;
    const float cardW = (rTraceMetrics.w - (cardGap * 3.f)) / 4.f;
    for (int i = 0; i < 4; ++i) {
        const Rect card = {rTraceMetrics.x + (cardW + cardGap) * i, rTraceMetrics.y, cardW, rTraceMetrics.h};
        drawPanel(this, card, Color(250, 250, 251), Color(216, 220, 225), fMetrics.panelRadius * 0.7f);
        fontSize(fMetrics.captionSize);
        fillColor(Color(108, 113, 121));
        textAlign(ALIGN_LEFT | ALIGN_MIDDLE);
        text(card.x + 10.f, card.y + 13.f, metricCards[i].label, nullptr);
        fontSize(fMetrics.valueSize);
        fillColor(Color(33, 37, 41));
        textAlign(ALIGN_LEFT | ALIGN_MIDDLE);
        text(card.x + 10.f, card.y + card.h - 13.f, metricCards[i].value, nullptr);
    }

    drawEnvelopeTrace(rTracePlot);

    char densityText[64];
    std::snprintf(densityText, sizeof(densityText), "Density 300 ms: %.2f ev/ms", fDen300);
    fontSize(fMetrics.captionSize);
    fillColor(Color(96, 102, 110));
    textAlign(ALIGN_LEFT | ALIGN_MIDDLE);
    text(rTracePlot.x, rTracePlot.y - 12.f, densityText, nullptr);

    textAlign(ALIGN_LEFT | ALIGN_MIDDLE);
    text(rTracePlot.x,
         rTracePlot.y + rTracePlot.h + 14.f,
         "Fixed/Diff highlight follows plugin state; amber outline marks a pending click until state syncs.",
         nullptr);
}

void UITinyFdnReverb::updateSliderDrag(const SliderSpec& spec, const float x) noexcept
{
    const Rect& r = this->*spec.rect;
    const float t = clampf((x - (r.x + 10.f)) / (r.w - 20.f), 0.f, 1.f);
    const float value = spec.minValue + (t * (spec.maxValue - spec.minValue));

    if (spec.paramIndex == kNoParam) {
        applyMatrixMorphFromUI(value);
        return;
    }

    this->*spec.value = value;
    setParam(spec.paramIndex, value);
}

void UITinyFdnReverb::onNanoDisplay()
{
    const float W = getWidth();
    const float H = getHeight();

    beginPath();
    rect(0.f, 0.f, W, H);
    fillColor(Color(249, 249, 247));
    fill();

    beginPath();
    rect(0.f, 0.f, W, fMetrics.padding + fMetrics.titleSize + 18.f);
    fillColor(Color(241, 244, 247));
    fill();

    fontSize(fMetrics.titleSize);
    fillColor(Color(28, 33, 39));
    textAlign(ALIGN_LEFT | ALIGN_MIDDLE);
    text(fMetrics.padding,
         fMetrics.padding + (fMetrics.titleSize * 0.5f),
         "Tiny FDN Reverb Demo UI",
         nullptr);

    fontSize(fMetrics.captionSize);
    fillColor(Color(103, 109, 118));
    textAlign(ALIGN_LEFT | ALIGN_MIDDLE);
    text(fMetrics.padding,
         fMetrics.padding + fMetrics.titleSize + 2.f,
         "UI truth comes from plugin state; active matrix view is read-only and live.",
         nullptr);

    std::size_t toggleCount = 0u;
    const ToggleSpec* toggleSpecs = getToggleSpecs(toggleCount);
    for (std::size_t i = 0; i < toggleCount; ++i)
        drawToggleControl(toggleSpecs[i]);

    std::size_t sliderCount = 0u;
    const SliderSpec* sliderSpecs = getSliderSpecs(sliderCount);
    for (std::size_t i = 0; i < sliderCount; ++i)
        drawSliderControl(sliderSpecs[i]);

    drawSmallButton(rAdvancedBtn,
                    fShowAdvanced ? "Advanced ON" : "Advanced OFF",
                    fShowAdvanced ? Color(73, 133, 220) : Color(124, 132, 143));

    if (fShowAdvanced) {
        drawPanel(this, rAdvancedPanel, Color(241, 243, 246), Color(210, 216, 222), fMetrics.panelRadius);
        const float pad = fMetrics.padding * 0.85f;
        drawSectionTitle({rAdvancedPanel.x + pad, rAdvancedPanel.y + pad, 0.f, 0.f}, "ADVANCED CONTROLS");

        drawPanel(this, rPresetStrip, Color(248, 249, 251), Color(214, 219, 224), fMetrics.panelRadius * 0.8f);
        const char* labels[4] = {"H+Prime", "H+Spread", "House+Prime", "House+Spread"};
        const float segW = rPresetStrip.w / 4.f;
        for (int i = 0; i < 4; ++i) {
            const bool active = ((i < 2 ? 0 : 1) == fMatrixType) && ((i % 2) == fDelaySet);
            const Rect seg = {rPresetStrip.x + segW * i + 2.f, rPresetStrip.y + 2.f, segW - 4.f, rPresetStrip.h - 4.f};
            drawPanel(this,
                      seg,
                      active ? Color(68, 131, 219) : Color(230, 234, 239),
                      active ? Color(53, 116, 203) : Color(214, 219, 224),
                      fMetrics.panelRadius * 0.6f);
            fontSize(fMetrics.captionSize);
            fillColor(active ? Color(255, 255, 255) : Color(58, 63, 71));
            textAlign(ALIGN_CENTER | ALIGN_MIDDLE);
            text(seg.x + seg.w * 0.5f, seg.y + seg.h * 0.5f, labels[i], nullptr);
        }

        drawSmallButton(rBurst, "Burst", Color(222, 118, 91));
        drawSmallButton(rPing, "Ping", Color(113, 95, 206));
    }

    drawActiveStatePanel();
    drawTracePanel();

    fontSize(fMetrics.captionSize);
    fillColor(Color(103, 109, 118));
    textAlign(ALIGN_RIGHT | ALIGN_MIDDLE);
    text(W - fMetrics.padding, H - fMetrics.padding * 0.5f, kPluginVersionText, nullptr);
}

bool UITinyFdnReverb::onMouse(const MouseEvent& ev)
{
    const float x = ev.pos.getX();
    const float y = ev.pos.getY();

    if (ev.press) {
        if (pointIn(rAdvancedBtn, x, y)) {
            if (fDragging != DragTarget::None) {
                const SliderSpec* spec = getSliderSpec(fDragging);
                if (spec != nullptr && spec->paramIndex != kNoParam)
                    endEdit(spec->paramIndex);
                fDragging = DragTarget::None;
            }
            fShowAdvanced = !fShowAdvanced;
            layout();
            repaint();
            return true;
        }

        std::size_t toggleCount = 0u;
        const ToggleSpec* toggleSpecs = getToggleSpecs(toggleCount);
        for (std::size_t i = 0; i < toggleCount; ++i) {
            const ToggleSpec& spec = toggleSpecs[i];
            const Rect& r = this->*spec.rect;
            if (!pointIn(r, x, y))
                continue;

            const int nextState = (x < (r.x + r.w * 0.5f)) ? 0 : 1;
            if (spec.paramIndex == PluginTinyFdnReverb::paramHouseholderMode) {
                applyHouseholderModeFromUI(nextState);
            } else if (spec.paramIndex == PluginTinyFdnReverb::paramDelaySet) {
                fDelaySet = nextState;
                beginEdit(spec.paramIndex);
                setParam(spec.paramIndex, float(nextState));
                endEdit(spec.paramIndex);
            } else if (spec.paramIndex == PluginTinyFdnReverb::paramMetalBoost) {
                fMetallic = nextState;
                beginEdit(spec.paramIndex);
                setParam(spec.paramIndex, float(nextState));
                endEdit(spec.paramIndex);
            }

            repaint();
            return true;
        }

        if (fShowAdvanced && pointIn(rPresetStrip, x, y)) {
            const float segW = rPresetStrip.w / 4.f;
            const int index = int((x - rPresetStrip.x) / segW);
            if (index >= 0 && index < 4) {
                const int matrixChoice = index < 2 ? 0 : 1;
                const int delayChoice = index % 2;
                beginEdit(PluginTinyFdnReverb::paramDelaySet);
                setParam(PluginTinyFdnReverb::paramDelaySet, float(delayChoice));
                endEdit(PluginTinyFdnReverb::paramDelaySet);
                fDelaySet = delayChoice;
                applyMatrixMorphFromUI(float(matrixChoice));
                repaint();
            }
            return true;
        }

        if (fShowAdvanced && pointIn(rPing, x, y)) {
            beginEdit(PluginTinyFdnReverb::paramPing);
            setParam(PluginTinyFdnReverb::paramPing, 1.0f);
            endEdit(PluginTinyFdnReverb::paramPing);
            return true;
        }

        if (fShowAdvanced && pointIn(rBurst, x, y)) {
            beginEdit(PluginTinyFdnReverb::paramExciteNoise);
            setParam(PluginTinyFdnReverb::paramExciteNoise, 1.0f);
            endEdit(PluginTinyFdnReverb::paramExciteNoise);
            return true;
        }

        const DragTarget dragOrder[] = {
            DragTarget::Morph,
            DragTarget::Rt60,
            DragTarget::Size,
            DragTarget::Damp,
            DragTarget::Mix,
            DragTarget::Mod,
            DragTarget::Detune,
        };

        for (const DragTarget target : dragOrder) {
            const SliderSpec* spec = getSliderSpec(target);
            if (spec == nullptr)
                continue;

            const Rect& r = this->*spec->rect;
            if (!pointIn(r, x, y))
                continue;

            fDragging = target;
            if (spec->paramIndex != kNoParam)
                beginEdit(spec->paramIndex);
            updateSliderDrag(*spec, x);
            repaint();
            return true;
        }
    } else if (fDragging != DragTarget::None) {
        const SliderSpec* spec = getSliderSpec(fDragging);
        if (spec != nullptr && spec->paramIndex != kNoParam)
            endEdit(spec->paramIndex);
        fDragging = DragTarget::None;
        return true;
    }

    return false;
}

bool UITinyFdnReverb::onMotion(const MotionEvent& ev)
{
    if (fDragging == DragTarget::None)
        return false;

    const SliderSpec* spec = getSliderSpec(fDragging);
    if (spec == nullptr)
        return false;

    updateSliderDrag(*spec, ev.pos.getX());
    repaint();
    return true;
}

// Manual smoke-test checklist:
// 1. Click Fixed/Diff and confirm the blue highlight follows the plugin-reported active mode.
// 2. Change Householder mode from the host or reopen the UI and confirm the toggle/state panel stay in sync.
// 3. Verify labels remain readable at the default 1120x640 size and at the enforced minimum size.
// 4. Resize the UI within constraints and confirm no overlap/clipping in the top bar, active state panel, or trace panel.
// 5. Switch Fixed/Diff and confirm the Active State panel updates its routing/config text and matrix/vector display.

UI* createUI()
{
    return new UITinyFdnReverb();
}

END_NAMESPACE_DISTRHO
