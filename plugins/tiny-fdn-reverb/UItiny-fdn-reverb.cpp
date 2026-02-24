/*
 * Tiny FDN Reverb — minimal UI (NanoVG)
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
#include "UItiny-fdn-reverb.hpp"
#include "NanoVG.hpp"   // NVGcontext + nvg* API
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <mutex>
// Uncomment to enable heavy logging (NOT RT-safe!)
// #define TFDN_ENABLE_LOG 1

#if defined(TFDN_ENABLE_LOG)
static FILE* gTFDNLog = nullptr;
static std::once_flag gTFDNOnce;
static void tfdn_open_log() { gTFDNLog = std::fopen("/tmp/tfdn.log", "a"); }
#define DBG(...) do { \
    std::call_once(gTFDNOnce, tfdn_open_log); \
    if (gTFDNLog) { std::fprintf(gTFDNLog, __VA_ARGS__); std::fprintf(gTFDNLog, "\n"); std::fflush(gTFDNLog);} \
} while(0)
#else
#define DBG(...) do {} while(0)
#endif

// [BOILERPLATE: DPF namespace macros]
START_NAMESPACE_DISTRHO
using namespace DGL; // nvg* symbols

UITinyFdnReverb::UITinyFdnReverb()
: UI(1040, 520) // +40px for extra row
{
    // Try a few common macOS fonts; pick the first that loads.
    int fid = -1;

#if defined(__APPLE__)
    const char* cands[] = {
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf"
    };
#else
    const char* cands[] = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    };
#endif

    for (unsigned i = 0; i < sizeof(cands)/sizeof(cands[0]); ++i) {
        const int id = createFontFromFile("ui", cands[i]);
        if (id >= 0) { fid = id; break; }
    }
    if (fid >= 0) {
        fontFaceId(fid);   // select the loaded font
    }

    layout();
    fUiTrace.fill(0.0f);
#if DISTRHO_PLUGIN_WANT_DIRECT_ACCESS
    fPluginInstance = static_cast<const PluginTinyFdnReverb*>(getPluginInstancePointer());
#endif
}

void UITinyFdnReverb::layout() {
    const float PAD = 16.f;
    const float W = getWidth(), H = getHeight();

    const float headerH = 28.f;
    rLayerMatrix = { PAD, 4.f, 250.f, 20.f };
    rAdvancedBtn = { W - PAD - 128.f, 4.f, 128.f, 20.f };

    rPreset = { PAD, headerH + 8.f, W - 2*PAD, 28.f };

    // Two-column layout
    const float COL_GAP = 16.f;
    const float leftW  = (W - 3*PAD) * 0.55f;        // ~55% for controls
    const float rightW = (W - 3*PAD) - leftW;        // rest for visuals
    const float leftX  = PAD;
    const float rightX = PAD + leftW + COL_GAP;

    // Row heights
    const float TOG_H = 28.f;
    const float SL_H  = 28.f;

    // Buttons
    rPing  = { rPreset.x + rPreset.w - 70.f, rPreset.y + 4.f, 62.f, rPreset.h - 8.f };
    rBurst = { rPing.x - 72.f, rPing.y, 62.f, rPing.h };

    // Top toggles (left column)
    rMatrix = { leftX, rPreset.y + rPreset.h + 10.f, (leftW - PAD)/2.f, TOG_H };
    rDelay  = { rMatrix.x + rMatrix.w + 10.f, rMatrix.y, (leftW - PAD)/2.f, TOG_H };
    rMetal  = { leftX, rMatrix.y + rMatrix.h + 10.f, leftW, TOG_H };

    // Sliders (left column)
    rRt60 = { leftX, rMetal.y + rMetal.h + 10.f, leftW, SL_H };
    rSize = { leftX, rRt60.y + rRt60.h + 10.f,   leftW, SL_H };
    rDamp = { leftX, rSize.y + rSize.h + 10.f,   leftW, SL_H };
    rMix  = { leftX, rDamp.y + rDamp.h + 10.f,   leftW, SL_H };
    rMod  = { leftX, rMix.y  + rMix.h  + 10.f,   leftW, SL_H };
    rDet  = { leftX, rMod.y  + rMod.h  + 10.f,   leftW, SL_H };

    // Decay panel (left column bottom)
    const float decH = 100.f;
    rDecay = { leftX, H - decH - 2*PAD, leftW, decH };
    rRing  = { leftX, rDecay.y + rDecay.h + 8.f, leftW, 22.f };

    // Right column: two matrix tiles stacked
    const float tileH = (H - rPreset.y - rPreset.h - 4*PAD) * 0.45f;
    rMatH  = { rightX, rMatrix.y, rightW, tileH };
    rMatHo = { rightX, rMatH.y + rMatH.h + 10.f, rightW, tileH };

    // Morph slider under right tiles
    rMorph = { rightX, rMatHo.y + rMatHo.h + 10.f, rightW, SL_H };
}

void UITinyFdnReverb::uiIdle()
{
    using clock = std::chrono::steady_clock;
    const clock::time_point now = clock::now();
    if (! fUiTickInit) {
        fLastUiTick = now;
        fUiTickInit = true;
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - fLastUiTick).count();
    if (elapsed >= 33) { // ~30 Hz refresh
        pullTraceSamples();
        fLastUiTick = now;
        repaint();
    }
}

void UITinyFdnReverb::pushTraceSample(float value) noexcept
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
        fPluginInstance = static_cast<const PluginTinyFdnReverb*>(getPluginInstancePointer());

    if (fPluginInstance != nullptr) {
        const uint32_t writeIndex = fPluginInstance->getEnvTraceWriteIndex();
        if (! fTraceReadInit) {
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
        return;
    }
#endif
    pushTraceSample(fWetEnv);
}

void UITinyFdnReverb::drawEnvelopeTrace(const Rect& r)
{
    beginPath();
    roundedRect(r.x, r.y, r.w, r.h, 6.f);
    fillColor(Color(245,245,245));
    fill();

    beginPath();
    roundedRect(r.x + 8.f, r.y + 8.f, r.w - 16.f, r.h - 16.f, 5.f);
    fillColor(Color(252,252,252));
    fill();

    const float x0 = r.x + 12.f;
    const float y0 = r.y + 12.f;
    const float w  = r.w - 24.f;
    const float h  = r.h - 24.f;

    beginPath();
    moveTo(x0, y0 + h * 0.75f);
    lineTo(x0 + w, y0 + h * 0.75f);
    strokeColor(Color(228,228,228));
    strokeWidth(1.f);
    stroke();

    beginPath();
    moveTo(x0, y0 + h * 0.50f);
    lineTo(x0 + w, y0 + h * 0.50f);
    strokeColor(Color(220,220,220));
    strokeWidth(1.f);
    stroke();

    beginPath();
    moveTo(x0, y0 + h * 0.25f);
    lineTo(x0 + w, y0 + h * 0.25f);
    strokeColor(Color(228,228,228));
    strokeWidth(1.f);
    stroke();

    if (fUiTraceCount >= 2u) {
        const uint32_t count = fUiTraceCount;
        const uint32_t start = (fUiTraceWrite - count) & (kUiTraceSize - 1u);

        beginPath();
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t idx = (start + i) & (kUiTraceSize - 1u);
            const float v = std::sqrt(clampf(fUiTrace[idx] * 12.f, 0.f, 1.f));
            const float x = x0 + (count > 1u ? (w * float(i) / float(count - 1u)) : 0.f);
            const float y = y0 + h - v * h;
            if (i == 0u) moveTo(x, y); else lineTo(x, y);
        }
        strokeColor(Color(35, 120, 210));
        strokeWidth(2.f);
        stroke();
    }

    fontSize(12.f);
    fillColor(Color(60,60,60));
    textAlign(ALIGN_LEFT | ALIGN_MIDDLE);
    text(r.x + 10.f, r.y - 8.f, "Wet Tail Envelope (block RMS)", nullptr);
}


void UITinyFdnReverb::parameterChanged(uint32_t index, float value) {
    switch (index) {
    case PluginTinyFdnReverb::paramRt60:       fRt60 = value; break;
    case PluginTinyFdnReverb::paramMix:        fMix = value; break;
    case PluginTinyFdnReverb::paramMatrixType: fMatrixType = int(std::lround(value)); break;
    case PluginTinyFdnReverb::paramDelaySet:   fDelaySet   = int(std::lround(value)); break;
    case PluginTinyFdnReverb::paramSize:         fSize   = value; break;
    case PluginTinyFdnReverb::paramDampHz:       fDampHz = value; break;
    case PluginTinyFdnReverb::paramMatrixMorph:  fMorph  = value; break;

    case PluginTinyFdnReverb::paramModDepth:  fModDepth = value; break;
    case PluginTinyFdnReverb::paramDetune:    fDetune   = value; break;
    case PluginTinyFdnReverb::paramMetalBoost:fMetallic = int(std::lround(value)); break;

    case PluginTinyFdnReverb::paramEDTms:         fEDTms     = value; break;
    case PluginTinyFdnReverb::paramRT60est:       fRT60est   = value; break;
    case PluginTinyFdnReverb::paramDensity100ms:  fDen100    = value; break;
    case PluginTinyFdnReverb::paramDensity300ms:  fDen300    = value; break;
    case PluginTinyFdnReverb::paramRinginess:     fRinginess = value; break;
    case PluginTinyFdnReverb::paramWetEnv:        fWetEnv    = value; break;
    default: break;
    }
}

static void drawLabel(UITinyFdnReverb* self, float x, float y, const char* txt,
                      float size=30.f, int r=20, int g=20, int b=20)
{
    self->fontSize(size);
    self->fillColor(r,g,b);
    self->textAlign(UITinyFdnReverb::ALIGN_LEFT | UITinyFdnReverb::ALIGN_MIDDLE);
    self->text(x, y, txt, nullptr);
}

// simple rounded-rect background
static void drawPanel(UITinyFdnReverb* self, float x, float y, float w, float h, int rr=6,
                      int r=235, int g=235, int b=235)
{
    self->beginPath();
    self->roundedRect(x, y, w, h, float(rr));
    self->fillColor(r,g,b);
    self->fill();
}

static void drawDensityBars(UITinyFdnReverb* self,
                            const UITinyFdnReverb::Rect& r, float d100, float d300)
{
    const float maxD = 50.f;
    const float barW = 18.f;
    const float gap  = 10.f;
    const float baseY = r.y + r.h - 12.f;
    const float x0 = r.x + r.w - (2*barW + gap) - 10.f;

    auto clamp01 = [](float v){ return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); };

    auto bar = [&](float x, float d, const char* lab) {
        const float t = clamp01(d / maxD);
        const float h = (r.h - 28.f) * t;
        self->beginPath();
        self->roundedRect(x, baseY - h, barW, h, 3.f);
        self->fillColor(Color(120,180,255));
        self->fill();
        self->fontSize(11.f);
        self->fillColor(Color(50,50,50));
        self->textAlign(UITinyFdnReverb::ALIGN_CENTER | UITinyFdnReverb::ALIGN_MIDDLE);
        self->text(x + barW*0.5f, baseY + 8.f, lab, nullptr);
    };

    bar(x0,                d100, "100");
    bar(x0 + barW + gap,   d300, "300");
}

void UITinyFdnReverb::drawSlider(const Rect& r, const char* label,
                                 float v, float vmin, float vmax)
{
    // panel
    drawPanel(this, r.x, r.y, r.w, r.h, 6, 235,235,235);

    const float t = (v - vmin) / (vmax - vmin);
    const float knobX = r.x + 8 + (r.w - 16) * clampf(t,0.f,1.f);

    // track
    beginPath();
    roundedRect(r.x+8, r.y + r.h*0.5f - 3, r.w-16, 6, 3);
    fillColor(Color(210,210,210));
    fill();

    // fill
    beginPath();
    roundedRect(r.x+8, r.y + r.h*0.5f - 3, (knobX - (r.x+8)), 6, 3);
    fillColor(Color(120,180,255));
    fill();

    // knob
    beginPath();
    circle(knobX, r.y + r.h*0.5f, 9.f);
    fillColor(Color(40,40,40));
    fill();

    // left label + right value (with units)
    char buf[64];
    if (std::strcmp(label, "RT60") == 0)          std::snprintf(buf, sizeof(buf), "%.2f s", v);
    else if (std::strcmp(label, "Damp (Hz)") == 0)std::snprintf(buf, sizeof(buf), "%.0f Hz", v);
    else                                          std::snprintf(buf, sizeof(buf), "%.0f %%", v*100.f);

    drawLabel(this, r.x + 10,         r.y - 4, label, 13.f, 50,50,50);
    drawLabel(this, r.x + r.w - 70.f, r.y - 4, buf,   13.f, 50,50,50);
}

void UITinyFdnReverb::drawToggle(const Rect& r, const char* label,
                                 const char* a, const char* b, int v)
{
    drawPanel(this, r.x, r.y, r.w, r.h, 6, 235,235,235);

    const float half = r.w * 0.5f;

    // left option
    beginPath();
    roundedRect(r.x+4, r.y+4, half-8, r.h-8, 4.f);
    fillColor(v==0 ? 120:210, v==0 ? 180:210, v==0 ? 255:210);
    fill();

    // right option
    beginPath();
    roundedRect(r.x+half+4, r.y+4, half-8, r.h-8, 4.f);
    fillColor(v==1 ? 120:210, v==1 ? 180:210, v==1 ? 255:210);
    fill();

    drawLabel(this, r.x+10, r.y-8, label, 12.f, 60,60,60);
    drawLabel(this, r.x+half*0.5f - 20, r.y + r.h*0.5f, a, 12.f);
    drawLabel(this, r.x+half + half*0.5f - 24, r.y + r.h*0.5f, b, 12.f);
}

void UITinyFdnReverb::drawDecay(const Rect& r, float rt60)
{
    drawPanel(this, r.x, r.y, r.w, r.h, 6, 245,245,245);

    const float tMax = 2.0f;
    const int   N = 100;

    beginPath();
    for (int i=0; i<=N; ++i) {
        const float t = tMax * float(i) / float(N);
        const float e = std::exp(-t * 6.91f / std::max(0.05f, rt60));
        const float x = r.x + (t/tMax) * r.w;
        const float y = r.y + (1.0f - e) * r.h;
        if (i==0) moveTo(x, y); else lineTo(x, y);
    }

    // EDT marker as a dot on the analytic curve at t = EDT/1000
    if (rt60 > 0.01f) {
        const float tEdt = std::max(0.f, std::min(2.0f, fEDTms/1000.f)); // needs fEDTms accessible here
        const float e    = std::exp(-tEdt * 6.91f / std::max(0.05f, rt60));
        const float x    = r.x + (tEdt/2.0f) * r.w;
        const float y    = r.y + (1.0f - e) * r.h;
        beginPath(); circle(x, y, 3.f); fillColor(Color(220,80,80)); fill();
    }
    strokeWidth(2.f);
    strokeColor(40,40,40);
    stroke();

    drawLabel(this, r.x+10, r.y-8, "Decay (analytic)", 12.f, 60,60,60);
}

// ✅ Re-add the matrix tile helper
static void drawMatrixTile(UITinyFdnReverb* self, const UITinyFdnReverb::Rect& r,
                           const char* title, const float m[4][4], bool active)
{
    // panel with subtle highlight if active
    self->beginPath();
    self->roundedRect(r.x, r.y, r.w, r.h, 6.f);
    self->fillColor(active ? Color(238,246,255) : Color(245,245,245));
    self->fill();

    // title
    self->fontSize(14.f);
    self->fillColor(Color(40,40,40));
    self->textAlign(UITinyFdnReverb::ALIGN_LEFT | UITinyFdnReverb::ALIGN_MIDDLE);
    self->text(r.x + 10.f, r.y + 14.f, title, nullptr);

    // grid
    const float gx = r.x + 10.f, gy = r.y + 28.f;
    const float gw = r.w - 20.f, gh = r.h - 38.f;
    const float cw = gw / 4.f,   ch = gh / 4.f;

    for (int i=0;i<4;++i) for (int j=0;j<4;++j) {
        const float v = m[i][j]; // expect ±0.5
        const int R = (v < 0.f) ? int(180.f * (-v / 0.5f)) : 0;
        const int B = (v > 0.f) ? int(180.f * ( v / 0.5f)) : 0;
        self->beginPath();
        self->rect(gx + j*cw, gy + i*ch, cw-2.f, ch-2.f);
        self->fillColor(Color(240 - R/3, 240 - (R+B)/6, 240 - B/3));
        self->fill();
        // sign marker
        self->fontSize(12.f);
        self->fillColor(Color(50,50,50));
        self->textAlign(UITinyFdnReverb::ALIGN_CENTER | UITinyFdnReverb::ALIGN_MIDDLE);
        const char* s = (v >= 0.f) ? "+" : "–";
        self->text(gx + j*cw + cw*0.5f, gy + i*ch + ch*0.5f, s, nullptr);
    }
}

// Ringiness meter: 0..1 bar with label
void UITinyFdnReverb::drawRingMeter(const Rect& r, float v)
{
    // background panel
    drawPanel(this, r.x, r.y, r.w, r.h, 6, 235,235,235);

    // label
    char buf[64];
    std::snprintf(buf, sizeof(buf), "Ringiness: %.2f", v);
    fontSize(12.f);
    fillColor(Color(60,60,60));
    textAlign(ALIGN_LEFT | ALIGN_MIDDLE);
    text(r.x + 10.f, r.y - 4.f, buf, nullptr);

    // bar background
    const float innerX = r.x + 8.f;
    const float innerW = r.w - 16.f;
    const float innerY = r.y + r.h*0.5f - 4.f;
    const float h      = 8.f;

    beginPath();
    roundedRect(innerX, innerY, innerW, h, 3.f);
    fillColor(Color(220,220,220));
    fill();

    // bar fill (0..1)
    const float t = clampf(v, 0.f, 1.f);
    beginPath();
    roundedRect(innerX, innerY, innerW * t, h, 3.f);
    // reddish to stand out from the blue sliders
    fillColor(Color(240,120,120));
    fill();
}

void UITinyFdnReverb::onNanoDisplay()
{
    // background + header
    beginPath(); rect(0, 0, getWidth(), getHeight()); fillColor(Color(250,250,250)); fill();
    beginPath(); rect(0, 0, getWidth(), 28); fillColor(Color(245,245,245)); fill();
    fontSize(16.f); fillColor(Color(30,30,30)); textAlign(ALIGN_LEFT | ALIGN_MIDDLE);
    text(12, 14, "Tiny FDN Reverb v1.12 — Dal Santo core", nullptr);

    // Layer 1 skeleton controls (always visible).
    drawToggle(rLayerMatrix, "Layer 1 Matrix", "Hadamard", "House", fMatrixType);
    drawPanel(this, rAdvancedBtn.x, rAdvancedBtn.y, rAdvancedBtn.w, rAdvancedBtn.h, 4, 235,235,235);
    beginPath();
    roundedRect(rAdvancedBtn.x+2.f, rAdvancedBtn.y+2.f, rAdvancedBtn.w-4.f, rAdvancedBtn.h-4.f, 4.f);
    fillColor(fShowAdvanced ? Color(120,180,255) : Color(210,210,210));
    fill();
    fontSize(12.f);
    fillColor(Color(40,40,40));
    textAlign(ALIGN_CENTER | ALIGN_MIDDLE);
    text(rAdvancedBtn.x + rAdvancedBtn.w*0.5f, rAdvancedBtn.y + rAdvancedBtn.h*0.5f,
         fShowAdvanced ? "Advanced: ON" : "Advanced: OFF", nullptr);

    Rect traceRect = rDecay;
    Rect ringRect = rRing;
    if (! fShowAdvanced) {
        const float top = rPreset.y;
        traceRect = { 16.f, top, getWidth() - 32.f, getHeight() - top - 82.f };
        ringRect  = { 16.f, traceRect.y + traceRect.h + 10.f, getWidth() - 32.f, 22.f };
    }

    if (fShowAdvanced) {
        // preset strip
        drawPanel(this, rPreset.x, rPreset.y, rPreset.w, rPreset.h, 6, 235,235,235);
        const char* labs[4] = { "H+Prime", "H+Spread", "House+Prime", "House+Spread" };
        const int target[4][2] = {{0,0},{0,1},{1,0},{1,1}};
        const float segW = rPreset.w / 4.f;
        for (int s=0; s<4; ++s) {
            const float sx = rPreset.x + s*segW + 3.f;
            const bool on = (fMatrixType==target[s][0] && fDelaySet==target[s][1]);
            beginPath(); roundedRect(sx, rPreset.y+3.f, segW-6.f, rPreset.h-6.f, 4.f);
            fillColor(on ? Color(120,180,255) : Color(210,210,210)); fill();
            fontSize(12.f); fillColor(Color(35,35,35)); textAlign(ALIGN_CENTER|ALIGN_MIDDLE);
            text(sx + (segW-6.f)*0.5f, rPreset.y + rPreset.h*0.5f, labs[s], nullptr);
        }

        // Buttons
        beginPath(); roundedRect(rBurst.x, rBurst.y, rBurst.w, rBurst.h, 4.f);
        fillColor(Color(255,150,120)); fill();
        fontSize(12.f); fillColor(Color(255,255,255));
        textAlign(ALIGN_CENTER|ALIGN_MIDDLE);
        text(rBurst.x + rBurst.w*0.5f, rBurst.y + rBurst.h*0.5f, "Burst", nullptr);

        beginPath(); roundedRect(rPing.x, rPing.y, rPing.w, rPing.h, 4.f);
        fillColor(Color(170,130,255)); fill();
        fontSize(12.f); fillColor(Color(255,255,255));
        textAlign(ALIGN_CENTER|ALIGN_MIDDLE);
        text(rPing.x + rPing.w*0.5f, rPing.y + rPing.h*0.5f, "Ping", nullptr);

        // toggles + sliders (left column)
        drawToggle(rMatrix, "Matrix", "Hadamard", "House", fMatrixType);
        drawToggle(rDelay,  "Delay",  "Prime",    "Spread", fDelaySet);

        // Metallic Boost (single toggle row)
        drawPanel(this, rMetal.x, rMetal.y, rMetal.w, rMetal.h, 6, 235,235,235);
        beginPath(); roundedRect(rMetal.x+4, rMetal.y+4, rMetal.w-8, rMetal.h-8, 4.f);
        fillColor(fMetallic ? Color(240,120,120) : Color(210,210,210)); fill();
        fontSize(12.f); fillColor(Color(35,35,35)); textAlign(ALIGN_CENTER|ALIGN_MIDDLE);
        text(rMetal.x + (rMetal.w-8)*0.5f, rMetal.y + rMetal.h*0.5f, "Over-spread delays", nullptr);

        drawSlider(rRt60, "RT60",     fRt60,   0.20f, 8.00f);
        drawSlider(rSize, "Size",     fSize,   0.50f, 2.00f);

        // Damping shows Hz
        drawPanel(this, rDamp.x, rDamp.y, rDamp.w, rDamp.h, 6, 235,235,235);
        {
            const float t = (fDampHz - 1500.f) / (12000.f - 1500.f);
            const float knobX = rDamp.x + 8 + (rDamp.w - 16) * clampf(t,0.f,1.f);
            beginPath(); roundedRect(rDamp.x+8, rDamp.y + rDamp.h*0.5f - 3, rDamp.w-16, 6, 3); fillColor(Color(210,210,210)); fill();
            beginPath(); roundedRect(rDamp.x+8, rDamp.y + rDamp.h*0.5f - 3, (knobX - (rDamp.x+8)), 6, 3); fillColor(Color(120,180,255)); fill();
            beginPath(); circle(knobX, rDamp.y + rDamp.h*0.5f, 9.f); fillColor(Color(40,40,40)); fill();
            char hz[64]; std::snprintf(hz, sizeof(hz), "%.0f Hz", fDampHz);
            drawLabel(this, rDamp.x + 10, rDamp.y - 4, "Damp (Hz)", 13.f, 50,50,50);
            drawLabel(this, rDamp.x + rDamp.w - 80, rDamp.y - 4, hz, 13.f, 50,50,50);
        }

        drawSlider(rMix,  "Mix",       fMix,      0.00f, 1.00f);
        drawSlider(rMod,  "Mod Depth", fModDepth, 0.00f, 1.00f);
        drawSlider(rDet,  "Detune",    fDetune,   0.00f, 1.00f);

        // matrix tiles (right column)
        float Hm[4][4] = {
            {+0.5f,+0.5f,+0.5f,+0.5f},
            {+0.5f,-0.5f,+0.5f,-0.5f},
            {+0.5f,+0.5f,-0.5f,-0.5f},
            {+0.5f,-0.5f,-0.5f,+0.5f}
        };
        float Ho[4][4];
        for(int i=0;i<4;++i) for(int j=0;j<4;++j) Ho[i][j] = (i==j)?0.5f:-0.5f;

        const bool actH  = (fMatrixType==0 && fMorph < 0.5f);
        const bool actHo = (fMatrixType==1 ||  fMorph >= 0.5f);
        drawMatrixTile(this, rMatH,  "Matrix: Hadamard (signed sums)", Hm, actH);
        drawMatrixTile(this, rMatHo, "Matrix: Householder (reflection)", Ho, actHo);

        // Morph slider (right column)
        drawSlider(rMorph, "Morph (H ⟷ House)", fMorph, 0.0f, 1.0f);

        // captions under toggles
        fontSize(12.f); fillColor(Color(60,60,60)); textAlign(ALIGN_LEFT | ALIGN_MIDDLE);
        char line[64];
        std::snprintf(line, sizeof(line), "Matrix: %s", fMatrixType ? "Householder" : "Hadamard");
        text(rMatrix.x, rMatrix.y + rMatrix.h + 12.f, line, nullptr);
        std::snprintf(line, sizeof(line), "Delay set: %s", fDelaySet ? "Spread" : "Prime");
        text(rDelay.x,  rDelay.y  + rDelay.h  + 12.f, line, nullptr);
    }

    drawEnvelopeTrace(traceRect);
    drawDensityBars(this, traceRect, fDen100, fDen300);
    drawRingMeter(ringRect, fRinginess);

    // metrics readout
    fontSize(12.f); fillColor(Color(50,50,50)); textAlign(ALIGN_LEFT|ALIGN_MIDDLE);
    char m1[128], m2[128];
    if (fEDTms > 0.f && fRT60est > 0.f)
        std::snprintf(m1, sizeof(m1), "EDT: %.0f ms   RT60(est): %.2f s", fEDTms, fRT60est);
    else
        std::snprintf(m1, sizeof(m1), "EDT: N/A   RT60(est): N/A");
    std::snprintf(m2, sizeof(m2), "Echo density: 100 ms = %.2f ev/ms   300 ms = %.2f ev/ms", fDen100, fDen300);
    text(traceRect.x, traceRect.y + traceRect.h + 14.f, m1, nullptr);
    text(traceRect.x, traceRect.y + traceRect.h + 30.f, m2, nullptr);
}



static bool pointIn(const UITinyFdnReverb::Rect& r, float x, float y) {
    return (x>=r.x && x<=r.x+r.w && y>=r.y && y<=r.y+r.h);
}

bool UITinyFdnReverb::onMouse(const MouseEvent& ev) {
    const float x = ev.pos.getX();
    const float y = ev.pos.getY();

    if (ev.press) {
        if (pointIn(rAdvancedBtn, x, y)) {
            if (fShowAdvanced && fDragging != DRAG_NONE) {
                if (fDragging == DRAG_RT60)   endEdit(PluginTinyFdnReverb::paramRt60);
                if (fDragging == DRAG_SIZE)   endEdit(PluginTinyFdnReverb::paramSize);
                if (fDragging == DRAG_DAMP)   endEdit(PluginTinyFdnReverb::paramDampHz);
                if (fDragging == DRAG_MIX)    endEdit(PluginTinyFdnReverb::paramMix);
                if (fDragging == DRAG_MORPH)  endEdit(PluginTinyFdnReverb::paramMatrixMorph);
                if (fDragging == DRAG_MOD)    endEdit(PluginTinyFdnReverb::paramModDepth);
                if (fDragging == DRAG_DETUNE) endEdit(PluginTinyFdnReverb::paramDetune);
                fDragging = DRAG_NONE;
            }
            fShowAdvanced = !fShowAdvanced;
            repaint();
            return true;
        }

        if (pointIn(rLayerMatrix, x, y)) {
            const float half = rLayerMatrix.w*0.5f;
            fMatrixType = (x < rLayerMatrix.x+half) ? 0 : 1;
            beginEdit(PluginTinyFdnReverb::paramMatrixType);
            setParam(PluginTinyFdnReverb::paramMatrixType, float(fMatrixType));
            endEdit(PluginTinyFdnReverb::paramMatrixType);
            repaint();
            return true;
        }

        if (!fShowAdvanced)
            return false;

        // sliders
        if (pointIn(rRt60, x, y)) { fDragging = DRAG_RT60; beginEdit(PluginTinyFdnReverb::paramRt60); return true; }
        if (pointIn(rSize, x, y))  { fDragging = DRAG_SIZE; beginEdit(PluginTinyFdnReverb::paramSize); return true; }
        if (pointIn(rDamp, x, y))  { fDragging = DRAG_DAMP; beginEdit(PluginTinyFdnReverb::paramDampHz); return true; }
        if (pointIn(rMix,  x, y))  { fDragging = DRAG_MIX;  beginEdit(PluginTinyFdnReverb::paramMix); return true; }
        if (pointIn(rMorph,x, y))  { fDragging = DRAG_MORPH;beginEdit(PluginTinyFdnReverb::paramMatrixMorph); return true; }
        if (pointIn(rMod,  x, y))  { fDragging = DRAG_MOD;  beginEdit(PluginTinyFdnReverb::paramModDepth); return true; }
        if (pointIn(rDet,  x, y))  { fDragging = DRAG_DETUNE;beginEdit(PluginTinyFdnReverb::paramDetune); return true; }

        // Ping / Burst
        if (pointIn(rPing, x, y)) {
            beginEdit(PluginTinyFdnReverb::paramPing);
            setParam(PluginTinyFdnReverb::paramPing, 1.0f);
            endEdit(PluginTinyFdnReverb::paramPing);
            return true;
        }
        if (pointIn(rBurst, x, y)) {
            beginEdit(PluginTinyFdnReverb::paramExciteNoise);
            setParam(PluginTinyFdnReverb::paramExciteNoise, 1.0f);
            endEdit(PluginTinyFdnReverb::paramExciteNoise);
            return true;
        }

        // toggles
        if (pointIn(rMatrix, x, y)) {
            const float half = rMatrix.w*0.5f;
            fMatrixType = (x < rMatrix.x+half) ? 0 : 1;

            beginEdit(PluginTinyFdnReverb::paramMatrixType);
            setParam(PluginTinyFdnReverb::paramMatrixType, float(fMatrixType));
            endEdit(PluginTinyFdnReverb::paramMatrixType);
            DBG("[UI] Matrix toggle → %d", fMatrixType);

            // Snap Morph to match (0=Hadamard, 1=Householder)
            const float snap = fMatrixType ? 1.0f : 0.0f;
            fMorph = snap;
            beginEdit(PluginTinyFdnReverb::paramMatrixMorph);
            setParam(PluginTinyFdnReverb::paramMatrixMorph, snap);
            endEdit(PluginTinyFdnReverb::paramMatrixMorph);

            repaint();
            return true;
        }
        if (pointIn(rDelay, x, y)) {
            const float half = rDelay.w*0.5f;
            fDelaySet = (x < rDelay.x+half) ? 0 : 1;
            beginEdit(PluginTinyFdnReverb::paramDelaySet);
            setParam(PluginTinyFdnReverb::paramDelaySet, float(fDelaySet));
            endEdit(PluginTinyFdnReverb::paramDelaySet);
            DBG("[UI] Delay toggle → %d (%s)", fDelaySet, fDelaySet? "Spread":"Prime");
            repaint(); return true;
        }
        if (pointIn(rMetal, x, y)) {
            fMetallic = !fMetallic;
            beginEdit(PluginTinyFdnReverb::paramMetalBoost);
            setParam(PluginTinyFdnReverb::paramMetalBoost, float(fMetallic));
            endEdit(PluginTinyFdnReverb::paramMetalBoost);
            repaint(); return true;
        }

        // preset strip
        if (pointIn(rPreset, x, y)) {
            const float segW = rPreset.w / 4.f;
            const int s = int((x - rPreset.x) / segW);
            if (s >= 0 && s < 4) {
                const int m = (s < 2) ? 0 : 1;
                const int d = (s % 2 == 0) ? 0 : 1;
                beginEdit(PluginTinyFdnReverb::paramMatrixType);
                setParam(PluginTinyFdnReverb::paramMatrixType, float(m));
                endEdit(PluginTinyFdnReverb::paramMatrixType);
                beginEdit(PluginTinyFdnReverb::paramDelaySet);
                setParam(PluginTinyFdnReverb::paramDelaySet, float(d));
                endEdit(PluginTinyFdnReverb::paramDelaySet);
                DBG("[UI] Preset click → Matrix=%d Delay=%d", m, d);
                // Snap Morph to match (0=Hadamard, 1=Householder)
                beginEdit(PluginTinyFdnReverb::paramMatrixMorph);
                setParam(PluginTinyFdnReverb::paramMatrixMorph, float(m));
                endEdit(PluginTinyFdnReverb::paramMatrixMorph);
                repaint();
            }
            return true;
        }
    } else {
        if (!fShowAdvanced) {
            fDragging = DRAG_NONE;
            return false;
        }

        // release
        if (fDragging == DRAG_RT60)   { endEdit(PluginTinyFdnReverb::paramRt60);        fDragging = DRAG_NONE; return true; }
        if (fDragging == DRAG_SIZE)   { endEdit(PluginTinyFdnReverb::paramSize);        fDragging = DRAG_NONE; return true; }
        if (fDragging == DRAG_DAMP)   { endEdit(PluginTinyFdnReverb::paramDampHz);      fDragging = DRAG_NONE; return true; }
        if (fDragging == DRAG_MIX)    { endEdit(PluginTinyFdnReverb::paramMix);         fDragging = DRAG_NONE; return true; }
        if (fDragging == DRAG_MORPH)  { endEdit(PluginTinyFdnReverb::paramMatrixMorph); fDragging = DRAG_NONE; return true; }
        if (fDragging == DRAG_MOD)    { endEdit(PluginTinyFdnReverb::paramModDepth);    fDragging = DRAG_NONE; return true; }
        if (fDragging == DRAG_DETUNE) { endEdit(PluginTinyFdnReverb::paramDetune);      fDragging = DRAG_NONE; return true; }
    }
    return false;
}


bool UITinyFdnReverb::onMotion(const MotionEvent& ev) {
    if (!fShowAdvanced || fDragging == DRAG_NONE) return false;
    const float x = ev.pos.getX();

    auto sliderT = [&](const Rect& r) {
        return clampf((x - (r.x+8)) / (r.w - 16), 0.f, 1.f);
    };

    if (fDragging == DRAG_RT60) {
        const float t = sliderT(rRt60);
               const float v = 0.20f + t * (8.00f - 0.20f);
        fRt60 = v; setParam(PluginTinyFdnReverb::paramRt60, v);
    } else if (fDragging == DRAG_SIZE) {
        const float t = sliderT(rSize);
        const float v = 0.50f + t * (2.00f - 0.50f);
        fSize = v; setParam(PluginTinyFdnReverb::paramSize, v);
    } else if (fDragging == DRAG_DAMP) {
        const float t = sliderT(rDamp);
        const float v = 1500.0f + t * (12000.0f - 1500.0f);
        fDampHz = v; setParam(PluginTinyFdnReverb::paramDampHz, v);
    } else if (fDragging == DRAG_MIX) {
        const float t = sliderT(rMix);
        const float v = t;
        fMix = v; setParam(PluginTinyFdnReverb::paramMix, v);
    } else if (fDragging == DRAG_MORPH) {
        const float t = sliderT(rMorph);
        const float v = t;
        fMorph = v; setParam(PluginTinyFdnReverb::paramMatrixMorph, v);
    } else if (fDragging == DRAG_MOD) {
        const float t = sliderT(rMod);
        const float v = t;
        fModDepth = v; setParam(PluginTinyFdnReverb::paramModDepth, v);
    } else if (fDragging == DRAG_DETUNE) {
        const float t = sliderT(rDet);
        const float v = t;
        fDetune = v; setParam(PluginTinyFdnReverb::paramDetune, v);
    }
    repaint();
    return true;
}


// === BOILERPLATE BEGIN: DPF UI factory ===
UI* createUI() { return new UITinyFdnReverb(); }
// === BOILERPLATE END ===

END_NAMESPACE_DISTRHO
