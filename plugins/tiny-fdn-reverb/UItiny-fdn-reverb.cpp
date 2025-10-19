/*
 * Tiny FDN Reverb — minimal UI (NanoVG)
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
#include "UItiny-fdn-reverb.hpp"
#include "NanoVG.hpp"   // NVGcontext + nvg* API
#include <cmath>

START_NAMESPACE_DISTRHO
using namespace DGL; // nvg* symbols

UITinyFdnReverb::UITinyFdnReverb()
: UI(1040, 480)
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
}

void UITinyFdnReverb::layout() {
    const float PAD = 16.0f;
    const float W = getWidth(), H = getHeight();
    const float SL_H = 28.0f;
    const float BTN_H = 28.0f;
    const float DEC_H = 90.0f;

    rMatrix = { PAD, PAD, (W - 3*PAD)/2.0f, BTN_H };
    rDelay  = { rMatrix.x + rMatrix.w + PAD, PAD, (W - 3*PAD)/2.0f, BTN_H };

    rRt60 =  { PAD, rMatrix.y + rMatrix.h + PAD, W - 2*PAD, SL_H };
    rMix  =  { PAD, rRt60.y + rRt60.h + PAD*0.75f, W - 2*PAD, SL_H };

    rDecay = { PAD, H - DEC_H - PAD, W - 2*PAD, DEC_H };
}

void UITinyFdnReverb::parameterChanged(uint32_t index, float value) {
    switch (index) {
    case PluginTinyFdnReverb::paramRt60:       fRt60 = value; break;
    case PluginTinyFdnReverb::paramMix:        fMix = value; break;
    case PluginTinyFdnReverb::paramMatrixType: fMatrixType = int(std::lround(value)); break;
    case PluginTinyFdnReverb::paramDelaySet:   fDelaySet   = int(std::lround(value)); break;
    default: break;
    }
    repaint();
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
    if (std::strcmp(label, "RT60") == 0) std::snprintf(buf, sizeof(buf), "%.2f s", v);
    else                                 std::snprintf(buf, sizeof(buf), "%.0f %%", v*100.f);

    // draw inside the slider row so it’s always visible
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
    strokeWidth(2.f);
    strokeColor(40,40,40);
    stroke();

    drawLabel(this, r.x+10, r.y-8, "Decay (analytic)", 12.f, 60,60,60);
}

void UITinyFdnReverb::onNanoDisplay()
{
    // background
    beginPath();
    rect(0, 0, getWidth(), getHeight());
    fillColor(Color(250,250,250));
    fill();

    // header strip
    beginPath();
    rect(0, 0, getWidth(), 28);
    fillColor(Color(245,245,245));
    fill();
    fontSize(42.f);
    fillColor(Color(30,30,30));
    textAlign(ALIGN_LEFT | ALIGN_MIDDLE);
    text(12, 14, "Tiny FDN Reverb — RT60/Mix, Matrix, Delay-Set", nullptr);

    // widgets
    drawToggle(rMatrix, "Matrix", "Hadamard", "House", fMatrixType);
    drawToggle(rDelay,  "Delay",  "Prime",    "Spread", fDelaySet);
    drawSlider(rRt60, "RT60", fRt60, 0.20f, 8.00f);
    drawSlider(rMix,  "Mix",  fMix,  0.00f, 1.00f);
    drawDecay(rDecay, fRt60);

    // dynamic captions under the toggles
    char line[64];
    fontSize(38.f);
    fillColor(Color(60,60,60));
    textAlign(ALIGN_LEFT | ALIGN_MIDDLE);

    std::snprintf(line, sizeof(line), "Matrix: %s", fMatrixType ? "Householder" : "Hadamard");
    text(rMatrix.x, rMatrix.y + rMatrix.h + 14.f, line, nullptr);

    std::snprintf(line, sizeof(line), "Delay set: %s", fDelaySet ? "Spread" : "Prime");
    text(rDelay.x,  rDelay.y  + rDelay.h  + 14.f, line, nullptr);
}


static bool pointIn(const UITinyFdnReverb::Rect& r, float x, float y) {
    return (x>=r.x && x<=r.x+r.w && y>=r.y && y<=r.y+r.h);
}

bool UITinyFdnReverb::onMouse(const MouseEvent& ev) {
    const float x = ev.pos.getX();
    const float y = ev.pos.getY();
    if (ev.press) {
        // sliders
        if (pointIn(rRt60, x, y)) {
            fDragging = DRAG_RT60; beginEdit(PluginTinyFdnReverb::paramRt60);
            return true;
        }
        if (pointIn(rMix, x, y)) {
            fDragging = DRAG_MIX;  beginEdit(PluginTinyFdnReverb::paramMix);
            return true;
        }
        // toggles
        if (pointIn(rMatrix, x, y)) {
            const float half = rMatrix.w*0.5f;
            fMatrixType = (x < rMatrix.x+half) ? 0 : 1;
            beginEdit(PluginTinyFdnReverb::paramMatrixType);
            setParam(PluginTinyFdnReverb::paramMatrixType, float(fMatrixType));
            endEdit(PluginTinyFdnReverb::paramMatrixType);
            repaint();
            return true;
        }
        if (pointIn(rDelay, x, y)) {
            const float half = rDelay.w*0.5f;
            fDelaySet = (x < rDelay.x+half) ? 0 : 1;
            beginEdit(PluginTinyFdnReverb::paramDelaySet);
            setParam(PluginTinyFdnReverb::paramDelaySet, float(fDelaySet));
            endEdit(PluginTinyFdnReverb::paramDelaySet);
            repaint();
            return true;
        }
    } else {
        // mouse release
        if (fDragging == DRAG_RT60) { endEdit(PluginTinyFdnReverb::paramRt60); fDragging = DRAG_NONE; return true; }
        if (fDragging == DRAG_MIX)  { endEdit(PluginTinyFdnReverb::paramMix);  fDragging = DRAG_NONE; return true; }
    }
    return false;
}

bool UITinyFdnReverb::onMotion(const MotionEvent& ev) {
    if (fDragging == DRAG_NONE) return false;
    const float x = ev.pos.getX();
    const Rect& r = (fDragging == DRAG_RT60 ? rRt60 : rMix);
    const float t = clampf((x - (r.x+8)) / (r.w - 16), 0.f, 1.f);

    if (fDragging == DRAG_RT60) {
        const float v = 0.20f + t * (8.00f - 0.20f);
        fRt60 = v;
        setParam(PluginTinyFdnReverb::paramRt60, v);
    } else if (fDragging == DRAG_MIX) {
        const float v = t; // 0..1
        fMix = v;
        setParam(PluginTinyFdnReverb::paramMix, v);
    }
    repaint();
    return true;
}

UI* createUI() { return new UITinyFdnReverb(); }

END_NAMESPACE_DISTRHO
