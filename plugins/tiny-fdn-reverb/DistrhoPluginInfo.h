/*
 * tiny-fdn-reverb audio effect based on DISTRHO Plugin Framework (DPF)
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2025 Fernando de Souza Araujo <faraujo0080@gmail.com>
 */

#ifndef DISTRHO_PLUGIN_INFO_H
#define DISTRHO_PLUGIN_INFO_H

// === BOILERPLATE BEGIN: Standard DPF plugin info macros (values customized) ===
// The plugin name.
// This is used to identify your plugin before a Plugin instance can be created.
#define DISTRHO_PLUGIN_NAME  "tiny-fdn-reverb"
// The plugin brand name. Used for the LV2 metadata and VST3 UI interface.
// Must be a valid C++ identifier, i.e. can not contain spaces or dashes.
#define DISTRHO_PLUGIN_BRAND "N/A"
// The plugin URI when exporting in LV2 format.
// See https://lv2plug.in/book/#_manifest_ttl_in
#define DISTRHO_PLUGIN_URI   "https://codeberg.org/fernandoAra/tiny-fdn-reverb#tiny-fdn-reverb"
// The plugin id when exporting in CLAP format, in reverse URI form
#define DISTRHO_PLUGIN_CLAP_ID "com.github.fernandoara.tiny_fdn_reverb"

#define DISTRHO_PLUGIN_HAS_UI        1
#define DISTRHO_UI_USE_NANOVG        1

#define DISTRHO_PLUGIN_IS_RT_SAFE       1
#define DISTRHO_PLUGIN_NUM_INPUTS       2
#define DISTRHO_PLUGIN_NUM_OUTPUTS      2
#define DISTRHO_PLUGIN_WANT_TIMEPOS     0
#define DISTRHO_PLUGIN_WANT_PROGRAMS    1
#define DISTRHO_PLUGIN_WANT_MIDI_INPUT  0
#define DISTRHO_PLUGIN_WANT_MIDI_OUTPUT 0
#define DISTRHO_PLUGIN_WANT_DIRECT_ACCESS 1

// See http://lv2plug.in/ns/lv2core#ref-classes
#define DISTRHO_PLUGIN_LV2_CATEGORY "lv2:AmplifierPlugin"
// See https://github.com/DISTRHO/DPF/blob/1504e7d327bfe0eac6a889cecd199c963d35532f/distrho/DistrhoInfo.hpp#L717
#define DISTRHO_PLUGIN_VST3_CATEGORIES "Fx|Tools|Stereo"
// See https://github.com/DISTRHO/DPF/blob/1504e7d327bfe0eac6a889cecd199c963d35532f/distrho/DistrhoInfo.hpp#L761
#define DISTRHO_PLUGIN_CLAP_FEATURES "audio-effect", "utility", "stereo"
// === BOILERPLATE END ===

#endif // DISTRHO_PLUGIN_INFO_H
