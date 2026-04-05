#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kN = 4;

struct Preset {
    std::string configId;
    int sampleRate = 48000;
    std::array<int, kN> delaySamples {1499, 2377, 3217, 4421};
    std::array<float, kN> gains {0.0f, 0.0f, 0.0f, 0.0f};
    std::string matrixType = "householder";
    std::array<float, kN> u {0.5f, 0.5f, 0.5f, 0.5f};
    std::array<float, kN> b {0.25f, 0.25f, 0.25f, 0.25f};
    std::array<float, kN> cL {0.5f, -0.5f, 0.5f, -0.5f};
    std::array<float, kN> cR {0.5f, 0.5f, -0.5f, -0.5f};
    float modDepth = 0.0f;
    float detune = 0.0f;
    float dampHz = 1.0e9f;
};

std::string trim(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

bool findJsonValueStart(const std::string& text, const std::string& key, size_t& pos) {
    const std::string token = "\"" + key + "\"";
    const size_t kpos = text.find(token);
    if (kpos == std::string::npos) {
        return false;
    }
    size_t cpos = text.find(':', kpos + token.size());
    if (cpos == std::string::npos) {
        throw std::runtime_error("Malformed JSON: missing ':' for key " + key);
    }
    ++cpos;
    while (cpos < text.size() && std::isspace(static_cast<unsigned char>(text[cpos]))) {
        ++cpos;
    }
    pos = cpos;
    return true;
}

double parseNumberToken(const std::string& token, const std::string& key) {
    const std::string cleaned = trim(token);
    if (cleaned.empty()) {
        throw std::runtime_error("Empty numeric token for key " + key);
    }
    char* end = nullptr;
    const double value = std::strtod(cleaned.c_str(), &end);
    if (end == cleaned.c_str()) {
        throw std::runtime_error("Failed to parse numeric token for key " + key);
    }
    return value;
}

double parseJsonNumber(const std::string& text, const std::string& key) {
    size_t pos = 0;
    if (!findJsonValueStart(text, key, pos)) {
        throw std::runtime_error("Missing key: " + key);
    }
    size_t end = pos;
    while (end < text.size()) {
        const char ch = text[end];
        if (ch == ',' || ch == '}' || std::isspace(static_cast<unsigned char>(ch))) {
            break;
        }
        ++end;
    }
    return parseNumberToken(text.substr(pos, end - pos), key);
}

double parseJsonNumberOr(const std::string& text, const std::string& key, double fallback) {
    size_t pos = 0;
    if (!findJsonValueStart(text, key, pos)) {
        return fallback;
    }
    size_t end = pos;
    while (end < text.size()) {
        const char ch = text[end];
        if (ch == ',' || ch == '}' || std::isspace(static_cast<unsigned char>(ch))) {
            break;
        }
        ++end;
    }
    return parseNumberToken(text.substr(pos, end - pos), key);
}

std::string parseJsonString(const std::string& text, const std::string& key, const std::string& fallback = "") {
    size_t pos = 0;
    if (!findJsonValueStart(text, key, pos)) {
        return fallback;
    }
    if (pos >= text.size() || text[pos] != '"') {
        throw std::runtime_error("Expected string value for key " + key);
    }
    const size_t end = text.find('"', pos + 1);
    if (end == std::string::npos) {
        throw std::runtime_error("Unterminated string value for key " + key);
    }
    return text.substr(pos + 1, end - pos - 1);
}

std::vector<double> parseJsonArray(const std::string& text, const std::string& key) {
    size_t pos = 0;
    if (!findJsonValueStart(text, key, pos)) {
        throw std::runtime_error("Missing key: " + key);
    }
    if (pos >= text.size() || text[pos] != '[') {
        throw std::runtime_error("Expected array value for key " + key);
    }
    const size_t end = text.find(']', pos + 1);
    if (end == std::string::npos) {
        throw std::runtime_error("Unterminated array value for key " + key);
    }

    std::vector<double> values;
    const std::string body = text.substr(pos + 1, end - pos - 1);
    size_t start = 0;
    while (start < body.size()) {
        size_t comma = body.find(',', start);
        if (comma == std::string::npos) {
            comma = body.size();
        }
        const std::string token = trim(body.substr(start, comma - start));
        if (!token.empty()) {
            values.push_back(parseNumberToken(token, key));
        }
        start = comma + 1;
    }
    return values;
}

template <typename T>
std::array<T, kN> toFixedArray(const std::vector<double>& values, const std::string& key) {
    if (values.size() != kN) {
        throw std::runtime_error("Expected " + std::to_string(kN) + " values for " + key);
    }
    std::array<T, kN> out {};
    for (int i = 0; i < kN; ++i) {
        out[i] = static_cast<T>(values[static_cast<size_t>(i)]);
    }
    return out;
}

Preset loadPreset(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open preset file: " + path);
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string json = buffer.str();

    Preset p;
    p.configId = parseJsonString(json, "config_id", "unknown_config");
    p.sampleRate = static_cast<int>(parseJsonNumber(json, "sr"));
    p.delaySamples = toFixedArray<int>(parseJsonArray(json, "delay_samples"), "delay_samples");
    p.gains = toFixedArray<float>(parseJsonArray(json, "gains"), "gains");
    p.matrixType = parseJsonString(json, "matrix_type", "householder");
    p.u = toFixedArray<float>(parseJsonArray(json, "u"), "u");
    p.b = toFixedArray<float>(parseJsonArray(json, "b"), "b");
    p.cL = toFixedArray<float>(parseJsonArray(json, "cL"), "cL");
    p.cR = toFixedArray<float>(parseJsonArray(json, "cR"), "cR");
    p.modDepth = static_cast<float>(parseJsonNumberOr(json, "mod_depth", 0.0));
    p.detune = static_cast<float>(parseJsonNumberOr(json, "detune", 0.0));
    p.dampHz = static_cast<float>(parseJsonNumberOr(json, "damp_hz", 1.0e9));
    return p;
}

void normalizeU(std::array<float, kN>& u) {
    double norm2 = 0.0;
    for (const float v : u) {
        norm2 += static_cast<double>(v) * static_cast<double>(v);
    }
    const double norm = std::sqrt(std::max(norm2, 1e-20));
    for (float& v : u) {
        v = static_cast<float>(v / norm);
    }
}

template <typename T>
void applyHadamard4(const std::array<T, kN>& in, std::array<T, kN>& out) {
    const T a = in[0];
    const T b = in[1];
    const T c = in[2];
    const T d = in[3];
    out[0] = T(0.5) * (+a + b + c + d);
    out[1] = T(0.5) * (+a - b + c - d);
    out[2] = T(0.5) * (+a + b - c - d);
    out[3] = T(0.5) * (+a - b - c + d);
}

template <typename T>
void applyHouseholder(const std::array<T, kN>& in, const std::array<float, kN>& u, std::array<T, kN>& out) {
    T dot = T(0);
    for (int i = 0; i < kN; ++i) {
        dot += in[i] * static_cast<T>(u[i]);
    }
    for (int i = 0; i < kN; ++i) {
        out[i] = in[i] - T(2) * static_cast<T>(u[i]) * dot;
    }
}

void writeLE16(std::ofstream& out, uint16_t value) {
    const char bytes[2] = {
        static_cast<char>(value & 0xFF),
        static_cast<char>((value >> 8) & 0xFF),
    };
    out.write(bytes, 2);
}

void writeLE32(std::ofstream& out, uint32_t value) {
    const char bytes[4] = {
        static_cast<char>(value & 0xFF),
        static_cast<char>((value >> 8) & 0xFF),
        static_cast<char>((value >> 16) & 0xFF),
        static_cast<char>((value >> 24) & 0xFF),
    };
    out.write(bytes, 4);
}

void writeWavFloat32(const std::string& wavPath, const std::vector<float>& samples, int sampleRate, int channels) {
    std::ofstream out(wavPath, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open WAV for writing: " + wavPath);
    }
    const uint32_t dataBytes = static_cast<uint32_t>(samples.size() * sizeof(float));
    const uint32_t riffBytes = 36u + dataBytes;
    const uint32_t byteRate = static_cast<uint32_t>(sampleRate * channels * sizeof(float));
    const uint16_t blockAlign = static_cast<uint16_t>(channels * sizeof(float));

    out.write("RIFF", 4);
    writeLE32(out, riffBytes);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    writeLE32(out, 16u);
    writeLE16(out, 3u); // IEEE float
    writeLE16(out, static_cast<uint16_t>(channels));
    writeLE32(out, static_cast<uint32_t>(sampleRate));
    writeLE32(out, byteRate);
    writeLE16(out, blockAlign);
    writeLE16(out, 32u);
    out.write("data", 4);
    writeLE32(out, dataBytes);
    out.write(reinterpret_cast<const char*>(samples.data()), static_cast<std::streamsize>(dataBytes));
}

void writeMetadataJson(
    const std::string& path,
    const std::string& presetPath,
    const std::string& wavPath,
    const Preset& preset,
    int samples,
    float seconds,
    int channels
) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open metadata JSON for writing: " + path);
    }

    out << "{\n";
    out << "  \"preset_path\": \"" << presetPath << "\",\n";
    out << "  \"wav_path\": \"" << wavPath << "\",\n";
    out << "  \"config_id\": \"" << preset.configId << "\",\n";
    out << "  \"matrix_type\": \"" << preset.matrixType << "\",\n";
    out << "  \"sample_rate\": " << preset.sampleRate << ",\n";
    out << "  \"seconds\": " << seconds << ",\n";
    out << "  \"samples\": " << samples << ",\n";
    out << "  \"channels\": " << channels << ",\n";
    out << "  \"delay_samples\": [" << preset.delaySamples[0] << ", " << preset.delaySamples[1] << ", "
        << preset.delaySamples[2] << ", " << preset.delaySamples[3] << "],\n";
    out << "  \"mod_depth\": " << preset.modDepth << ",\n";
    out << "  \"detune\": " << preset.detune << ",\n";
    out << "  \"damp_hz\": " << preset.dampHz << ",\n";
    out << "  \"note\": \"Offline tiny-FDN renderer; intentionally close to plugin core but not sample-identical.\"\n";
    out << "}\n";
}

void printUsage(const char* name) {
    std::cerr << "Usage: " << name << " <preset.json> <out.wav> [seconds]\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            printUsage(argv[0]);
            return 1;
        }

        const std::string presetPath = argv[1];
        const std::string outWavPath = argv[2];
        const float seconds = (argc >= 4) ? std::max(0.01f, std::stof(argv[3])) : 4.0f;

        Preset preset = loadPreset(presetPath);
        normalizeU(preset.u);
        // This renderer is intentionally LTI for transfer-function comparisons.
        if (std::fabs(preset.modDepth) > 1.0e-7f || std::fabs(preset.detune) > 1.0e-7f) {
            throw std::runtime_error(
                "gen_ir.cpp only supports LTI compare mode: set mod_depth=0 and detune=0 in preset");
        }
        if (preset.dampHz < 1.0e8f) {
            throw std::runtime_error(
                "gen_ir.cpp compare mode requires damping disabled (set damp_hz very high, e.g., 1e9)");
        }
        for (int i = 0; i < kN; ++i) {
            preset.delaySamples[i] = std::max(1, preset.delaySamples[i]);
        }
        const int totalSamples = std::max(1, static_cast<int>(std::lround(seconds * preset.sampleRate)));

        std::array<std::vector<double>, kN> buffers;
        std::array<int, kN> index {};
        for (int i = 0; i < kN; ++i) {
            buffers[i].assign(static_cast<size_t>(preset.delaySamples[i]), 0.0);
            index[i] = 0;
        }

        std::vector<float> out;
        out.resize(static_cast<size_t>(totalSamples) * 2u, 0.0f);

        // No allocations in this loop.
        for (int n = 0; n < totalSamples; ++n) {
            const double impulse = (n == 0) ? 1.0 : 0.0;

            std::array<double, kN> x {};
            std::array<double, kN> g {};
            std::array<double, kN> fb {};
            for (int i = 0; i < kN; ++i) {
                x[i] = buffers[i][static_cast<size_t>(index[i])];
                g[i] = static_cast<double>(preset.gains[i]) * x[i];
            }

            if (preset.matrixType == "hadamard") {
                applyHadamard4(g, fb);
            } else {
                applyHouseholder(g, preset.u, fb);
            }

            for (int i = 0; i < kN; ++i) {
                buffers[i][static_cast<size_t>(index[i])] =
                    fb[i] + static_cast<double>(preset.b[i]) * impulse;
                ++index[i];
                if (index[i] >= preset.delaySamples[i]) {
                    index[i] = 0;
                }
            }

            double yL = 0.0;
            double yR = 0.0;
            for (int i = 0; i < kN; ++i) {
                yL += static_cast<double>(preset.cL[i]) * x[i];
                yR += static_cast<double>(preset.cR[i]) * x[i];
            }

            out[static_cast<size_t>(2 * n)] = static_cast<float>(yL);
            out[static_cast<size_t>(2 * n + 1)] = static_cast<float>(yR);
        }

        writeWavFloat32(outWavPath, out, preset.sampleRate, 2);
        const std::string metaPath = outWavPath + ".json";
        writeMetadataJson(metaPath, presetPath, outWavPath, preset, totalSamples, seconds, 2);

        std::cout << "Wrote WAV: " << outWavPath << "\n";
        std::cout << "Wrote metadata: " << metaPath << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
