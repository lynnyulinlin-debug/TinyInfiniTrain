#include "example/common/tokenizer.h"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "glog/logging.h"

namespace infini_train {

constexpr uint32_t kGpt2Eot = 50256;
constexpr uint32_t kLLaMA3Eot = 128001;
constexpr uint64_t kRandomU32Multiplier = 0x2545F4914F6CDD1Dull;
constexpr float kF32Divisor = 16777216.0f; // 2^24
constexpr uint64_t kRngState = 1337;

using Version = Tokenizer::Version;

const std::unordered_map<uint32_t, uint32_t> kEotMap = {
    {20240328, kGpt2Eot},   // GPT-2
    {20240801, kLLaMA3Eot}, // LLaMA-3
};

const std::unordered_map<uint32_t, std::vector<uint32_t>> kPromptMap = {
    // e.g. "The meaning of life is"
    // ref: https://tiktokenizer.vercel.app/
    {20240328, std::vector<uint32_t>{464, 3616, 286, 1204, 318}}, // GPT-2
    {20240801, std::vector<uint32_t>{791, 7438, 315, 2324, 374}}, // LLaMA-3
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

unsigned int RandomU32(uint64_t &state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (state * kRandomU32Multiplier) >> 32;
}

float RandomF32(uint64_t &state) { // random float32 in [0,1)
    return (RandomU32(state) >> 8) / kF32Divisor;
}

int SampleMult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from RandomF32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

Tokenizer::Tokenizer(const std::string &filepath) {
    /* ===================================== 作业 =====================================
    TODO：实现Tokenizer二进制文件加载

    文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | VOCAB TABLE                           |
    | magic(4B) | version(4B) | vocab_size(4B) | reserved(1012B) | token词表数据       |
    ----------------------------------------------------------------------------------
    ===================================== 作业 ===================================== */
    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open tokenizer file: " << filepath;

    auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    CHECK_EQ(header.size(), 1024u) << "Invalid tokenizer header size: " << filepath;

    magic_number_ = BytesToType<uint32_t>(header, 0);
    const uint32_t version = BytesToType<uint32_t>(header, 4);
    (void)version;

    vocab_size_ = BytesToType<uint32_t>(header, 8);
    CHECK_GT(vocab_size_, 0u) << "Invalid vocab_size in tokenizer: " << filepath;

    CHECK(kEotMap.contains(magic_number_)) << "Unknown tokenizer magic_number: " << magic_number_;
    eot_token_ = kEotMap.at(magic_number_);

    token_table_.clear();
    token_table_.reserve(vocab_size_);

    // Token 格式: [uint8 len][bytes]
    for (uint32_t i = 0; i < vocab_size_; ++i) {
        uint8_t len = 0;
        ifs.read(reinterpret_cast<char *>(&len), sizeof(uint8_t));
        CHECK(ifs.good()) << "Failed to read token length at id=" << i;

        std::string token;
        token.resize(len);
        if (len > 0) {
            ifs.read(token.data(), len);
            CHECK(ifs.good()) << "Failed to read token bytes at id=" << i << ", len=" << static_cast<int>(len);
        }
        token_table_.push_back(std::move(token));
    }

    CHECK_EQ(token_table_.size(), static_cast<size_t>(vocab_size_));
 
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    /* ===================================== 作业 =====================================
    TODO：实现token_id到文本的转换
    功能描述：根据token_id返回对应的文本片段
    ===================================== 作业 ===================================== */
    //return "";
    CHECK_LT(token_id, token_table_.size()) << "token_id out of range: " << token_id;
    return token_table_[token_id];
}

void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
    std::vector<int64_t> dims;
    dims.assign({batch_size, sequence_length});
    // x_tensor (FLAGS_batch_size, FLAGS_sequence_length) eq:(4, 64)
    infini_train::Tensor x_tensor = infini_train::Tensor(dims, DataType::kINT64);
    int64_t *x_buff = static_cast<int64_t *>(x_tensor.DataPtr());
    for (int i = 0; i < batch_size * sequence_length; ++i) { x_buff[i] = eot_token_; }

    // Give some contexts: "The meaning of life is "
    auto prompt = kPromptMap.at(magic_number_);
    auto prompt_len = prompt.size();
    for (int i = 0; i < prompt_len; ++i) { x_buff[i] = prompt[i]; }
    std::cout << "The meaning of life is";

    auto x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
    uint64_t kRngState = kRngState;
    LOG(INFO) << "start generate text:";
    for (int t = prompt_len; t < text_length; t++) {
        /* ===================================== 作业 =====================================
        TODO：实现单步文本生成逻辑
        HINT：调用model.Forward推理获取logits，根据推理结果进行随机采样，调用Decode获取文本结果
        ===================================== 作业 ===================================== */
          //std::cout << std::endl;
        // 修复局部 kRngState 自初始化的不确定性：首步强制重置为固定种子，保证可复现
    if (t == static_cast<int>(prompt_len)) {
        kRngState = 1337;
    }

    // 超出上下文窗口时滑窗：整体左移一格
    if (t >= static_cast<int>(sequence_length)) {
        for (uint32_t b = 0; b < batch_size; ++b) {
            int64_t *row = x_buff + b * sequence_length;
            for (uint32_t i = 0; i + 1 < sequence_length; ++i) {
                row[i] = row[i + 1];
            }
            row[sequence_length - 1] = eot_token_;
        }
        t = static_cast<int>(sequence_length) - 1;
    }

    // Forward：GPT2::Forward 只需要 idx（bs, seq_len）
    auto x_dev = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
    auto outputs = model.Forward({x_dev});
    CHECK(!outputs.empty());
    auto logits = outputs[0];
    CHECK(logits != nullptr);

    // logits: (bs, seq_len, vocab) -> CPU
    auto cpu_logits = logits->To(Device(DeviceType::kCPU, 0));
    const auto &ld = cpu_logits.Dims();
    CHECK_EQ(ld.size(), 3u);
    CHECK_EQ(static_cast<uint32_t>(ld[0]), batch_size);
    CHECK_EQ(static_cast<uint32_t>(ld[1]), sequence_length);

    const int vocab = static_cast<int>(ld[2]);
    const float *logits_ptr = static_cast<const float *>(cpu_logits.DataPtr());

    // 取前一位置预测当前：pos=t-1（最小为0）
    const int pos = std::max(0, t - 1);

    std::vector<float> probs(vocab);

    for (uint32_t b = 0; b < batch_size; ++b) {
        const float *row = logits_ptr + (static_cast<int64_t>(b) * sequence_length + pos) * vocab;

        // stable softmax
        float max_logit = row[0];
        for (int i = 1; i < vocab; ++i) {
            if (row[i] > max_logit) max_logit = row[i];
        }
        float sum = 0.0f;
        for (int i = 0; i < vocab; ++i) {
            probs[i] = std::exp(row[i] - max_logit);
            sum += probs[i];
        }
        CHECK_GT(sum, 0.0f);
        for (int i = 0; i < vocab; ++i) probs[i] /= sum;

        float coin = RandomF32(kRngState);
        int next_id = SampleMult(probs.data(), vocab, coin);
        CHECK_GE(next_id, 0);
        CHECK_LT(next_id, vocab);

        x_buff[b * sequence_length + t] = static_cast<int64_t>(next_id);

        // 只打印 batch=0，避免多 batch 输出混乱
        if (b == 0) {
            std::cout << Decode(static_cast<uint32_t>(next_id));
        }
    }
    }
  
}
} // namespace infini_train
