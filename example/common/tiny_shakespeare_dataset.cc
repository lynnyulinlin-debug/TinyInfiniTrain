#include "example/common/tiny_shakespeare_dataset.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <cstring>
#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace {
using DataType = infini_train::DataType;
using TinyShakespeareType = TinyShakespeareDataset::TinyShakespeareType;
using TinyShakespeareFile = TinyShakespeareDataset::TinyShakespeareFile;

const std::unordered_map<int, TinyShakespeareType> kTypeMap = {
    {20240520, TinyShakespeareType::kUINT16}, // GPT-2
    {20240801, TinyShakespeareType::kUINT32}, // LLaMA 3
};

const std::unordered_map<TinyShakespeareType, size_t> kTypeToSize = {
    {TinyShakespeareType::kUINT16, 2},
    {TinyShakespeareType::kUINT32, 4},
};

const std::unordered_map<TinyShakespeareType, DataType> kTypeToDataType = {
    {TinyShakespeareType::kUINT16, DataType::kUINT16},
    {TinyShakespeareType::kUINT32, DataType::kINT32},
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

TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    /* =================================== 作业 ===================================
       TODO：实现二进制数据集文件解析
       文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | DATA (tokens)                        |
    | magic(4B) | version(4B) | num_toks(4B) | reserved(1012B) | token数据           |
    ----------------------------------------------------------------------------------
       =================================== 作业 =================================== */
           std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open dataset file: " << path;

    auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    CHECK_EQ(header.size(), 1024u) << "Invalid header size: " << path;

    const uint32_t magic = BytesToType<uint32_t>(header, 0);
    const uint32_t version = BytesToType<uint32_t>(header, 4);
    (void)version; // 当前实现不依赖 version，但保留解析以符合格式

    const uint32_t num_toks_u32 = BytesToType<uint32_t>(header, 8);
    const size_t num_toks = static_cast<size_t>(num_toks_u32);

    CHECK(kTypeMap.contains(static_cast<int>(magic))) << "Unknown dataset magic number: " << magic;
    const auto tok_type = kTypeMap.at(static_cast<int>(magic));
    CHECK(kTypeToSize.contains(tok_type));
    const size_t tok_size = kTypeToSize.at(tok_type);

    CHECK_GT(sequence_length, 0u);
    const size_t num_full_samples = num_toks / sequence_length;
    CHECK_GE(num_full_samples, 2u) << "Not enough samples for x/y shift. "
                                   << "num_toks=" << num_toks << ", seq_len=" << sequence_length;

    const size_t used_toks = num_full_samples * sequence_length;
    const size_t data_bytes = num_toks * tok_size;

    auto raw = ReadSeveralBytesFromIfstream(data_bytes, &ifs);
    CHECK_EQ(raw.size(), data_bytes) << "Failed to read token data: " << path;

    if (used_toks != num_toks) {
        LOG(WARNING) << "num_toks not divisible by sequence_length, truncating tail. "
                     << "num_toks=" << num_toks << ", used_toks=" << used_toks;
    }

    // 转成 int64_t，保证与 operator[] 的 sizeof(int64_t) 偏移一致
    std::vector<int64_t> tokens_i64(used_toks);
    if (tok_type == TinyShakespeareType::kUINT16) {
        for (size_t i = 0; i < used_toks; ++i) {
            const uint16_t v = BytesToType<uint16_t>(raw, i * tok_size);
            tokens_i64[i] = static_cast<int64_t>(v);
        }
    } else if (tok_type == TinyShakespeareType::kUINT32) {
        for (size_t i = 0; i < used_toks; ++i) {
            const uint32_t v = BytesToType<uint32_t>(raw, i * tok_size);
            tokens_i64[i] = static_cast<int64_t>(v);
        }
    } else {
        CHECK(false) << "Unsupported TinyShakespeareType.";
    }

    TinyShakespeareFile file;
    file.type = tok_type;
    file.dims = {static_cast<int64_t>(num_full_samples), static_cast<int64_t>(sequence_length)};
    file.tensor = infini_train::Tensor(file.dims, DataType::kINT64);
    std::memcpy(file.tensor.DataPtr(), tokens_i64.data(), used_toks * sizeof(int64_t));
    return file;
}
} // namespace

// TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length) {
//     // =================================== 作业 ===================================
//     // TODO：初始化数据集实例
//     // HINT: 调用ReadTinyShakespeareFile加载数据文件
//     // =================================== 作业 ===================================
//     CHECK_EQ(text_file_.dims.size(), 2u);
//     CHECK_EQ(static_cast<size_t>(text_file_.dims[1]), sequence_length_);
//     CHECK_GE(num_samples_, 2u);
// }
TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
    : text_file_(ReadTinyShakespeareFile(filepath, sequence_length)),
      sequence_length_(sequence_length),
      sequence_size_in_bytes_(sequence_length * sizeof(int64_t)),
      num_samples_(static_cast<size_t>(text_file_.dims[0]) - 1) {
    // =================================== 作业 ===================================
    // TODO：初始化数据集实例
    // HINT: 调用ReadTinyShakespeareFile加载数据文件
    // =================================== 作业 ===================================
    CHECK_EQ(text_file_.dims.size(), 2u);
    CHECK_EQ(static_cast<size_t>(text_file_.dims[1]), sequence_length_);
    CHECK_GE(num_samples_, 2u);
}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, text_file_.dims[0] - 1);
    std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
    // x: (seq_len), y: (seq_len) -> stack -> (bs, seq_len) (bs, seq_len)
    return {std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_, dims),
            std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_ + sizeof(int64_t),
                                                   dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
