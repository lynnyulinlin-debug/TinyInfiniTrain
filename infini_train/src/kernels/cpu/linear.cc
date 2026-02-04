#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <numeric>
#include <tuple>
#include <algorithm>


#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    // =================================== 作业 ===================================
    // TODO：实现CPU上的矩阵乘法前向计算
    // REF:
    // =================================== 作业 ===================================
    CHECK(input);
    CHECK(other);
    CHECK(input->Dtype() == DataType::kFLOAT32);
    CHECK(other->Dtype() == DataType::kFLOAT32);
    CHECK(input->GetDevice().Type() == DeviceType::kCPU);
    CHECK(other->GetDevice().Type()== DeviceType::kCPU);

    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    CHECK_GE(input_dims.size(), 2);
    CHECK_GE(other_dims.size(), 2);

    // inner dim check: [..., M, K] x [..., K, N]
    CHECK_EQ(input_dims.back(), other_dims[other_dims.size() - 2]);

    const int64_t m = input_dims[input_dims.size() - 2];
    const int64_t k = input_dims[input_dims.size() - 1];
    const int64_t n = other_dims[other_dims.size() - 1];

    // compute "batch" = product of leading dims (all except last 2)
    int64_t input_batch = 1;
    for (size_t i = 0; i + 2 < input_dims.size(); ++i) {
        input_batch *= input_dims[i];
    }
    int64_t other_batch = 1;
    for (size_t i = 0; i + 2 < other_dims.size(); ++i) {
        other_batch *= other_dims[i];
    }

    // allow same batch, or broadcast when one side batch == 1
    CHECK(input_batch == other_batch || input_batch == 1 || other_batch == 1)
        << "Batch dimensions must match or be 1 for broadcasting.";

    const int64_t batch_size = std::max(input_batch, other_batch);

    // output dims = broadcasted batch dims + [m, n]
    std::vector<int64_t> output_dims;
    const auto &batch_dims = (input_dims.size() >= other_dims.size()) ? input_dims : other_dims;
    for (size_t i = 0; i + 2 < batch_dims.size(); ++i) {
        output_dims.push_back(batch_dims[i]);
    }
    output_dims.push_back(m);
    output_dims.push_back(n);

    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    float *out_data = static_cast<float *>(output->DataPtr());
    const float *in_data = static_cast<const float *>(input->DataPtr());
    const float *ot_data = static_cast<const float *>(other->DataPtr());

    // if broadcast, stride is 0; else normal stride
    const int64_t input_stride = (input_batch == 1) ? 0 : (m * k);
    const int64_t other_stride = (other_batch == 1) ? 0 : (k * n);
    const int64_t output_stride = m * n;

    for (int64_t b = 0; b < batch_size; ++b) {
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            A(in_data + b * input_stride, m, k);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            B(ot_data + b * other_stride, k, n);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            C(out_data + b * output_stride, m, n);
        C = A * B;
    }

    return output;


}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    // =================================== 作业 ===================================
    // TODO：实现CPU上的矩阵乘法反向传播
    // REF:
    // =================================== 作业 ===================================
    CHECK(input);
    CHECK(other);
    CHECK(grad_output);
    CHECK(input->Dtype()== DataType::kFLOAT32);
    CHECK(other->Dtype()== DataType::kFLOAT32);
    CHECK(grad_output->Dtype()== DataType::kFLOAT32);
    CHECK(input->GetDevice().Type()== DeviceType::kCPU);
    CHECK(other->GetDevice().Type()== DeviceType::kCPU);
    CHECK(grad_output->GetDevice().Type()== DeviceType::kCPU);

    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const auto &go_dims = grad_output->Dims();
    CHECK_GE(input_dims.size(), 2);
    CHECK_GE(other_dims.size(), 2);
    CHECK_EQ(input_dims.back(), other_dims[other_dims.size() - 2]);

    const int64_t m = input_dims[input_dims.size() - 2];
    const int64_t k = input_dims[input_dims.size() - 1];
    const int64_t n = other_dims[other_dims.size() - 1];
    CHECK_EQ(go_dims[go_dims.size() - 2], m);
    CHECK_EQ(go_dims[go_dims.size() - 1], n);

    int64_t input_batch = 1;
    for (size_t i = 0; i + 2 < input_dims.size(); ++i) {
        input_batch *= input_dims[i];
    }
    int64_t other_batch = 1;
    for (size_t i = 0; i + 2 < other_dims.size(); ++i) {
        other_batch *= other_dims[i];
    }
    CHECK(input_batch == other_batch || input_batch == 1 || other_batch == 1);

    const int64_t batch_size = std::max(input_batch, other_batch);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32);

    const float *in_data = static_cast<const float *>(input->DataPtr());
    const float *ot_data = static_cast<const float *>(other->DataPtr());
    const float *go_data = static_cast<const float *>(grad_output->DataPtr());
    float *gi_data = static_cast<float *>(grad_input->DataPtr());
    float *go2_data = static_cast<float *>(grad_other->DataPtr());

    const int64_t input_stride = (input_batch == 1) ? 0 : (m * k);
    const int64_t other_stride = (other_batch == 1) ? 0 : (k * n);
    const int64_t go_stride = m * n;

    // broadcast needs accumulation
    if (input_batch == 1 && batch_size > 1) {
        grad_input->Fill<float>(0.0f);
    }
    if (other_batch == 1 && batch_size > 1) {
        grad_other->Fill<float>(0.0f);
    }

    for (int64_t b = 0; b < batch_size; ++b) {
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            dC(go_data + b * go_stride, m, n);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            A(in_data + b * input_stride, m, k);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            B(ot_data + b * other_stride, k, n);

        // dA = dC * B^T
        if (input_batch == 1 && batch_size > 1) {
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                dA(gi_data, m, k);
            dA += dC * B.transpose();
        } else {
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                dA(gi_data + b * (m * k), m, k);
            dA = dC * B.transpose();
        }

        // dB = A^T * dC
        if (other_batch == 1 && batch_size > 1) {
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                dB(go2_data, k, n);
            dB += A.transpose() * dC;
        } else {
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                dB(go2_data + b * (k * n), k, n);
            dB = A.transpose() * dC;
        }
    }

    return {grad_input, grad_other};


}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {
    /*
    transpose:  output = input * weight^T + bias
    output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]

    !transpose: output = input * weight + bias
    output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    const int out_features = weight_dims[transpose ? 0 : 1];

    if (bias) {
        const auto &bias_dims = bias->Dims();
        CHECK_EQ(bias_dims.size(), 1);
        CHECK_EQ(bias_dims[0], out_features);
    }

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    if (transpose) {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix().transpose();
    } else {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix();
    }

    if (bias) {
        output->EigenMatrix().rowwise() += bias->EigenVector();
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output, const bool bias) {
    /*
    transpose: grad_input = grad_output * weight
    grad_input[*, in_features] = grad_output[*, out_features] * weight[out_features, in_features]
    grad_weight[out_features, in_features] = grad_output[*, out_features]^T * input[*, in_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)

    !transpose: grad_input = grad_output * weight^T
    grad_input[*, in_features] = grad_output[_, out_features] * weight[in_features, out_features]^T
    grad_weight[in_features, out_features] = input[*, in_features]^T * grad_output[*, out_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32);
    }

    if (transpose) {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix();
        grad_weight->EigenMatrix() = grad_output->EigenMatrix().transpose() * input->EigenMatrix();
    } else {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix().transpose();
        grad_weight->EigenMatrix() = input->EigenMatrix().transpose() * grad_output->EigenMatrix();
    }
    if (bias) {
        grad_bias->EigenVector() = grad_output->EigenMatrix().colwise().sum();
    }

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_LINEAR_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_LINEAR_KERNEL(MatmulForward)
REGISTER_CPU_LINEAR_KERNEL(MatmulBackward)
REGISTER_CPU_LINEAR_KERNEL(LinearForward)
REGISTER_CPU_LINEAR_KERNEL(LinearBackward)

#undef REGISTER_CPU_LINEAR_KERNEL