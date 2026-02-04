#include <cstddef>
#include <memory>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
        static_cast<float *>(tensor->DataPtr())[idx] += rate * static_cast<const float *>(gradient->DataPtr())[idx];
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF:
    // =================================== 作业 ===================================
    //   m = beta1*m + (1-beta1)*g
    //   v = beta2*v + (1-beta2)*g^2
    //   m_hat = m / (1 - beta1^t)
    //   v_hat = v / (1 - beta2^t)
    //   param -= lr * m_hat / (sqrt(v_hat) + eps)

    const int64_t n = grad->NumElements();

    const float *g_ptr = static_cast<const float *>(grad->DataPtr());
    float *p_ptr = static_cast<float *>(param->DataPtr());
    float *m_ptr = static_cast<float *>(m->DataPtr());
    float *v_ptr = static_cast<float *>(v->DataPtr());

    // Use double pow to match the test's std::pow behavior closely.
    const float bc1 = 1.0f - static_cast<float>(std::pow(static_cast<double>(beta1), static_cast<double>(t)));
    const float bc2 = 1.0f - static_cast<float>(std::pow(static_cast<double>(beta2), static_cast<double>(t)));

    for (int64_t i = 0; i < n; ++i) {
        const float g = g_ptr[i];

        const float mi = beta1 * m_ptr[i] + (1.0f - beta1) * g;
        const float vi = beta2 * v_ptr[i] + (1.0f - beta2) * g * g;

        m_ptr[i] = mi;
        v_ptr[i] = vi;

        const float m_hat = mi / bc1;
        const float v_hat = vi / bc2;

        p_ptr[i] -= learning_rate * m_hat / (static_cast<float>(std::sqrt(static_cast<double>(v_hat))) + eps);
    }
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
