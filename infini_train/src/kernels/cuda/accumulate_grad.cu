#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"
#include <cmath>
#include "cuda_runtime_api.h"

namespace infini_train::kernels::cuda {

__global__ void AccumulateGradKernel(const float *grad_ptr, float rate, float *tensor_ptr, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor_ptr[idx] += rate * grad_ptr[idx];
    }
}

// ===== 新增：Adam kernel =====
__global__ void AdamAccumulateGradKernel(const float *g_ptr, float *p_ptr, float *m_ptr, float *v_ptr,
                                          size_t n, float lr, float beta1, float beta2, float eps,
                                          float one_minus_beta1_pow_t, float one_minus_beta2_pow_t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 1. 读取数据到寄存器
        float grad = g_ptr[idx];
        float m = m_ptr[idx];
        float v = v_ptr[idx];
        
        // 2. 更新一阶矩：m = β1*m + (1-β1)*g
        m = beta1 * m + (1.0f - beta1) * grad;
        
        // 3. 更新二阶矩：v = β2*v + (1-β2)*g²
        v = beta2 * v + (1.0f - beta2) * grad * grad;
        
        // 4. 写回动量
        m_ptr[idx] = m;
        v_ptr[idx] = v;
        
        // 5. 偏差修正
        float m_hat = m / one_minus_beta1_pow_t;
        float v_hat = v / one_minus_beta2_pow_t;
        
        // 6. 更新参数
        p_ptr[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);

    }
}

void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    size_t num_elements = gradient->NumElements();

    const float *grad_ptr = static_cast<const float *>(gradient->DataPtr());
    float *tensor_ptr = static_cast<float *>(tensor->DataPtr());

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AccumulateGradKernel<<<num_blocks, threads_per_block>>>(grad_ptr, rate, tensor_ptr, num_elements);
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF:
    // =================================== 作业 ===================================

    // 获取元素数量和数据指针
    size_t n = grad->NumElements();
    const float *g_ptr = static_cast<const float *>(grad->DataPtr());
    float *p_ptr = static_cast<float *>(param->DataPtr());
    float *m_ptr = static_cast<float *>(m->DataPtr());
    float *v_ptr = static_cast<float *>(v->DataPtr());
    
    // 在 CPU 端预计算偏差修正系数，避免 GPU 每个线程重复计算 pow()
    // bc1 = 1 - β1^t,  bc2 = 1 - β2^t
    float bc1 = 1.0f - powf(beta1, static_cast<float>(t));
    float bc2 = 1.0f - powf(beta2, static_cast<float>(t));
    
    // 配置 kernel 启动参数
    int threads_per_block = 256;  // 每个 block 256 个线程（常用配置）
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;  // 向上取整
    
    // 启动 kernel
    AdamAccumulateGradKernel<<<num_blocks, threads_per_block>>>(
        g_ptr, p_ptr, m_ptr, v_ptr, n, learning_rate, beta1, beta2, eps, bc1, bc2);

}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL