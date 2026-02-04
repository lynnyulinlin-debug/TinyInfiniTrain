#pragma once

#include <iostream>
#include <map>
#include <type_traits>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/device.h"

namespace infini_train {
class KernelFunction {
public:
    template <typename FuncT> explicit KernelFunction(FuncT &&func) : func_ptr_(reinterpret_cast<void *>(func)) {}

    template <typename RetT, class... ArgsT> RetT Call(ArgsT... args) const {
        // =================================== 作业 ===================================
        // TODO：实现通用kernel调用接口
        // 功能描述：将存储的函数指针转换为指定类型并调用
        // =================================== 作业 ===================================
        using FuncT = RetT (*)(ArgsT...);
        // TODO: 实现函数调用逻辑
        // 从统一的 void*还原函数指针类型
        auto func = reinterpret_cast<FuncT>(func_ptr_);
        return func(args...);
    }

private:
    void *func_ptr_ = nullptr;
};

class Dispatcher {
public:
    using KeyT = std::pair<DeviceType, std::string>;

    static Dispatcher &Instance() {
        static Dispatcher instance;
        return instance;
    }

    const KernelFunction &GetKernel(KeyT key) const {
        CHECK(key_to_kernel_map_.contains(key))
            << "Kernel not found: " << key.second << " on device: " << static_cast<int>(key.first);
        return key_to_kernel_map_.at(key);
    }

    template <typename FuncT> void Register(const KeyT &key, FuncT &&kernel) {
        // =================================== 作业 ===================================
        // TODO：实现kernel注册机制
        // 功能描述：将kernel函数与设备类型、名称绑定
        // =================================== 作业 ===================================

        CHECK(!key_to_kernel_map_.contains(key))
        << "Kernel already registered: " << key.second << " on device: " << static_cast<int>(key.first);
        // 将 kernel 函数封装为 KernelFunction，　统一类型后， 并存入 map
        key_to_kernel_map_.emplace(key, KernelFunction(std::forward<FuncT>(kernel)));
    }

private:
    std::map<KeyT, KernelFunction> key_to_kernel_map_;
};
} // namespace infini_train

//#define REGISTER_KERNEL(device, kernel_name, kernel_func)                                                      
    // =================================== 作业 ===================================
    // TODO：实现自动注册宏
    // 功能描述：在全局静态区注册kernel，避免显式初始化代码
    // =================================== 作业 ===================================
// 第一层：连接辅助宏，强制 __LINE__ 展开
#define REGISTER_KERNEL_CONCAT(device, kernel_name, kernel_func, line) \
    static struct KernelRegistrar_##kernel_name##_##line {             \
        KernelRegistrar_##kernel_name##_##line() {                     \
            ::infini_train::Dispatcher::Instance().Register(           \
                std::make_pair(device, #kernel_name), kernel_func);    \
        }                                                              \
    } g_kernel_reg_##kernel_name##_##line{};

// 第二层：展开 __LINE__
#define REGISTER_KERNEL_EXPAND(device, kernel_name, kernel_func, line) \
    REGISTER_KERNEL_CONCAT(device, kernel_name, kernel_func, line)

// 第三层：用户接口
#define REGISTER_KERNEL(device, kernel_name, kernel_func) \
    REGISTER_KERNEL_EXPAND(device, kernel_name, kernel_func, __LINE__)
    