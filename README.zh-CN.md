# Model Quantization 模型量化

<div align="center">

[English](README.md) | [简体中文](README.zh-CN.md)

</div>

> 记录自本人于 2024.6-2024.8 在中兴通讯实习期间的学习笔记

## 基本概念
> Source: [模型量化详解](https://blog.csdn.net/WZZ18191171661/article/details/103332338)；[深度学习模型量化（低精度推理）大总结](https://blog.csdn.net/zlgahu/article/details/104662203)
简而言之，所谓的模型量化就是将浮点存储（运算）转换为整型存储（运算）的一种模型压缩技术。简单直白点讲，即原来表示一个权重需要使用float32表示，量化后只需要使用int8来表示就可以了，仅仅这一个操作，我们就可以获得接近4倍的网络加速！
维基百科中关于量化（quantization）的定义是: 量化是将数值 x 映射到 y 的过程，其中 x 的定义域是一个大集合(通常是连续的)，而 y 的定义域是一个小集合（通常是可数的）。8-bit 低精度推理中， 我们将一个原本 FP32 的 weight/activation 浮点数张量转化成一个 int8/uint8 张量来处理。模型量化会带来如下两方面的好处：
- **减少内存带宽和存储空间：**
深度学习模型主要是记录每个 layer（比如卷积层/全连接层） 的 weights 和 bias, FP32 模型中，每个 weight 数值原本需要 32-bit 的存储空间，量化之后只需要 8-bit 即可。因此，模型的大小将直接降为将近 1/4。不仅模型大小明显降低， activation 采用 8-bit 之后也将明显减少对内存的使用，这也意味着低精度推理过程将明显减少内存的访问带宽需求，提高高速缓存命中率，尤其对于像 batch-norm， relu，elmentwise-sum 这种内存约束(memory bound)的 element-wise 算子来说，效果更为明显。
- **提高系统吞吐量（throughput），降低系统延时（latency）：**
直观理解，试想对于一个 专用寄存器宽度为 512 位的 SIMD 指令，当数据类型为 FP32 而言一条指令能一次处理 16 个数值，但是当我们采用 8-bit 表示数据时，一条指令一次可以处理 64 个数值。因此，在这种情况下，可以让芯片的理论计算峰值增加 4 倍。在CPU上，英特尔至强可扩展处理器的 AVX-512 和 VNNI 高级矢量指令支持低精度和高精度的累加操作。
## PyTorch模型量化
官方文档：[QUANTIZATION](https://pytorch.org/docs/master/quantization.html)
[PyTorch的量化](https://zhuanlan.zhihu.com/p/299108528)
## ONNX模型量化
官方文档：[Quantize ONNX Models](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
动态量化与静态量化：[模型量化（3）：ONNX 模型的静态量化和动态量化](https://blog.csdn.net/m0_63642362/article/details/124741589?)
### 量化方式
> Quantization in ONNX Runtime refers to 8 bit linear quantization of an ONNX model.
ONNXRuntime 支持两种模型量化方式：
- 动态量化：
对于动态量化，缩放因子（Scale）和零点（Zero Point）是在推理时计算的，并且特定用于每次激活。因此它们更准确，但引入了额外的计算开销
- 静态量化：
对于静态量化，它们使用校准数据集离线计算。所有激活都具有相同的缩放因子（Scale）和零点（Zero Point）
方法选择：通常，建议**对 RNN 和基于 Transformer 的模型使用动态量化，对 CNN 模型使用静态量化**。
### 量化格式
量化ONNX模型有两种表示方式：
- 面向操作符（QOperator）：
所有量化操作符都有自己的ONNX定义，如QLinearConv、MatMulInteger等。
- 面向张量（QDQ；Quantize和DeQuantize）：
这种格式在原始操作符之间插入DeQuantizeLinear(QuantizeLinear(tensor))来模拟量化和反量化过程。
在静态量化中，QuantizeLinear和DeQuantizeLinear操作符也携带量化参数。
在动态量化中，插入了一个ComputeQuantizationParameters函数原型来实时计算量化参数。

以下方式生成的模型采用QDQ格式：
- 使用quantize_static进行量化的模型，设置quant_format=QuantFormat.QDQ。
- 从TensorFlow转换或从PyTorch导出的量化感知训练（QAT）模型。
- 从TFLite和其他框架转换的量化模型。
对于后面提到的两种情况，你不需要使用量化工具来量化模型。ONNX Runtime 可以直接作为量化模型来运行它们。
### 量化步骤
必读：[ONNX Runtime Quantization Example](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md) （其中例子为静态量化）
