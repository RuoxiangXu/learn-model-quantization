# Model Quantization

<div align="center">

[English](README.md) | [简体中文](README.zh-CN.md)

</div>

> Notes collected during my internship at ZTE (June–August 2024)

## Basic Concepts
> Source: [Model Quantization Explained](https://blog.csdn.net/WZZ18191171661/article/details/103332338); [Deep Learning Model Quantization (Low Precision Inference) Summary](https://blog.csdn.net/zlgahu/article/details/104662203)

In simple terms, model quantization is a model compression technique that converts floating-point storage (computation) to integer storage (computation). To put it simply, a weight that originally required float32 representation can now be represented using int8 after quantization. With just this operation alone, we can achieve nearly 4x network acceleration!

According to Wikipedia's definition of quantization: Quantization is the process of mapping a numerical value x to y, where the domain of x is a large set (usually continuous), and the domain of y is a small set (usually countable). In 8-bit low-precision inference, we convert an original FP32 weight/activation floating-point tensor into an int8/uint8 tensor for processing. Model quantization brings the following two benefits:

- **Reduced Memory Bandwidth and Storage Space:**
Deep learning models primarily record the weights and bias of each layer (such as convolutional layers/fully connected layers). In FP32 models, each weight value originally requires 32-bit storage space, which only needs 8-bit after quantization. Therefore, the model size will directly reduce to nearly 1/4. Not only is the model size significantly reduced, but using 8-bit for activation will also significantly reduce memory usage, which means the low-precision inference process will significantly reduce memory access bandwidth requirements and improve cache hit rate, especially for memory-bound element-wise operators like batch-norm, relu, and elementwise-sum.

- **Improved System Throughput and Reduced System Latency:**
Intuitively, consider a SIMD instruction with a dedicated register width of 512 bits. When the data type is FP32, one instruction can process 16 values at a time, but when we use 8-bit to represent data, one instruction can process 64 values at a time. Therefore, in this case, the theoretical computational peak of the chip can be increased by 4 times. On CPUs, Intel Xeon Scalable Processors' AVX-512 and VNNI advanced vector instructions support low-precision and high-precision accumulation operations.

## PyTorch Model Quantization
Official Documentation: [QUANTIZATION](https://pytorch.org/docs/master/quantization.html)
[PyTorch Quantization](https://zhuanlan.zhihu.com/p/299108528)

## ONNX Model Quantization
Official Documentation: [Quantize ONNX Models](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
Dynamic vs Static Quantization: [Model Quantization (3): Static and Dynamic Quantization of ONNX Models](https://blog.csdn.net/m0_63642362/article/details/124741589?)

### Quantization Methods
> Quantization in ONNX Runtime refers to 8 bit linear quantization of an ONNX model.

ONNX Runtime supports two model quantization methods:

- **Dynamic Quantization:**
For dynamic quantization, the scale factor (Scale) and zero point (Zero Point) are calculated during inference and are specific to each activation. Therefore, they are more accurate but introduce additional computational overhead.

- **Static Quantization:**
For static quantization, they are calculated offline using a calibration dataset. All activations have the same scale factor (Scale) and zero point (Zero Point).

Method Selection: Generally, it is recommended to **use dynamic quantization for RNN and Transformer-based models, and static quantization for CNN models**.

### Quantization Formats
There are two ways to represent quantized ONNX models:

- **Operator-Oriented (QOperator):**
All quantized operators have their own ONNX definitions, such as QLinearConv, MatMulInteger, etc.

- **Tensor-Oriented (QDQ; Quantize and DeQuantize):**
This format inserts DeQuantizeLinear(QuantizeLinear(tensor)) between original operators to simulate the quantization and dequantization process.
In static quantization, QuantizeLinear and DeQuantizeLinear operators also carry quantization parameters.
In dynamic quantization, a ComputeQuantizationParameters function prototype is inserted to calculate quantization parameters in real-time.

The following methods generate models in QDQ format:
- Models quantized using quantize_static with quant_format=QuantFormat.QDQ.
- Quantization-aware training (QAT) models converted from TensorFlow or exported from PyTorch.
- Quantized models converted from TFLite and other frameworks.

For the latter two cases mentioned above, you don't need to use quantization tools to quantize the model. ONNX Runtime can run them directly as quantized models.

### Quantization Steps
Must Read: [ONNX Runtime Quantization Example](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md) (The example is for static quantization)