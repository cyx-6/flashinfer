====================
Kernel Library Guide
====================

FlashInfer now uses `apache-tvm-ffi`_ as python binding instead of PyTorch C++ extensions.

Tensor
======

Tensor is the most important input for a kernel libaray. In PyTorch C++ extensions, kernel library usually takes ``at::Tensor`` as tensor input. In TVM FFI, we introduce two types of tensor, ``ffi::Tensor`` and ``ffi::TensorView``.

Tensor and TensorView
---------------------

Both ``ffi::Tensor`` and ``ffi::TensorView`` are designed to represent tensors in TVM FFI eco-system. The main difference is whether it is an owning tensor pointer.

:ffi::Tensor:
 ``ffi::Tensor`` is a completely onwing tensor pointer, pointing to TVM FFI tensor object. TVM FFI handles the lifetime of ``ffi::Tensor`` by retaining a strong reference. 

:ffi::TensorView:
 ``ffi::TensorView`` is a light weight non-owning tnesor pointer, pointeing to a TVM FFI tensor or external tensor object. TVM FFI does not retain its reference. So users are responsible for ensuring the lifetime of tensor object to which the ``ffi::TensorView`` points. 

TVM FFI can automatically convert the input tensor at Python side, e.g. `torch.Tensor`, to both ``ffi::Tensor`` or ``ffi::TensorView`` at C++ side, depends on the C++ function arguments. However, for more flexibility and better compatibility, we **recommand** to use ``ffi::TensorView`` in practice. Since some frameworks, like JAX, cannot provide strong referenced tensor, as ``ffi::Tensor`` expected.

Tensor as Argument
------------------

Typically, we expect that all tensors are pre-allocated at Python side and passed in via TVM FFI, including the output tensor. And TVM FFI will convert them into ``ffi::TensorView`` at runtime. For the optional arguments, ``ffi::Optional`` is the best practice. Here is an example of a kernel definition at C++ side and calling at Python side.

.. code-block:: c++

 // Kernel definition
 void func(ffi::TensorView input, ffi::Optional<ffi::Tensor> optional_input, ffi::TensorView output, ffi::TensorView workspace);

.. code-block:: python

 # Kernel calling
 input = torch.tensor(...)
 output = torch.empty(...)
 workspace = torch.empty(...)
 func(input, None, output, workspace)

Ideally, we expect the kernel function to have ``void`` as return type. However, if it is necessary to return the ``ffi::Tensor`` anyway, please pay attention to convert the output ``ffi::Tensor`` to original tensor type at Python side, like ``torch.from_dlpack``.

Tensor Methods
--------------

``ffi::TensorView`` and ``ffi::Tensor`` expose the same set of methods for attributes retrieval, and aligned to most ``at::Tensor`` interface.

+----------------------------------+-----------------------------------+
|ffi::Tensor/ffi::TensorView       |at::Tensor                         |
+---------------+------------------+---------------+-------------------+
|Return         |Method            |Return         |Method             |
+---------------+------------------+---------------+-------------------+
|int32_t        |ndim()            |int64_t        |dim()              |
+---------------+------------------+---------------+-------------------+
|ffi::DLDataType|dtype()           |at::ScalarType |scalar_type()      |
+---------------+------------------+---------------+-------------------+
|ffi::ShapeView |shape()           |at::IntArrayRef|sizes()            |
+---------------+------------------+---------------+-------------------+
|ffi::ShapeView |strides()         |at::IntArrayRef|strides()          |
+---------------+------------------+---------------+-------------------+
|int64_t        |size(size_t idx)  |int64_t        |size(int64_t dim)  |
+---------------+------------------+---------------+-------------------+
|int64_t        |stride(size_t idx)|int64_t        |stride(int64_t dim)|
+---------------+------------------+---------------+-------------------+
|int64_t        |numel()           |int64_t        |numel()            |
+---------------+------------------+---------------+-------------------+
|uint64_t       |byte_offset()     |int64_t        |storage_offset()   |
+---------------+------------------+---------------+-------------------+
|void*          |data_ptr()        |void*          |data_ptr()         |
+---------------+------------------+---------------+-------------------+
|ffi::DLDevice  |device()          |at::Device     |device()           |
+---------------+------------------+---------------+-------------------+
|bool           |IsContiguous()    |bool           |is_contiguous()    |
+---------------+------------------+---------------+-------------------+




.. _apache-tvm-ffi: https://github.com/apache/tvm-ffi

