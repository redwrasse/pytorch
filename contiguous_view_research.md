# PyTorch Contiguous() and View() Issues Research

Research conducted on 2025-11-11 exploring interesting problems related to `contiguous()` and `view()` operations in PyTorch, with emphasis on mathematically complex operations.

## Table of Contents

1. [FFT and Signal Processing Issues](#fft-and-signal-processing-issues)
2. [Matrix Operations (Matmul, Einsum)](#matrix-operations-matmul-einsum)
3. [Attention Mechanisms and Transformers](#attention-mechanisms-and-transformers)
4. [Memory Layout and Channels](#memory-layout-and-channels)
5. [Normalization Layers](#normalization-layers)
6. [Torch.compile and Inductor Issues](#torchcompile-and-inductor-issues)
7. [Indexing Operations (Gather, Scatter)](#indexing-operations-gather-scatter)
8. [Unfold and Im2col Operations](#unfold-and-im2col-operations)
9. [Complex Number Operations](#complex-number-operations)
10. [PrimTorch and Compiler Infrastructure](#primtorch-and-compiler-infrastructure)

---

## FFT and Signal Processing Issues

### Issue #71321: torch.fft.fft/ifft give incorrect output on contiguous permuted tensors
**Status:** Closed
**Severity:** High - Correctness Issue

**Problem:**
`torch.fft.fft2` and `torch.fft.ifft2` produce incorrect outputs when operating on permuted tensors that have been made contiguous. The bug manifests when FFT operations are applied along corresponding dimensions after permutation.

**Key Code Example:**
```python
X1 = torch.randn((1, 100, 50, 8), dtype=torch.complex64)
x1 = torch.fft.ifft2(X1, norm="ortho", dim=(1, 2))

X2 = X1.permute((0, 3, 1, 2)).contiguous()
x2 = torch.fft.ifft2(X2, norm="ortho", dim=(-2, -1))
x2 = x2.permute(0, 2, 3, 1)

# Error: 0.0048 (should be ~0)
```

**Root Cause:**
- Spectral functions assume (but never check) that input tensor is contiguous
- If `.contiguous()` is removed, the error is 0
- Bug is related to memory layout for complex tensors
- MKL may choose different algorithms based on memory layout

**Impact:**
- Complex64 shows larger errors (~1e-6 fractional) than complex128
- Different normalization modes produce varying error magnitudes
- Affects any FFT-based computations including signal processing and convolution implementations

---

### Issue #106623: Meta implementations of FFT operators often have incorrect strides
**Status:** Open

**Problem:**
The `aten._fft_c2r` and `aten._fft_r2c` meta implementations currently return contiguous strides in all cases, which is not consistent with eager mode behavior.

**Impact:**
- Inconsistency between meta functions and eager execution
- Affects shape inference and compilation pipelines
- Can cause unexpected behavior in torch.export and torch.compile

---

### Issue #38413: CUDA irfft may be doing unnecessary cloning of input
**Status:** Open

**Problem:**
Out-of-place complex-to-real FFT will overwrite input buffer if custom strides are set by the user.

**Implications:**
- Relates to contiguity requirements for FFT operations
- May cause unexpected memory behavior
- Unnecessary clones hurt performance

---

## Matrix Operations (Matmul, Einsum)

### Issue #30303: reshape with non-contiguous input has wrong gradient on CUDA (breaking einsum)
**Status:** Closed - Fixed
**Severity:** Critical - Gradient Correctness

**Problem:**
`einsum` with non-contiguous tensor input cannot back-propagate correct gradients on CUDA. The issue stems from incorrect gradient computation when using reshape or `as_strided` operations on non-contiguous tensors.

**Key Code Example:**
```python
import torch
torch.manual_seed(42)
device = torch.device('cuda:0')

_x = torch.rand(1, 2, device=device)
x1 = _x.expand(2, -1).requires_grad_()

# Non-contiguous gradient (WRONG)
print(torch.autograd.grad(x1.reshape(1, 2, 2), x1,
      torch.eye(2, device=x1.device)[None]))
# Output: [[0.5, 0.5], [0.5, 0.5]]

# Contiguous gradient (CORRECT)
print(torch.autograd.grad(x1.contiguous().reshape(1, 2, 2), x1,
      torch.eye(2, device=x1.device)[None]))
# Output: [[1.0, 0], [0, 1.0]]
```

**Root Cause:**
- `as_strided_backward` produces invalid gradients under the assumption that the next backward step will be `SumToSize` collapsing the expanded dimensions
- This breaks when gradients are accessed directly at leaf nodes
- The issue specifically affects CUDA tensors

**Impact:**
- Affects all operations using einsum with non-contiguous inputs
- Breaks many mathematical operations that rely on correct gradients
- Can silently produce wrong training results

---

### Issue #18862: matmul uses too much memory in some batched cases
**Status:** Open

**Problem:**
Unnecessary contiguous call in matmul code where `tensor1.expand(tensor1_expand_size).contiguous().view()` could potentially use `reshape()` instead to avoid copying.

**Impact:**
- Memory inefficiency in batched matrix multiplication
- Performance degradation due to unnecessary copies

---

### Issue #16594: matrix multiplication sometimes can launch very unoptimized kernels
**Status:** Open

**Problem:**
Matmul with `(4, 4) matmul (4, 400K)` non-contiguous tensor takes 40ms while contiguous tensor takes 0.5ms - 80x performance difference!

**Impact:**
- Massive performance degradation with non-contiguous tensors
- No warning to users about the performance issue
- Can make certain computational patterns unexpectedly slow

---

## Attention Mechanisms and Transformers

### Issue #112577: [Breaking change 2.1] Passing non-contiguous inputs to SDPA on CUDA device with the mem-efficient attention backend returns garbage
**Status:** Closed - Fixed in 2.2.0
**Severity:** Critical - Breaking Change

**Problem:**
A breaking change between PyTorch 2.0.1 and 2.1 affecting Scaled Dot-Product Attention (SDPA). The memory-efficient attention backend outputs wrong results when passing non-contiguous key/value tensors, while the math backend works correctly.

**Behavior Differences:**
- PyTorch 2.0.1: dispatches to math backend for fp32 inputs with custom attention masks
- PyTorch 2.1+: dispatches to memory-efficient attention, which produces incorrect results

**Test Results:**
- CPU math backend: 0% difference with non-contiguous tensors
- CUDA math backend: 0% difference
- CUDA memory-efficient backend: ~58.9% mean relative difference!

**Technical Details:**
The issue occurs specifically when:
- Using fp32 data type
- Running on CUDA devices
- Applying custom attention masks
- Passing non-contiguous key/value tensors through expanded operations

**Impact:**
- Broke existing code in PyTorch 2.1
- Required explicit `.contiguous()` workarounds
- Affected transformer models using custom attention masks

---

### Issue #159126: torch.export with nn.Transformer creates a non-contiguous memory tensor for aten.view
**Status:** Open
**Severity:** High

**Problem:**
`torch.export.export()` issue when exporting a transformer layer model where the exported program includes `torch.ops.aten.view` ops with an input tensor that doesn't keep a contiguous memory_format.

**Root Cause:**
The `.view()` call requires the tensor to be contiguous in memory, but the input tensor has non-contiguous strides due to a previous permute operation.

**Impact:**
- Breaks export functionality for transformer models
- Makes it difficult to deploy models using torch.export

---

### Issue #148827: Inductor may permute inputs to flex attention, leading to assertion error
**Status:** Open

**Problem:**
For flex attention, inputs must be contiguous, but inductor seems to permute inputs under certain conditions which then results in an assertion error. Even when the eager version is contiguous, the compilation may fail due to query not being contiguous.

**Impact:**
- Affects FlexAttention compilation
- Can break working eager mode code when compiled

---

### Issue #134471: Cryptic error on non contiguous input to compiled flex attention
**Status:** Open

**Problem:**
Runtime error when using flex attention with compiled model and non-contiguous input with a cryptic error that makes it hard to pinpoint.

**Impact:**
- Poor user experience due to unclear error messages
- Difficult to debug

---

### Issue #109607: "RuntimeError: (*bias): last dimension must be contiguous" with F.scaled_dot_product_attention + torch.compile
**Status:** Open

**Problem:**
Issues with `F.scaled_dot_product_attention` and `torch.compile` where the last dimension must be contiguous.

**Impact:**
- Limits usability of torch.compile with attention mechanisms

---

## Memory Layout and Channels

### Issue #113437: [1, C, H, W] channels_last contiguous tensor view as [1, C, H*W] mismatch new strides
**Status:** Closed - Working as Intended

**Problem:**
When reshaping a `[1,3,4,5]` channels_last tensor to `[1,3,20]`, the resulting stride is `(3, 1, 3)` rather than the expected `(60, 1, 3)`.

**Test Cases:**
- `[2,3,4,5]` NCHW format: stride `(60, 20, 1)` ✓
- `[2,3,4,5]` NHWC format: stride `(60, 1, 3)` ✓
- `[1,3,4,5]` NCHW format: stride `(60, 20, 1)` ✓
- `[1,3,4,5]` NHWC format: stride `(3, 1, 3)` ✗ (anomalous)

**Resolution:**
PyTorch collaborator closed as expected behavior: "The stride for dimensions of size 1 can be anything (since it is never used)."

**Insight:**
This issue demonstrates the subtlety of stride semantics when dimensions have size 1.

---

### Issue #82060: Wrong output of single-channel channels_last convolution with channels_first input
**Status:** Open
**Severity:** High - Correctness

**Problem:**
Running convolution with `torch.channels_last` memory format for the weights but contiguous input produces wrong output.

**Impact:**
- Correctness issue with mixed memory formats
- Can silently produce incorrect results

---

### Issue #32088: Batch convolutional layer with 5d weight tensor that is not contiguous
**Status:** Open

**Problem:**
A batch of weight tensors with shape `(b, out_features, in_features, kernel_height, kernel_width)` which is not contiguous cannot view without copying. The view to 4D requires expensive copy operations when tensors are not contiguous.

**Impact:**
- Performance degradation in batched convolution
- Memory overhead from copying

---

### Issue #43195: one_hot tensors are channels_last but marked as contiguous
**Status:** Open

**Problem:**
`one_hot` tensors are marked as contiguous but actually use channels_last layout.

**Impact:**
- Confusion about tensor layout
- May affect performance of subsequent operations

---

## Normalization Layers

### Issue #33168: Conv Transposed + Layer Norm = "RuntimeError: columns needs to be contiguous"
**Status:** Closed - Fixed

**Problem:**
Runtime error when combining transposed convolution with layer normalization in PyTorch 1.4. The error occurs during backpropagation: "RuntimeError: columns needs to be contiguous"

**Reproduction:**
1. Applying a `ConvTranspose1d` layer to input
2. Transposing dimensions and applying `LayerNorm`
3. Calling `.backward()` on the result

Even explicitly calling `.contiguous()` after each operation didn't prevent the failure.

**Resolution:**
Fixed through PR #33462 ("Make slow_conv_transpose2d_backward tensors contiguous"), which addressed the underlying memory format problem in the transposed convolution backward pass.

**Impact:**
- Broke common architectural patterns combining conv transpose and layer norm
- Fixed in nightly builds as of February 2020

---

### Issue #70440: 3D Convolution Transpose + Layer norm leads to error RuntimeError: ones needs to be contiguous
**Status:** Open

**Problem:**
Similar to #33168 but for 3D convolutions. "RuntimeError: ones needs to be contiguous" when combining 3D transpose convolution with LayerNorm.

**Impact:**
- Affects 3D models (video, medical imaging)
- Workaround requires manual `.contiguous()` calls

---

### Issue #28201: Optimize GroupNorm in PyTorch
**Status:** Open

**Problem:**
GroupNorm implementation reshapes the input and uses BatchNorm to compute moments, which is inefficient for both CPU and CUDA. The reshape operations may trigger contiguous calls.

**Impact:**
- Performance issues with GroupNorm
- Unnecessary memory operations

---

## Torch.compile and Inductor Issues

### Issue #137372: inductor shape padding of activations is bad for compiled autograd
**Status:** Closed - Fixed
**Severity:** High

**Problem:**
Inductor's shape padding optimization for activations causes failures in compiled autograd.

**Error:**
"Cannot view a tensor with shape torch.Size([s0, s2]) and strides (s3, 1) as a tensor with shape (s0*s2,)!"

**Root Cause:**
1. **Intermediate activations get padded**: Inductor optimizes stride layouts for performance (e.g., `empty_strided_cuda((8, 1308), (1312, 1))`) on tensors saved for backward that aren't user-visible outputs.

2. **Non-contiguous tensors fail view operations**: During compiled autograd's backward graph tracing, the padded activation attempts a `.view(-1)` operation, which requires contiguity and fails.

3. **FX tracing materializes the view**: Unlike standard autograd (which lowers views directly), compiled autograd must trace views as intermediate nodes, exposing stride incompatibilities.

**Key Insight:**
The error only manifests with compiled autograd enabled. Without it, views are lowered directly without materializing as intermediate FX nodes.

**Resolution:**
Resolved by ensuring the AOT backward module applies post-grad passes (specifically `view_to_reshape`) before compiled autograd inlines it into the backward graph.

---

### Issue #100086: [Inductor] Striding problems arising from complex operations in torch.compile
**Status:** Closed - Completed
**Module:** complex, oncall: pt2

**Problem:**
Using `torch.compile` with the Inductor backend fails when complex operations (specifically rotary embeddings via complex multiplication) are embedded in a larger model.

**Error:**
"Tensor must have a last dimension with stride 1"

**Occurs when:**
1. Complex tensor operations are performed using `torch.view_as_complex()` and `torch.view_as_real()`
2. These operations are followed by tensor reshaping and matrix multiplication

**Key Observations:**
- When the rotary embedding section is disabled, the code compiles successfully
- Isolates the problem to how Inductor handles stride information from complex operations
- PyTorch minifier couldn't fully minimize the reproduction case, suggesting interactions between multiple operations

---

### Issue #90454: Inductor bug because of aten.clone lowering doesn't support memory_format
**Status:** Open

**Problem:**
Related to permute operations followed by view calls. Inductor's lowering of `aten.clone` doesn't properly support memory_format parameter.

**Impact:**
- Can cause incorrect compilation behavior
- May silently produce wrong results

---

### Issue #146390: CPU-specific Inductor Error with `view` on `torch.nn.Embedding` output
**Status:** Open

**Problem:**
Works with CUDA but fails on CPU when using view operations on embedding outputs.

**Impact:**
- Platform-specific behavior
- Makes code non-portable

---

### Issue #151156: Inductor doesn't support tensor.view(dtype).copy_()
**Status:** Open
**Priority:** High - Correctness

**Problem:**
Inductor doesn't support the pattern `tensor.view(dtype).copy_(...)`.

**Impact:**
- High priority correctness issue
- Limits supported patterns in compiled code

---

## Indexing Operations (Gather, Scatter)

### Issue #36956: advanced indexing backwards is broken on (some) non-contiguous index tensors
**Status:** Closed - Fixed

**Problem:**
PyTorch's backward pass for advanced indexing fails when using non-contiguous index tensors with certain memory formats.

**Error:**
"RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead."

**Reproduction Code:**
```python
import torch
shape = (2,8,1,2)
i=torch.randint(1, shape, device='cuda').contiguous(memory_format=torch.channels_last)
x=torch.randn(shape, requires_grad=True, device='cuda')
x[i].sum().backward()
```

**Root Cause:**
In `aten/src/ATen/native/cuda/Indexing.cu` at line 109. Previously, in PyTorch 1.4, the indexing operation consistently returned contiguous tensors. After moving functionality to ATen in later versions, the function can now return non-contiguous tensors, which causes problems during the backward pass.

**Resolution:**
Fixed through PRs #36957 and #36959 that ensure either the remainder or the linear index of advanced indexing operations remain contiguous during backpropagation.

---

### Issue #1193: gather / scatter dont respect non-contigous out Tensor
**Status:** Open

**Problem:**
`torch.gather`, `torch.scatter`, and `torch.index_select` don't respect non-contiguous output tensors.

**Impact:**
- Can produce unexpected results when output tensor is pre-allocated with non-standard layout
- No error or warning is raised

---

### Issue #22513: No assertion when using scatter_ on a non-contiguous tensor
**Status:** Open

**Problem:**
`scatter_` behaves unexpectedly on non-contiguous tensors, with the output differing from equivalent contiguous tensors. The core problem is about overlapping memory locations rather than just contiguity, which is hard to efficiently detect.

**Impact:**
- Silent incorrect behavior
- Difficult to debug

---

### Issue #1631: Gather backward is incorrect with repeated indices
**Status:** Open

**Problem:**
The gather function gives incorrect gradients on both CPU and GPU when using repeated indices, with no warnings or errors raised.

**Impact:**
- Incorrect gradients can break training
- No indication of the problem to users

---

## Unfold and Im2col Operations

### Issue #60466: PyTorch unfold could be faster
**Status:** Open

**Problem:**
PyTorch's `unfold` implementation is significantly slower than a custom implementation using `as_strided`.

**Performance Comparison:**
- Speed: The stride-based approach is up to 20x faster depending on kernel and stride parameters
- Memory: Uses approximately half the memory compared to `F.unfold`

**Specific test cases:**
- Kernel=64, stride=8: ~48% faster, ~53% memory usage
- Kernel=2048, stride=190: ~96% faster, ~46% memory usage

**Technical Approach:**
The proposed solution leverages `torch.Tensor.as_strided()` to create views of the input tensor rather than materializing copies. This technique manipulates stride information to expose the unfolded structure without allocating additional memory.

**Related Findings:**
`torch.Tensor.unfold()` (different from `F.unfold()`) performs approximately 1.5x faster than the stride-based approach while maintaining comparable memory efficiency.

**Impact:**
- Performance bottleneck in operations using unfold
- Opportunity for significant optimization

---

### Issue #33452: 2nd derivative for im2col and col2im not implemented
**Status:** Open

**Problem:**
Second derivative for `im2col_backward` is not implemented, causing issues with gradient penalties.

**Impact:**
- Breaks double backward through convolution operations
- Limits applicability of certain regularization techniques

---

### Issue #44989: Support unfold for integral types (long, byte etc) tensors
**Status:** Open

**Problem:**
Unfold internally calls `torch._C._nn.im2col`, which produces error `"im2col_out_cpu" not implemented for 'Long'` for integral tensor types.

**Impact:**
- Limits unfold to floating point types
- No support for integer-based operations

---

### Issue #98143: [pt2] `add` + `unfold` + `abs_` produces wrong results
**Status:** Open
**Severity:** Correctness

**Problem:**
In compiled code, unfold is implemented using `torch.ops.aten.as_strided`, and the combination of `add` + `unfold` + `abs_` produces incorrect results.

**Impact:**
- Correctness issue in PT2
- Silent wrong results

---

## Complex Number Operations

### Issue #150050: `torch.view_as_complex()` does not work on memory layout produced by `torch.contiguous()` after transpose
**Status:** Open
**Labels:** module: complex, triaged

**Problem:**
`torch.view_as_complex()` fails after calling `transpose().contiguous()` on a tensor, despite the contiguous operation appearing to succeed.

**Working case:**
```python
x = torch.rand(336, 1, 2)
print_strides(x)  # Output: 2 2 1
torch.view_as_complex(x)  # Success!
```

**Failing case:**
```python
x = torch.rand(336, 2, 1)
x = x.transpose(1, 2).contiguous()
print_strides(x)  # Output: 2 1 1
torch.view_as_complex(x)  # RuntimeError
```

**Error Message:**
"Tensor must have a stride divisible by 2 for all but last dimension"

**Technical Details:**
The function requires non-final dimension strides to be even multiples, but the contiguous operation produces strides of `[2, 1, 1]` where the middle dimension's stride (1) violates this requirement.

**Impact:**
- Breaks common patterns with complex numbers
- `.contiguous()` doesn't guarantee compatibility with `view_as_complex()`

---

## PrimTorch and Compiler Infrastructure

### Issue #84618: [primTorch] view size is not compatible with input tensor's size and stride
**Status:** Closed - Fixed

**Problem:**
Discrepancy in tensor memory layout handling between PyTorch's eager mode and primTorch when processing strided tensors through layer normalization.

**Core Issue:**
"With eager mode, `layer_norm` returns a contiguous tensor when its input is a strided tensor. However, primTorch will return a strided tensor like the input."

**Impact:**
This incompatibility causes runtime failures when downstream operations (like `linear`) attempt to reshape the output, triggering: "RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces)."

**Resolution Path:**
1. Explicitly calling `.contiguous()` on layer norm outputs
2. Mapping `view` operations to `reshape` during tracing to handle non-contiguous tensors automatically

**Resolution:**
Fixed after PR #84799 implemented returning contiguous tensors from the native layer norm reference implementation.

---

### Issue #78050: RFC: [primTorch] Stride-agnostic Operator Semantics
**Status:** Open - RFC

**Problem:**
Discusses the primTorch project's approach to stride consistency and view semantics.

**Impact:**
- Fundamental design decisions for PyTorch's compiler infrastructure
- Affects how strides are handled throughout the stack

---

## Additional Notable Issues

### Issue #3653: Support view() on batch dimensions for non-contiguous tensors?
**Status:** Open

**Problem:**
Discusses the need for view operations on batch dimensions and the extra `.contiguous()` calls required for non-contiguous tensors.

**Impact:**
- Particularly important for 4D+ batch matrix multiplication
- Requires explicit contiguous calls, hurting performance

---

### Issue #764: view() after transpose() raises non contiguous error
**Status:** Closed

**Problem:**
Classic example of the common issue where calling `.view()` after `.transpose()` fails due to non-contiguity.

**Impact:**
- Very common user pain point
- Requires understanding of memory layout

---

### Issue #28090: [discussion] Smarter version of torch.reshape (can avoid realloc in some cases)
**Status:** Open

**Problem:**
Discussion about avoiding unnecessary reallocations in reshape operations.

**Impact:**
- Performance optimization opportunity
- Could reduce memory usage

---

### Issue #141295: Memory efficient reshape for stride-0 tensors
**Status:** Open

**Problem:**
If a reshape that cannot be done as a view is performed on a stride-0 tensor (broadcasted), the whole tensor is made full sized and contiguous, resulting in a huge allocation.

**Impact:**
- Memory explosion with broadcasted tensors
- Can cause OOM errors unexpectedly

---

### Issue #62027: `x.to(memory_format=torch.contiguous_format)` does not always return a contiguous tensor
**Status:** Open

**Problem:**
Calling `.to(memory_format=torch.contiguous_format)` doesn't always produce a truly contiguous tensor.

**Impact:**
- Confusing API behavior
- May not achieve intended effect

---

### Issue #116333: `.contiguous()` doesn't work with one non-singleton dimension
**Status:** Open

**Problem:**
When calling `.contiguous()` on a tensor with one non-singleton dimension, it doesn't change stride.

**Impact:**
- Surprising behavior
- May not fix contiguity issues as expected

---

## Summary of Key Patterns

### Common Themes:

1. **Gradient Correctness Issues**: Many issues involve incorrect gradients when using non-contiguous tensors (e.g., #30303, #36956, #1631)

2. **Compiler Infrastructure Challenges**: torch.compile and Inductor have multiple issues handling non-contiguous tensors and view operations (e.g., #137372, #100086, #146390)

3. **Attention Mechanism Complexity**: Attention operations (SDPA, FlexAttention) have strict contiguity requirements that can break with compilation (e.g., #112577, #148827, #159126)

4. **Memory Layout Ambiguity**: Channels_last and other memory formats interact poorly with view operations (e.g., #113437, #82060, #43195)

5. **Performance Cliffs**: Non-contiguous tensors can cause massive performance degradation (80x slower in some cases) without warning (e.g., #16594)

6. **Complex Number Special Cases**: Complex operations have additional stride requirements that interact poorly with standard contiguity operations (e.g., #150050, #71321, #100086)

7. **FFT Bugs**: FFT operations assume contiguity but don't check for it, leading to silent correctness issues (e.g., #71321)

8. **Normalization Layer Issues**: LayerNorm and BatchNorm interact poorly with transposed convolutions (e.g., #33168, #70440)

### Most Severe Issues:

1. **#30303** - Gradient correctness bug in einsum/reshape
2. **#112577** - SDPA breaking change causing incorrect results
3. **#71321** - FFT producing wrong results on contiguous permuted tensors
4. **#36956** - Advanced indexing backward pass failures
5. **#137372** - Compiled autograd failures with view operations

### Areas Needing Improvement:

1. Better documentation of contiguity requirements
2. More warnings when operations will be slow due to non-contiguity
3. Automatic detection and handling of view-incompatible strides
4. Consistency between eager and compiled modes
5. Better error messages pointing to the actual problem
6. More comprehensive testing of non-contiguous paths
