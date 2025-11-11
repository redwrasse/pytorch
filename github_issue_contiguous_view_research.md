# Comprehensive Research on PyTorch Contiguous() and View() Issues

**Research Date:** 2025-11-11
**Scope:** 50+ GitHub issues analyzed from pytorch/pytorch
**Focus:** Mathematically complex operations and their interaction with tensor memory layout

## Executive Summary

This research identifies critical patterns of failures, correctness bugs, and performance issues related to `contiguous()` and `view()` operations in PyTorch. The findings reveal systematic problems affecting:

- **Gradient correctness** (silent training failures)
- **Numerical accuracy** (FFT, attention mechanisms)
- **Performance cliffs** (up to 80x slowdown)
- **Compiler infrastructure** (eager vs compiled mode inconsistencies)

## 🔴 Critical Severity Issues

### 1. Gradient Correctness Bug in Einsum/Reshape
**Issue:** [pytorch/pytorch#30303](https://github.com/pytorch/pytorch/issues/30303)
**Status:** Fixed
**Impact:** Critical - produces incorrect gradients

Non-contiguous tensors through `reshape` or `as_strided` produce wrong gradients on CUDA:

```python
_x = torch.rand(1, 2, device='cuda')
x1 = _x.expand(2, -1).requires_grad_()

# Non-contiguous: WRONG gradient [[0.5, 0.5], [0.5, 0.5]]
torch.autograd.grad(x1.reshape(1, 2, 2), x1, torch.eye(2, device=x1.device)[None])

# Contiguous: CORRECT gradient [[1.0, 0], [0, 1.0]]
torch.autograd.grad(x1.contiguous().reshape(1, 2, 2), x1, torch.eye(2, device=x1.device)[None])
```

**Root Cause:** `as_strided_backward` produces invalid gradients assuming next backward step will collapse expanded dimensions.

---

### 2. SDPA Memory-Efficient Backend Returns Garbage
**Issue:** [pytorch/pytorch#112577](https://github.com/pytorch/pytorch/issues/112577)
**Status:** Fixed in PyTorch 2.2.0
**Impact:** Breaking change - 59% error rate

PyTorch 2.1 introduced a breaking change where Scaled Dot-Product Attention's memory-efficient backend outputs **wrong results** with non-contiguous key/value tensors:

- **CUDA math backend:** 0% error ✓
- **CUDA mem-efficient backend:** 58.9% mean relative error ✗

**User Impact:** Broke existing transformer code requiring explicit `.contiguous()` workarounds.

---

### 3. FFT Operations Produce Incorrect Results
**Issue:** [pytorch/pytorch#71321](https://github.com/pytorch/pytorch/issues/71321)
**Status:** Closed
**Impact:** Numerical correctness

`torch.fft.fft2` and `torch.fft.ifft2` produce incorrect outputs on permuted tensors made contiguous:

```python
X1 = torch.randn((1, 100, 50, 8), dtype=torch.complex64)
x1 = torch.fft.ifft2(X1, norm="ortho", dim=(1, 2))

X2 = X1.permute((0, 3, 1, 2)).contiguous()
x2 = torch.fft.ifft2(X2, norm="ortho", dim=(-2, -1))
x2 = x2.permute(0, 2, 3, 1)

# Error: 0.0048 (should be ~0)
# Without .contiguous(), error is 0!
```

**Root Cause:** FFT functions assume contiguity but never check for it. MKL may choose different algorithms based on memory layout.

---

### 4. Advanced Indexing Backward Broken
**Issue:** [pytorch/pytorch#36956](https://github.com/pytorch/pytorch/issues/36956)
**Status:** Fixed
**Impact:** Breaks backpropagation

Advanced indexing backward pass fails with non-contiguous index tensors:

```python
shape = (2,8,1,2)
i = torch.randint(1, shape, device='cuda').contiguous(memory_format=torch.channels_last)
x = torch.randn(shape, requires_grad=True, device='cuda')
x[i].sum().backward()  # RuntimeError!
```

---

### 5. Inductor Compiled Autograd Failures
**Issue:** [pytorch/pytorch#137372](https://github.com/pytorch/pytorch/issues/137372)
**Status:** Fixed
**Impact:** Breaks torch.compile with autograd

Inductor's shape padding optimization breaks view operations in compiled autograd:

```
Cannot view a tensor with shape torch.Size([s0, s2]) and strides (s3, 1)
as a tensor with shape (s0*s2,)!
```

Padding activations for performance creates non-contiguous tensors that fail `.view(-1)` during backward tracing.

---

## 📊 Performance Critical Issues

### 80x Performance Degradation with Non-Contiguous Matmul
**Issue:** [pytorch/pytorch#16594](https://github.com/pytorch/pytorch/issues/16594)
**Status:** Open

Matrix multiplication performance cliff:
- **Contiguous tensor:** 0.5ms
- **Non-contiguous tensor:** 40ms (80x slower!)

**Problem:** No warning to users, silent performance degradation.

---

### Unfold Implementation 20x Slower Than Necessary
**Issue:** [pytorch/pytorch#60466](https://github.com/pytorch/pytorch/issues/60466)
**Status:** Open

Current `F.unfold` implementation is significantly slower than an `as_strided`-based approach:

| Configuration | Speed Improvement | Memory Usage |
|--------------|-------------------|--------------|
| kernel=64, stride=8 | 48% faster | 53% of original |
| kernel=2048, stride=190 | 96% faster | 46% of original |

**Opportunity:** Unfold could use views instead of materializing copies.

---

## 🧠 Attention Mechanism Issues

Multiple attention-related contiguity issues found:

1. **[#159126](https://github.com/pytorch/pytorch/issues/159126)** - torch.export with nn.Transformer creates non-contiguous tensors for aten.view
2. **[#148827](https://github.com/pytorch/pytorch/issues/148827)** - Inductor permutes inputs to flex attention causing assertion errors
3. **[#134471](https://github.com/pytorch/pytorch/issues/134471)** - Cryptic errors with compiled flex attention and non-contiguous inputs
4. **[#109607](https://github.com/pytorch/pytorch/issues/109607)** - "last dimension must be contiguous" with SDPA + torch.compile

**Pattern:** Attention mechanisms have strict contiguity requirements that break during compilation.

---

## 🎨 Memory Layout Challenges

### Channels Last Format Issues

**[#113437](https://github.com/pytorch/pytorch/issues/113437)** - Channels_last tensors with batch_size=1 produce unexpected strides:

```python
x = torch.rand(1, 3, 4, 5).to(memory_format=torch.channels_last)
y = x.view(1, 3, 20)
# Stride: (3, 1, 3) instead of expected (60, 1, 3)
```

**Resolution:** Working as intended - stride for size-1 dimensions can be anything.

**[#82060](https://github.com/pytorch/pytorch/issues/82060)** - Wrong output with mixed memory formats in convolution:
- Weights: channels_last
- Input: contiguous
- Result: **Incorrect output**

---

## 🔧 Normalization Layer Problems

**[#33168](https://github.com/pytorch/pytorch/issues/33168)** - ConvTranspose + LayerNorm = "RuntimeError: columns needs to be contiguous"

Fixed in PyTorch 1.4+, but similar issue persists for 3D:

**[#70440](https://github.com/pytorch/pytorch/issues/70440)** - 3D ConvTranspose + LayerNorm still broken

---

## 🔍 Complex Number Operations

**[#150050](https://github.com/pytorch/pytorch/issues/150050)** - `view_as_complex()` fails after `transpose().contiguous()`

```python
x = torch.rand(336, 2, 1)
x = x.transpose(1, 2).contiguous()
# Stride: [2, 1, 1]
torch.view_as_complex(x)  # RuntimeError!
```

**Error:** "Tensor must have a stride divisible by 2 for all but last dimension"

**Problem:** `.contiguous()` doesn't guarantee compatibility with `view_as_complex()` requirements.

---

**[#100086](https://github.com/pytorch/pytorch/issues/100086)** - Striding problems with complex operations in torch.compile

Rotary embeddings using `view_as_complex()` and `view_as_real()` fail compilation with:
> "Tensor must have a last dimension with stride 1"

---

## 🏗️ Compiler Infrastructure Issues

### PrimTorch vs Eager Mode Inconsistencies

**[#84618](https://github.com/pytorch/pytorch/issues/84618)** - Layer norm stride inconsistency

- **Eager mode:** Returns contiguous tensor from layer_norm with strided input
- **PrimTorch:** Returns strided tensor like input

Causes downstream failures in Linear layers:
```
RuntimeError: view size is not compatible with input tensor's size and stride
```

---

### Inductor Issues Summary

1. **[#90454](https://github.com/pytorch/pytorch/issues/90454)** - aten.clone lowering doesn't support memory_format
2. **[#146390](https://github.com/pytorch/pytorch/issues/146390)** - CPU-specific error with view on Embedding output (works on CUDA)
3. **[#151156](https://github.com/pytorch/pytorch/issues/151156)** - tensor.view(dtype).copy_() not supported (HIGH PRIORITY)

---

## 📦 Indexing Operations

**[#1193](https://github.com/pytorch/pytorch/issues/1193)** - gather/scatter don't respect non-contiguous output tensors

**[#22513](https://github.com/pytorch/pytorch/issues/22513)** - No assertion when using scatter_ on non-contiguous tensor
- **Silent incorrect behavior**
- Core problem: overlapping memory locations

**[#1631](https://github.com/pytorch/pytorch/issues/1631)** - Gather backward incorrect with repeated indices
- Wrong gradients with no warning

---

## 🔄 Unfold and Im2col

**[#33452](https://github.com/pytorch/pytorch/issues/33452)** - 2nd derivative for im2col/col2im not implemented
- Breaks double backward through convolutions
- Limits gradient penalty techniques

**[#98143](https://github.com/pytorch/pytorch/issues/98143)** - add + unfold + abs_ produces wrong results in PT2
- **Correctness issue in compiled mode**

**[#44989](https://github.com/pytorch/pytorch/issues/44989)** - Unfold not supported for integral types
- Only works with floating point

---

## 🔑 Key Patterns Identified

### 1. Silent Failures
Many operations fail silently or produce incorrect results without warnings:
- FFT with permuted tensors
- Gather with repeated indices
- Scatter on non-contiguous tensors
- Matmul 80x performance degradation

### 2. Compiler Mode Inconsistencies
Eager mode works, compiled mode fails:
- Flex attention contiguity
- Complex operations with torch.compile
- PrimTorch vs eager stride handling

### 3. Gradient Correctness
Multiple gradient bugs with non-contiguous tensors:
- Einsum/reshape (#30303)
- Advanced indexing (#36956)
- Gather with repeated indices (#1631)

### 4. Memory Layout Assumptions
Operations assume contiguity but don't check:
- FFT operations (#71321)
- SDPA mem-efficient backend (#112577)
- Scatter operations (#22513)

### 5. Documentation Gaps
Many operations lack clear documentation about:
- Contiguity requirements
- Performance characteristics with non-contiguous inputs
- Behavior differences between memory formats

---

## 💡 Recommendations

### For PyTorch Core Development

1. **Add Runtime Checks**
   - FFT operations should verify contiguity or handle non-contiguous inputs
   - Indexing operations should warn about non-contiguous outputs

2. **Performance Warnings**
   - Warn when operations will be 10x+ slower due to memory layout
   - Add telemetry to identify common slow patterns

3. **Compiler Consistency**
   - Ensure eager and compiled modes have identical stride semantics
   - Test all operations with non-contiguous inputs in PT2

4. **Better Error Messages**
   - "view size is not compatible" should suggest using `.reshape()` or `.contiguous()`
   - Include stride information in error messages

5. **Documentation Improvements**
   - Document contiguity requirements for all operations
   - Add "Performance Considerations" section covering memory layout
   - Provide migration guide for memory format transitions

### For Research and Development

1. **Systematic Testing**
   - Create test suite covering all operations with:
     - Contiguous inputs
     - Non-contiguous inputs (transpose, permute, expand)
     - Different memory formats (contiguous, channels_last, etc.)

2. **Optimization Opportunities**
   - Unfold can be 20x faster with as_strided approach
   - Many operations could use views instead of copies

3. **Compiler Development**
   - Need comprehensive stride handling in Inductor
   - Better integration between AOT autograd and compiled autograd

---

## 📚 Full Research Document

Complete analysis with code examples and technical details: `contiguous_view_research.md` in branch `claude/pytorch-contiguous-view-research-011CV2WMvvpRHp72ePAhq9AD`

---

## 🏷️ Issue Categories

### By Severity
- **Critical (Correctness):** #30303, #112577, #71321, #36956, #98143, #82060
- **High (Performance):** #16594, #60466
- **Medium (UX):** #150050, #134471, #33168

### By Component
- **Autograd:** #30303, #36956, #1631, #137372
- **FFT:** #71321, #106623, #38413
- **Attention:** #112577, #159126, #148827, #134471, #109607
- **Convolution:** #82060, #32088, #33168, #70440
- **Indexing:** #36956, #1193, #22513, #1631
- **Compiler:** #137372, #100086, #90454, #146390, #151156, #84618
- **Complex:** #150050, #100086, #71321

### By Status
- **Fixed:** #30303, #112577, #36956, #137372, #84618, #33168
- **Open (High Priority):** #151156, #16594, #82060, #98143
- **Open (Optimization):** #60466, #18862

---

## 📈 Impact Assessment

### User-Facing Impact
- **Training Failures:** Incorrect gradients can silently break model training
- **Inference Errors:** FFT and attention bugs produce wrong outputs
- **Performance Issues:** 80x slowdowns without warning hurt production workloads
- **Migration Pain:** Breaking changes require code updates

### Development Impact
- **Compiler Reliability:** Multiple PT2 issues limit torch.compile adoption
- **Testing Gaps:** Non-contiguous code paths undertested
- **API Consistency:** Different behaviors between operations

---

## 🎯 Next Steps

1. **Prioritize Critical Bugs**
   - Any remaining gradient correctness issues
   - Silent numerical errors (FFT, attention)
   - Breaking changes in stable releases

2. **Improve Observability**
   - Add performance warnings
   - Better error messages
   - Runtime validation where needed

3. **Systematic Testing**
   - Expand test coverage for non-contiguous paths
   - Test all memory formats systematically
   - Validate eager/compiled consistency

4. **Documentation**
   - Memory layout guide for users
   - Performance best practices
   - Migration guide for memory format changes

---

**Research Repository:** https://github.com/redwrasse/pytorch (branch: claude/pytorch-contiguous-view-research-011CV2WMvvpRHp72ePAhq9AD)
