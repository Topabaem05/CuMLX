# Primary sources and scope of reuse

검토일: 2026-09-28. API profile: MLX 0.32.2. `main` 소스는 구현 의도를 확인하기 위한 참조이며 설치된 wheel과 동일 commit이라고 가정하지 않습니다. GPU 실행으로 API compatibility를 확인하지 못했습니다.

- **S1 — MLX metal_kernel API**
  https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html
  Function-body JIT, synthesized argument types, header and safe math compile options.
- **S2 — MLX cuda_kernel API**
  https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.cuda_kernel.html
  CUDA JIT body/header API and launch arguments.
- **S3 — MLX quantize API**
  https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html
  NVFP4/MXFP4 block/scale formats; uint32 native packing and format-specific return values.
- **S4 — MLX quantized_matmul API**
  https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html
  Native comparator; transpose, group_size, mode, bits. The documented API has no global_scale argument.
- **S5 — NVIDIA TransformerEngine NVFP4**
  https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html
  E2M1, E4M3 G16 scales and FP32 tensor multiplier; training includes 2D weights, padded layouts, RHT and stochastic rounding for gradients. Our own deterministic 1D inference quantizer does not implement this full recipe.
- **S6 — MLX custom Metal kernel guide**
  https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
  Function body, generated signature, simdgroup examples.
- **S7 — MLX native CUDA custom kernel implementation**
  https://github.com/ml-explore/mlx/blob/main/mlx/backend/cuda/custom_kernel.cpp
  Inspected text: https://raw.githubusercontent.com/ml-explore/mlx/main/mlx/backend/cuda/custom_kernel.cpp
  grid is a logical-thread extent, converted to CUDA block count; constants become compile-time values. Headers are emitted outside the generated kernel namespace.
- **S8 — MLX native Metal custom kernel implementation**
  https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/custom_kernel.cpp
  Inspected text: https://raw.githubusercontent.com/ml-explore/mlx/main/mlx/backend/metal/custom_kernel.cpp
  Row-contiguous input handling, max workgroup checking and dispatch_threads behavior.
- **S9 — Apple threads and threadgroups**
  https://developer.apple.com/documentation/metal/creating-threads-and-threadgroups
  Workgroup semantics for launch and shared-memory design.

No third-party kernel source was copied into this release. Kernel bodies and common helpers were authored for the stated algebra and canonical contract. The headers under kernels/ contain only our math helpers plus standard include directives, not bundled vendor SDK headers. The earlier user-provided ZIPs are identified by hashes in provenance/SOURCE-ARCHIVES.json and retained outside Git; their claims are not GPU verification evidence.
