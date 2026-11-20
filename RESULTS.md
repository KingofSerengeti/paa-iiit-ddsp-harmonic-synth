# RESULTS: fused CUDA harmonic synthesizer vs faithful PyTorch port (RTX 5070)

Date: 2026-11-21. All numbers below are copied by script from `fused_results.csv`, `baseline_faithful_results.csv`
and `test_fused_log.txt` in this folder. Code state: fused_ddsp.py as of commit `e7daecf`.

## Environment
- GPU: NVIDIA GeForce RTX 5070, driver 595.99.02
- torch 2.11.0+cu128 (CUDA 12.8), Python 3.12, venv `paa-env`
- Kernel compiler: system nvcc, `Cuda compilation tools, release 12.0, V12.0.140`. This nvcc cannot target sm_120, so the extension is built as
  **compute_90 PTX and JIT-compiled to sm_120 by the driver** at load time (`TORCH_CUDA_ARCH_LIST=9.0+PTX`, see `fused_ddsp.py`).
  Not tested: a native sm_120 build with nvcc >= 12.8.
- Precision: fused kernel inputs/outputs fp32; frame phase offsets fp64 (F values per batch item); Nyquist cutoff decided in fp64.
- Reference for errors: `ddsp_port.harmonic_synthesis` in fp64 (default cumsum path), batch item by batch item.
- Opponent: `faithful_angular_compile` = `torch.compile(harmonic_synthesis(..., use_angular_cumsum=True))`, measured in the same run.

## 1. Correctness test (`python test_fused.py`, verbatim from test_fused_log.txt)
```
GPU: NVIDIA GeForce RTX 5070 | torch 2.11.0+cu128
--- B=2 K=100 T=1024
  fused (skip, sinpif)   max_abs=1.119e-06  rel_l2=4.383e-07
  fused no-skip          max_abs=1.119e-06  rel_l2=4.328e-07
  fused fast-sin         max_abs=1.203e-06  rel_l2=4.631e-07
  angular eager fp32     max_abs=2.352e-04  rel_l2=1.051e-04
--- B=2 K=100 T=16000
  fused (skip, sinpif)   max_abs=1.716e-06  rel_l2=5.102e-07
  fused no-skip          max_abs=1.716e-06  rel_l2=5.105e-07
  fused fast-sin         max_abs=1.894e-06  rel_l2=5.487e-07
  angular eager fp32     max_abs=1.067e-03  rel_l2=6.221e-04
--- B=1 K=128 T=64000
  fused (skip, sinpif)   max_abs=1.794e-06  rel_l2=5.611e-07
  fused no-skip          max_abs=1.779e-06  rel_l2=5.608e-07
  fused fast-sin         max_abs=1.898e-06  rel_l2=5.937e-07
  angular eager fp32     max_abs=2.288e-03  rel_l2=1.643e-03
--- B=8 K=100 T=64000
  fused (skip, sinpif)   max_abs=3.499e-06  rel_l2=5.852e-07
  fused no-skip          max_abs=3.499e-06  rel_l2=5.856e-07
  fused fast-sin         max_abs=3.663e-06  rel_l2=6.149e-07
  angular eager fp32     max_abs=4.629e-03  rel_l2=1.933e-03
```
`test_fused.py` prints errors and has no assert; its docstring sets the expectation "fused rel_l2 around 1e-6 to 1e-5".
Fused rel_l2 in this log: 4.328e-07 to 6.149e-07 for all variants and configs.

## 2. Main config (8,100,64000), rows from fused_results.csv
| impl | ms_median | ms_std | peak_extra_MB | energy_mJ | energy_std | max_abs_err_vs_fp64 | rel_l2_err |
|---|---|---|---|---|---|---|---|
| fused | 0.223 | 0.0047 | 2.0 | 23.568 | 0.992 | 3.499e-06 | 5.852e-07 |
| fused_noskip | 0.3565 | 0.0556 | 2.0 | 37.654 | 1.504 | 3.499e-06 | 5.856e-07 |
| fused_fastsin | 0.2175 | 0.0386 | 2.0 | 18.846 | 0.646 | 3.663e-06 | 6.149e-07 |
| fused_kernel_only | 0.1761 | 0.0053 | 2.0 | 22.147 | 0.749 | 3.499e-06 | 5.852e-07 |
| fused_block64 | 0.3288 | 0.0584 | 2.0 | 23.267 | 0.464 | 3.499e-06 | 5.852e-07 |
| fused_block128 | 0.3209 | 0.0585 | 2.0 | 22.922 | 0.209 | 3.499e-06 | 5.852e-07 |
| fused_block512 | 0.3299 | 0.0759 | 2.0 | 22.217 | 0.121 | 3.499e-06 | 5.852e-07 |
| fused_block1024 | 0.2757 | 0.0546 | 2.0 | 25.011 | 0.137 | 3.499e-06 | 5.852e-07 |
| faithful_angular_compile | 2.0964 | 0.0059 | 198.1 | 447.135 | 8.971 | 3.310e-03 | 1.262e-03 |

## 3. All rows, fused_results.csv (this run, `python bench_fused.py`, log in bench_fused_log.txt)
| impl | B | K | T | ms_median | ms_std | samples_per_s | peak_extra_MB | energy_mJ | energy_std | max_abs_err_vs_fp64 | rel_l2_err | active_harmonic_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fused | 1 | 100 | 1024 | 0.0574 | 0.0253 | 17847183 | 0.0 | 2.344 | 0.138 | 2.717e-07 | 5.297e-07 | 0.319 |
| fused_noskip | 1 | 100 | 1024 | 0.059 | 0.0712 | 17353580 | 0.0 | 2.307 | 0.019 | 2.717e-07 | 5.271e-07 | 0.319 |
| fused_fastsin | 1 | 100 | 1024 | 0.0561 | 0.0739 | 18254421 | 0.0 | 2.372 | 0.123 | 2.593e-07 | 5.660e-07 | 0.319 |
| fused_kernel_only | 1 | 100 | 1024 | 0.035 | 0.0025 | 29250458 | 0.0 | 0.278 | 0.06 | 2.717e-07 | 5.297e-07 | 0.319 |
| faithful_angular_compile | 1 | 100 | 1024 | 0.0482 | 0.0018 | 21248340 | 1.2 | 2.662 | 0.13 | 8.398e-06 | 1.750e-05 | 0.319 |
| fused | 1 | 100 | 16000 | 0.0871 | 0.0017 | 183621005 | 0.1 | 2.56 | 0.207 | 1.794e-06 | 6.137e-07 | 0.316 |
| fused_noskip | 1 | 100 | 16000 | 0.0685 | 0.0602 | 233644860 | 0.1 | 3.135 | 0.276 | 1.779e-06 | 6.131e-07 | 0.316 |
| fused_fastsin | 1 | 100 | 16000 | 0.0595 | 0.0788 | 268817210 | 0.1 | 2.567 | 0.111 | 1.898e-06 | 6.454e-07 | 0.316 |
| fused_kernel_only | 1 | 100 | 16000 | 0.033 | 0.01 | 485201364 | 0.1 | 0.816 | 0.088 | 1.794e-06 | 6.137e-07 | 0.316 |
| faithful_angular_compile | 1 | 100 | 16000 | 0.1185 | 0.004 | 135007424 | 6.2 | 17.635 | 0.69 | 4.037e-04 | 1.931e-04 | 0.316 |
| fused | 8 | 100 | 64000 | 0.223 | 0.0047 | 2295717020 | 2.0 | 23.568 | 0.992 | 3.499e-06 | 5.852e-07 | 0.37 |
| fused_noskip | 8 | 100 | 64000 | 0.3565 | 0.0556 | 1436136811 | 2.0 | 37.654 | 1.504 | 3.499e-06 | 5.856e-07 | 0.37 |
| fused_fastsin | 8 | 100 | 64000 | 0.2175 | 0.0386 | 2354152891 | 2.0 | 18.846 | 0.646 | 3.663e-06 | 6.149e-07 | 0.37 |
| fused_kernel_only | 8 | 100 | 64000 | 0.1761 | 0.0053 | 2908033327 | 2.0 | 22.147 | 0.749 | 3.499e-06 | 5.852e-07 | 0.37 |
| fused_block64 | 8 | 100 | 64000 | 0.3288 | 0.0584 | 1557177579 | 2.0 | 23.267 | 0.464 | 3.499e-06 | 5.852e-07 | 0.37 |
| fused_block128 | 8 | 100 | 64000 | 0.3209 | 0.0585 | 1595532511 | 2.0 | 22.922 | 0.209 | 3.499e-06 | 5.852e-07 | 0.37 |
| fused_block512 | 8 | 100 | 64000 | 0.3299 | 0.0759 | 1552192503 | 2.0 | 22.217 | 0.121 | 3.499e-06 | 5.852e-07 | 0.37 |
| fused_block1024 | 8 | 100 | 64000 | 0.2757 | 0.0546 | 1857010124 | 2.0 | 25.011 | 0.137 | 3.499e-06 | 5.852e-07 | 0.37 |
| faithful_angular_compile | 8 | 100 | 64000 | 2.0964 | 0.0059 | 244228198 | 198.1 | 447.135 | 8.971 | 3.310e-03 | 1.262e-03 | 0.37 |
| fused | 8 | 64 | 64000 | 0.2193 | 0.006 | 2335084640 | 2.0 | 22.802 | 1.008 | 3.499e-06 | 5.852e-07 | 0.579 |
| fused_noskip | 8 | 64 | 64000 | 0.2263 | 0.004 | 2262123545 | 2.0 | 29.075 | 0.866 | 3.499e-06 | 5.856e-07 | 0.579 |
| fused_fastsin | 8 | 64 | 64000 | 0.2179 | 0.0503 | 2349658549 | 2.0 | 18.351 | 0.598 | 3.663e-06 | 6.149e-07 | 0.579 |
| fused_kernel_only | 8 | 64 | 64000 | 0.1743 | 0.0067 | 2936857541 | 2.0 | 21.683 | 0.727 | 3.499e-06 | 5.852e-07 | 0.579 |
| faithful_angular_compile | 8 | 64 | 64000 | 1.1086 | 0.0045 | 461853766 | 128.1 | 246.716 | 6.614 | 2.907e-03 | 7.938e-04 | 0.579 |
| fused | 8 | 128 | 64000 | 0.2203 | 0.0044 | 2324061325 | 2.0 | 23.178 | 0.781 | 3.499e-06 | 5.852e-07 | 0.289 |
| fused_noskip | 8 | 128 | 64000 | 0.2825 | 0.0503 | 1812312434 | 2.0 | 48.479 | 0.282 | 3.499e-06 | 5.856e-07 | 0.289 |
| fused_fastsin | 8 | 128 | 64000 | 0.3295 | 0.0591 | 1554077017 | 2.0 | 18.336 | 1.061 | 3.663e-06 | 6.149e-07 | 0.289 |
| fused_kernel_only | 8 | 128 | 64000 | 0.1761 | 0.0055 | 2907504878 | 2.0 | 21.88 | 0.669 | 3.499e-06 | 5.852e-07 | 0.289 |
| faithful_angular_compile | 8 | 128 | 64000 | 2.0589 | 0.0038 | 248675024 | 252.2 | 440.287 | 3.097 | 3.310e-03 | 1.262e-03 | 0.289 |
| fused | 16 | 100 | 64000 | 0.3893 | 0.0011 | 2630497286 | 4.0 | 46.496 | 1.757 | 2.793e-06 | 6.094e-07 | 0.394 |
| fused_noskip | 16 | 100 | 64000 | 0.4232 | 0.0342 | 2419476765 | 4.0 | 78.893 | 3.009 | 2.797e-06 | 6.100e-07 | 0.394 |
| fused_fastsin | 16 | 100 | 64000 | 0.3892 | 0.0385 | 2631146105 | 4.0 | 37.969 | 1.562 | 2.658e-06 | 6.383e-07 | 0.394 |
| fused_kernel_only | 16 | 100 | 64000 | 0.344 | 0.0038 | 2976744085 | 3.9 | 44.421 | 0.631 | 2.793e-06 | 6.094e-07 | 0.394 |
| faithful_angular_compile | 16 | 100 | 64000 | 4.1192 | 0.0214 | 248594852 | 394.9 | 898.45 | 22.397 | 2.955e-03 | 1.059e-03 | 0.394 |

Row meanings (from bench_fused.py): `fused` = default (skip inactive harmonics, sinpif, block 256, includes the
fp64 frame-offset computation in PyTorch); `fused_noskip` = loop over all K; `fused_fastsin` = `__sinf` on range-reduced angle;
`fused_kernel_only` = kernel launch only, phase offsets precomputed; `fused_blockN` = block size N; timing = CUDA events,
median of 50 after 10 warmup (`baseline_ddsp.time_ms`); energy = NVML board energy with repetition amplification (`baseline_ddsp.energy_mj`).

## 4. Earlier baseline, baseline_faithful_results.csv (Oct 30 run, file unchanged)
| impl | B | K | T | ms_median | ms_std | samples_per_s | peak_extra_MB | energy_mJ | energy_std | max_abs_err_vs_fp64 | rel_l2_err | active_harmonic_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| faithful_eager | 1 | 100 | 1024 | 0.2792 | 0.0085 | 3667412 | 2.7 | 9.789 | 0.702 | 1.387e-04 | 1.408e-04 | 0.319 |
| faithful_angular_eager | 1 | 100 | 1024 | 0.2976 | 0.0156 | 3440860 | 4.6 | 11.049 | 0.775 | 9.851e-05 | 1.231e-04 | 0.319 |
| faithful_angular_compile | 1 | 100 | 1024 | 0.0464 | 0.0043 | 22076578 | 1.2 | 2.696 | 0.163 | 7.854e-06 | 1.718e-05 | 0.319 |
| faithful_eager | 1 | 100 | 16000 | 2.1783 | 0.0052 | 7345218 | 43.1 | 127.133 | 4.593 | 2.054e-02 | 1.272e-02 | 0.316 |
| faithful_angular_eager | 1 | 100 | 16000 | 0.6234 | 0.2165 | 25666693 | 43.1 | 35.646 | 2.545 | 4.900e-04 | 4.797e-04 | 0.316 |
| faithful_angular_compile | 1 | 100 | 16000 | 0.1178 | 0.0012 | 135851104 | 6.2 | 17.549 | 0.638 | 4.085e-04 | 1.934e-04 | 0.316 |
| faithful_eager | 8 | 100 | 64000 | 31.2381 | 0.3358 | 16390243 | 1378.1 | 3144.213 | 42.993 | 7.527e-01 | 2.975e-01 | 0.37 |
| faithful_angular_eager | 8 | 100 | 64000 | 13.7968 | 0.2264 | 37110140 | 1378.3 | 2241.583 | 83.981 | 4.629e-03 | 1.933e-03 | 0.37 |
| faithful_angular_compile | 8 | 100 | 64000 | 2.0093 | 0.0104 | 254813584 | 198.1 | 481.486 | 20.854 | 2.867e-03 | 8.228e-04 | 0.37 |
| faithful_eager | 8 | 64 | 64000 | 25.78 | 0.2644 | 19860382 | 885.9 | 2273.474 | 86.379 | 7.527e-01 | 2.975e-01 | 0.579 |
| faithful_angular_eager | 8 | 64 | 64000 | 8.2164 | 0.1356 | 62314033 | 886.0 | 1353.603 | 34.558 | 4.629e-03 | 1.933e-03 | 0.579 |
| faithful_angular_compile | 8 | 64 | 64000 | 1.0436 | 0.0087 | 490601909 | 128.1 | 219.387 | 6.192 | 3.310e-03 | 1.262e-03 | 0.579 |
| faithful_eager | 8 | 128 | 64000 | 33.9844 | 0.1413 | 15065757 | 1757.8 | 3548.738 | 196.578 | 7.527e-01 | 2.975e-01 | 0.289 |
| faithful_angular_eager | 8 | 128 | 64000 | 16.214 | 0.0954 | 31577713 | 1758.1 | 2662.134 | 52.052 | 4.629e-03 | 1.933e-03 | 0.289 |
| faithful_angular_compile | 8 | 128 | 64000 | 2.0422 | 0.0096 | 250707081 | 252.2 | 408.782 | 5.87 | 2.913e-03 | 1.152e-03 | 0.289 |
| faithful_eager | 16 | 100 | 64000 | 42.5752 | 0.5101 | 24051533 | 2746.6 | 5175.757 | 249.974 | 7.651e-01 | 2.903e-01 | 0.394 |
| faithful_angular_eager | 16 | 100 | 64000 | 27.5948 | 0.38 | 37108461 | 2747.0 | 4452.7 | 103.189 | 4.810e-03 | 2.435e-03 | 0.394 |
| faithful_angular_compile | 16 | 100 | 64000 | 4.1572 | 0.0394 | 246320579 | 394.9 | 977.836 | 35.386 | 2.557e-03 | 7.992e-04 | 0.394 |

Note: `faithful_angular_compile` appears in both files from different runs, and the values differ (e.g. (8,100,64000): 2.0093 ms
on Oct 30 vs 2.0964 ms today; rel_l2 8.228e-04 vs 1.262e-03). Section 3 rows are the same-session comparison.

## 5. Failures, problems and caveats
1. First build failed: `nvcc fatal : Unsupported gpu architecture 'compute_120'` (system nvcc 12.0). Worked around with
   compute_90 PTX + driver JIT (no software installed). Native sm_120 build not tested.
2. First test run: at (8,100,64000) all three fused variants had max_abs 2.800e-03, rel_l2 5.330e-05. Cause found: batch item 7,
   t=45932, f0 = 363.6364 Hz = 8000/22, so harmonic 22 is at Nyquist; the kernel's fp32 `k*freq < nyq` test kept it active while the
   fp64 reference zeroed it. Fixed in the kernel (fp64 per-sample cutoff with the reference's formula); test unchanged.
3. `bench_fused.py` run: 0 FAILED lines, 34 rows written.
4. Several fused rows have ms_std of the same order as the gap between them (e.g. fused_block64/128/512 at the main config:
   ms_std 0.0584/0.0585/0.0759 ms; fused_noskip at (1,100,1024): ms_std 0.0712 ms). Rankings among those rows are not resolved by this data.
5. `torch.compile` of the default (non-angular) path still fails with an Inductor error (see handoff 2); not retested tonight and not in this benchmark.
6. Nsight Compute profiling NOT done: only Nsight Compute 2022.4.1 is installed (`/usr/lib/nsight-compute/ncu`), which does not
   support Blackwell. No DRAM / SM throughput numbers exist.
7. One GPU only (RTX 5070). Energy is board-level NVML. Validation of `ddsp_port.py` against the TensorFlow original is still pending.
8. Only one control generator (`make_controls`, seed 0, f0 in 100-400 Hz) was used for all results.
