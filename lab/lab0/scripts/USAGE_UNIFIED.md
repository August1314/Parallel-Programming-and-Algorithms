# 统一测试入口

当前保留 unified 入口作为实现主入口，同时补回旧文件名作为兼容包装：

```bash
./lab/lab1/scripts/run_matrix_mul_unified v6_mkl 512 512 512 20250401
./lab/lab1/scripts/benchmark_lab1_unified.py --sizes 512x512x512 --repeat 1 --include-mkl

./lab/lab1/scripts/run_matrix_mul v6_mkl 512 512 512 20250401
./lab/lab1/scripts/benchmark_lab1.py --sizes 512x512x512 --repeat 1 --include-mkl
```

说明：

- `run_matrix_mul` 与 `benchmark_lab1.py` 仅做兼容转发；
- `v1-v5` 仍走宿主机原生实现；
- `v6_mkl` 默认走 Docker `linux/amd64` oneMKL 容器；
- 如需强制走宿主机本地二进制，可设置 `LAB1_V6_MODE=local`；
- 统一基准脚本通过 `--v6-mode {docker,local}` 控制 `v6` 路由。
