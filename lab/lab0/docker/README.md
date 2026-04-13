# v6 Docker 运行说明

当前机器为 Apple Silicon，不能原生运行 Intel macOS 版 oneMKL。`v6_mkl` 通过 Docker 的 `linux/amd64` 仿真环境完成：

```bash
./lab/lab1/scripts/run_v6_docker.sh 512 512 512 20250401
```

说明：

- 镜像基于 Intel 官方 `oneAPI Base Toolkit` 容器。
- 容器内使用 `icpx -O3 -std=c++17 -qmkl` 构建 `v6_mkl`。
- 该结果适合做功能完成与接口验证，不应与宿主机原生 `v1-v5` 做严格公平的性能比较。
