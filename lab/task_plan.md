# lab2 任务规划

## 目标
基于 lab1 的 MPI-v1 点对点矩阵乘法，在 `lab/lab2` 下建立完整独立实验模块，满足 lab2 要求：引入 MPI 集合通信、使用 MPI_Type_create_struct 聚合元数据通信，并交付代码实现、benchmark 数据、绘图代码与可视化更丰富的报告草稿。

## 当前阶段
- [x] 阶段1：探索现有仓库与题目要求
- [x] 阶段2：澄清 lab2 交付范围
- [x] 阶段3：提出候选设计方案
- [x] 阶段4：形成并确认设计
- [ ] 阶段5：进入实现规划

## 已确认设计
- 在 `lab/lab2` 下独立建立 `src/`、`scripts/`、`results/`、`report/`、`tests/`、`Makefile`、`pyproject.toml`、`CLAUDE.md`
- 实现两个版本：`mpi_v1_p2p` 与 `mpi_v2_collective`
- 代码结构预留第三版扩展点，但本次不实现第三版
- `mpi_v2_collective` 使用 `MPI_Scatterv` / `MPI_Bcast` / `MPI_Gatherv`
- 使用 `MPI_Type_create_struct` 广播聚合后的运行配置元数据
- benchmark、绘图、报告资产全部在 `lab2` 内闭环
- Python 工作流统一使用 `uv`

## 已知约束
- 必须参考 lab0 与 lab1 的现有结构和输出契约
- 设计阶段不能直接开始实现
- 需要保持 CLI/benchmark 输出为稳定 key=value 风格
- 报告需要可视化内容丰富，因此必须生成多类图表

## 遇到的问题
- Read 工具不能直接读取 docx/pptx 二进制内容，已改用受控脚本提取 OOXML 文本
