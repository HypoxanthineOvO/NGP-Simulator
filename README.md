# Instant NGP Cycle-Accurate Simulator
## Intro
[CICC 2024](https://ieeexplore.ieee.org/document/10529071) 的非官方 Cycle Accurate Simulator，旨在复现其主要计算流程，为后续新架构的设计做准备。

暂时只支持时序仿真。量化部分的仿真由另外的 PyTorch 部分实现；访存的仿真正在探索中。

## Quick Start
在 `main.cpp` 中配置好数据集的名称、分辨率和时钟频率等信息，然后：
```bash
xmake
./main
```
即可生成数据。其 Output 同时会 Dump 在一个 `.txt` 文件中。
