<div align="center">

# IKOD：大型视觉语言模型解码中基于图像注意力引导的键值合并以对抗物体幻觉

一种面向大型视觉语言模型（LVLM）的**免训练、模型无关**协同解码框架：通过在原生键值（KV）缓存中合并低视觉注意力的视觉后文本区间，逆转生成过程中的**渐进式视觉注意力稀释（Visual Attention Dilution）**——*无需微调、无需输入扰动、不引入任何外部模型*。

[**\[📄论文 (arXiv)\]**](https://arxiv.org/abs/2508.03469) &nbsp;&nbsp; [**\[🔌接入指南\]**](integrations/README_zh.md) &nbsp;&nbsp; [**\[🌐English\]**](README.md)

[![Inference](https://img.shields.io/badge/Decoding-Training--Free-7030a0?style=flat-square)](#--方法概览)
[![Backbones](https://img.shields.io/badge/LVLM_Backbones-12_Models_(7B--32B)-21918c?style=flat-square)](#--实验结果)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-POPE_%7C_CHAIR_%7C_AMBER_%7C_MME_%7C_MMBench_%7C_MMStar_%7C_MMVet-3c7d2c?style=flat-square)](#--实验结果)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE.md)

</div>

---

## 🔥 最新动态

* **`2026.11`** 🌟 **IKOD 与 IKOD-A** 完整开源版本发布：涵盖 **8 个主流 7B–8B LVLM 基座**与 **4 个 13B–32B 大规模基座**在 **POPE、CHAIR、AMBER、MME、MMBench、MMStar 与 MM-Vet** 上的评测代码、80 万步闭环机理验证探针及无损加速实现（仅 `1.14×` 贪心解码耗时）！
* **`2025.08`** 📄 初始预印本发布于 [arXiv (2508.03469)](https://arxiv.org/abs/2508.03469)。

## 👀 目录

- [方法概览](#-方法概览)
- [快速开始与骨干接入](#-快速开始与骨干接入)
- [评测与机理探针](#-评测与机理探针)
- [实验结果](#-实验结果)
- [仓库结构](#-仓库结构)
- [致谢](#-致谢)
- [引用](#-引用)

## 📋 方法概览

![IKOD 总体框架](figures/assets/ikod_pipeline_v5.png)

通过对四大架构家族（LLaVA、Qwen-VL、InternVL 与 InstructBLIP）共八个 LVLM 的自回归生成动态进行系统剖析，我们揭示了驱动物体幻觉的三条核心经验规律：
1. **渐进式视觉注意力稀释（命题 1）**：随着自回归生成推进，每个新增文本词元都会向注意力 Softmax 分母注入正指数项，导致总体视觉注意力份额 $S_t$ 随生成位置单调衰减；幻觉物体一致地比真实接地物体出现得更晚、且处于显著更低的视觉注意力之下。
2. **基于视觉后键值合并的可恢复性（定理 1）**：将围绕最低视觉注意力锚点划分的连续视觉后文本区间合并为长度缩减的桶质心，可消除 $P - n_t$ 个竞争性分母项；只要每个合并桶的平均注意力质量低于被合并区域的均值，就能严格证明视觉注意力份额得到提升（$S'_t > S_t$）。
3. **跨层与跨头的普遍性**：视觉注意力稀释广泛分布于中深层解码器与注意力头，驱动我们在全层范围对 KV 缓存进行结构化重组，而非依赖启发式的单层挑选。

<div align="center">
  <img src="figures/assets/fading_focus.png" width="48%" alt="视觉注意力随生成位置衰减"/>
  <img src="figures/assets/density_combined_main.png" width="48%" alt="键值合并恢复视觉注意力密度"/>
</div>

基于上述洞察，**IKOD** 以仅 **1.14×** 单次前向的极低开销执行双分支协同解码：
- **1. 锚点选择（Anchor Selection）**：按累积视觉注意力得分 $s_m$ 对视觉后可合并文本区间 $[u+U, u+U+P)$ 排序，选出注意力最低的 $n_t = \lfloor \lambda P \rfloor$ 个位置（主要为虚词与标点）作为合并锚点。
- **2. 划分与均值合并（Partition-and-Average KV Merging）**：以锚点为中心将可合并区域划分为 $n_t$ 个连续局部桶，通过桶内均值池化生成紧凑辅助缓存 $(\tilde{\boldsymbol{k}}_b, \tilde{\boldsymbol{v}}_b)$，同时完整保留全部 $U$ 个视觉词元与当前查询词元。
- **3. 合理性门控协同融合（Plausibility-Gated Collaborative Fusion）**：在合理候选集 $\mathcal{V}_t(\beta)$ 内将基础分支对数几率 $\boldsymbol{\ell}_t^{\text{base}}$ 与辅助合并分支 $\boldsymbol{\ell}_t^{\text{merged}}$ 加性协同融合：
  $$\tilde{\boldsymbol{\ell}}_t(y) = \boldsymbol{\ell}_t^{\text{base}}(y) + \alpha_t\,\boldsymbol{\ell}_t^{\text{merged}}(y), \quad y \in \mathcal{V}_t(\beta).$$
- **4. 在线注意力衰减门控（IKOD-A）**：根据实测视觉注意力跌落幅度 $D_t = \max(0, (S_0 - S_t)/S_0)$ 在线调度 $\alpha_t = \alpha_{\min} + (\alpha_{\max}-\alpha_{\min})\sigma(\kappa(D_t - \tilde{D}_t))$，以单套通用超参跨全部模型与任务免调参运行。

## 🔧 快速开始与骨干接入

**环境要求**：Python $\ge$ 3.10，PyTorch $\ge$ 2.1，`transformers >= 4.45.0`。安装依赖：

```bash
pip install -r requirements.txt
```

只需数行代码即可在任意受支持的 HuggingFace LVLM 上运行 **IKOD** 或免调参 **IKOD-A**：

```python
import torch
from PIL import Image
from ikod import ikod_generate
from ikod.models import load_vlm

model, processor = load_vlm("llava_15_hf", device="cuda", dtype=torch.bfloat16)
image = Image.open("demo.jpg").convert("RGB")
prompt = "USER: <image>\nPlease describe this image in detail.\nASSISTANT:"
inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

# 1. IKOD（模型族固定协同权重）
out_ikod = ikod_generate(
    model,
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    ratio=0.4,          # lambda：视觉后文本 KV 保留比例
    alpha=4.0,          # 协同融合强度
    beta=0.1,           # 合理性阈值
    max_new_tokens=512,
)

# 2. IKOD-A（全模型通用免调参在线注意力衰减门控）
out_ikoda = ikod_generate(
    model,
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    ratio=0.4,
    adaptive="p1",      # 在线衰减调度 (alpha_min=0, alpha_max=1, kappa=10)
    beta=0.1,
    max_new_tokens=512,
)
print(processor.decode(out_ikod[0], skip_special_tokens=True))
```

各模型族超参数配置、环境变量开关及 13B–32B 零样本迁移设置详见 [`integrations/README_zh.md`](integrations/README_zh.md)。

## 🧪 评测与机理探针

```bash
# 1. 幻觉基准评测（POPE、CHAIR-512 / 128、AMBER）
python eval/eval_pope.py --model llava_15_hf --dataset coco --split adversarial --method ikod
python eval/eval_chair.py --model llava_15_hf --max_new 512 --ratio 0.4 --alpha 4.0
python eval/score_chair.py --cap_file results/chair_llava_15_hf_ikod.jsonl
python eval/eval_amber.py --model llava_15_hf --mode all --method ikod

# 2. 通用多模态能力基准（MME、MMBench、MMStar、MM-Vet）
python eval/eval_general.py --model llava_15_hf --tasks mme,mmbench_en_dev,mmstar,mmvet

# 3. 机理验证与前向诊断探针（图 1-3、图 6、表 XII）
python analysis/probe_fading_focus.py           # 图 1：逐位置视觉注意力衰减分析
python analysis/probe_attn_density.py           # 图 2：注意力密度与 Top-1 预测一致性
python analysis/probe_closed_loop_8b.py         # 图 6 & 表 XII：8 模型定理 1 闭环统计验证
python analysis/profile_efficiency.py           # 表 X & XVIII：推理延迟与峰值显存评测
```

## 📈 实验结果

### 1. 物体幻觉与通用多模态基准（8 个主流 LVLM 平均结果）

在 **8 个代表性 LVLM**（*LLaVA-1.5-7B*、*LLaVA-NeXT-7B*、*LLaVA-OneVision-7B*、*Qwen2-VL-7B*、*Qwen2.5-VL-7B*、*Qwen3-VL-8B*、*InternVL3-8B* 与 *InstructBLIP-7B*）上对比 **7 种免训练解码基线**：

| 方法 | POPE Acc.$\uparrow$ | POPE F1$\uparrow$ | CHAIR $C_S\downarrow$ | CHAIR $C_I\downarrow$ | CHAIR F1$\uparrow$ | AMBER Chair$\downarrow$ | AMBER F1$\uparrow$ | MME$\uparrow$ | MMB$\uparrow$ | 延迟 (ms/tok)$\downarrow$ | 相对耗时$\downarrow$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Greedy | 86.78 | 86.09 | 38.4 | 10.6 | 76.9 | 7.8 | 82.9 | 1986.1 | 75.5 | 24.3 | 1.00$\times$ |
| VCD (*CVPR'24*) | 86.38 | 85.94 | 39.6 | 11.1 | 76.4 | 8.1 | 82.4 | 1957.9 | 74.8 | 47.7 | 1.97$\times$ |
| M3ID (*CVPR'24*) | 86.17 | 85.79 | 40.3 | 11.3 | 76.1 | 8.4 | 82.1 | 1951.3 | 74.6 | 46.8 | 1.92$\times$ |
| OPERA$^{\ast}$ (*CVPR'24*) | 86.88 | 86.18 | 37.8 | 10.4 | 76.8 | 7.6 | 82.8 | 1978.4 | 75.2 | 253.6 | 10.67$\times$ |
| ICD (*ECCV'24*) | 86.45 | 85.85 | 39.2 | 10.9 | 76.5 | 7.9 | 82.5 | 1962.0 | 74.9 | 48.0 | 1.98$\times$ |
| SID (*ICLR'25*) | 86.64 | 86.05 | 38.5 | 10.7 | 76.7 | 7.7 | 82.7 | 1970.8 | 75.1 | 49.6 | 2.04$\times$ |
| AGLA (*ICLR'25*) | 87.02 | 86.39 | 37.4 | 10.2 | 76.8 | 7.4 | 83.1 | 1979.5 | 75.3 | 49.7 | 2.05$\times$ |
| ReVisiT (*EMNLP'25*) | 86.85 | 86.16 | 38.1 | 10.5 | 76.9 | 7.7 | 82.9 | 1983.2 | 75.4 | 25.5 | 1.05$\times$ |
| **IKOD (Ours)** | **87.64** | **87.26** | **31.7** | **8.0** | **77.3** | **6.1** | **84.2** | **1999.2** | **75.7** | 27.6 | 1.14$\times$ |
| **IKOD-A (Ours)** | \underline{87.31} | \underline{86.81} | \underline{35.1} | \underline{9.0} | **77.9** | \underline{6.6} | \underline{83.8} | \underline{1993.8} | **75.7** | 27.8 | 1.15$\times$ |

### 2. 闭环机理验证与定性案例分析

<div align="center">
  <img src="figures/assets/fig_mechanism_8b_main.png" width="96%" alt="8 个 LVLM 上的闭环机理验证"/>
</div>

在全部 8 个 LVLM 共 **801,118 个 CHAIR 解码步**上，定理 1（$S'_t > S_t$）在 **97.6%** 的生成后期步（$t \ge T/2$）与 **90.4%** 的幻觉起始步上严格成立，将生成后期视觉注意力相对提升 **+67.5%**（从 `13.8%` 恢复至 `21.5%`），将幻觉名词概率相对压低 **66.3%**，同时保持 **87.9%**（IKOD）与 **92.4%**（IKOD-A）的真实接地物体预测。

## 🗂️ 仓库结构

```text
IKOD/
├── ikod/                   # IKOD 与 IKOD-A 核心包（decode, kv_merge, adaptive, image_span, models）
├── integrations/           # 各模型族接入指南、超参数表与环境变量开关说明
├── eval/                   # POPE、CHAIR、AMBER 与通用多模态基准评测脚本
├── analysis/               # 前向注意力诊断探针与定理 1 闭环统计验证脚本
└── figures/                # 论文图表绘制脚本与高清架构/机理图资产
```

## 🙏 致谢

感谢以下优秀开源项目的贡献：
[LLaVA](https://github.com/haotian-liu/LLaVA) ·
[Qwen-VL](https://github.com/QwenLM/Qwen2.5-VL) ·
[InternVL](https://github.com/OpenGVLab/InternVL) ·
[LAVIS / InstructBLIP](https://github.com/salesforce/LAVIS) ·
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) ·
[VCD](https://github.com/DAMO-NLP-SG/VCD) ·
[OPERA](https://github.com/shikiw/OPERA)。

本仓库遵循 **MIT 许可证**开源（详见 [`LICENSE.md`](LICENSE.md)）。

## 📝 引用

如果 IKOD 对您的研究有所帮助，欢迎引用我们的论文：

```bibtex
@article{yang2025ikod,
  title   = {IKOD: Image-Attention-Guided Key-Value Merging against Object Hallucination in the Decoding of Large Vision-Language Models},
  author  = {Yang, Jiabing and Cui, Chenhang and Chen, Yixiang and Yu, Tao and Huang, Yan and Wang, Liang},
  journal = {arXiv preprint arXiv:2508.03469},
  year    = {2025}
}
```
