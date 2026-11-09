# 将 IKOD 接入视觉语言模型（LVLM）骨干

[English](README.md) · **简体中文**

IKOD 与 IKOD-A 直接在 HuggingFace 原生 `DynamicCache` 上执行自回归协同解码，不修改任何模型权重，无需外部检测器或辅助模型，并在整个解码过程中完整保留全部视觉 Token。

将 IKOD 接入任意视觉语言骨干仅需三个要素：
1. **视觉词元区间 `[u, u+U)`**：由 `ikod/image_span.py` 中的 `locate_image_span(input_ids, image_token_id)` 在前向传播时自动定位。
2. **视觉后可合并区域 `[u+U, u+U+P)`**：视觉区间之后、受保护的最近查询词元之前的全部文本 KV 条目（`P = t - u - U - 1`）。
3. **协同融合权重**：可使用模型族固定标量 `alpha`（**IKOD**），或完全免调参的在线注意力衰减门控 `alpha_t`（**IKOD-A**，`--adaptive p1`）。

```python
from ikod import ikod_generate, load_vlm

model, processor = load_vlm("llava_15_hf")
output_ids = ikod_generate(
    model,
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    ratio=0.4,          # lambda：视觉后文本 KV 保留比例
    alpha=4.0,          # 固定协同权重（或传入 adaptive="p1" 启用 IKOD-A）
    beta=0.1,           # 合理性截断阈值
    max_new_tokens=512,
)
```

---

## 环境变量开关

| 变量名 | 默认值 | 说明 |
|---|---|---|
| `IKOD_FAST_MERGE` | `1` | 启用向量化分桶均值合并 + 视觉前缀 KV 跨步复用 + 合理性单元素短路无损加速 |
| `IKOD_CKPT_DIR` | `./checkpoints` | 本地模型权重目录（未找到时自动回退至 HuggingFace Hub） |
| `IKOD_DATA_DIR` | `./data` | 评测数据集根目录（COCO、POPE、AMBER） |
| `IKOD_RESULTS_DIR` | `./results` | 生成结果与评测报告输出目录 |
