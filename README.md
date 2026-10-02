# isoassign

面向实验室分析流程（Python 调用，非命令行工具）的**竞争式同位素峰簇归属内核**。
提交分子式候选、centroid 峰表与一次测量的质量校准，得到可解释、可复核的峰簇归属。

## 它解决的核心问题

手工按最近质量逐峰归属会产生两个错误：

1. **共用峰重复计数** —— 两个候选的理论峰窗重叠时，同一观测峰被两个候选各算一次，
   于是每个候选"看起来都被支持"。
2. **强度被忽略** —— 单个大噪声峰足以让一个候选在最近质量规则下显得成立。

`isoassign` 把**所有候选的全部理论峰**与**全部观测峰**放入同一张二分图，用
SciPy 匈牙利算法做**全局最大权一对一匹配**：一个观测峰至多属于一个理论峰，
一个理论峰至多占用一个观测峰。边权由*预测的*同位素相对丰度与质量误差共同决定，
因此单个大噪声峰无法压过一整组相干峰簇。

## 设计原则

| 原则 | 实现 |
|---|---|
| 理论值、原始观测、最终决定三方可核对 | `TheoreticalPeak` / `PeakMatch.observed_mz_raw` / `CandidateReport` 同时保留 |
| 校准不能改写原始 m/z | `PeakTable.mz` 不可变；校准只在匹配时叠加，报告同时给出 raw 与 calibrated |
| 可缓存理论计算与测量状态分离 | `TheoryLibrary` 不含任何峰表/校准；`assign()` 是纯函数 |
| 改校准可重判，但不能串样本观测身份 | 峰 id 全局唯一（`T0001:P000001`），结果指纹包含表身份 |
| 增删候选触发全局重新协调 | 不做局部追加；每次调用对当前候选全集重新求解 |
| 相同输入与容差结果稳定 | 确定性平局规则；提供 `fingerprint`，对候选顺序、峰行顺序不敏感 |
| 失败必须明确 | 无效公式/加合物 → `invalid_formula`；预算超限 → `budget_exceeded`；图过大 → `resource_limit`；均带诊断，其他候选照常给出 |
| 弱同位素缺失不能误杀真候选 | 证据按"强峰覆盖 + 同位素图形余弦 + 单同位素峰"综合判断，不要求每个理论峰都有观测 |
| 资源不随原子数失控 | 同位素卷积带丰度剪枝（幂内绝对地板、跨元素相对地板），运行/内存由峰簇形状决定而非原子数 |

## 快速上手

```python
from isoassign import (TheoryLibrary, PeakTable, Calibration,
                      AssignmentSettings, assign)

# 1) 理论库：一次构建，反复用于不同峰表/校准版本（可 pickle 持久化）
lib = TheoryLibrary()
lib.add("glucose", "C6H12O6", "[M+H]+")
lib.add("lactate", "C3H6O3",  "[M-H]-")

# 2) 一次测量：原始 m/z 不可变
table = PeakTable([(181.07066, 1000.0), (182.07402, 65.0)],
                  table_id="run-2026-10-02-a")

# 3) 本次测量的校准（版本化）：恒等 / 线性 ppm / 双锁质量峰
cal = Calibration.from_ppm(-3.0, version="cal-2026-10-02")

# 4) 全局互斥归属
result = assign(lib.theories(), table, cal,
                settings=AssignmentSettings(ppm_tolerance=5.0))

for c in result.candidates:
    print(c.candidate_id, c.status)
    for m in c.matches:
        print("  theo", m.theoretical_mz, "raw", m.observed_mz_raw,
              "cal", m.observed_mz_calibrated, f"{m.error_ppm:+.2f} ppm")
    print("  FOR    :", c.supporting_evidence)
    print("  AGAINST:", c.opposing_evidence)
```

候选状态：`supported` / `contested`（通过阈值但有强共享峰靠平局拿到）/
`weak` / `unsupported` / `rejected`（强度图形明显不符）/
`invalid_formula` / `budget_exceeded` / `resource_limit`。

竞争解释同样保留在报告里：

- `report.contenders` —— 本该属于本候选、但被对手拿走的峰（含双方峰强度与误差）；
- `report.overlapping_claims` —— 本候选拿到、但对手理论峰也能解释的峰；
- `result.unassigned` —— 未归属观测峰，含其容差窗内的候选峰（`nearby`）。

## 关键 API

- `Formula("Ca(OH)2")` / `Formula({"C": 6})`：解析、校验、Hill 规范式、单同位素精确质量。
- `Adduct("[M+H]+")` 或 `Adduct("[M+Li]+", delta={"Li": 1}, charge=1)`；
  m/z 用电子质量校正（`[M+H]+ = M + proton_mass`）。
- `isotope_distribution(formula, IsotopeBudget(...))`：带预算的剪枝卷积，
  返回精细结构质量、概率、相对丰度、丢失概率。
- `TheoryLibrary`：`add/remove/get/theories/save/load`，缓存键为
  （组成、加合物、电荷）。
- `Calibration.identity / .linear / .from_ppm / .lock_mass`，均带 `version`。
- `assign(theories, peak_table, calibration, settings) -> AssignmentResult`。
- `AssignmentResult.to_dict()`：JSON 安全的完整审计结构。

## 参考值核对

葡萄糖 `C6H12O6`：

- 中性单同位素质量 **180.0633881 Da**（与 `pyteomics.mass.calculate_mass` 一致）；
- `[M+H]+` 单峰 **m/z 181.07066**；
- M+1 精细结构：¹³C₁ **6.489%**、¹⁷O₁ 0.229%、²H₁ 0.138%（与 pyteomics
  `isotopologues` 在足够低阈值下逐项一致；注意 pyteomics 默认 5e-4 阈值会漏掉
  ¹⁷O/²H，全量枚举则会组合爆炸——见 `tests/test_isotopes.py`）。

同位素丰度数据来自 pyteomics 自带的 NIST 表；卷积/匹配/裁决内核为本项目实现，
SciPy（`linear_sum_assignment`）用于全局最优匹配。

## 预算控制

```python
IsotopeBudget(min_rel_abundance=1e-5,   # 尾峰剪枝地板
              max_peaks=300,            # 峰数硬上限（最强者保留，单同位素强制保留）
              hard_pair_limit=2_000_000,# 单步卷积对数量硬上限（超限抛 BudgetError）
              accepted_dropped_fraction=0.01)  # 丢失天然概率超此值则标记 budget_exceeded
```

50 000 个原子的分子与 6 碳糖的峰数同量级（≤ `max_peaks`）、亚秒完成；
无法在预算内给可靠结果时，候选被明确标记为 `budget_exceeded` 而不是返回空集。

## 测试

```bash
python -m pytest            # 43 个测试
```

覆盖：公式解析与精确质量、葡萄糖参考值与 pyteomics 独立交叉验证、预算与可扩展性、
校准与原始身份隔离、共享峰一对一互斥、增删候选重新协调、确定性与指纹、
噪声/缺失弱峰鲁棒性、失败诊断隔离、两组人为叠加峰簇端到端核对、报告 JSON 可审计。

## 依赖

numpy、scipy、pyteomics（同位素数据表与独立判据）。无数据库、无网络、无网页、无 CLI。
