# LoopyCuts + SAC 强化学习项目完整交接文档
**版本：2026-08-14 / Phase 2E-B Resource Feasibility Audit**  
**用途：新对话无缝接续、当前事实基线、代码与实验纪律说明**

> 新对话必须优先以“当前真实源码 + 最新运行输出 + 冻结 manifest / regression”为准。本文件用于传递已经查清楚的总体架构、语义、实验结果和下一步路线，但不能替代源码。任何新 patch 前都应先查看当前文件，而不是根据历史版本猜接口。

---

## A. 项目目标与核心研究问题

本项目在完整复现 LoopyCuts 的基础上，将强化学习引入 **Stage-2 volumetric cutter 的 loop execution order**。

当前不是：
- 用 AI 重新生成 surface loops；
- 替换 Stage1；
- 学一个静态 loop score 后一次排序。

当前研究的是：

\[
a_t=\pi(s_t), \qquad a_t\in A_{\mathrm{legal}}(s_t)
\]

其中：
- `s_t` 为当前 volumetric cutting state；
- `a_t` 是当前 C++ RL server 输出的 authoritative legal loop；
- 每一步 cut 会改变 tetrahedral mesh、convergence、meta mesh、后续 legality 与 finalization；
- 因此 Stage2 是真正的 state-dependent sequential decision process。

当前最合理的论文叙事：

> Stage1 已经产生几何合理的 candidate loops，但 Stage2 的 serialized traversal 无法根据当前 volumetric state 动态调整。不同执行顺序会显著改变 tetrahedral refinement、cuts/reverts、block decomposition complexity、finalization stability、内存和运行时间。因此学习一个动态 Stage2 loop-selection policy，有机会减少 over-decomposition、tet explosion、reverts 和资源消耗，同时保持或提高最终 full-hex 成功率。

Random degradation 只证明 **order sensitivity / learning headroom**，不能当作 RL superiority。

---

## B. 机器、环境与路径

### B1. 系统
- Windows + WSL2
- Ubuntu 22.04
- VS Code Linux
- 16 logical CPUs
- RAM ≈ 7.6 GiB
- Swap ≈ 33 GiB

最近实测空闲状态：
```text
MemTotal      ≈ 7.6 GiB
MemAvailable  ≈ 6.2 GiB
SwapTotal     ≈ 33 GiB
```

Swap：
```text
/dev/sdc                 2 GiB
/swapfile_loopycuts_32g 32 GiB
```

重要：系统空闲时也可能已有约 2.1~2.2 GiB system swap used。因此不能用系统 SwapUsed 判断单个 LoopyCuts episode 是否换页，必须看 `volumetric_cutter` 自己 `/proc/<pid>/status` 的 `VmSwap`。

### B2. 项目路径
```text
/home/yjk/codes/LoopyCuts
/home/yjk/codes/loopycuts_preprocessor
/home/yjk/codes/loopycuts_rl
/home/yjk/codes/rl_refs/{ASMR,AMBER,SD-SAC,tianshou}
```

### B3. RL Python 环境
```text
venv:      /home/yjk/codes/loopycuts_rl/.venv
Python:    3.11.15
uv:        0.12.3
Tianshou:  2.0.1
Gymnasium: 0.28.1
NumPy:     1.24.4
Torch:     2.1.1+cpu
SciPy:     1.10.1
```

Tianshou source：
```text
/home/yjk/codes/rl_refs/tianshou
```

历史记录 commit：
```text
f2402056...
```
精确 SHA 以后用 `git rev-parse HEAD` 重新确认。

---

## C. LoopyCuts 原始总体流程

```text
source OBJ
  ↓
几何/拓扑检查
  ↓
保特征重网格
  ↓
triangular surface mesh
  ↓
SHARP feature
  ↓
GODF / 4-RoSy
  ↓
Stage1 loop_distribution
  ↓
_loop.txt
  ↓
Stage2 volumetric_cutter
  ↓
cuts / convergence / block decomposition
  ↓
finalization
  ↓
subdivision
  ↓
smoothing
  ↓
poly classification
  ↓
hex / hybrid mesh
```

核心 executable：
```text
Stage1:
/home/yjk/codes/LoopyCuts/loop_distribution/loop_distributor

Stage2:
/home/yjk/codes/LoopyCuts/volumetric_cutter/volumetric_cutter
```

Windows GODF 使用过 `ffgen.exe`。

---

## D. Stage1 已经查清楚的关键事实

### D1. `_loop.txt`
`loop_distribution/loop_splitter.h::SaveLoopInfo()` 序列化最终 loop 信息，包括：
- 最终 serialized order；
- type；
- Closed/Open；
- Cross OK/FAIL；
- segments；
- sharp bit。

Stage1 内部一些信息不会保存：
```text
Essential
AvgDistance
SolveSharp
SolveLoop
SolveInterCross
...
```

### D2. serialized ID 不是最早的简单 SampleStep rank
`LoopSplitter::InitLoopSequences()` 会重新组织序列，feature loop 与部分 `closed + ConcaveNonSampled` 会重排。

所以 Observation 中：
```text
serialized_position
```
表示 **最终 original Stage2 `_loop.txt` order**，不是绝对几何重要性。

### D3. Stage1 工程 fallback
BracketInches 前处理为处理 SHARP overlap 等问题加入过 guarded heuristic：
```text
LOOPYCUTS_RESOLVE_NON_DIRECT_DIRECTION_OVERLAP=1
LOOPYCUTS_RESOLVE_BOTH_SHARP_ENDS_NEAREST=1
LOOPYCUTS_FALLBACK_PARTIAL_CONCAVE_UNSAMPLED=1
LOOPYCUTS_FALLBACK_AMBIGUOUS_CONCAVE_UNSAMPLED=1
```
这些是工程 fallback，不应在论文中描述成已理论证明的新算法。

---

## E. 原始 Stage2 与 path dependence

原 `run_batch` 大体：
```text
CONVEX:
    skip

CONCAVE:
    always process if valid

REGULAR:
    process only while !converged
```

关键：
- 一旦 convergence 后 REGULAR 被跳过；
- 后续 CONCAVE 可以再次破坏 convergence；
- 已跳过 REGULAR 不重新回来。

Stage2 path dependence 已从源码确认：
- CONCAVE `find_mates()` 会消耗其他 loop；
- HRBF 作用于当前 tet mesh；
- cut 改变后续 tet geometry；
- topology/bubble validators 依赖当前 decomposition；
- reverted cut 不完全 rollback tessellation；
- labeling / clusters 变化；
- MeshExtractor 每一步面对当前 decomposition；
- finalization 还会额外 cut/undo/revert。

因此动态 RL policy 有真实算法依据。

---

## F. 顺序敏感性已验证

Cylinder Plate：
```text
Original:       4 committed → 88 hex
Reverse:        4 committed → 88 hex
Random seed1:  16 committed → 744 hex
Random seed2:  12 committed → 400 hex
Random seed3:  15 committed + 1 revert → 472 hex
Random seed4:  13 committed → 320 hex
```

说明只改 Stage2 order 就可产生数倍 decomposition 差异。

---

## G. Stage2 C++ RL 基础设施

已完成：
- reusable `execute_cut_step()`；
- persistent `-rl-server`；
- external order mode；
- headless finalization；
- `FINALIZE_EVAL` no-save path。

### G1. execute_cut_step
已有 `cut_step.h/cpp`（实际路径以仓库为准）。

回归：
- Cylinder 与原 run_batch 一致；
- random seed3 loop22 正确 `REVERTED`；
- 原 batch final 输出 byte-identical。

当前视为 FROZEN。

### G2. Loop dynamic fields
重要字段：
```text
type
closed
flawed
used
reverted
srf_bubble
active
Nico_bug
...
```

`used` 不一定等于 agent selected。mate consumption / crease handling 等可能自动消耗 loop，所以 Observation 另有 `executed`。

---

## H. C++ RL server 协议

输出：
```text
[RL] READY
[RL] STATE
[RL] ACTIONS
[RL] STEP_RESULT
[RL] FINALIZE_BEGIN
[RL] FINAL_RESULT
[RL] FINALIZE_END
[RL] BYE
```

命令：
```text
STATE
STEP <loop_id>
FINALIZE <output_dir>
FINALIZE_EVAL
QUIT
```

核心纪律：
> `[RL] ACTIONS` 是 legality 唯一权威；Python 不自己推导。

---

## I. RL V1 episode semantics — FROZEN

`regular_phase_closed`：
```text
initial = false
第一次任一步达到 convergence → true
之后永远 true
```

phase 关闭前：
```text
legal = valid CONCAVE + valid REGULAR
```

关闭后：
```text
legal = remaining valid CONCAVE
```

terminal：
```text
没有 legal actions
```

不是：
```text
converged == true
```

selection success：
```text
terminal && current converged
```

BracketInches：
```text
first convergence: loop 28
later: loops 81..89
loop 87 breaks convergence
terminal after 89
converged=0
selection_success=0
```

Dynamic reopen 暂留未来 V2/ablation，不在当前版本实现。

---

## J. Python bridge

文件：
```text
bridge/cpp_client.py
```

核心：
```python
LoopyCutsClient(
    executable,
    mesh_file,
    loop_file,
    echo_logs=False,
)
```

构造即启动：
```text
volumetric_cutter mesh loop -rl-server
```

并等待 READY/STATE/ACTIONS；返回后 `client.state` 与 `client.actions` 可用。

异常：
```text
RLServerProtocolError
RLServerProcessError
```

`RLServerProcessError` 保存：
```text
phase
return_code
expected_prefix
lines
signal_number
signal_name
```

POSIX `SIGABRT → return_code=-6`。

支持 context manager。

---

## K. Finalization — 已审计

`finalize_block_decomposition` 不只是 export，还可能：
- extra cuts；
- undo；
- revert；
- failed loop → crease；
- 多次迭代；
- 某些情况下回退到 hybrid snapshot。

MeshExtractor convergence：
```text
topological + geometric
```
不等于最终 full-hex。

典型：
```text
Deckel:
selection_success=1
最终 NON_FULL_HEX 512/518
```

---

## L. FINALIZE_EVAL — FROZEN

`FINALIZE_EVAL` 与 `FINALIZE` 执行相同 geometry pipeline。

差别：
```text
FINALIZE      保存结果
FINALIZE_EVAL 不保存文件
```

full-hex 时两者都仍执行 `poly_fix_orientation()`。

因此 `FINALIZE_EVAL` 是真实 finalization outcome，不是 surrogate。

---

## M. FinalizationOutcome — FROZEN

文件：
```text
finalization/outcome.py
```

核心：
```python
evaluate_terminal_finalization(client)
```

分类：
```text
FULL_HEX
NON_FULL_HEX
FINALIZATION_CRASH
```

真实非零 C++ process failure → `FINALIZATION_CRASH`。  
Protocol/infrastructure failure 不转换成 mesh quality label，继续 raise。

Ground truth：
```text
Cylinder Original: FULL_HEX 88/88
Cylinder seed3:    FULL_HEX 472/472
Bracket Original:  FINALIZATION_CRASH, SIGABRT -6
Deckel Original:   NON_FULL_HEX 512/518
Eraser Original:   FULL_HEX 3884/3884
Bimba Original:    FULL_HEX 1560/1560
```

Bracket 的已知 assertion：
```text
b_verts.size() <= this->num_verts()+1
```
不是 OOM。

---

## N. Observation V1 — FROZEN

```text
MAX_LOOPS=331
```

Unseen >331：reset error，不 silent truncate。

结构：
```python
{
  "obs": {
    "global": float32[16],
    "loops":  float32[331,14],
    "exists": bool[331],
  },
  "mask": bool[331],
}
```

Global 16：
```text
0 step_fraction
1 available_fraction
2 legal_concave_fraction
3 legal_regular_fraction
4 converged
5 regular_phase_closed
6 log current/initial verts
7 log tets
8 log1p mm_verts
9 log1p mm_edges
10 log1p mm_faces
11 log1p mm_polys
12 diagnostics_valid
13 log1p nonmanifold
14 log1p highgenus
15 log1p buggy
```

Loop 14：
```text
0 serialized_position
1 CONCAVE
2 REGULAR
3 CONVEX
4 closed
5 flawed
6 log1p segments
7 sharp_fraction
8 legal
9 used
10 reverted
11 executed
12 nico_bug
13 top_relevant
```

`exists` 用于真实/padding 区分。

---

## O. Gym Env

```text
envs/loopycuts_env.py
```

特点：
- `Discrete(331)`；
- authoritative C++ ACTIONS；
- persistent C++ process；
- static metadata parse once；
- dynamic state each step；
- invalid action 在进入 C++ 前拒绝；
- base Env 不 finalization；
- base Env 保持纯 Stage2 selection MDP。

---

## P. Tianshou masked Discrete SAC 基础集成

已有 thin `MaskedDiscreteSACPolicy.forward`：
- invalid logits mask；
- 不在 sampled action 后做 remap；
- terminal public mask all false；
- distribution 内部仅为数值构造允许 fallback；
- critic 有 unwrap adapter。

真实环境已完成：
```text
Collector
ReplayBuffer
Discrete SAC update
```
smoke/regression。

后续进入 Actor/Critic 前必须重新找出当前实际 SAC integration 文件，不根据历史名字猜。

---

## Q. Transition Metrics — FROZEN

```text
rewards/transition_metrics.py
```

主要字段：
```text
step
loop_id
status
committed
reverted
step_cost
log_tet_growth
log_vert_growth
step_time
convergence_delta
first_convergence
phase_closed_this_step
terminal
selection_success
terminal_failure
diagnostics_delta_valid
delta_log_nonmanifold
delta_log_high_genus
delta_log_buggy_chains
post_log_nonmanifold
post_log_high_genus
post_log_buggy_chains
delta_log_mm_polys
available_before
available_after
available_drop
```

注意 reverted cut 的 tet refinement 不完全 rollback。

---

## R. Reward V1 / V2

### R1. Selection Reward V1 — FROZEN
\[
r =
-\frac{1}{N_0}
-\Delta\log T
-0.10I_{\mathrm{revert}}
-I_{\Delta conv=-1}
+I_{\Delta conv=+1,\ not\ first}
+3I_{\mathrm{selection\ success}}
-3I_{\mathrm{terminal\ failure}}
\]

### R2. Reward V2 — FROZEN
去掉 selection terminal proxy，dense 保持：
```text
-1/N0
-log tet growth
-0.1 revert
-1 convergence loss
+1 convergence recovery
```

真实 final outcome：
```text
FULL_HEX             +3
NON_FULL_HEX         -3
FINALIZATION_CRASH   -4
```

文件：
```text
rewards/reward_v2.py
```

当前 Reward / Finalization subsystem 不应无故修改。

---

## S. Wrappers 与 Collector regression

```text
envs/finalization_eval_wrapper.py
envs/final_reward_wrapper.py
```

当前结构：
```text
FinalRewardWrapper(
  FinalizationEvalWrapper(
    LoopyCutsEnv(...)
  )
)
```

FinalizationEvalWrapper 只在：
```python
terminated and not truncated
```
执行 `FINALIZE_EVAL`。

Deckel Reward V2 Collector regression：
```text
23 steps
episode return ≈ -4.9574907
offline 与 replay 一致
最后 terminated=true
truncated 全 false
terminal obs_next mask count=0
```

---

## T. 74-model corpus

```text
74/74 parsed
loops min/median/max = 30 / 81.5 / 331
actionable min/median/max = 22 / 63 / 199
C=364, R=4881, V=2026, singular=0
```

原 corpus：
```text
/home/yjk/loopycuts_test/rl_corpus/loop_corpus.csv
/home/yjk/loopycuts_test/rl_corpus/loop_corpus.json
```

**路径纪律：**
Author 文件命名不规则，例如 Plate1/2/3/4 的实际 mesh 名并不等于目录名。所有后续 mesh/loop paths 必须来自 frozen manifest，不能自己拼。

---

## U. Dataset Split V2 — FROZEN

```text
engineering_calibration 5
train                   49
dev                     10
test                    10
```

文件：
```text
/home/yjk/codes/loopycuts_rl/data/manifests/dataset_split_v2.csv
/home/yjk/codes/loopycuts_rl/data/manifests/dataset_split_v2.json
/home/yjk/codes/loopycuts_rl/data/manifests/dataset_split_v2.sha256
```

SHA256：
```text
CSV:
e7bc6ba976417d427d9a105f5b90a54c304c08fdd0baff542e083c8f42ff826b

JSON:
1c8333605e7946bea332b2ec49b2ef42f98667ca2a662ec570a91e0414b6feec
```

### U1. leakage / family audit
External cylinder 与 author cylinder near-duplicate，因此 author cylinder 从 test → engineering calibration。

`tris_open / tris_closed` near same-family，二者均放 train。

当前没有足够证据 family-lock：
```text
bimba / busto_bimba
Plate1..4
bone1 / bone_femur
```

---

## V. Frozen model membership

Engineering：
```text
bimba
deckel
BracketInches
eraser_ball
cylinder_plate
```

Dev：
```text
kiss
sphinx
bone1
ellipse
vessel
busto_bimba
sculpt
beveled_shoulder_2
pinion
rod
```

Blind Test：
```text
hanger
chinese_lion
clef
pig
Plate1
cube_carved
wrench
halved_oblique_scarf_3
gyroidpuzzle
mech_piece
```

### Blind-test 纪律
当前禁止 test：
```text
Original
Random
Heuristic
SAC
FINALIZE_EVAL
```

最终 Phase4 才使用：
```text
--allow-held-out-test
```

---

## W. Baseline policies

```text
policies/simple.py
```

Original：
```python
return min(actions)
```

Random：
```python
self.rng.choice(actions)
```

Random 是每步从**当前 authoritative legal set**采样，不是预先 permutation。

ReplayPolicy 仅 regression/debug。

---

## X. Generic baseline runner

```text
evaluation/baseline_audit.py
```

流程：
```text
frozen manifest
→ hash verification
→ split/model selection
→ test seal
→ LoopyCutsClient
→ run_episode(finalize=False)
→ selection terminal
→ FINALIZE_EVAL
→ classify outcome
→ record row
→ immediately persist CSV/JSON
```

支持：
```text
--split
--policy
--seed
--models
--output-root
--max-steps
--allow-held-out-test
--resume
--overwrite
--echo-logs
```

---

## Y. Engineering Original baseline

```text
Bimba:
24 steps, 23 commit, 1 revert
tet_ratio 2.812954
FULL_HEX 1560/1560

Deckel:
23 steps, 19 commit, 4 revert
tet_ratio 3.478774
NON_FULL_HEX 512/518

BracketInches:
38 steps, 32 commit, 6 revert
selection_success=0
tet_ratio 5.396908
FINALIZATION_CRASH

Eraser:
39 commit
tet_ratio 4.845146
FULL_HEX 3884/3884

Cylinder:
4 commit
tet_ratio 1.588845
FULL_HEX 88/88
```

---

## Z. Engineering Random seed0 baseline

```text
Bimba:
33 steps, 30 commit, 3 revert
tet_ratio 4.010502
FULL_HEX 2728/2728

Deckel:
74 steps, 58 commit, 16 revert
selection_success=0
tet_ratio 25.744195
NON_FULL_HEX 4874/4934

Bracket:
90 steps, 72 commit, 18 revert
tet_ratio 17.703602
NON_FULL_HEX 10034/10093

Eraser:
75 commit
tet_ratio 8.694064
FINALIZATION_CRASH

Cylinder:
7 commit
tet_ratio 1.789266
FULL_HEX 184/184
```

Random seed count 尚未冻结。

---

## AA. Passive ResourceMonitor — 当前最新代码状态

文件：
```text
runtime/resource_monitor.py
```

它是**纯 instrumentation，不是 watchdog**。

不会：
```text
kill
terminate
SIGTERM
SIGKILL
timeout
改变 action
改变 reward
改变 C++
改变 terminated/truncated
```

读取：
```text
/proc/<pid>/status
/proc/meminfo
```

`ResourceStats`：
```text
samples
peak_rss_mb
peak_process_swap_mb
min_mem_available_mb
max_system_swap_used_mb
monitor_elapsed_s
```

### AA1. snapshot()
已加入：
```python
ResourceMonitor.snapshot()
```

只返回当前累计统计，不停止监控。

自测：
```text
A samples=4
B samples=8
C samples=10
```
证明 snapshot 后继续采样。

### AA2. 备份
```text
runtime/resource_monitor.py.before_snapshot
evaluation/baseline_audit.py.before_passive_resource_monitor
evaluation/baseline_audit.py.before_selection_resource_snapshot
```

暂时不要删除。

---

## AB. Baseline resource integration

baseline 在 client 初始化完成后启动 monitor：
```text
sample_interval_s=1.0
```

在 selection terminal、FINALIZE_EVAL 前：
```python
selection_resource_stats = resource_monitor.snapshot()
```

FINALIZE_EVAL 后：
```python
resource_stats = resource_monitor.stop()
```

因此目前记录：

Selection-only：
```text
selection_resource_samples
selection_peak_rss_mb
selection_peak_process_swap_mb
selection_min_mem_available_mb
```

Whole episode：
```text
resource_samples
peak_rss_mb
peak_process_swap_mb
min_mem_available_mb
max_system_swap_used_mb
resource_monitor_elapsed_s
```

---

## AC. Passive monitor regression

Bimba Original 与 Random 在监控前后 trajectory 完全一致，证明当前 passive instrumentation 没有改变 LoopyCuts MDP。

Bimba resource：
```text
Original:
whole peak RSS 2390.3 MiB
process swap 0
min available 3366.2 MiB

Random:
whole peak RSS 3339.9 MiB
process swap 0
min available 2378.7 MiB
```

---

## AD. Deckel resource：当前最重要的新结果

Original：
```text
Selection:
peak RSS              677.5 MiB
process VmSwap          0.0 MiB
min MemAvailable     5089.7 MiB

Whole:
peak RSS             1308.9 MiB
process VmSwap          0.0 MiB
min MemAvailable     4451.6 MiB
```

Random seed0：
```text
Selection:
peak RSS             4689.6 MiB
process VmSwap          0.0 MiB
min MemAvailable     1093.9 MiB

Whole:
peak RSS             6968.6 MiB
process VmSwap       2260.4 MiB
min MemAvailable       49.9 MiB
```

Timing：
```text
selection ≈ 209 s
finalization ≈ 139 s
total ≈ 362 s
```

解释：
- Random selection 已接近 RAM 极限；
- selection 本身尚未 swap；
- 巨大 terminal decomposition 的 FINALIZE_EVAL 触发约 2.26 GiB process swap；
- early SAC exploration 有潜在资源风险；
- 但当前**尚未实现 SAC watchdog**。

---

## AE. 当前 watchdog 决策

当前没有：
```text
ResourceLimitWrapper
RESOURCE_LIMIT reward
RESOURCE_LIMIT terminal
RESOURCE_LIMIT truncation
SAC process kill
```

用户要求：
> Watchdog 只有在必要、且不会严重破坏 SAC 正常学习时才考虑。如果只是少数极端 train model 造成问题，优先把它们在正式训练前归类为 stress/high，而不是为了保留所有模型强行改变 SAC。

当前优先：
```text
passive monitoring
→ train resource feasibility
→ train eligibility
→ curriculum
```

---

## AF. Train model 未来分类

建议：
```text
train_eligible
train_high
train_stress
```

必须在正式 SAC training 前冻结。

不能：
```text
SAC 学不好 → 事后删除
```

不能只按：
```text
actionable > N
```
直接删除。

应综合：
- static loops/actionable；
- Original resource；
- 少量受控 Random pilot；
- 实际 runtime / tet growth；
- 当前机器可行性。

---

## AG. 当前可能的 curriculum

如果 resource audit 支持：
```text
Phase A:
safe/small/medium train_eligible

Phase B:
加入更复杂但 Original 正常的 train_high

Phase C:
可选 train_stress adaptation
```

这比 SAC 初期在全部 49 train 上近随机探索更稳健。

---

## AH. Actor/Critic 设计方向（尚未实现）

Loop encoder：
```text
14 → 64 → 128
```

Global：
```text
16 → 64 → 128
```

Masked set context：
```text
masked mean
masked max
```

每 loop concatenate：
```text
loop 128 + global 128 + mean 128 + max 128 = 512
```

Actor：
```text
512 → 256 → 128 → 1
```
→ `[B,331]` logits + authoritative mask。

Critic1/2：类似 per-loop Q scorer。

初版：
- Actor/Critic 独立；
- LayerNorm；
- ReLU 或 SiLU 统一；
- dropout=0；
- 无 learned loop-ID embedding；
- padding excluded by `exists`。

不建议第一版：
```text
flat 4650→MLP→331
Transformer
GNN
```

---

## AI. 尚未冻结的 SAC 超参数

尚未冻结：
```text
gamma
alpha
n-step
training schedule
checkpoint selection
random baseline seed count
```

历史 gamma candidate：
```text
0.99
0.995
0.997
```

Entropy 初版更倾向 fixed alpha。

---

## AJ. 当前 Phase 状态

```text
Phase 0   reproduction                         ✓
Phase 0.5 order sensitivity                    ✓
Phase 1A  cut helper                           ✓ FROZEN
Phase 1B  C++ RL server                        ✓
Phase 1C  bridge                               ✓
Phase 1D  dynamic policies                     ✓
Phase 1E  RL V1 semantics                      ✓ FROZEN
Phase 1F  terminal/finalization/crash           ✓
Phase 2A  corpus                               ✓
Phase 2C  Observation V1                       ✓ FROZEN
Phase 2D  metrics/reward/finalization           ✓ FROZEN
Phase 2E-A Dataset Split V2                    ✓ FROZEN
Phase 2E-B Original/Random baseline             ✓ infrastructure
                                               ✓ engineering runs
                                               ✓ passive resources
                                               ← CURRENT
Phase 2E-C heuristic baseline                  not started
Phase 2F   Actor/Critic                        not started
Phase 2G   SAC integration regression          not started
Phase 3    formal training                     not started
Phase 4    blind test                          not started
```

---

## AK. 当前准确停点与下一步

最后完成：
```text
Deckel Original/Random:
selection-vs-finalization resource measurement
```

当前决定：
```text
暂不加 SAC watchdog
```

**下一条实际命令：从 frozen train manifest 列 top-15 complexity。**

```bash
cd ~/codes/loopycuts_rl
source .venv/bin/activate

python - <<'PY'
import csv
from pathlib import Path

path = Path("data/manifests/dataset_split_v2.csv")
rows = []

with path.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["split"] != "train":
            continue
        rows.append({
            "model": row["model"],
            "loops": int(row["parsed_loops"]),
            "actionable": int(row["actionable_nonconvex"]),
            "concave": int(row["concave"]),
            "regular": int(row["regular"]),
            "convex": int(row["convex"]),
            "mesh": row["mesh_file"],
            "loops_file": row["loop_file"],
        })

rows.sort(
    key=lambda x: (x["actionable"], x["loops"]),
    reverse=True,
)

print(
    f"{'rank':>4} {'model':<28} {'loops':>6} "
    f"{'action':>7} {'C':>4} {'R':>4} {'V':>4}"
)
print("-" * 70)

for i, row in enumerate(rows[:15], 1):
    print(
        f"{i:>4} {row['model']:<28} {row['loops']:>6} "
        f"{row['actionable']:>7} {row['concave']:>4} "
        f"{row['regular']:>4} {row['convex']:>4}"
    )
PY
```

不要直接运行这 15 个模型。

下一阶段：
1. 查看真实 top-15；
2. 选少量 tail representatives；
3. 第一轮只跑 Original + Passive ResourceMonitor；
4. 判断模型本身是否资源昂贵；
5. 必要时仅少量 Random seed0；
6. 冻结 `train_eligibility_v1.csv`；
7. 再进入 heuristic baseline / Actor-Critic。

---

## AL. 推荐 train_eligibility_v1 schema

```text
model
train_role
static_loops
static_actionable
concave
regular
convex
original_num_steps
original_tet_ratio
original_selection_wall_time
original_finalization_wall_time
original_selection_peak_rss_mb
original_selection_peak_swap_mb
original_peak_rss_mb
original_peak_swap_mb
pilot_policy
pilot_seed
pilot_num_steps
pilot_tet_ratio
pilot_selection_peak_rss_mb
pilot_peak_rss_mb
pilot_peak_swap_mb
reason
```

role：
```text
train_eligible
train_high
train_stress
```

正式训练前 freeze + SHA256。

---

## AM. Phase 2E-C Heuristic baseline

尚未实现。

目的：
> 证明 SAC 的提升不是“任意 state-dependent heuristic 都能做到”。

原则：
- 不使用 SAC 看不到的 oracle 信息；
- 使用相同 C++ legality；
- 使用相同 FINALIZE_EVAL taxonomy；
- 与 Original/Random/SAC 同样评价。

---

## AN. Phase 2F / 2G 之前必须重新查看当前代码

进入 Actor/Critic 前至少重新读取：
```text
envs/loopycuts_env.py
envs/finalization_eval_wrapper.py
envs/final_reward_wrapper.py
rewards/reward_v2.py
rewards/transition_metrics.py
policies/simple.py
bridge/cpp_client.py
当前 masked SAC integration 文件
```

不要根据交接文档猜接口。

---

## AO. 正式训练路线

```text
3A engineering / safe small subset overfit
3B train_eligible multi-model
3C dev checkpoint selection
3D curriculum train_high
3E optional stress adaptation
```

---

## AP. Final Phase4

只有以下全部冻结后：
```text
Actor/Critic architecture
Reward
gamma
alpha
training protocol
curriculum
train eligibility
checkpoint selection
Random seed count
Heuristic
```

才运行 test：
```text
Original
Random
Heuristic
SAC
```

---

## AQ. 最终论文应报告的主要指标

不能只报告 reward。

至少：
```text
FULL_HEX rate
NON_FULL_HEX rate
FINALIZATION_CRASH rate
final_hex / final_total_polys
selected steps
committed
reverted
tet_ratio
vert_ratio
selection runtime
finalization runtime
total runtime
selection peak RSS
whole peak RSS
process VmSwap
```

---

## AR. 新对话必须遵守的纪律

1. 修改前先看真实源码。
2. FROZEN 模块不要悄悄改。
3. 如要改，应新建明确 V2/V3。
4. 不运行 blind test。
5. 不根据 test outcome 调参数。
6. 不根据 SAC reward 删除 train model。
7. 不根据 model 名猜 mesh path。
8. 路径只从 frozen manifest 获取。
9. 不只按 actionable 数删模型。
10. 当前没有 SAC watchdog。
11. Passive ResourceMonitor 只是 instrumentation。
12. 当前 terminated/truncated semantics 未改变。
13. Random seed count 未冻结。
14. Actor/Critic 尚未实现。
15. 正式 SAC training 尚未开始。
16. 不确定时索要源码/日志，不猜。

---

## AS. 进入新对话后推荐先查看的当前源码

Resource monitor：
```bash
sed -n '1,380p' \
~/codes/loopycuts_rl/runtime/resource_monitor.py
```

Baseline runner：
```bash
cd ~/codes/loopycuts_rl

grep -n \
"def run_model\|ResourceMonitor\|selection_resource_stats\|resource_stats\|def main" \
evaluation/baseline_audit.py

sed -n '1,230p' evaluation/baseline_audit.py
sed -n '880,1130p' evaluation/baseline_audit.py
sed -n '1300,1540p' evaluation/baseline_audit.py
```

实际行号若变化，以 `grep -n` 为准。

---

## AT. 代码版本传递原则

新对话判断事实的优先级：

```text
当前真实源码
>
最新终端输出
>
frozen manifest / regression artifacts
>
本交接文档
>
一般经验
```

这条非常重要。

---

# 结束
