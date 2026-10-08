# SIHSP-TT

Skill-Initialized Hierarchical Self-Play for Two-Robot Table Tennis。

当前阶段是单台 Panthera 机械臂的随机单次回球：确定性准备后，PPO 在参考横拍轨迹上学习挥拍时机和小幅修正。训练使用 UniLab Manager-Based 环境与 RSL-RL PPO，物理后端为 MuJoCo。

## 项目结构

| 路径 | 职责 |
|---|---|
| `src/`、`tests/` | 不依赖仿真器的球状态、预测、导轨与转腰轨迹及测试 |
| `assets/` | 当前 Panthera URDF 与本地机器人资源 |
| `UniLab/` | 内置的环境、任务、准备控制、观测、reward、训练、评估与可视化 |
| `UniSim/` | 内置的仿真器公共接口、逐物理步观测与受限轨迹跟踪 |
| `docs/preparation.md` | 早期预测和准备阶段的说明与验证记录 |
| `runs/` | 当前模型、加载配置和新运行输出；仅提交当前可复现模型文件 |

UniLab 和 UniSim 已作为源码内置在本仓库中。UniLab 通过相对 editable 依赖引用根项目和 `../UniSim`；Linux 克隆本仓库后可在相同目录结构下同步依赖并继续训练。机器人 URDF 和 meshes 也随仓库提供。

## 当前任务与入口

正式入口为 `panthera_horizontal/mujoco`，注册名为 `PantheraHorizontalReturn`。导轨、转腰、高度准备保持确定性，link5 固定；PPO 控制挥拍时机、link6 拍面和 link1/link2/link3/link4 的位置残差。当前 URDF 边界保持有效，导轨参数仍是仿真假设。

控制分工、reward、训练命令和验收边界见 [回球训练说明](UniLab/src/unilab/tasks/manipulation/table_tennis/RETURN_TRAINING.md)。旧 `panthera_return/mujoco` 保留为直接速度控制对照，不能替代当前模型的动作语义。

在项目根目录进入 UniLab。Linux 首次配置时：

```bash
cd UniLab
uv sync --locked --group table-tennis --extra mujoco --extra motrix --extra uni_rl
```

如果机器没有 MuJoCo 原生窗口环境，可以继续训练并设置 `training.no_play=true`；可视化需要 Linux 图形环境和对应的 MuJoCo viewer 支持。

Panthera 的 XML、URDF 和 meshes 必须已在本地，依赖同步不会自动恢复自定义机器人资源。

## 可视化查看已训练模型

在 `UniLab/` 中运行。当前模型保存在 `runs/stage1/`，`run_config.json` 必须与模型同目录：

```bash
uv run --no-sync mjpython -m unilab.scripts.play_interactive \
  --algo ppo --task panthera_horizontal --sim mujoco \
  interactive.action_mode=policy \
  algo.load_run=../runs/stage1 \
  algo.checkpoint=49 interactive.speed=0.5 \
  interactive.camera_follow_body=false interactive.camera_distance=8 \
  interactive.camera_elevation=-20 interactive.camera_azimuth=90
```

空格暂停/继续，`N` 单步，`+`/`-` 调整速度；关闭窗口退出。`mjpython` 用于 macOS 原生 MuJoCo 窗口。

## 独立评估

在 `UniLab/` 中评估同一批随机发球：

```bash
uv run --no-sync python scripts/tools/evaluate_panthera_return.py \
  --task panthera_horizontal/mujoco --mode policy \
  --checkpoint ../runs/stage1/model_49.pt \
  --episodes 1000 --num-envs 32 --seed 20007 --paired-serves \
  --output-dir ../runs/eval_model49_seed20007
```

输出为 `summary.json`、`episodes.jsonl`、`serve_bank.json`。快速检查可将球数改为 100；新样本应更换 seed。每次运行应另选输出目录以保留证据。

## 当前证据与待处理项

2026-10-07 初版完成 51,200 transitions，独立种子 20007 的 1,000 球正式结果为 972/1000，95% Wilson 区间约 95.98%–98.06%。历史实验原始记录已清理，当前仅保留这一版模型及其运行配置、训练摘要。

28 条正式“下网失败”已逐球重放：继续记录后均首次落在对方台内。已定位完整离网后的重复球底高度条件产生误判；修正尚未实施，正式统计保持原值。当前数值动作门槛也尚未按视频标定。有限样本不证明整个分布达到 100% 或实机可用。

## 验证代码

在 `UniLab/` 中运行：

```bash
uv run --no-sync pytest -q ../tests tests/tasks/manipulation/table_tennis tests/tasks/test_panthera_return_training.py
UNILAB_LOCAL_UNISIM="$(cd ../UniSim && pwd)" make check
UNILAB_LOCAL_UNISIM="$(cd ../UniSim && pwd)" make test
```

本地 UniSim 使用已有的来源验证配置。部分通用框架测试还需要其他机器人资源；缺失资源应单独报告。
