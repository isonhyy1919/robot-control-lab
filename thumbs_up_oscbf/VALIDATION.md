# 打包验证记录

验证日期：2026-09-07。环境：Windows、Python 3.14.4，依赖版本见 `requirements.txt`。

- 组合 URDF 的全部网格引用均可在本目录中解析，MuJoCo 成功加载模型。
- 点赞动作执行 `--no-viewer --fast --print-pose-check` 通过。
- 重新导出 202 个轨迹采样点，与打包的当前输入 CSV 逐行一致。
- 使用目录内模型和轨迹运行 OSCBF：1341 个控制采样步；过滤后障碍物及关节限位违规采样步均为 0。
- `validate_original_video.py`、`validate_joint_limit_comparison.py`、`validate_a4_limit_comparison.py` 均通过。
- 9 个 MP4 均可读取元数据、统计帧数，并解码首帧与末帧。
- Python 文件语法、首页和视频展示页的本地链接、文件大小检查通过。
- 归档实验 CSV 和 MP4 与原项目的 SHA-256 一致。JSON 仅将本机路径改为目录内路径。

此次新运行结果保存在打包工作区的临时验证目录，未覆盖仓库中的历史实验结果。运行通过仅说明上述仿真与文件检查通过，实机执行未验证。
