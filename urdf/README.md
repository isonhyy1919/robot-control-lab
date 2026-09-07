# URDF 模型文件

本目录集中存放机器人控制平台的 URDF 描述、网格资源和预览图。

## 文件说明

- `new_base_only.urdf`：台架模型。
- `linkerhand_l20lite_right.urdf`：L20Lite 右手模型。
- `ARM7_urdf_local.urdf`：七轴机械臂模型。
- `new_base_single_arm_right_hand.urdf`：台架、机械臂和右手组合模型。
- `combined_robot_preview.png`：组合模型零关节位姿预览。

## 资源路径

为便于仓库维护，四个 URDF 中的网格引用统一指向 `meshes/`。请勿随意移动该目录，否则模型加载时会找不到 STL 文件。

## 使用提示

可使用 RViz、Gazebo 或其他兼容 URDF 的可视化工具加载组合模型。开始控制或仿真前，请检查关节限位、碰撞模型、坐标系方向以及控制器配置。
