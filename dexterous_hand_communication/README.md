# 灵巧手通讯（Windows / PCAN-USB）

本目录整理本地灵巧手通讯项目所需的 SDK 源码、配置、联调脚本、依赖和 API 文档。

## 当前设备配置

- 右手，Windows，PEAK PCAN-USB，通道 `PCAN_USBBUS1`。
- SDK 配置记录：本项目 L20 Lite 固件自报 `LHT10`，按 **L10 蜗轮协议**控制，位置向量为 10 项。
- `LinkerHand/config/setting.yaml` 已保留右手 `JOINT: L10`、`TOUCH: False` 和 PCAN 通道设置。
- 这是当前项目的设备适配信息，不代表所有 L20 Lite 都使用相同协议；更换设备时需核对设备身份。

## 文件结构

```text
dexterous_hand_communication/
├── README.md
├── SOURCE.md
└── vendor/linkerhand-python-sdk/
    ├── LinkerHand/                       # 通讯 API、CAN/RS485 实现及配置
    ├── test_l20_lite_l10_protocol.py      # 当前设备入口；默认仅读取
    ├── test_l20_read.py                  # 20 项 L20 协议读取脚本
    ├── test_l20_small_move.py            # 20 项 L20 协议动作脚本
    ├── requirements-windows-l20.txt      # Windows PCAN 通讯最小依赖
    ├── requirements.txt                 # 上游可选 GUI/仿真等完整依赖清单
    ├── example/                         # 上游 GUI、状态读取及动作示例
    └── doc/API-Reference.md              # 上游 API 文档
```

## 安装与读取

先安装与 Python 位数匹配的 PEAK PCAN 驱动和 PCAN-Basic 运行库，连接设备电源与 USB-CAN。
在仓库根目录打开 PowerShell，执行：

```powershell
cd dexterous_hand_communication/vendor/linkerhand-python-sdk
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows-l20.txt
.\.venv\Scripts\python.exe test_l20_lite_l10_protocol.py
```

此脚本默认读取 10 项位置，不发送动作指令。脚本显式指定右手、L10 协议和 PCAN 通道；更换通道或左右手时，需要同步修改脚本中的 `LinkerHandApi(...)` 参数，仅改 YAML 不会覆盖这些显式参数。

## 单关节小幅动作

确认设备固定、运动空间空闲且没有其他程序同时控制后，执行：

```powershell
.\.venv\Scripts\python.exe test_l20_lite_l10_protocol.py --move --joint index --delta -4 --speed 30
```

此命令基于当前读数修改食指目标，再读取结果。`--move` 才会启用动作；`--delta` 和 `--speed` 可调整，建议先使用上面的默认小幅参数。位置值为协议数值 0–255，并非角度。

## 保留的 L20 协议脚本

`test_l20_read.py` 和 `test_l20_small_move.py` 使用 `hand_joint="L20"`，要求 20 项位置反馈，仅适用于确认采用该协议的设备。当前配置的 LHT10 设备优先使用 `test_l20_lite_l10_protocol.py`。

注意：`test_l20_small_move.py` 读取通过后会直接发送动作，没有 `--move` 开关。三个 `test_*.py` 都是需要真实硬件的联调程序，不是自动化单元测试。

## 可选示例与来源

上游 GUI 和其他型号示例保留在 `example/`，其依赖不包含在最小依赖清单中。上游 `requirements.txt` 含多个仿真、视觉和其他硬件依赖，使用可选功能时再按需安装。

详见 [来源与整理说明](SOURCE.md)、[上游中文说明](vendor/linkerhand-python-sdk/README_CN.md) 和 [API 文档](vendor/linkerhand-python-sdk/doc/API-Reference.md)。

本次归档只验证文件完整性、Python 语法与导入，不连接硬件或执行动作；硬件行为需在实际设备上验证。
