# Lepton Pi 热像仪 — 上手操作手册

如何进入树莓派、确认相机活着、取流看图、干净关机。相机曾因**接触虚接**假死过
(见 `webcam-input/.../IR_ASSISTED_TELEOP_PROGRESS.md` 的 Phase H)——文末排障顺序很重要。

> **铁律:只用官方 C++ LeptonModule 二进制 + 项目 sanctioned 工具。**
> **不要**用临时拼的 `python3 -c "import spidev..."` 直读、也不要用 scratchpad 里的
> 一次性 listener。原因:临时 python 直读 SPI 会和 C++ streamer 抢 `spidev0.0`,
> 把控制器搅进不可中断态(D-state),导致 streamer 起不来、要重启才能清。
> Pi 侧取流一律走 **`raspberrypi_video_network`(C++)**;笔记本侧只用项目自带的
> **`view_ir_camera.py --lepton-udp`**(它是该 C++ streamer 的官方配套接收端)。

---

## 0. 硬件与网络

```
笔记本 (192.168.50.1) ──直连网线── 树莓派 (192.168.50.2, 用户 anujn)
                                      └─ FLIR Lepton 3.1R + Breakout v2.0
                                         (SPI: CE0/spidev0.0, 20MHz, mode 3; I2C addr 0x2a)
```

- 相机通过 **UDP 8080** 把 160×120 uint16 TLinear 热帧发到笔记本(每帧 4 个
  10004-byte UDP 数据报，含 footer telemetry；0.01 K/count)。
- SPI 片选是 **CE0 / spidev0.0**(不是 0.1)。
- Pi 由 J3 供电(黄=VIN→Pi pin1 3.3V,黑=GND→Pi pin6)。给 Pi 用**扎实的 5V/3A+ 电源**,
  弱电源会导致 Pi 反复复位、SSH 时通时断。

---

## 1. 进入树莓派

```bash
ssh anujn@192.168.50.2
```

连不上先确认:
- Pi 通电、**耐心等满 60s 开完**(`shutdown -h` 是彻底关机,要物理拔插 USB-C 才重开;
  刚 `reboot` 后 sshd 没起会报 `Connection refused`——等一会再连,别抢半开机)。
- `ping 192.168.50.2`(应 0% 丢包)。

---

## 2. 确认相机活着(健康检查)

**只用两样:`i2cdetect`(标准系统工具,查 CCI)+ 官方 C++ streamer(查 VoSPI)。**

**(a) I2C / CCI —— SSH 进 Pi 后:**
```bash
sudo modprobe i2c-dev
i2cdetect -y 1            # 期望 0x2a 处显示 "2a"(-- = 相机没 boot)
```

**(b) VoSPI —— 直接跑官方 C++ streamer 看它是否稳定出帧**(不要用 python 直读探针):
```bash
~/Project/LeptonModule/software/build/raspberrypi_video_network -net 192.168.50.1 -port 8080
```
应先打印 `Lepton verified: AGC=disabled, Raw14 cooked TLinear=0.01K, manual FFC,
telemetry footer`，再打印网络地址/端口；持续不退出且笔记本侧 viewer 收到完整帧 =
VoSPI 正常。仅进程存活不等于已收齐完整帧。
秒退 / 报 `can't open device` = SPI 被占或 spidev 卡住(见 §5b)。

活着的判据:`i2cdetect` 见 `0x2a` + 上电有 **FFC 快门"咔哒"声** + streamer 稳定运行、
笔记本侧 viewer 能出图。

---

## 3. 取流 + 看图(官方链路)

**Pi 侧 —— 官方 C++ streamer(推荐前台跑,能直接看日志/报错):**
```bash
ssh anujn@192.168.50.2
~/Project/LeptonModule/software/build/raspberrypi_video_network -net 192.168.50.1 -port 8080
# 让它开着别关
```

**笔记本侧 —— 项目自带接收端(注意是 worktree 那份,才有 --lepton-udp):**
```bash
cd $WORKSPACE_ROOT
env -u PYTHONPATH .venv-lerobot/bin/python \
  webcam-input/.worktrees/ir-hand-pressure-so101-teleop/lerobot_teleoperator_so101_webcam/view_ir_camera.py \
  --lepton-udp 8080
```

**便捷等价(脚本替你 ssh 起 C++ streamer):**
```bash
cd $WORKSPACE_ROOT
./scripts/run_lepton_stream.sh start     # ssh 起官方 streamer
./scripts/run_lepton_stream.sh status    # 确认 raspberrypi_video_network 在跑
./scripts/run_lepton_stream.sh stop      # 停
```
`probe` 子命令已移除。查健康只用上面 §2 的 `i2cdetect` + C++ streamer。

**受控手动 FFC(只用 C++，可在 streamer 运行时执行):**
```bash
ssh anujn@192.168.50.2 \
  '~/Project/LeptonModule/software/build/raspberrypi_video_network -ffc-only'
```
此命令先只读核验 `AGC=disabled`、TLinear/manual/telemetry 配置，再触发 FFC；不会重放
video/telemetry 配置。期望输出 `Manual FFC complete`，viewer telemetry 应出现
`complete → imminent/in_progress → complete`，随后 `since_last_ffc` 归零。trial 前调用，
不要在 trial 中调用。

**其它官方 C++ 工具(备查):**
- `raspberrypi_video`(Qt GUI)——需要显示器,headless 用不了。
- `raspberrypi_capture`——存 PGM,但源码是 **80×60(Lepton 2.x)**,**不适配 3.1R 的 160×120**,别用。
- 3.1R headless 的正解就是 `raspberrypi_video_network` + 上面的 viewer。

---

## 4. 干净关机

```bash
ssh anujn@192.168.50.2 'sudo shutdown -h now'
# 等 Pi 的 ACT 绿灯停止闪烁、稳定熄灭后再拔电源,防 SD 卡损坏。
```

---

## 5. 排障 —— 相机"假死"(i2cdetect `--`、streamer 不出帧),先查物理

**大概率是接触虚接,不是坏件。按顺序(断电、用万用表/示波器,不是软件):**

1. **J3 电源接触(头号嫌疑)。** 空载量 J3 pin1–pin2 会显示 3.3V,**但不代表带载 OK**——
   相机 boot 要 ~150mA,虚接会带载压降起不来。**重插/焊实 J3 电源线。**
2. **模块座接触。** 断电把 Lepton 模块从 Molex 座子拔出、**压平正**插回(别歪)。
3. **J5–J9 跳线帽**都在、都压实(把 25MHz 时钟 / 2.8V / 1.2V / 上电时序送进模块)。
4. 仍死 → 示波器在 J2 排针测:**pin18=MASTER_CLK(25MHz 该有)**、
   **pin17=RESET_L / pin20=PW_DWN_L(该为高 ~2.8V)**。都正常才怀疑模块芯片。

**不要**一上来判"模块坏、买新板"——本项目验证过,基本是接触问题,免费能修。
诊断要点:**空载 DMM 电压正常 ≠ 芯片带载能起来。**

### 5b. 官方 streamer 秒退 / 不出帧,但 `i2cdetect` 见 0x2a

不是相机坏——是 **Pi 的 `spidev0.0` / SPI 控制器被卡进不可中断态**。最常见诱因:
**用临时 python 直读 SPI 和 C++ streamer 抢总线**(所以才有文首那条铁律)。

**处理:**
1. **别再跑任何 python spidev 探针**,只用 C++ streamer。
2. `ssh anujn@192.168.50.2 'sudo reboot'`,**等满 60s 开完**再起 streamer(重启清掉卡住的 spidev)。
3. 起 streamer 前确保没有旧 streamer 占 SPI。优先在笔记本运行
   `./scripts/run_lepton_stream.sh stop`;若必须在 Pi 上操作,使用锚定全路径的
   `pkill -f '^$LEPTON_PI_BIN( |$)'`。

Phase 1 streamer 会先保持 CS idle 200 ms 做 VoSPI 软重同步；只有连续 5 次软重同步失败
才尝试 camera reboot。若需回滚，2026-07-21 前的 binary 保存在 Pi：
`~/Project/LeptonModule/software/build/raspberrypi_video_network.pre-phase1-20260721-c6224841`。

重启后仍秒退 / 仍无帧,才回到 §5 查物理(J3 带载电源、模块座、25MHz)。

---

## 6. 相关文件

| 用途 | 路径 |
| --- | --- |
| **Pi 端官方 C++ streamer** | `~/Project/LeptonModule/software/build/raspberrypi_video_network -net 192.168.50.1 -port 8080` |
| **笔记本接收/查看**(sanctioned) | `webcam-input/.worktrees/ir-hand-pressure-so101-teleop/.../view_ir_camera.py --lepton-udp 8080` |
| 便捷起停脚本(start/stop/status) | `scripts/run_lepton_stream.sh` |
| 修复/诊断全过程 | `webcam-input/lerobot_teleoperator_so101_webcam/IR_ASSISTED_TELEOP_PROGRESS.md` (Phase H) |
| Breakout 引脚定义 | `~/Downloads/DS_16912_FLiR...Breakout_Board_V2.pdf`(J2 2×10 / J3 电源 / J5–J9 跳线) |

> 已弃用且不要使用:临时 `python3 -c "import spidev..."` 直读、scratchpad 的一次性
> listener、`scripts/lepton_spi_matrix.py`。`run_lepton_stream.sh probe` 已被移除。
