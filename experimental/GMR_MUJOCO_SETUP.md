# GMR + MuJoCo 部署说明

这是实验性 `FreeMoCap -> GMR -> MuJoCo` bridge 的最短部署流程。

## 1. 初始化 GMR

在仓库根目录运行：

```powershell
git submodule update --init --recursive external/GMR
```

把 GMR 安装进当前虚拟环境：

```powershell
.\.venv\Scripts\python.exe -m pip install -e external\GMR
```

## 2. 检查并应用 SciPy patch

先检查当前环境是否需要 patch：

```powershell
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'experimental'); import gmr_runtime; print(gmr_runtime.gmr_scipy_patch_state())"
```

如果输出是 `missing`，应用 patch：

```powershell
git -C external/GMR apply ../../experimental/patches/gmr_scipy_compat.patch
```

再验证结果：

```powershell
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'experimental'); import gmr_runtime; print(gmr_runtime.validate_gmr_runtime(require_mujoco=True).patch_state)"
```

输出应为 `patched` 或 `not_needed`。

## 3. 启动 MuJoCo 可视化

GUI：

```powershell
.\.venv\Scripts\python.exe experimental\skellycam_live_bridge_gui.py
```

GUI 默认勾选 `MuJoCo viewer`。

CLI 示例：

```powershell
.\.venv\Scripts\python.exe experimental\freemocap_to_gmr_bridge.py `
  --source skellycam `
  --calibration-toml <camera_calibration.toml> `
  --camera-ids 0,1 `
  --tracker pose `
  --model-complexity 0 `
  --parallel-camera-tracking `
  --actual-human-height 1.6 `
  --mujoco-viewer `
  --mujoco-fps 30
```

## 注意

- 不要提交 `external/GMR` 里的 patch 后文件；只提交本仓库里的 [experimental/patches/gmr_scipy_compat.patch]即可。
- 应用 patch 后，`external/GMR` 可能会显示两个 modified 文件，这是当前 SciPy 不支持 `scalar_first=True` 时的预期状态。
- 不加 `--mujoco-viewer` 时，ZMQ/sim2real bridge 仍然照常工作。
