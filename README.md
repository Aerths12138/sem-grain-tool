# SEM Grain Tool（SEM 晶粒识别工具）

SEM Grain Tool 是一个面向 Windows 的本地网页应用，用于扫描电子显微镜（SEM）
图像中的颗粒/晶粒分割、尺寸统计、人工修正和几何导出。双击 EXE 后，程序只在
本机 `127.0.0.1` 启动服务，并自动打开默认浏览器；上传的图像和分析结果不会发送
到互联网。

## 主要功能

- 使用 CPSAM / Cellpose 识别颗粒和晶粒
- 自动识别标尺，或通过两点人工标定像素尺寸
- 统计颗粒数量、面积、质心和等效直径
- 输出粒径分布 CSV 和统计图
- 在浏览器中删除、合并、新增和分割掩膜，支持撤销/重做
- 按指定面积比例和间距重建不重叠圆形
- 输出以毫米为单位的 DXF 和对应参数 JSON，可用于 COMSOL 等软件
- GPU 不可用、驱动异常或显存不足时自动回退 CPU
- 任务可取消，并带有超时和子进程清理机制

## 下载与启动

预编译程序请从 [GitHub Releases](https://github.com/Aerths12138/sem-grain-tool/releases)
下载。推荐使用 `SEMGrainToolGPU` 完整包：它内置固定版本的 PyTorch、CUDA 11.8、
Cellpose 和 CPSAM 模型，同时保留 CPU 执行能力。只打算使用 CPU 的电脑也可以下载
体积更小的 `SEMGrainToolCPU.7z`。

1. 下载所选版本的全部 `SEMGrainToolGPU.7z.*` 或 `SEMGrainToolCPU.7z.*` 分卷。
2. 使用 7-Zip 从 `.7z.001` 开始解压，其他分卷必须放在同一目录。
3. 保持解压后的目录结构不变，双击 `SEMGrainToolGPU.exe`。
4. 程序会打开一个随机端口的 `http://127.0.0.1:端口/` 页面。
5. 使用结束后点击页面右上角的“关闭软件”。

CPU 包的解压方式相同：下载全部 `SEMGrainToolCPU.7z.*`，从 `.001` 开始解压，
然后双击其中的 `SEMGrainTool.exe`。

> 不要只复制 EXE。`_internal` 目录包含模型、Python 运行时和依赖库，必须和 EXE
> 放在一起。首次启动和首次识别通常会比后续操作慢。

## GPU 与兼容模式

目标电脑不需要单独安装 Python、PyTorch、Cellpose 或 CUDA Toolkit。

| 目标电脑情况 | 实际运行方式 |
| --- | --- |
| NVIDIA 显卡、驱动可用、显存不少于 4 GB | 可选择 GPU |
| 没有 NVIDIA 显卡或没有 NVIDIA 驱动 | 自动使用 CPU |
| NVIDIA 显存小于 4 GB | 为避免 CPSAM 显存溢出，自动使用 CPU |
| GPU 推理发生 CUDA/兼容性错误 | 自动重新使用 CPU 识别 |

GPU 加速仍依赖目标电脑上的 NVIDIA 驱动。4 GB 是当前的安全门槛，不同图像尺寸
仍可能需要更多显存；运行时回退机制会处理大部分 CUDA 失败。

## 使用流程

1. 在网页中上传 SEM 图像。
2. 自动识别标尺，或输入标尺长度并在图中选择两个端点。
3. 选择是否优先使用 GPU，然后开始晶粒识别。
4. 检查识别叠加图；必要时进入掩膜编辑器修正。
5. 查看颗粒统计和粒径分布。
6. 设置面积比例、圆间距并导出圆形预览、CSV、JSON 和 DXF。

运行结果保存在程序目录中的以下文件夹：

- `cellpose_dataset/`：导入和标准化后的图像
- `cellpose_results_pretrained/`：分割掩膜与叠加图
- `particle_analysis/`：颗粒统计 CSV 和图表
- `coordinate_shape_outputs/`：圆形重建和 DXF

## 源码结构

- `pyinstaller_packaging/sem_grain_web_app.py`：本地 HTTP 服务和网页接口
- `pyinstaller_packaging/web_static/index.html`：浏览器界面
- `pyinstaller_packaging/sem_grain_app_frozen.py`：处理流程和掩膜编辑后端
- `device_probe.py`：独立 CUDA、驱动和显存探测
- `detect_scale_bar.py`：标尺识别
- `analyze_particles.py`：颗粒测量和粒径分布
- `generate_shapes_from_particles.py`：圆形重建与 DXF 导出
- `pyinstaller_packaging/`：PyInstaller 构建、测试和验收工具
- `docs/`：使用、环境和版本发布文档
- `legacy/`：旧桌面界面及早期分割、几何实验脚本；不参与当前网页版构建

仓库根目录只保留当前网页版运行、构建和常用 Cellpose 工具。历史实验没有删除，
统一归档在 `legacy/`，方便需要时追溯。

## 从源码测试

项目使用便携 CPU 环境时，可运行网页端测试：

```powershell
.\SEM_Grain_Tool_Portable\env\python.exe -m unittest pyinstaller_packaging.test_sem_grain_web_app
```

测试覆盖文件上传、比例标定、掩膜编辑、设备探测和 GPU 失败后的 CPU 重试。

## 构建发布包

构建前将 CPSAM 权重放到：

```text
cellpose_cache/models/cpsam
```

CPU 构建：

```powershell
.\SEM_Grain_Tool_Portable\env\python.exe pyinstaller_packaging\build_cpu_release.py
```

固定 CUDA 11.8 环境的 GPU 构建：

```powershell
E:\conda_envs\sem_sam_env\python.exe pyinstaller_packaging\build_gpu_release.py
```

构建脚本会先检查 Torch 类型和模型资源，然后运行网页测试、PyInstaller 构建及
打包后验收。详细说明见
[pyinstaller_packaging/README.md](pyinstaller_packaging/README.md)。

## 常见问题

### 双击后没有立即看到页面

首次启动需要加载依赖并进行设备探测，请等待片刻，同时检查安全软件是否拦截。
服务仅监听 `127.0.0.1`，不需要开放公网端口。

### 目标电脑已经安装了不同版本的 Torch/CUDA，会冲突吗？

不会。发布包使用自己的运行时，不读取目标电脑的 Python 或 Torch 环境。

### 为什么检测到了 NVIDIA 显卡却仍显示 CPU？

常见原因是驱动不可用、显存少于 4 GB，或 CUDA 烟雾测试失败。页面设备状态会显示
具体原因。

### 可以只下载源码直接运行吗？

可以，但需要自行准备兼容的 Python 环境、依赖和 CPSAM 权重。普通用户建议直接
下载 Release 成品。

## 隐私与安全

- 服务只绑定本机回环地址 `127.0.0.1`。
- API 使用每次启动随机生成的访问令牌。
- 图像、掩膜和统计结果全部保存在本地。
- 本项目不会自动上传 SEM 图像或分析数据。
