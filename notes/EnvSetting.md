```bash
# Create and activate conda environment
mamba create -n dcvla python=3.10 -y
mamba activate dcvla

# Install PyTorch. Below is a sample command to do this, but you should check the following link
# to find installation instructions that are specific to your compute platform:
# https://pytorch.org/get-started/locally/
mamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  # UPDATE ME!

# Clone and install the openvla repo
cd src/openvla # pwd: ~/dc-vla/src/openvla
pip install -e .

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO # pwd: ~/dc-vla/LIBERO
# 注意：使用 --config-settings editable_mode=compat 是为了兼容 pip 25.x 版本
# 新版本 pip 默认使用 PEP 660 可编辑安装，但 LIBERO 项目只有 setup.py 没有 pyproject.toml
# 使用兼容模式可以避免 "ModuleNotFoundError: No module named 'libero'" 错误
pip install -e . --config-settings editable_mode=compat

cd src/openvla # pwd: ~/dc-vla/src/openvla
pip install -r experiments/robot/libero/libero_requirements.txt

# 之后会报这个错误
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# tensorflow 2.15.0 requires numpy<2.0.0,>=1.23.5, but you have numpy 2.2.6 which is incompatible.

# 然后`pip install numpy==1.26`即可，之后会报numpy与tensorflow的冲突，不用管
pip install numpy==1.26
```

## 常见问题排查

### LIBERO 路径配置问题

如果遇到类似以下错误：
```
FileNotFoundError: [Errno 2] No such file or directory: '/home/xxx/Documents/xxx/LIBERO/libero/libero/./init_files/...'
```

这是因为 LIBERO 会在首次导入时在 `~/.libero/config.yaml` 中保存路径配置。如果你移动了项目目录或重新安装，需要删除旧的配置文件让其重新生成：

```bash
# 删除旧的配置文件
rm ~/.libero/config.yaml

# 重新生成配置文件（会提示是否自定义数据集路径，一般选 N）
conda activate dcvla
echo "N" | python -c "from libero.libero import benchmark"
```

### EGL 渲染问题

如果遇到类似以下错误：
```
ImportError: Cannot initialize a EGL device display. This likely means that your EGL driver does not support the PLATFORM_DEVICE extension
libEGL warning: MESA-LOADER: failed to open nouveau
```

这是因为 MuJoCo/robosuite 需要配置正确的渲染后端。需要完成以下三个步骤：

#### 步骤 1：设置环境变量

在 `~/.bashrc` 中添加以下环境变量：

```bash
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
echo 'export PYOPENGL_PLATFORM=egl' >> ~/.bashrc
source ~/.bashrc
```

#### 步骤 2：添加用户到 render 和 video 组

```bash
sudo usermod -a -G render $USER
sudo usermod -a -G video $USER
```

**重要**：执行后需要完全退出并重新登录（或重启）才能生效。

#### 步骤 3：创建 NVIDIA EGL vendor 配置文件

如果系统尝试使用 Mesa/nouveau 驱动而不是 NVIDIA 驱动，需要创建 NVIDIA EGL 配置文件：

```bash
# 创建临时文件
cat > /tmp/10_nvidia.json << 'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF

# 复制到系统目录
sudo cp /tmp/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# 验证
ls -la /usr/share/glvnd/egl_vendor.d/
```

这个配置文件告诉系统优先使用 NVIDIA EGL 库而不是 Mesa。

## 问题总结

本次环境配置过程中遇到的主要问题：

### 1. LIBERO 模块导入失败
**错误**：`ModuleNotFoundError: No module named 'libero'`

**原因**：pip 25.x 版本改变了可编辑安装的方式，使用 PEP 660 标准，但 LIBERO 项目只有 `setup.py` 没有 `pyproject.toml`，导致生成的 editable finder 配置不正确。

**解决**：使用兼容模式安装 `pip install -e . --config-settings editable_mode=compat`

### 2. LIBERO 路径错误
**错误**：`FileNotFoundError: [Errno 2] No such file or directory: '/home/xxx/dc-vla/LIBERO/...'`

**原因**：LIBERO 首次导入时会在 `~/.libero/config.yaml` 中保存路径，如果项目目录移动或重新安装，配置文件中的路径会过期。

**解决**：删除 `~/.libero/config.yaml` 并重新导入 LIBERO 让其重新生成配置。

### 3. EGL 渲染初始化失败
**错误**：`ImportError: Cannot initialize a EGL device display`

**原因**：三个方面的问题：
- 缺少 `MUJOCO_GL=egl` 环境变量
- 用户不在 `render` 和 `video` 组，无法访问 GPU 设备
- 系统缺少 NVIDIA EGL vendor 配置，默认使用 Mesa 驱动

**解决**：
1. 设置环境变量到 `~/.bashrc`
2. 添加用户到 `render` 和 `video` 组
3. 创建 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` 配置文件
