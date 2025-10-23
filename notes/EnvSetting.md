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
pip install -e .

cd src/openvla # pwd: ~/dc-vla/src/openvla
pip install -r experiments/robot/libero/libero_requirements.txt

pip install numpy==1.26
```

之后会报这个错误

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorflow 2.15.0 requires numpy<2.0.0,>=1.23.5, but you have numpy 2.2.6 which is incompatible.

然后`pip install numpy==1.26`即可，之后会报numpy与tensorflow的冲突，不用管