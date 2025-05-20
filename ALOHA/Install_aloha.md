
# Install aloha env
```
conda create -n aloha python=3.10.12
conda activate aloha
pip install torchvision
pip install torch
pip install pyquaternion
pip install pyyaml
pip install rospkg
pip install pexpect
pip install mujoco==2.3.7
pip install dm_control==1.0.14
pip install opencv-python
pip install matplotlib
pip install einops
pip install packaging
pip install h5py
pip install ipython
pip install diffusers
pip install decord
cd act/detr && pip install -e .
cd ..
``` 
# Install Cosmos env and Download checkpoints from Hugging Face
https://github.com/NVIDIA/Cosmos-Tokenizer
```
git clone https://github.com/NVIDIA/Cosmos-Tokenizer.git
cd Cosmos-Tokenizer
apt-get install -y ffmpeg git-lfs
git lfs pull
pip3 install -e .
cd ..
```

# Download dataset and Change dataset path 
1. Download the dataset from [ALOHA_Data](https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O)

2. Modify `constants.py Line 5` to your own dataset path

# Run the codes

## Training script
```
bash script/train_eval.sh sim_insertion_human 20000 0 0
# bash script/train_eval.sh <task_name> <num_steps> <seed> <cuda_id>
```
## Evaluation script
```
bash script/eval.sh sim_insertion_human 20000 0 0 0 
# bash script/train_eval.sh <task_name> <num_steps> <seed> <cuda_id> <ckpt_type>
```
