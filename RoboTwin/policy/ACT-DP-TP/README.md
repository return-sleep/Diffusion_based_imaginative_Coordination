### Install
```
cd policy/ACT-DP-TP
cd detr
pip install -e .
cd ..
cd Cosmos-Tokenizer
pip install -e .
#upload policy/ACT-DP-TP/Cosmos-Tokenizer/pretrained_ckpts
```
### Command
```
#data_dir: policy/ACT-DP-TP/data_zarr
cd policy/ACT-DP-TP
bash scripts/act_dp_tp/train.sh bottle_adjust 300 20 20 0
```
### ICCV2025 Version
```
git checkout ICCV25
```
