# Prototypical Transformer As Unified Motion Learners (ICML 2024)



> **Abstract:** *In this work, we introduce the Prototypical Transformer (ProtoFormer), a general and unified framework that approaches various motion tasks from a prototype perspective. ProtoFormer seamlessly integrates prototype learning with Transformer by thoughtfully considering motion dynamics, introducing two innovative designs. First, Cross-Attention Prototyping discovers prototypes based on signature motion patterns, providing transparency in understanding motion scenes. Second, Latent Synchronization guides feature representation learning via prototypes, effectively mitigating the problem of motion uncertainty. Empirical results demonstrate that our approach achieves competitive performance on popular motion tasks such as optical flow and scene depth. Furthermore, it exhibits generality across various downstream tasks, including object tracking and video stabilization.*


>
> <p align="center">
> <img width="940" src="resources/framework.jpg">
> </p>



## Installation

For installation, please create a new environment and follows these steps:

```
pip install torch==1.9.1+cu111 torchaudio==0.9.1 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install matplotlib
pip install tensorboard
pip install scipy
pip install opencv-python
pip install yacs
pip install loguru
pip install einops
pip install timm==0.4.12
pip install imageio
```

## Dataset preparation

By default `datasets.py` will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder

```Shell
├── datasets
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
```


## Training

To train your own model, please apply the following command. Give on FlyingChair on optical flow as an example.

```
sh ./tools/dist_train.sh configs/resnet/resnet50_8xb32_in1k_centroids.py 8 \
  --work-dir SCRATCH_DIR 
```

General case:

```
sh ./tools/dist_train.sh configs/(resnet/swin_transformer)/xxxxxx.py 8 \
  --work-dir SCRATCH_DIR
```

| Dataset | Command | 
|:-----------:|:-----------:|
|MSR-VTT-9k|`python train.py  --arch=clip_stochastic    --exp_name=MSR-VTT-9k --videos_dir={VIDEO_DIR}  --batch_size=32 --noclip_lr=3e-5 --transformer_dropout=0.3  --dataset_name=MSRVTT --msrvtt_train_file=9k     --stochasic_trials=20 --gpu='0' --num_epochs=5  --support_loss_weight=0.8`|
|LSMDC|`python train.py --arch=clip_stochastic   --exp_name=LSMDC --videos_dir={VIDEO_DIR}  --batch_size=32 --noclip_lr=1e-5 --transformer_dropout=0.3 --dataset_name=LSMDC    --stochasic_trials=20 --gpu='0'  --num_epochs=5  --stochastic_prior=normal --stochastic_prior_std=3e-3`|
|DiDeMo|`python train.py  --num_frame=12 --raw_video  --arch=clip_stochastic   --exp_name=DiDeMo --videos_dir={VIDEO_DIR} --batch_size=32 --noclip_lr=1e-5 --transformer_dropout=0.4 --dataset_name=DiDeMo     --stochasic_trials=20 --gpu='0' --num_epochs=5`| 


## Infer

Download [trained weights](https://drive.google.com/drive/folders/1zCT10t09mXw-8iLqDvkmxR46lOD5dsv4?usp=sharing)

```
# Single-gpu testing
pip list | grep "mmcv\|mmcls\|^torch"
python tools/test.py local_config_file.py model.pth --out result.pkl --metrics accuracy
```

## License
ProtoFormer is under the Apache License.


## Citation

If you find the work is useful, please cite it as:

```
@inproceedings{wang2023visual,
  title={Visual recognition with deep nearest centroids},
  author={Wang, Wenguan and Han, Cheng and Zhou, Tianfei and Liu, Dongfang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
```


## Acknowledgement

This project is built on the following codes. Great thanks to them!
- [RAFT](https://github.com/princeton-vl/RAFT)
- [GMA](https://github.com/zacjiang/GMA)
- [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official)
- [R-MSFM](https://github.com/jsczzzk/R-MSFM)