# Rethinking Interactive Image Segmentation with Low Latency, High Quality, and Diverse Prompts
Pytorch implementation for paper [Rethinking Interactive Image Segmentation with Low Latency, High Quality, and Diverse Prompts](https://arxiv.org/), CVPR 2024. <br>
Qin Liu, Jaemin Cho, Mohit Bansal, Marc Niethammer <br>
UNC-Chapel Hill <br>

<p align="center">
  <img src="./assets/medal.gif" alt="drawing", height="180"/>
  <img src="./assets/bicyclestand.gif" alt="drawing", height="180"/>
  <img src="./assets/crack.gif" alt="drawing", height="180"/>

</p>


#### [Paper](https://arxiv.org/) | [Demo Videos](https://drive.google.com/drive/folders/13tOhSYFCY2Ue8QR5rR8EEWHXGE75Zkxo?usp=sharing)

## Installation
The code is tested with ``python=3.10``, ``torch=2.2.0``, ``torchvision=0.17.0``.
```
git clone https://github.com/uncbiag/SegNext
cd SegNext
```
Now, create a new conda environment and install required packages accordingly.
```
conda create -n segnext python=3.10
conda activate segnext
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
## Getting Started
First, download three model weights: vitb_sax1 (408M), vitb_sax2 (435M), and vitb_sax2_ft (435M). These weights will be automatically saved to the ``weights`` folder.
```
python download.py
``` 
Run interactive GUI with the downloaded weights. The ``assets`` contains images for demo.
```
./run_demo.sh
``` 

## Datasets
We train and test our method on three datasets: DAVIS, COCO+LVIS, and HQSeg-44K.

| Dataset   |                      Description             |           Download Link              |
|-----------|----------------------------------------------|:------------------------------------:|
|DAVIS      |  345 images with one object each (test)      |  [DAVIS.zip (43 MB)][DAVIS]          |
|HQSeg-44K  |  44320 images (train); 1537 images (val)     |  [official site][HQSeg]              |
|COCO+LVIS* |  99k images with 1.5M instances (train)      |  [original LVIS images][LVIS] + <br> [combined annotations][COCOLVIS_annotation] |

[HQSeg]: https://huggingface.co/sam-hq-team/sam-hq-training/tree/main/data
[LVIS]: https://www.lvisdataset.org/dataset
[DAVIS]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/DAVIS.zip
[COCOLVIS_annotation]: https://github.com/saic-vul/ritm_interactive_segmentation/releases/download/v1.0/cocolvis_annotation.tar.gz

Don't forget to change the paths to the datasets in [config.yml](config.yml) after downloading and unpacking.

(*) To prepare COCO+LVIS, you need to download original LVIS v1.0, then download and unpack 
pre-processed annotations that are obtained by combining COCO and LVIS dataset into the folder with LVIS v1.0. (The combined annotations are prepared by [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation).)

## Evaluation
We provide a script (``run_eval.sh``) to evaluate our presented models. The following command runs the NoC evaluation on all test datasets.
```
python ./segnext/scripts/evaluate_model.py --gpus=0 --checkpoint=./weights/vitb_sa2_cocolvis_hq44k_epoch_0.pth --datasets=DAVIS,HQSeg44K
```

<table>
    <thead align="center">
        <tr>
            <th rowspan="2"><span style="font-weight:bold">Train</span><br><span style="font-weight:bold">Dataset</span></th>
            <th rowspan="2">Model</th>
            <th colspan="4">HQSeg-44K</th>
            <th colspan="4">DAVIS</th>
        </tr>
        <tr>
            <td>5-mIoU</td>
            <td>NoC90</td>
            <td>NoC95</td>
            <td>NoF95</td>
            <td>5-mIoU</td>
            <td>NoC90</td>
            <td>NoC95</td>
            <td>NoF95</td>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td rowspan="1">COCO+LVIS</td>
            <td align="left"><a href="https://drive.google.com/uc?export=download&id=1eqkd5-J9MELGIw2WRcT5hejsnGc5oO30">vitb-sax1 (408 MB)</a></td>
            <td>85.41</td>
            <td>7.47</td>
            <td>11.94</td>
            <td>731</td>
            <td>90.13</td>
            <td>5.46</td>
            <td>13.31</td>
            <td>177</td>
        </tr>
        <tr>
            <td rowspan="1">COCO+LVIS</td>
            <td align="left"><a href="https://drive.google.com/uc?export=download&id=1oxwCm4bFby6RgltO_tl54BqRN9tojylT">vitb-sax2 (435 MB)</a></td>
            <td>85.71</td>
            <td>7.18</td>
            <td>11.52</td>
            <td>700</td>
            <td>89.85</td>
            <td>5.34</td>
            <td>12.80</td>
            <td>163</td>
        </tr>
        <tr>
            <td rowspan="1">COCO+LVIS+HQ</td>
            <td align="left"><a href="https://drive.google.com/uc?export=download&id=1yDN3mwBBO33TlA0KRdO2s07Q5HWXR6nt">vitb-sax2 (435 MB)</a></td>
            <td>91.75</td>
            <td>5.32</td>
            <td>9.42</td>
            <td>583</td>
            <td>91.87</td>
            <td>4.43</td>
            <td>10.73</td>
            <td>123</td>
        </tr>
    </tbody>
</table>

For SAT latency evaluation, please refer to [eval_sat_latency.ipynb](./notebooks/eval_sat_latency.ipynb).

## Training
We provide a script (``run_train.sh``) for training our models on the HQSeg-44K dataset. You can start training with the following commands. By default we use 4 A6000 GPUs for training.
```
# train vitb-sax1 model on coco+lvis 
MODEL_CONFIG=./segnext/models/default/plainvit_base1024_cocolvis_sax1.py
torchrun --nproc-per-node=4 --master-port 29504 ./segnext/train.py ${MODEL_CONFIG} --batch-size=16 --gpus=0,1,2,3

# train vitb-sax2 model on coco+lvis 
MODEL_CONFIG=./segnext/models/default/plainvit_base1024_cocolvis_sax2.py
torchrun --nproc-per-node=4 --master-port 29505 ./segnext/train.py ${MODEL_CONFIG} --batch-size=16 --gpus=0,1,2,3

# finetune vitb-sax2 model on hqseg-44k 
MODEL_CONFIG=./segnext/models/default/plainvit_base1024_hqseg44k_sax2.py
torchrun --nproc-per-node=4 --master-port 29506 ./segnext/train.py ${MODEL_CONFIG} --batch-size=12 --gpus=0,1,2,3 --weights ./weights/vitb_sa2_cocolvis_epoch_90.pth

```

## Citation
```
TBD
```
