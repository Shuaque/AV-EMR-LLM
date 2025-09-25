# Pre-processing

We provide a pre-processing pipeline in this repository for detecting and cropping mouth regions of interest (ROIs) as well as corresponding audio waveforms for CMLR


## Setup

1. Install all dependency-packages.

```Shell
pip install -r requirements.txt
pip install scikit-video, pydub
pip install "numpy<1.24"
pip install scikit-image

```

2. Install [retinaface](./tools) tracker,you can put another detector in `/detectors`:

- `cd ../data/preparation/detectors/retinaface/`
- Install [ibug.face_detection](https://github.com/hhj1897/face_detection)

```Shell
git clone https://github.com/hhj1897/face_detection.git
cd face_detection
pip install -e .
cd ..
```
Recommendation: manually download to the specified directory, since errors frequently occur with `ibug/face_detection/retina_face/weights/Resnet50_Final.pth`.

- Install [*`ibug.face_alignment`*](https://github.com/hhj1897/face_alignment)

```
git clone https://github.com/hhj1897/face_alignment.git
cd face_alignment
pip install -e .
cd ..
```
- Reference mean face download from: [Line]( https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/blob/master/preprocessing/20words_mean_face.npy)


## Pre-processing CMLR

To pre-process the CMLR dataset, plrase follow these steps:

1. Download the CMLR dataset from the official website.

2. Download pre-computed landmarks below. If you leave `landmarks-dir` empty, landmarks will be provided with the used of `detector`.

| File Name              | Source URL                                                                              | File Size  |
|------------------------|-----------------------------------------------------------------------------------------|------------|
| CMLR_landmarks.zip     |[GoogleDrive](https://bit.ly/) or [BaiduDrive](https://bit.lyh)(key: mi3c) |     18GB   |


3. Run the following command to pre-process dataset:

```Shell
python preprocess_lrs2lrs3.py \
    --data-dir [data_dir] \
    --landmarks-dir [landmarks_dir] \
    --detector [detector] \
    --root-dir [root_dir] \
    --dataset [dataset] \
    --gpu_type [gpu_type] \
    --subset [subset] \
    --seg-duration [seg_duration] \
    --groups [n] \
    --job-index [j]
```

### Arguments
- `data-dir`: Directory of original dataset.
- `landmarks-dir`: Path to the directory containing landmarks files. If the `landmarks-dir` is specified, face detector will not be used.
- `detector`: Type of face detector. Valid values are: `mediapipe` and `retinaface`. Default: `retinaface`.
- `root-dir`: Root directory of preprocessed dataset.
- `dataset`: Name of dataset. Valid values are: `lrs2` and `lrs3`.
- `gpu_type`: Type of GPU to use. Valid values are `cuda` and `mps`. Default: `cuda`.
- `subset`: Subset of dataset. For `lrs2`, the subset can be `train`, `val`, and `test`. For `lrs3`, the subset can be `train` and `test`.
- `seg-duration`: Length of the maximal segment in seconds. Default: `16`.
- `groups`: Number of groups to split the dataset into.
- `job-index`: Job index for the current group. Valid values are an integer within the range of `[0, n)`.

### Steps
Several steps are defined through the option `--step`, and each step can be executed independently.

`Step 0.` To perform bulk decompression of the CMLR corpus:

    '''
    Make sure you have the following directory structure:

    /.../.../CMLR-CORPUS/
    ├── audio
    │   ├── s1.zip
    ├── video
    │   ├── s1.zip
    ├── text.zip
    ├── train.csv
    ├── val.csv
    ├── test.csv

    '''

`step 1.` To split dataset to `train, val, test` in `../datasets`

`step 2.` Generate file ID lists and text labels from the dataset, and trim video/audio using new time boundaries (default 15 ms; trimming applied if exceeded, optional).



`step 3.` 如果需要生成detection.zip 


av-hubert:
1. 生成 文件ID lrs3/file.list, lrs3/label.list
步骤 1，将 LRS3 中的长话语分成pretraining较短的话语，生成它们的时间边界和标签
步骤 2，根据新的时间边界修剪视频和音频
步骤 3，提取音频用于 trainval 和测试分割
步骤 4，生成文件 ID 列表和相应的文本转录。 ${nshard}和${rank}仅在步骤 2 和 3 中使用。这会将所有视频分片到${nshard}和处理${rank}第分片中，其中 rank 是中的整数[0,nshard-1]。

2. 检测面部特征点并裁剪嘴部 ROI

3. 每个片段的帧数