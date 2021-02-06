# *MFQE 2.0: A New Approach for Multi-frame Quality Enhancement on Compressed Video* (TPAMI 2019)

- [*MFQE 2.0: A New Approach for Multi-frame Quality Enhancement on Compressed Video* (TPAMI 2019)](#mfqe-20-a-new-approach-for-multi-frame-quality-enhancement-on-compressed-video-tpami-2019)
  - [0. Background](#0-background)
  - [1. Pre-request](#1-pre-request)
    - [1.1. Environment](#11-environment)
    - [1.2. Dataset](#12-dataset)
  - [2. Test](#2-test)
  - [3. Training](#3-training)
  - [4. Q&A](#4-qa)
  - [5. License & Citation](#5-license--citation)
  - [6. See more](#6-see-more)

## 0. Background

Official repository of [[*MFQE 2.0: A New Approach for Multi-frame Quality Enhancement on Compressed Video*]](http://arxiv.org/abs/1902.09707), TPAMI 2019. [[速览 (中文)]](https://github.com/RyanXingQL/Blog/blob/master/posts/mfqev2.md)

- The first **multi-frame** quality enhancement approach for compressed videos.
- The first to consider and utilize the **quality fluctuation** feature of compressed videos.
- Enhance low-quality frames using **neighboring high-quality** frames.

![Demo](https://user-images.githubusercontent.com/34084019/105737566-10a31b00-5f71-11eb-9d2c-19780ab94ab1.png)

Feel free to contact: ryanxingql@gmail.com.

## 1. Pre-request

### 1.1. Environment

- Python 3.5
- TensorFlow 1.8 (1.13/14 is ok but with warnings)
- TFLearn
- Scikit-image (for calculating PSNR and SSIM)

### 1.2. Dataset

We open-source our lossless video dataset, including 108 videos for training and 18 common test sequences (recommended by ITU-T) for test.

Download link: [[DropBox]](https://www.dropbox.com/sh/d04222pwk36n05b/AAC9SJ1QypPt79MVUZMosLk5a?dl=0) [[百度网盘 (mfqe)]](https://pan.baidu.com/s/1oBZf75bFGRanLmQQLAg4Ew)

<details>

<summary><b>How to compress videos</b></summary>

We have also provided the video compression toolbox in the dataset link.

First edit `option.yml` in `video_compression/`, i.e., `dir_dataset` and `qp`.

Then run:

```bash
cd video_compression/
chmod +x TAppEncoderStatic
python unzip_n_compress.py
```

Finally, we will get:

```tex
MFQEv2_dataset/
├── train_108/
│   ├── raw/
│   └── HM16.5_LDP/
│       └── QP37/
├── test_18/
│   ├── raw/
│   └── HM16.5_LDP/
│       └── QP37/
├── video_compression/
│   └── ...
└── README.md
```

</details>

## 2. Test

1. Download the `data` and `model` folders at [[Google Drive]](https://drive.google.com/drive/folders/1L-d4ptHZWV_jLl6KGvY81CochKcoY4wj?usp=sharing) [[百度网盘 (mfqe)]](https://pan.baidu.com/s/1gE-VnMTgRW-57QiUwVNlVQ).
2. Put your test videos at `data/test/raw` and `data/test/compressed`. All videos in `data/test/compressed` will be enhanced one by one.
3. Change information in `main_test.py`, e.g., QP and GPU index.
4. Run `main_test.py`.

**Note: Enhancing class A sequences *Traffic* and *PeopleOnStreet* may lead to OOM.** See Q&A.

The average improved PSNR and improved SSIM results will be recorded in `out/record_test.txt`, which are the same as that in our paper:

![result](https://user-images.githubusercontent.com/34084019/105737588-16006580-5f71-11eb-9820-b974aeca1917.png)

## 3. Training

We have released our data processing and training code.

**Note: MFQEv2 may be hard to train.** MFQEv2 network may be sensitive to training data, PQF label and training method. We design our network (i.e., MFQEv2 network) as simple as possible to implement our MFQE strategy; therefore, it is very compact (with only 255K params) but not robust. We highly recommend using a more robust (e.g., wider and deeper) network with MFQE strategy.

<details>

<summary><b>Overview</b></summary>

For non-PQF enhancement:

- `main_extract_TrainingSet_NP`
  - Obtain patch pairs (pre-PQF, mid-non-PQF and sub-PQF patch pairs) from videos.
  - Shuffle and stack these pairs into `.npy` files.
- `main_train_NP`: Train non-PQF enhancement model at QP=37 (training model at QP=37 may be easier than other QPs).
- `main_train_FineTune_NP`: Train non-PQF enhancement models at other QPs by fine-tuning QP=37 model.

For PQF enhancement:

- `main_extract_TrainingSet_P`
  - Obtain patch pairs (pre-PQF, mid-PQF and sub-PQF patch pairs) from videos.
  - Shuffle and stack these pairs into `.npy` files.
- `main_train_FinetuneFromStep1_P`: Train PQF enhancement models by fine-tuning non-PQF enhancement models at corresponding QPs.

You can also train your own model by fine-tuning the open-source pre-trained model.

</details>

Unfortunately, the above training codes are written in different times and devices. To run these code properly, you may have to change some paths.

I'm sorry about the coarse training code of my first scientific work MFQEv2. If you're finding a more robust work for practical use, see my implementation of [[STDF (AAAI 2020)]](https://github.com/RyanXingQL/STDF-PyTorch).

## 4. Q&A

<details>

<summary><b>How to enhanced PQFs</b></summary>

In MFQEv2, PQFs are also enhanced using their neighboring PQFs. Note that the models for PQF enhancement and non-PQF enhancement are trained separately.

In MFQEv1, PQFs are enhanced using other image enhancement approaches.

</details>

<details>

<summary><b>How to detect PQFs</b></summary>

In the training stage, we use **ground truth PQF labels**, i.e., labels that generated by PSNR values. See our paper for more details.

In the testing stage, we provide PQF labels of all 18 videos in `data`. Besides, we have two more options for you:

1. Use **ground truth PQF labels** based on PSNR values. PSNR values can be simply obtained from either codecs or bit flow (e.g., log files generated by encoder include PSNR values). Also, we proved in our paper that the PSNR-generated label and detector-generated label have almost the same effect on the final enhancement performance.
2. Simply based on **QP values**. We find that among the 38 values, the QP value is of most importance. Therefore, one can also generate the PQF label according to the QP values. This option works especially for the LDP-encoded video. Note that: a low QP value usually indicates high image quality.

**Further explanation**

In the testing stage, we use a Bi-LSTM based detector to generate PQF labels. It requires a 38-dimension vector as the input. In particular, 36 values are generated by a NR-IQA method, and the other 2 values are QP and bitrate values.

In other words, MFQE approach requires coding information (QP and bitrate), which can simply be extracted from the encoder log file (e.g., log.txt generated by HM16.5).

Specifically:

1. We first encode the raw video by HM16.5 codec, and thus obtain compressed video and its log file (includes QP, bitrate and also PSNR value of each frame).
2. We generate the 36 values by the open-source MATLAB code of the NR-IQA method adopted in our paper.
3. We extract the QP and bitrate values, and combine them with the above 36 values. This way, we obtain a 38-dim vector for each frame.
4. We feed all 38-value vectors of this video into the BiLSTM detector, and finally obtain the PQF label of this video.

As we can see, the whole process is a bit complicated:

- One can use various codecs, and then obtain various log files with different format.
- The data loading process can vary.
- The IQA method is based on MATLAB, which is hard to be transferred into Python code used by MFQEv2.

Therefore, we omit the compression, IQA and detection processes, but instead provide you with the pre-generated 18 PQF labels as well as two simpler options as mentioned above.

</details>

<details>

<summary><b>Assign approximate QP label (If needed)</b></summary>

There may exist frames with different QPs in one video. You can prepare a `npy` file that **records the QP of each frame in one video**, and store it in folder `data/PQF_label` as  `ApprQP_VideoName.npy`.

Notice that we have only 5 models with QP22, 27, 32, 37, 42, so we should record the nearest QP for each frame. For example, if the QPs for 4 frames are: `21,28,25,33`, then we should record: `22,27,27,32`. That's why we call it "approximate QP label".

</details>

<details>

<summary><b>Modify unqualified label</b></summary>

In our MFQE approach, each non-PQF should be enhanced with the help of its neighboring two PQFs (previous one + subsequent one).

Let 0 denotes non-PQF and 1 denotes PQF. In some cases, the PQF label might be something like:

- 0 0 1 ... (The first 2 non-PQFs have no previous PQFs)
- ... 1 0 (The last non-PQF has no subsequent PQF)

Our solution: we simply let themselves to be the pre-PQF and sub-PQF, to manage the enhancement. The test code will automatically detect and fix this problem.

Similarly, the first PQF has no previous PQF, and the last PQF has no subsequent PQF. The first PQF serves as the previous PQF for itself, and the last PQF serves as the subsequent PQF for itself.

</details>

<details>

<summary><b>Two MF-CNN models are similar</b></summary>

There are two different models in `net_MFCNN.py`. `network2` is for QP = 22, 27, 32 and `network1` for QP = 37, 42. Correspondingly, there are two types of pre-trained model. The performances of these two networks are close. Feel free to use them.

</details>

<details>

<summary><b>OOM when testing class A videos</b></summary>

Even with a 2080Ti GPU, we cannot process `2560x1600` frames (i.e., test sequences *Traffic* and *PeopleOnStreet*) directly. We simply cut them into 4 patches for enhancement, combine the enhanced patches, and then calculate PSNR and SSIM. For simplicity, the patching and combination processes are omitted in the test code.

</details> 

<details>

<summary><b>Deal with scene switch</b></summary>

There may exist multiple scenes in one video. Frames in different scenes should not be fused. In this case, we can use SSIM to detect scene switch, and then cut the video into a few clips. Luckily, it seems that no scene switch exists in the 18 test sequences.

</details>

<details>

<summary><b>Enhance black frame</b></summary>

Enhancing black frames or other "plane" frames (all pixel values are the same) may lead to inf PSNR. Our solution:

1. If the middle frame is plane, skip it (do not enhance it).
2. If the pre- or sub-PQF is plane, simply let the middle frame itself to be its pre-PQF and sub-PQF for enhancement.

</details>

## 5. License & Citation

You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made.

```tex
@article{MFQEv2,
  doi = {10.1109/tpami.2019.2944806},
  url = {https://doi.org/10.1109%2Ftpami.2019.2944806},
  year = 2021,
  publisher = {Institute of Electrical and Electronics Engineers ({IEEE})},
  pages = {949--963},
  volume = {43},
  number = {3},
  author = {Zhenyu Guan and Qunliang Xing and Mai Xu and Ren Yang and Tie Liu and Zulin Wang},
  title = {{MFQE} 2.0: A New Approach for Multi-frame Quality Enhancement on Compressed Video},
  journal = {{IEEE} Transactions on Pattern Analysis and Machine Intelligence}
}
```

## 6. See more

- [[PyTorch implementation of STDF (AAAI 2020)]](https://github.com/RyanXingQL/STDF-PyTorch)
  - A **simple** yet **effective** video quality enhancement network.
  - Adopt **feature alignment** by multi-frame **deformable convolutions**, instead of motion estimation and motion compensation.

- [[RBQE (ECCV 2020)]](https://github.com/RyanXingQL/RBQE)
  - A **single blind** enhancement model for HEVC/JPEG-compressed images with a **wide range** of Quantization Parameters (QPs) or Quality Factors (QFs).
  - A **multi-output dynamic** network with **early-exit** mechanism for easy input.
  - A **Tchebichef-moments** based **NR-IQA** approach for early-exit decision. This IQA approach is highly interpretable and sensitive to blocking energy detection.
