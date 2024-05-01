# 深層学習を用いた Spitzer bubble の検出

Paper URL : ~~

日本語話者の方は、japanese Branchを参照ください。

<p style="display: inline">
  <!-- バックエンドの言語一覧 -->
  <img src="https://img.shields.io/badge/-Python-F2C63C.svg?logo=python&style=for-the-badge">
  <!-- インフラ一覧 -->
  <img src="https://img.shields.io/badge/-Docker-1488C6.svg?logo=docker&style=for-the-badge">
</p>

## Abstract

我々は、近赤外線データから Spitzer Bubble を検出できる深層学習モデルを提案する。モデルは、Single Shot MultiBox Detector を使用し、学習データには Milky Way Project (MWP)の Spitzer Bubble を用いた。我々は、 開発したモデルを天の川銀河の観測領域や、Cygnus X・LMC に適用し、MWP の未検出 Bubble の検出を行っ た。その結果、開発したモデルは、MWP-Bubble の中でも 8μm が 24μm を内包した Bubble では98%の検出率となり、新たに Bubble と判断した天体においても Spitzer Bubble の特徴をよく捉えた天体であることが確認できた。また、検出時間も MWP と比較して非常に短い時間に短縮することができ、科学的発見のペースを加速する大きな機会を提供することができる。



## <span style="color: green; ">Environment</span>
To set up the Docker environment for this project, follow these steps:

1. **Git clone this repository**

2. **Build the Docker image by running the following command in the terminal**

    ```bash
    ./build.sh
    ```

    This will create a Docker image with the tag `cuda-python`.

3. **Run a Docker container based on the image you just built. Change pathes of `docker-run.sh`.**

    ```bash
    ./docker-run.sh
    ```

    This will start a container and run your project inside it.

Make sure you have Docker installed on your machine before following these steps.

## <span style="color: green; ">Datasets</span>

#### Make Non-Ring data
To create Non-Ring data, you can refer to the `mak_circle_nan_fits.py` script in the `make_fits` folder. Follow these steps:

1. **Create fits where the spitzer bubble position is Nan to make Non-Ring data**

    Run `mak_circle_nan_fits.py`. Replace `PATH_TO_SPITZER_DATA` with the path to your spitzer data, and `PATH_TO_NANFITS_FILE` with the desired path for the output fits:

    ```bash
    cd make_fits
    export PATH_TO_SPITZER_DATA = /home/filament/jupyter/fits_data/spitzer_data
    export PATH_TO_NANFITS_FILE = /home/filament/jupyter/fits_data/ring_to_circle_nan_fits
    python mak_circle_nan_fits.py $PATH_TO_SPITZER_DATA $PATH_TO_NANFITS_FILE
    ```
    This will execute the script and generate fits where the spitzer bubble position is Nan.

2. **Create NonRing data**

    Run `make_NonRing.py`. Replace `PATH_TO_NANFITS_FILE` with path of created fits where the location of the spitzer bubble is Nan, and `PATH_TO_OUTPUT_NONRING_DATA` with the desired path for the output Non-Ring data:

    ```bash
    cd photoutils/Non_Ring
    export PATH_TO_OUTPUT_NONRING_DATA = /home/filament/jupyter/workspace/NonRing_png
    python make_NonRing.py $PATH_TO_NANFITS_FILE $PATH_TO_OUTPUT_NONRING_DATA
    ```

    <details><summary> <span style="color: blue; ">Non-Ring Clustering</span></summary>

    1. **Copy the NonRing data**

        Start by making a copy of the Non-Ring data you created above. This is to ensure that the original data remains unchanged during the clustering process. You can do this using a command like:

        ```bash
        export PATH_TO_NONRING_DATA_COPY = /home/filament/jupyter/workspace/NonRing_png_copy
        cp -r $PATH_TO_OUTPUT_NONRING_DATA $PATH_TO_NONRING_DATA_COPY
        ```

    2. **NonRing clustering**

        Run the clustering.py script to perform clustering on the Non-Ring data:

        ```bash
        python clustering.py class_num model_version $PATH_TO_NONRING_DATA_COPY
        ```
    </details>

#### Make validation data

To create Validation data, you can refer to the `make_val_data.py` script in the `photoutils/Val_data` directory. Follow these steps:

1. **Create Validation data**

    Run `make_val_data.py`. Replace `PATH_TO_SPITZER_DATA` with the path to your spitzer data, and `PATH_TO_OUTPUT_VALIDATION_DATA` with the desired path for the output Validation data:
    ```bash
    export PATH_TO_OUTPUT_VALIDATION_DATA = /home/filament/jupyter/workspace/cut_val_png
    python make_val_data.py $PATH_TO_SPITZER_DATA $PATH_TO_OUTPUT_VALIDATION_DATA
    ```


## <span style="color: green; ">Training model</span>

1. **Start learning**

    Run `train_main.py`. For spitzer_path, savedir_path, NonRing_data_path and validation_data_path, change the path to suit your environment accordingly:

    ```bash
    cd photoutils
    python train_main.py \
    --spitzer_path path_of_spitzer_data \
    --savedir_path path_of_savedir \
    --NonRing_data_path path_of_NonRing_data \
    --validation_data_path path_of_validation_data \
    --i 0 \
    --Ring_mini_batch 8 \
    --NonRing_mini_batch 8 \
    --Val_mini_batch 128 \
    --training_ring_catalogue MWP \
    --val_ring_catalogue MWP \
    --fscore f2_score
    --wandb_project Project_name \
    --wandb_name Run_name \
    ```

    <details><summary> <span style="color: blue; ">Run photoutils_clustering</span></summary>

    if you run `photoutils_clustering` script, follow these steps:

    1. **Run train_main.py in the `photoutils_clustering`**:

        Replace `class_num` with the determined number of classes. `NonRing_remove_class_list` and `NonRing_aug_num` are also replaced with a predetermined value:

        ```bash
        cd photoutils_clustering
        python train_main.py \
        --spitzer_path path_of_spitzer_data \
        --savedir_path path_of_savedir \
        --NonRing_data_path path_of_NonRing_data \
        --validation_data_path path_of_validation_data \
        --i 0 \
        --Ring_mini_batch 8 \
        --NonRing_mini_batch 8 \
        --Val_mini_batch 128 \
        --training_ring_catalogue MWP \
        --val_ring_catalogue MWP \
        --fscore f2_score
        --wandb_project Project_name \
        --wandb_name Run_name \
        --NonRing_class_num 10 \
        --NonRing_remove_class_list 5 9 \
        --NonRing_aug_num 1 1 1 1 1 0 1 1 1 0
        ```

        **Note**: Before executing the command, the clustered Non-Ring must be formed.

    </details>

### Acknowledgement
I would like to thank the members of FUGIN-AI for their comments on this research. I am grateful to Nakatani Shuyo / Cybozu Labs Inc. for his careful assistance on this code.

This research is supported by the National Institutes of Natural Sciences (NINS), Japan, through the inter-disciplinary collaboration project ”Reconstruction and Elucidation of the 3-D Spatial Structure of the Milky Way Galaxy Using Large-scale Molecular Cloud Data and Machine Learning / Elucidation of the 3-D Spatial Structure of the Milky Way Galaxy Based on Machine Learning and Deep Learning”.
