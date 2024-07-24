# Detection of Spitzer Bubbles Using Deep Learning

Paper URL : ~~

日本語話者の方は、japanese Branchを参照ください。

<p style="display: inline">
  <!-- バックエンドの言語一覧 -->
  <img src="https://img.shields.io/badge/-Python-F2C63C.svg?logo=python&style=for-the-badge">
  <!-- インフラ一覧 -->
  <img src="https://img.shields.io/badge/-Docker-1488C6.svg?logo=docker&style=for-the-badge">
</p>

## Abstract

We propose a deep learning model that can detect Spitzer bubbles comprehensively using two-wavelength near-infrared data acquired by the Spitzer Space Telescope. The model is based on the Single Shot MultiBox Detector as the object detection model and uses Spitzer bubbles detected by the Milky Way Project (MWP-Bubble) as the training and validation data. We found that only MWP bubbles with clear bubble structures should be used for the training data to improve the performance, and these data are subjected to normalization and data augmentation. We also use the data without bubbles to reduce the bias in the dataset by combining two methods: negative sampling and clustering. The model was optimized by hyperparameter search using Bayesian optimization. Applying this model to a test region of the galactic plane resulted in a 98 % detection rate for MWP-bubbles with 8 µm emission clearly encompassing 24 µm emission. In addition, we also tried to detect bubbles in the wide area of 1° ≦ |l| ≦ 65°, |b| ≦ 1°, including the training and validation regions, and the model detected 3006 bubbles, of which 1413 were newly detected. We also attempted to detect bubbles in the high-mass star-forming region Cygnus $X$ and in the external galaxies Large Magellanic Cloud (LMC) and NGC 628.  As a result, the model proved to be effective for detecting Spitzer bubbles in external galaxies, while it also detected Mira-type variable stars and other objects that are difficult to distinguish from Spitzer bubbles for compact sources. The detection process takes only a few hours, demonstrating the efficiency of the model in detecting bubble structures. The method used for the Spitzer bubble detection is also applied to detect shell-like structures observed only in the 8 µm emission band, and we detected 469 shell-like structures in the LMC and 143 in NGC 628.


## <span style="color: green; ">Environment</span>
To establish the Docker environment for this project, follow these steps:

1. **Git clone this repository**

2. **Build the Docker image by running the following command in the terminal**

    ```bash
    ./build.sh
    ```

    This will create a Docker image with the tag `cuda-python`.

3. **Run a Docker container based on the image you just built. Change pathes in `docker-run.sh`.**

    ```bash
    ./docker-run.sh
    ```

    This will start a container and run your project inside it.

Make sure you have Docker installed on your machine before following these steps.

## <span style="color: green; ">Datasets</span>

#### Make Non-Bubble data
About Non-Bubble data, see subsection 3.2 in the paper.

To create Non-Bubble data, please run `mak_circle_nan_fits.py` script in the `make_fits` folder following these steps.

1. **Create fits where the Spitzer bubble position is Nan to make Non-Bubble data**

    Run `mak_circle_nan_fits.py`. Replace `PATH_TO_SPITZER_DATA` with the path to your spitzer data, and `PATH_TO_NANFITS_FILE` with the desired path for the output fits:

    ```bash
    cd make_fits
    export PATH_TO_SPITZER_DATA = /home/filament/jupyter/fits_data/spitzer_data
    export PATH_TO_NANFITS_FILE = /home/filament/jupyter/fits_data/ring_to_circle_nan_fits
    python mak_circle_nan_fits.py $PATH_TO_SPITZER_DATA $PATH_TO_NANFITS_FILE
    ```
    This will execute the script and generate fits where the spitzer bubble position is Nan.

2. **Create Non-Bubbl data**

    Run `make_NonRing.py`. Replace `PATH_TO_NANFITS_FILE` with path of created fits where the location of the spitzer bubble is Nan, and `PATH_TO_OUTPUT_NONBUBBLE_DATA` with the desired path for the output Non-Bubble data:

    ```bash
    cd photoutils/Non_Ring
    export PATH_TO_OUTPUT_NONBUBBLE_DATA = /home/filament/jupyter/workspace/NonRing_png
    python make_NonRing.py $PATH_TO_NANFITS_FILE $PATH_TO_OUTPUT_NONBUBBLE_DATA
    ```

    <details><summary> <span style="color: blue; ">Non-Bubble Clustering</span></summary>

    1. **Copy the Non-Bubble data**

        Start by making a copy of the Non-Bubble data you created above. This is to ensure that the original data remains unchanged during the clustering process. You can do this using a command like:

        ```bash
        export PATH_TO_NONBUBBLE_DATA_COPY = /home/filament/jupyter/workspace/NonRing_png_copy
        cp -r $PATH_TO_OUTPUT_NONRING_DATA $PATH_TO_NONBUBBLE_DATA_COPY
        ```

    2. **Non-Bubble clustering**

        Run the clustering.py script to perform clustering on the Non-Bubble data:

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
I would like to thank all the FUGIN-AI members who were involved in this study, and I would also like to thank Shota Ueda for his generous support. I am grateful to Nakatani Shuyo / Cybozu Labs Inc. for his careful assistance on this code.

This work was supported by the 'Young interdisciplinary collaboration project' in the National Institutes of Natural Sciences (NINS), Japan. This work was also supported by JST SPRING, Grant Number JPMJSP2139.
