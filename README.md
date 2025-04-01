# Infrared bubble recognition in the Milky Way and beyond using deep learning

Paper URL : https://doi.org/10.1093/pasj/psaf008

日本語話者の方は、japanese Branchを参照ください。

<p style="display: inline">
  <!-- バックエンドの言語一覧 -->
  <img src="https://img.shields.io/badge/-Python-F2C63C.svg?logo=python&style=for-the-badge">
  <!-- インフラ一覧 -->
  <img src="https://img.shields.io/badge/-Docker-1488C6.svg?logo=docker&style=for-the-badge">
</p>

## Abstract

We propose a deep-learning model that can detect Spitzer bubbles accurately using two-wavelength near-infrared data acquired by the Spitzer Space Telescope and JWST. The model is based on the single-shot multibox detector as an object detection model, trained and validated using Spitzer bubbles identified by the Milky Way Project (MWP bubbles). We found that using only MWP bubbles with clear structures, along with normalization and data augmentation, significantly improved performance. To reduce the dataset bias, we also use data without bubbles in the dataset selected by combining two techniques: negative sampling and clustering. The model was optimized by hyperparameter tuning using Bayesian optimization. Applying this model to a test region of the Galactic plane resulted in a 98% detection rate for MWP bubbles with 8 µm emission clearly encompassing 24 µm emission. Additionally, we applied the model to a broader area of 1◦ ≤ |l| ≤ 65◦, |b| ≤ 1◦ including both training and validation regions, and the model detected 3006 bubbles, of which 1413 were newly detected. We also attempted to detect bubbles in the high-mass star-forming region Cygnus X, as well as in external galaxies, the Large Magellanic Cloud (LMC) and NGC 628. The model successfully detected Spitzer bubbles in these external galaxies, though it also detected Mira-type variable stars and other compact sources that can be difficult to distinguish from Spitzer bubbles. The detection process takes only a few hours, demonstrating the efficiency in detecting bubble structures. Furthermore, the method used for detecting Spitzer bubbles was applied to detect shell-like structures observable only in the 8 µm emission band, leading to the detection of 469 shell-like structures in the LMC and 143 in NGC 628.



## <span style="color: green; ">Environment</span>
Follow the steps below to build the Docker environment for training the models used in this study.

1. **Git clone this repository**

2. **Run the following command in terminal to build the Docker image.**

    ```bash
    ./build.sh
    ```

    As a result of executing this command, a Docker image with the tag `cuda-python` is generated.

3. **Next, create a Docker container from the image you have just built.**
    Run `docker-run.sh`, but you need to rewrite the path of the folder to be referenced to suit your own environment. The docker environment is then built by running the following command.

    ```bash
    ./docker-run.sh
    ```

    Note: The above example assumes that Docker has been installed. Please refer to the various articles on how to install docker.


## <span style="color: green; ">Datasets</span>

Non-Bubble data and validation data need to be created in advance of the training.

#### Make Non-Bubble data
About Non-Bubble data, see subsection 3.2 in the paper.

To create Non-Bubble data from Fits data, it is first necessary to generate fits where the area the MWP-Bubble exists is filled with NaN. Run the file `mak_circle_nan_fits.py` in the `make_fits` folder as following steps.

1. **Create fits where the area the MWP-Bubble exists is filled with NaN**

    Run `mak_circle_nan_fits.py`. Please replace `PATH_TO_SPITZER_DATA` with the path of your spitzer data, and `PATH_TO_NANFITS_FILE` with the folder where the new fits will be created. If you do not have the Spitzer data, please contact us at the e-mail address in the paper.

    ```bash
    cd make_fits
    export PATH_TO_SPITZER_DATA = /home/filament/jupyter/fits_data/spitzer_data
    export PATH_TO_NANFITS_FILE = /home/filament/jupyter/fits_data/ring_to_circle_nan_fits
    python mak_circle_nan_fits.py $PATH_TO_SPITZER_DATA $PATH_TO_NANFITS_FILE
    ```


2. **Create Non-Bubble data**

    Run `make_NonRing.py`. Please replace `PATH_TO_NANFITS_FILE` with the path of the fits filled with NaN in the area of the MWP-Bubble, and `PATH_TO_OUTPUT_NONBUBBLE_DATA` with the folder where the created Non-Bubble data is saved, respectively.

    ```bash
    cd photoutils/Non_Ring
    export PATH_TO_OUTPUT_NONBUBBLE_DATA = /home/filament/jupyter/workspace/NonRing_png
    python make_NonRing.py $PATH_TO_NANFITS_FILE $PATH_TO_OUTPUT_NONBUBBLE_DATA
    ```

    <details><summary> <span style="color: blue; ">Non-Bubble Clustering</span></summary>

    Please refer to subsection 4.3 of the paper for more details on clusterstering.
    1. **Copy the created Non-Bubble data**

        Make a copy of the Non-Bubble data created above, to ensure that the original data is not altered by clustering. If your OS is Linux, you can do this with the cp command.

        ```bash
        export PATH_TO_NONBUBBLE_DATA_COPY = /home/filament/jupyter/workspace/NonRing_png_copy
        cp -r $PATH_TO_OUTPUT_NONRING_DATA $PATH_TO_NONBUBBLE_DATA_COPY
        ```

    2. **Non-Bubble clustering**

        Run `clustering.py`. Please specify in `PATH_TO_NONRING_DATA_COPY` the path of the Non-Bubble data that you have just copied.

        ```bash
        python clustering.py class_num model_version $PATH_TO_NONRING_DATA_COPY
        ```
    </details>

#### Make validation data

Please refer to sub-subsection 3.2.2 of the paper for more details on validation data. To create validation data, use `make_val_data.py` in the `photoutils/Val_data` folder.

1. **Create Validation data**

    Run `make_val_data.py`. Please replace `PATH_TO_SPITZER_DATA` with the path where the spitzer data is located and `PATH_TO_OUTPUT_VALIDATION_DATA` with the folder where the created Validation data will be saved.
    
    ```bash
    export PATH_TO_OUTPUT_VALIDATION_DATA = /home/filament/jupyter/workspace/cut_val_png
    python make_val_data.py $PATH_TO_SPITZER_DATA $PATH_TO_OUTPUT_VALIDATION_DATA
    ```


## <span style="color: green; ">Training model</span>

1. **Start learning**

    Run `train_main.py`. Please replace as appropriate for your environment regarding `spitzer_path`, `savedir_path`, `NonRing_data_path` and `validation_data_path`.

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

    Run `train_main.py` in the `photoutils_clustering` folder. The command options are almost the same, but there are two different options.

    1. **Run train_main.py in the `photoutils_clustering`**:
        
        Please set `class_num` to the number of classes you have clustered Non-Bubble data. Please set `NonRing_remove_class_list` to the classes with Spitzer bubble characteristics and `NonRing_aug_num` to 0 for the classes in the NonRing_remove_class_list and 1 for the others. *It does not necessarily have to be 0 or 1, increase the number as much as you want to increase it. 

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

        **Note**: Before running the command, the clustered Non-Ring must be formed.

    </details>

### Acknowledgement
I would like to thank all the FUGIN-AI members who were involved in this study, and I would also like to thank Shota Ueda and Shuyo Nakatani (Cybozu Labs Inc.) for their generous support and careful assistance on this code.

This work was supported by the 'Young interdisciplinary collaboration project' in the National Institutes of Natural Sciences (NINS), Japan. This work was also supported by JST SPRING, Grant Number JPMJSP2139.
