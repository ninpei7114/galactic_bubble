# 深層学習を用いた Spitzer bubble の検出

Paper URL : ~~

## Abstract

我々は、近赤外線データから Spitzer Bubble を検出できる深層学習モデルを提案する。モデルは、Single Shot MultiBox Detector を使用し、学習データには Milky Way Project (MWP)の Spitzer Bubble を用いた。我々は、 開発したモデルを天の川銀河の観測領域や、Cygnus X・LMC に適用し、MWP の未検出 Bubble の検出を行っ た。その結果、開発したモデルは、MWP-Bubble の中でも 8μm が 24μm を内包した Bubble では98%の検出率 となり、新たに Bubble と判断した天体においても Spitzer Bubble の特徴をよく捉えた天体であることが確認できた。また、検出時間も MWP と比較して非常に短い時間に短縮することができ、科学的発見のペースを加速する大きな機会を提供することができる。



## <span style="color: green; ">Environment</span>
To set up the Docker environment for this project, follow these steps:

1. Git pull galactic bubble.

2. Build the Docker image by running the following command in the terminal:

    ```bash
    ./build.sh
    ```

    This will create a Docker image with the tag `cuda-python`.

3. Open the `docker-run.sh` and change pathes. Run a Docker container based on the image you just built by running the following command. :

    ```bash
    ./docker-run.sh
    ```

    This will start a container and run your project inside it.

Make sure you have Docker installed on your machine before following these steps.

## <span style="color: green; ">Datasets</span>

#### Make Non-Ring data
To create Non-Ring data, you can refer to the `mak_circle_nan_fits.py` script in the `make_fits` folder. Follow these steps:

1. **Create fits where the spitzer bubble position is Nan to make Non-Ring data**

    Run the Python script using the following command. Replace `path_of_spitzer_data` with the path to your spitzer data, and `path_of_Nanfits_file` with the desired path for the output fits:

    ```bash
    python mak_circle_nan_fits.py path_of_spitzer_data path_of_Nanfits_file
    ```
    This will execute the script and generate fits where the spitzer bubble position is Nan.

2. **Change directory to the `photoutils` directory**

    ```bash
    cd ../photoutils
    ```

3. **Run make_NonRing.py**

    Run the Python script using the following command. Replace `path_of_Nanfits_file` with created fits where the location of the spitzer bubble is Nan, and `path_of_output_file` with the desired path for the output Non-Ring data:

    ```bash
    python make_NonRing.py path_of_Nanfits_file path_of_output_file
    ```

    <details><summary> <span style="color: blue; ">Non-Ring Clustering</span></summary>

    1. **Copy the Non-Ring Data**

        Start by making a copy of the Non-Ring data you created above. This is to ensure that the original data remains unchanged during the clustering process. You can do this using a command like:

        ```bash
        cp -r /path/to/original/Non_Ring /path/to/copy/Non_Ring
        ```

    2. **Run clustering.py**

        Run the clustering.py script to perform clustering on the Non-Ring data. The command might look something like this:

        ```python
        python clustering.py class_num model_version /path/to/copy/Non_Ring
        ```
    </details>

#### Make validation data

To create Validation data, you can refer to the `make_val_data.py` script in the `photoutils/Val_data` directory. Follow these steps:

1. **Run make_val_data.py**

    Replace `path_of_spitzer_data` with the path to your spitzer data, and `path_of_output_file` with the desired path for the output Validation data:
    ```bash
    python make_val_data.py path_of_spitzer_data path_of_output_file
    ```


## <span style="color: green; ">Training model</span>

1. **Run train_main.py**

    Run Python script using the following command. For spitzer_path, savedir_path, NonRing_data_path and validation_data_path, change the path to suit your environment accordingly:

    ```bash
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

        Run the `photoutils_clustering` script using the following command. Replace `class_num` with the determined number of classes. `NonRing_remove_class_list` and `NonRing_aug_num` are also replaced with a predetermined value:

        ```bash
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
