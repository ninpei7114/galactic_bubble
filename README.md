# 深層学習を用いた Spitzer bubble の検出

Paper URL : ~~


<p style="display: inline">
  <!-- バックエンドの言語一覧 -->
  <img src="https://img.shields.io/badge/-Python-F2C63C.svg?logo=python&style=for-the-badge">
  <!-- インフラ一覧 -->
  <img src="https://img.shields.io/badge/-Docker-1488C6.svg?logo=docker&style=for-the-badge">
</p>

## Abstract

We propose a deep learning model that can detect Spitzer bubbles comprehensively using two-wavelength near-infrared data acquired by the Spitzer Space Telescope. The model is based on the Single Shot MultiBox Detector as the object detection model and uses Spitzer bubbles detected by the Milky Way Project (MWP-Bubble) as the training and validation data. We found that only MWP bubbles with clear bubble structures should be used for the training data to improve the performance, and these data are subjected to normalization and data augmentation. We also use the data without bubbles to reduce the bias in the dataset by combining two methods: negative sampling and clustering. The model was optimized by hyperparameter search using Bayesian optimization. Applying this model to a test region of the galactic plane resulted in a 98 % detection rate for MWP-bubbles with 8 µm emission clearly encompassing 24 µm emission. In addition, we also tried to detect bubbles in the wide area of 1° ≦ |l| ≦ 65°, |b| ≦ 1°, including the training and validation regions, and the model detected 3006 bubbles, of which 1413 were newly detected. We also attempted to detect bubbles in the high-mass star-forming region Cygnus $X$ and in the external galaxies Large Magellanic Cloud (LMC) and NGC 628.  As a result, the model proved to be effective for detecting Spitzer bubbles in external galaxies, while it also detected Mira-type variable stars and other objects that are difficult to distinguish from Spitzer bubbles for compact sources. The detection process takes only a few hours, demonstrating the efficiency of the model in detecting bubble structures. The method used for the Spitzer bubble detection is also applied to detect shell-like structures observed only in the 8 µm emission band, and we detected 469 shell-like structures in the LMC and 143 in NGC 628.


## <span style="color: green; ">Environment</span>
本研究で使用したモデルを学習するためのDocker環境を、以下のステップで構築します。

1. **Git clone this repository**

2. **terminalで以下のコマンドを実行し、Docker imageをbuildします。**

    ```bash
    ./build.sh
    ```

    このコマンドの実行結果として、`cuda-python`というタグが付いたDocker imageが生成されます。

3. **次に、先ほどbuildしたimageからDocker　containerを作成します。**

    `docker-run.sh`というファイルを実行しますが、参照するフォルダのpathを自身の環境に合わせて書き換える必要があります。
    その後、以下のコマンドを実行することで、docker環境が構築されます。

    ```bash
    ./docker-run.sh
    ```

（注意）
以上の実行例は、Dockerがinstallされていることが前提です。
install方法は様々な記事で紹介されているため、そちらをご参照ください。

## <span style="color: green; ">Datasets</span>
学習にあたり、事前にNon-Bubble dataとvalidation dataを作成する必要があります。

#### Non-Bubble dataの作成
Non-Bubble dataについては, 論文の subsection 3.2 をご参照ください.

FitsデータからNon-Bubble data作成のためには、まずMWP-Bubbleが存在する領域をNaNで埋められたfitsを生成する必要があります。
`make_fits` フォルダにある`mak_circle_nan_fits.py`を、以下の手順で実行してください。

1. **MWP-Bubbleの領域をNaNにしたfitsの作成**

    `mak_circle_nan_fits.py`を実行してください。その際、`PATH_TO_SPITZER_DATA`はspitzerのdataがあるpathに、`PATH_TO_NANFITS_FILE`は新しいfitsが作成がされるフォルダに各自変更してください。
    ＊Spitzer dataをお持ちでない方は、論文に記載のメールにご連絡ください。


    ```bash
    cd make_fits
    export PATH_TO_SPITZER_DATA = /home/filament/jupyter/fits_data/spitzer_data
    export PATH_TO_NANFITS_FILE = /home/filament/jupyter/fits_data/ring_to_circle_nan_fits
    python mak_circle_nan_fits.py $PATH_TO_SPITZER_DATA $PATH_TO_NANFITS_FILE
    ```

2. **Non-Bubble dataの作成**

    `make_NonRing.py`を実行してください。その際、`PATH_TO_NANFITS_FILE` は先ほど生成したMWP-Bubbleの領域をNaNで埋められたfitのpathに、`PATH_TO_OUTPUT_NONBUBBLE_DATA`には作成されたNon-Bubble dataを保存するフォルダに各自変更してください。

    ```bash
    cd photoutils/Non_Ring
    export PATH_TO_OUTPUT_NONBUBBLE_DATA = /home/filament/jupyter/workspace/NonRing_png
    python make_NonRing.py $PATH_TO_NANFITS_FILE $PATH_TO_OUTPUT_NONBUBBLE_DATA
    ```

    <details><summary> <span style="color: blue; ">Non-Bubble dataのclustering</span></summary>
    clusteringの詳細については、論文の subsection 4.3 をご参照ください.

    1. **作成したNon-Bubble dataをコピーする**

        上記で作成したNon-Bubble dataのコピーを作成してください。clusteringによりオリジナルデータが変更されないようにするためです。お使いのPCがLinuxであれば、cpコマンドで出来ます。

        ```bash
        export PATH_TO_NONBUBBLE_DATA_COPY = /home/filament/jupyter/workspace/NonRing_png_copy
        cp -r $PATH_TO_OUTPUT_NONRING_DATA $PATH_TO_NONBUBBLE_DATA_COPY
        ```

    2. **Non-Bubble data のclustering**

        `clustering.py`を実行してください。`PATH_TO_NONRING_DATA_COPY`には先ほどコピーしたNon-Bubble dataのpathを指定ください。

        ```bash
        python clustering.py class_num model_version $PATH_TO_NONRING_DATA_COPY
        ```
    </details>

#### Validation dataの作成
validation dataについては, 論文の sub-subsection 3.2.2 をご参照ください.
validation dataの作成には、`photoutils/Val_data`フォルダの`make_val_data.py`を使用します。

1. **Validation dataの作成**

    `make_val_data.py`を実行してください。その際、`PATH_TO_SPITZER_DATA`はspitzerのdataがあるpathに、`PATH_TO_OUTPUT_VALIDATION_DATA` には作成されたValidation dataを保存するフォルダに各自変更してください。

    ```bash
    export PATH_TO_OUTPUT_VALIDATION_DATA = /home/filament/jupyter/workspace/cut_val_png
    python make_val_data.py $PATH_TO_SPITZER_DATA $PATH_TO_OUTPUT_VALIDATION_DATA
    ```


## <span style="color: green; ">Training model</span>

1. **学習開始**

    `train_main.py`を実行してください。spitzer_path, savedir_path, NonRing_data_path and validation_data_pathに関しては、ご自身の環境に適宜変更してください。

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

    <details><summary> <span style="color: blue; ">photoutils_clusteringを実行する場合</span></summary>

    `photoutils_clustering`フォルダにある`train_main.py`を実行してください。コマンドのオプションはほとんど変わらないですが、２点異なるパラメータがあります。

    1. **Run train_main.py in the `photoutils_clustering`**:

        `class_num`をclusteringしたクラス数に変換してください。 `NonRing_remove_class_list`にはSpitzer bubbleの特徴があるクラスを、`NonRing_aug_num` には`NonRing_remove_class_list`のクラスを0、それ以外を1に設定してください。＊必ずしも0,1にする必要はなく、増やしたい分だけ数を大きくしてください。

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

        **Note**: このコマンドの実行には、事前にNon-Bubble dataのクラスタリングをしておく必要があります。

    </details>

### Acknowledgement
I would like to thank all the FUGIN-AI members who were involved in this study, and I would also like to thank Shota Ueda for his generous support. I am grateful to Nakatani Shuyo / Cybozu Labs Inc. for his careful assistance on this code.

This work was supported by the 'Young interdisciplinary collaboration project' in the National Institutes of Natural Sciences (NINS), Japan. This work was also supported by JST SPRING, Grant Number JPMJSP2139.
