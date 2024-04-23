FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04
# FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
# FROM nvcr.io/nvidia/cuda:11.7.0-cudnn8-devel-ubuntu20.04

ENV TZ=Asia/Tokyo
RUN  apt-key adv --keyserver keyserver.ubuntu.com --recv f23c5a6cf475977595c89f51ba6932366a755776 \
  && apt-get update && apt-get upgrade -y \
  && ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
  && apt-get install -y libgl1-mesa-dev libglib2.0-0 \
  && apt-get install -y python3-pip \
  && apt-get install -y python3.8 python3.8-distutils locales montage \
  && ln -s /usr/bin/python3.8 /usr/bin/python \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* 
RUN pip install --upgrade pip \ 
  && pip install torch torchvision torchaudio\
  && pip install ipython scikit-learn matplotlib pillow pandas  \
  && pip install astroquery astropy photutils aplpy \
  && pip install torchsummary torchviz \
  && pip install reproject scikit-image opencv-python faiss_gpu seaborn sympy \
  && pip install tqdm ipywidgets japanize_matplotlib \
  && pip install seaborn natsort npy_append_array \
  && pip install webdataset optuna wandb
