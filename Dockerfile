FROM nvcr.io/nvidia/pytorch:21.12-py3
RUN apt-get update && apt-get install -y openssh-server vim 

WORKDIR /workspace/app
RUN pip install openpyxl 
RUN pip install torchviz 
RUN pip install progressbar2
RUN pip install monai[all] 
RUN pip install SimpleITK
RUN pip install pytorch_lightning
RUN pip install catalyst==20.07
RUN git clone https://github.com/Project-MONAI/GenerativeModels.git
WORKDIR GenerativeModels
RUN python setup.py install
