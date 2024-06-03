data_dir=/raid/Vikash/Data/chest-x-ray-dataset-with-lung-segmentation-1.0.0/files
app_dir=/raid/Vikash/Tools/Generative/app
#data_dir=/raid/Vikash/Tools/CAII/Segmentation/app_2/ResizedImages

IMAGE=vikash/generative:0.1.0
docker build -t $IMAGE .
NV_GPU=0,1,2,3 nvidia-docker run -it --rm --shm-size=4g --ulimit memlock=-1 --ulimit stack=67108864 -v $data_dir:/workspace/data -v $app_dir:/workspace/app $IMAGE /bin/bash
