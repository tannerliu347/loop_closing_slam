bin_file=./build/kitti_test
seq=05
dataset_dir=/home/tannerliu/datasets/KITTI/dataset/sequences/$seq
param_file=temp/cvo_intensity_params_img_gpu0.yaml
method=meh
transform_file=temp/$seq.txt


# link library
export LD_LIBRARY_PATH=/home/tannerliu/unified_cvo_gpu/install/lib/UnifiedCvo-0.1/
export CUDA_VISIBLE_DEVICES=0

$bin_file $dataset_dir $param_file $seq $method $transform_file