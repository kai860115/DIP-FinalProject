method=$1
name=$2
LR_output_dir=$3
repair_output_path=$4

python lrxx.py --method ${method} --depth_image data/${name}/disp.png --mask data/${name}/mask.png --init_image data/${name}/tnnr.png --output_path ${LR_output_dir} --name ${name}

python repair.py --input $3/$1_result/$2/${method,,}.png --output ${repair_output_path}