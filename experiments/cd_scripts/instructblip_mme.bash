seed=${1:-100}
task=${2:-"mme"}    # pope, chair, mme
dataset_name=${3:-"coco"}  # coco, gqa, aokvqa
type=${4:-"random"}  # random, popular, adversaril
model_path=${5:-"./checkpoints/instructblip-vicuna-7b"}
alpha=${6:-1}
beta=${7:-0.1}
ratio=${8:-0.9}
do_sample=${9:-"False"}  # True or False
image_folder=${10:-"/data/mme/MME_Benchmark_release_version"}

if [[ $do_sample == "True" ]]; then
  decode=sample
else
  decode=greedy
fi

python ./gen/instructblip_mme.py \
--task ${task} \
--model-path ${model_path} \
--image-folder ${image_folder} \
--answers-folder ./output/instructblip/${task}/our/${decode}/1_${ratio}_${alpha}_${beta}_att_min_seed${seed} \
--alpha $alpha \
--beta $beta \
--ratio $ratio \
--do_sample $do_sample \
--seed ${seed}

