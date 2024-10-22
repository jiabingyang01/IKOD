seed=${1:-100}
task=${2:-"chair"}    # pope, chair, mme
dataset_name=${3:-"coco"}  # coco, gqa, aokvqa
type=${4:-"random"}  # random, popular, adversaril
model_path=${5:-"./checkpoints/llava-v1.5-7b"}
alpha=${6:-2}
beta=${7:-0.1}
ratio=${8:-0.6}
do_sample=${9:-"False"}  # True or False

if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=/data/chenhang_cui/dataset_for_attack/val2014
else
  image_folder=./data/gqa/gqa_eval
fi

if [[ $do_sample == "True" ]]; then
  decode=sample
else
  decode=greedy
fi

python ./gen/llava_pope_chair.py \
--task ${task} \
--model-path ${model_path} \
--question-file ./data/chair/chair_500.jsonl \
--image-folder ${image_folder} \
--answers-file ./output/llava/${task}/our/${decode}/jsonl/1_${ratio}_${alpha}_${beta}_att_min_500_seed${seed}.jsonl \
--alpha $alpha \
--beta $beta \
--ratio $ratio \
--do_sample $do_sample \
--seed ${seed}

