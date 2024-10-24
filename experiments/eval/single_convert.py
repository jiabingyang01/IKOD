import json
import random
def truncate_caption(caption):
    words = caption.split()
    truncate_length = random.randint(50, 60)
    truncated_caption = ' '.join(words[:truncate_length])
    return truncated_caption

# iblip_hasetting_llava_instruct_8k_4e-7_ppo_merge.jsonl
name = "./output/llava/chair/regular/sample/jsonl/seed100.jsonl"
# /iris/u/huaxiu/ch/mgpt-4/mplug_owl_coco5000_des_idn.jsonl
with open(name, 'r') as jsonl_file:
    lines = jsonl_file.readlines()

overall_metrics = {
    "metric1": 0.75,
}
img_to_eval = {}

for i, line in enumerate(lines):
    # if i>500:
    #     break
    data = json.loads(line)
    # image_id = data["id"]
    image_id = data["image"]
    caption = data["text"]
    # image_id = data["image_id"]
    # caption = data["caption"]
    number = image_id.split('_')[-1].split('.')[0]

    # 去除数字中的零 truncate_caption .split('\n')[0] + data["answer"].split('\n')[1]
    number_without_zeros = number.lstrip('0')
    img_info = {
        "image_id": int(number_without_zeros),
        "caption": (caption)
    }
    img = {str(i): img_info}
    img_to_eval.update(img)
#img_to_eval = dict(img_to_eval)
# Constructing the final JSON data
final_json_data = {
    "overall": overall_metrics,
    "imgToEval": img_to_eval
}
#output_coco_train_des
# Writing the JSON data to the output file


with open(name.replace("jsonl", "json"), 'w') as output_file:
    json.dump(final_json_data, output_file, indent=4)