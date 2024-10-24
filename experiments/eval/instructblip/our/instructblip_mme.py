import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pathlib import Path
from collections import defaultdict
import math
import types
# import kornia
from transformers import set_seed
from vcd_utils.vcd_add_noise import add_diffusion_noise
# from vcd_utils.vcd_sample import evolve_vcd_sampling
# evolve_vcd_sampling()
from utils.cache_generate import generate, sample, greedy_search
from utils.kv_cache import ElasticCache

def eval_model(args):
    # Model
    # disable_torch_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loads InstructBLIP model
    # For large_sized model,
    model = InstructBlipForConditionalGeneration.from_pretrained(args.model_path).cuda()
    processor = InstructBlipProcessor.from_pretrained(args.model_path)
    model.language_model.generate = types.MethodType(generate, model.language_model)
    model.language_model.sample = types.MethodType(sample, model.language_model)
    model.language_model.greedy_search = types.MethodType(greedy_search, model.language_model)
    k_seq_dim = v_seq_dim = 2
    kv_cache = ElasticCache(
    start_size=1,
    recent_size=1024,
    k_seq_dim=k_seq_dim,
    v_seq_dim=v_seq_dim,
    ratio= 0.5,
    layer_num=32 if "7b" in model_name else 40
    )
    data_path = args.image_folder
    answers_folder = args.answers_folder
    for path in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, path)): 
            # read data from file
            data = defaultdict(list)
            if os.path.isdir(os.path.join(data_path, path, "questions_answers_YN")) and os.path.isdir(os.path.join(data_path, path, "images")):
                image_root = os.path.join(data_path, path, "images")
                ann_path = os.path.join(data_path, path, "questions_answers_YN")
                for file in os.listdir(ann_path):
                    with open(os.path.join(ann_path, file), 'r', encoding='utf-8') as f:
                        for line in f:
                            sentences = line.strip().split('\t')
                            
                            question = sentences[0]
                            answer = sentences[1] if len(sentences) > 1 else None
                            qa=(question, answer)
                            data[str(file).replace(".txt", "")].append(qa)
            else:
                image_root = os.path.join(data_path, path)
                ann_path = os.path.join(data_path, path)
                for file in os.listdir(ann_path): 
                    if file.endswith(".txt"):
                        with open(os.path.join(ann_path, file), 'r', encoding='utf-8') as f:
                            for line in f:
                                sentences = line.strip().split('\t')
                                
                                question = sentences[0]
                                answer = sentences[1] if len(sentences) > 1 else None
                                qa=(question, answer)
                                data[str(file).replace(".txt", "")].append(qa)

            results = []
            for item in tqdm(data):
                try:
                    image_path = os.path.join(image_root, item+".jpg")
                    raw_image = Image.open(image_path).convert("RGB")
                except FileNotFoundError as e:
                    image_path = os.path.join(image_root, item+".png")
                    raw_image = Image.open(image_path).convert("RGB")
                    # print("image not found: ", item)
                    # continue
                for qa in data[item]:
                    qs = qa[0]
                    if model.config.mm_use_im_start_end:
                        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                    else:
                        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

                    conv = conv_templates[args.conv_mode].copy()
                    # For POPE
                    if args.task == "pope":
                        conv.append_message(conv.roles[0],  qs + " Please answer this question with one word.")
                    else:
                    # For generative tasks and MME
                        conv.append_message(conv.roles[0],  qs) 
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    image_token_len = 576
                    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                    prompt_len = input_ids.shape[1] + image_token_len - 1
                    image_start = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1][0]
                    image_end = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1][0] + image_token_len
                    token_position = {'image_start':image_start, "image_end": image_end, "prompt_len": prompt_len}
                    image_tensor = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]
                    
                    if args.use_cd:
                        image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
                    else:
                        image_tensor_cd = None      

                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=image_tensor.unsqueeze(0).half().cuda(),
                            kv_cache_criteria=kv_cache,
                            token_position = token_position,
                            alpha = args.alpha,
                            beta = args.beta,                
                            do_sample=args.do_sample,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            max_new_tokens=1024,
                            use_cache=True)

                    input_token_len = input_ids.shape[1]
                    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                    if n_diff_input_output > 0:
                        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                    outputs = outputs.strip()
                    if outputs.endswith(stop_str):
                        outputs = outputs[:-len(stop_str)]
                    outputs = outputs.strip()
                    results.append({"image": image_path, "question": qa[0], "gt_ans": qa[1], "pred_ans":outputs})
            with open(os.path.join(answers_folder, str(path)+".json"), "w", encoding='utf-8') as f:
                json.dump(results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="pope")    
    parser.add_argument("--model-path", type=str, default="/data/private_models/dpo_models/instructblip-vicuna-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--answers-folder", type=str, default="answer_folder")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--do_sample", type=bool, default=True)    
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)