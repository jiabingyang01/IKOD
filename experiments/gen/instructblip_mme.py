import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
from transformers import set_seed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pathlib import Path
from collections import defaultdict
import math
import types
# import kornia
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
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
    model_path = args.model_path
    model = InstructBlipForConditionalGeneration.from_pretrained(args.model_path).cuda()
    processor = InstructBlipProcessor.from_pretrained(args.model_path)
    model.language_model.generate = types.MethodType(generate, model.language_model)
    model.language_model.sample = types.MethodType(sample, model.language_model)
    model.language_model.greedy_search = types.MethodType(greedy_search, model.language_model)
    k_seq_dim = v_seq_dim = 2
    kv_cache = ElasticCache(
    start_size=1,
    recent_size=2048,
    k_seq_dim=k_seq_dim,
    v_seq_dim=v_seq_dim,
    ratio= args.ratio,
    layer_num=32 if "7b" in model_path else 40
    )
    do_sample = True if args.do_sample == "True" else False
    data_path = args.image_folder
    answers_folder = args.answers_folder
    os.makedirs(answers_folder, exist_ok=True)

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
                    if args.task == "pope":
                        prompt = qs +  " Please answer this question with one word."
                    else:
                    # For generative tasks and MME
                        prompt = qs

                    # prepare the image
                    # image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)

                    ## create a white image for contrastive decoding     
                    prompt_len = 40
                    image_start = 1
                    image_end = 33
                    token_position = {'image_start':image_start, "image_end": image_end, "prompt_len": prompt_len}
            

                    pixel_values = inputs["pixel_values"]
                    qformer_input_ids =  inputs["qformer_input_ids"]
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]
                    batch_size = pixel_values.shape[0]
                    image_embeds = model.vision_model(pixel_values, return_dict=True).last_hidden_state

                    image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

                    query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
                    query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
                    qformer_attention_mask = None
                    if qformer_attention_mask is None:
                        qformer_attention_mask = torch.ones_like(qformer_input_ids)
                    qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
                    query_outputs = model.qformer(
                        input_ids=qformer_input_ids,
                        attention_mask=qformer_attention_mask,
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_attention_mask,
                        return_dict=True,
                    )
                    query_output = query_outputs.last_hidden_state[:, : query_tokens.size(1), :]

                    language_model_inputs = model.language_projection(query_output)
                    language_attention_mask = torch.ones(
                        language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
                    )

                    if input_ids is None:
                        input_ids = (
                            torch.LongTensor([[model.config.text_config.bos_token_id]])
                            .repeat(batch_size, 1)
                            .to(image_embeds.device)
                        )
                    if attention_mask is None:
                        attention_mask = torch.ones_like(input_ids)
                    attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

                    # concatenate query embeddings with prompt embeddings
                    inputs_embeds = model.get_input_embeddings()(input_ids)
                    inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

                    outputs = model.language_model.generate(
                        inputs_embeds=inputs_embeds,
                        kv_cache_criteria=kv_cache,
                        token_position = token_position,
                        attention_mask=attention_mask,
                        alpha = args.alpha,
                        beta = args.beta,             
                        do_sample=do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_new_tokens=512,
                        use_cache=True)

                    outputs[outputs == 0] = 2
                    outputs = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                    results.append({"image": image_path, "question": qa[0], "gt_ans": qa[1], "pred_ans":outputs})
            with open(os.path.join(answers_folder, str(path)+".json"), "w", encoding='utf-8') as f:
                json.dump(results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="pope")    
    parser.add_argument("--model-path", type=str, default="./checkpoints/instructblip-vicuna-7b")
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
    parser.add_argument("--ratio", type=float, default=0.5)    
    parser.add_argument("--do_sample", type=str, default="True")    
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
