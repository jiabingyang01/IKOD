"""Generic native-HF lmms-eval wrapper for IKOD comprehensive / no-regression eval.

Why this exists: lmms-eval's stock per-model wrappers don't all work for our IKOD path.
 - llava_hf: inserts a literal "<image>" string that the llava-1.5 processor (tf>=4.52) does
   NOT expand -> "Image features and image tokens do not match: tokens 0, features 576".
 - internvl3: loads the OpenGVLab remote-code model and generates via model.chat(), which our
   ikod_generate (HF-native InternVLForConditionalGeneration, monkeypatched model.generate) cannot
   intercept.

This wrapper loads the native-HF model via ikod.models.load_vlm (bf16 + eager) and builds inputs
with proc.apply_chat_template([{image},{text}]) — the same proven path ikod.models.build_inputs uses
for CHAIR/POPE (validated on all 6 backbones). generate_until then calls self._model.generate, which
run_lmms_ikod monkeypatches to ikod_generate / a baseline. Single- and multi-process (accelerate
data-parallel) both supported, mirroring lmms_eval.models.simple.llava_hf.LlavaHf.
"""
import os
from typing import List

import PIL
import torch
from accelerate import Accelerator, DistributedType
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.models.simple.llava_hf import LlavaHf
from loguru import logger as eval_logger

from ikod.models import load_vlm, REGISTRY

# checkpoint dir basename -> registry key (e.g. "InternVL3-8B-hf" -> "internvl3")
_DIR2KEY = {os.path.basename(v["path"].rstrip("/")): k for k, v in REGISTRY.items()}


class IKODGenericHf(LlavaHf):
    def __init__(self, pretrained, model_key=None, device="cuda", dtype="bfloat16",
                 batch_size=1, device_map="", attn_implementation="eager", **kwargs):
        lmms.__init__(self)  # skip LlavaHf.__init__ (it loads a Llava model); set up the base only
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        if model_key is None:
            base = os.path.basename(str(pretrained).rstrip("/"))
            model_key = _DIR2KEY.get(base)
            assert model_key is not None, f"cannot map checkpoint '{base}' to a registry key; pass model_key="
        self.model_key = model_key
        self.pretrained = pretrained

        accelerator = Accelerator()
        if accelerator.num_processes > 1 and device_map == "":
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map

        model, proc, img_tok = load_vlm(model_key, device=str(self._device),
                                        dtype=torch.bfloat16, eager=(attn_implementation == "eager"))
        self._model = model
        self._image_processor = proc
        self._tokenizer = getattr(proc, "tokenizer", proc)
        try:
            self._tokenizer.padding_side = "left"
        except Exception:
            pass
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.chat_template = None
        self.use_cache = True
        self.max_frames_num = 32
        self._max_length = 4096

        if accelerator.num_processes > 1 and device_map == "":
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], \
                "Only DDP/FSDP/DEEPSPEED supported."
            # evaluation_mode=True -> does NOT wrap in DDP, so self._model.generate stays patchable
            self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            self.accelerator = accelerator
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
            if accelerator.is_local_main_process:
                eval_logger.info(f"IKODGenericHf[{model_key}]: {accelerator.num_processes} devices (data parallel)")
        else:
            self.accelerator = accelerator
            self._rank = 0
            self._world_size = 1
            eval_logger.info(f"IKODGenericHf[{model_key}]: single device {self._device}")

    def _build_inputs(self, context, visuals, task_type):
        """Build model inputs via apply_chat_template with proper image content (not literal <image>)."""
        if task_type == "image" and len(visuals) > 0:
            content = [{"type": "image"} for _ in visuals] + [{"type": "text", "text": context}]
            conv = [{"role": "user", "content": content}]
            prompt = self._image_processor.apply_chat_template(conv, add_generation_prompt=True)
            inputs = self._image_processor(images=list(visuals), text=prompt, return_tensors="pt")
        else:  # text-only (e.g. some ScienceQA items)
            conv = [{"role": "user", "content": [{"type": "text", "text": context}]}]
            prompt = self._image_processor.apply_chat_template(conv, add_generation_prompt=True)
            inputs = self._image_processor(text=prompt, return_tensors="pt")
        inputs = inputs.to(self._device)
        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._model.dtype)
        return inputs

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            if len(visuals) == 0:
                task_type = "text"
            elif isinstance(visuals[0], PIL.Image.Image):
                task_type = "image"
            else:
                task_type = "video"  # not expected for the no-regression suite
            gen_kwargs = all_gen_kwargs[0]
            until = [self.tok_decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
            assert self.batch_size_per_gpu == 1, "batch_size must be 1"
            context = contexts[0]

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            do_sample = gen_kwargs.get("temperature", 0) and gen_kwargs["temperature"] > 0

            try:
                inputs = self._build_inputs(context, visuals, task_type)
                cont = self._model.generate(
                    **inputs,
                    do_sample=bool(do_sample),
                    temperature=gen_kwargs["temperature"] if do_sample else None,
                    top_p=gen_kwargs.get("top_p", None),
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    pad_token_id=self.eot_token_id,
                    eos_token_id=self.eot_token_id,
                )
                cont = cont[:, inputs["input_ids"].shape[-1]:]
                text_outputs = self._tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            except Exception as e:
                if os.environ.get("IKOD_DEBUG"):
                    import traceback; traceback.print_exc()
                eval_logger.error(f"IKODGenericHf generate error: {e}")
                text_outputs = ""

            res.append(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
        res = re_ords.get_original(res)
        pbar.close()
        return res
