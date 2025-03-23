# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForCausalLM

from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek_vl.utils.io import load_pil_images

# specify the path to the model
#model_path = "deepseek-ai/deepseek-vl-7b-chat"
model_path = "deepseek-ai/deepseek-vl-1.3b-chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# steering_prompt = """You are driving a car. Your job is to keep on the second lane from right to left. Look at the road and decide what steering should be applied, from -1 (-70 degrees) to 1 (+70 degrees), in increments or decrements of 0.1, where negative is left, positive is right, and 0 is straight ahead. Reply only with the angle, inside <A> tags."""
# Assistant: 1
#steering_prompt = "You are driving a car. Describe how many lanes the road has."
#Assistant: The road has two lanes.
# steering_prompt = "You are driving a car. Look at the image of the road ahead and decide if you should steer left, right or straight."
#Assistant: The road ahead appears to be a straight stretch with no traffic or obstacles. There is a large Coca-Cola billboard on the right side of the road, but it is not obstructing the view of the road. The road is clear, and there are no visible signs of traffic or other vehicles. Therefore, based on the visual information provided, it is safe to assume that you should steer straight ahead.
steering_prompt = "You are driving a car. Look at the image of the road ahead and decide if you should steer left, right or straight. Reply with only one word."
#Assistant: Straight
steering_prompt = "You are driving a car. Look at the image of the road ahead and decide if you should steer left, right or straight. Reply with only one word."
prompt = "<image_placeholder>" + steering_prompt
# single image conversation example
conversation = [
    {
        "role": "User",
        "content": prompt,
        "images": ["200x200.jpg"],
    },
    {"role": "Assistant", "content": ""},
]

#conversation = [
#    {
#        "role": "User",
#        "content": "<image_placeholder>Describe each stage of this image.",
#        "images": ["./images/training_pipelines.jpg"],
#    },
#    {"role": "Assistant", "content": ""},
#]


# multiple images (or in-context learning) conversation example
# conversation = [
#     {
#         "role": "User",
#         "content": "<image_placeholder>A dog wearing nothing in the foreground, "
#                    "<image_placeholder>a dog wearing a santa hat, "
#                    "<image_placeholder>a dog wearing a wizard outfit, and "
#                    "<image_placeholder>what's the dog wearing?",
#         "images": [
#             "images/dog_a.png",
#             "images/dog_b.png",
#             "images/dog_c.png",
#             "images/dog_d.png",
#         ],
#     },
#     {"role": "Assistant", "content": ""}
# ]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
).to(vl_gpt.device)

# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# run the model to get the response
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True,
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)
