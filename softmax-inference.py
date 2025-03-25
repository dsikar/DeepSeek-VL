# https://x.com/i/grok?conversation=1904450040942272605
import os
import time
import torch
import argparse
from transformers import AutoModelForCausalLM
from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek_vl.utils.io import load_pil_images

# Environment setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Define folders
INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_predictions"

# Create folders if they don’t exist
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Image-based steering prediction script")
parser.add_argument("--debug", action="store_true", help="Enable debug mode to time inference")
args = parser.parse_args()
print(f"Debug mode: {args.debug}")

# Load model and processor once
model_path = "deepseek-ai/deepseek-vl-1.3b-chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

def get_steering_probabilities(image_path, steering_prompt, debug=False):
    prompt = "<image_placeholder>" + steering_prompt
    conversation = [
        {"role": "User", "content": prompt, "images": [image_path]},
        {"role": "Assistant", "content": ""}
    ]

    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    if debug:
        start_time = time.time()

    with torch.no_grad():
        outputs = vl_gpt.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            return_dict=True
        )
        logits = outputs.logits

    if debug:
        end_time = time.time()
        print(f"Inference time for {os.path.basename(image_path)}: {end_time - start_time:.3f} seconds")

    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)

    token_ids = {
        "Straight": 93169,
        "Left": 12312,
        "Right": 10122
    }
    steering_probs = {key: probs[token_ids[key]].item() for key in token_ids}

    # Tie-breaking: Mimic generate's preference for lower token ID
    max_prob = max(steering_probs.values())
    candidates = [key for key, prob in steering_probs.items() if prob == max_prob]
    predicted = min(candidates, key=lambda x: token_ids[x])  # Lowest token ID wins
    print(f"Predicted (tie-break by token ID): {predicted}")

    # Get top 20 predictions with probabilities, logits, and token IDs
    top_k = 20
    top_probs, top_ids = torch.topk(probs, top_k)
    top_logits = torch.topk(last_logits, top_k)[0]
    print(f"Top {top_k} predictions (word, probability, logit, token ID):")
    for i, (prob, token_id, logit) in enumerate(zip(top_probs, top_ids, top_logits)):
        token = tokenizer.decode([token_id.item()], skip_special_tokens=True)
        print(f"  {i+1}. {token}: prob={prob.item():.10f}, logit={logit.item():.10f}, token_id={token_id.item()}")
    prob_mass = top_probs.sum().item()
    print(f"Probability mass for top {top_k}: {prob_mass:.6f} ({prob_mass * 100:.3f}%)")
    
    return steering_probs

# Main loop
print("Starting image monitoring loop...")
while True:
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.png'))]
    if image_files:
        image_file = image_files[0]
        image_path = os.path.join(INPUT_FOLDER, image_file)
        try:
            with open("prompts/prompt.txt", "r") as f:
                steering_prompt = f.read()
            probs = get_steering_probabilities(image_path, steering_prompt, debug=args.debug)
            print(f"Processed {image_file}:")
            for key, value in probs.items():
                print(f"  {key}: {value:.10f}")
            output_file = os.path.splitext(image_file)[0] + ".txt"
            output_path = os.path.join(OUTPUT_FOLDER, output_file)
            with open(output_path, "w") as f:
                for key, value in probs.items():
                    f.write(f"{key}: {value:.10f}\n")
            os.remove(image_path)
            print(f"Removed {image_file} from input folder")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    else:
        time.sleep(0.1)


# Example output
# Predicted (tie-break by token ID): Left
# Top 20 predictions (word, probability, logit, token ID):
#   1. Straight: prob=0.3652343750, logit=22.1250000000, token_id=93169
#   2. Left: prob=0.3652343750, logit=22.1250000000, token_id=12312
#   3. Right: prob=0.1718750000, logit=21.3750000000, token_id=10122
#   4. Ste: prob=0.0141601562, logit=18.8750000000, token_id=7393
#   5. Middle: prob=0.0109863281, logit=18.6250000000, token_id=36504
#   6. No: prob=0.0097045898, logit=18.5000000000, token_id=3233
#   7. straight: prob=0.0086059570, logit=18.3750000000, token_id=40578
#   8. left: prob=0.0051879883, logit=17.8750000000, token_id=1354
#   9. Yes: prob=0.0035705566, logit=17.5000000000, token_id=5661
#   10. Direct: prob=0.0021667480, logit=17.0000000000, token_id=9451
#   11. Center: prob=0.0016860962, logit=16.7500000000, token_id=24863
#   12. None: prob=0.0014877319, logit=16.6250000000, token_id=11137
#   13. To: prob=0.0014877319, logit=16.6250000000, token_id=1898
#   14. right: prob=0.0010223389, logit=16.2500000000, token_id=1035
#   15. Stay: prob=0.0010223389, logit=16.2500000000, token_id=36632
#   16. L: prob=0.0009040833, logit=16.1250000000, token_id=43
#   17. Forward: prob=0.0007972717, logit=16.0000000000, token_id=34364
#   18. You: prob=0.0007057190, logit=15.8750000000, token_id=2054
#   19. Correct: prob=0.0006599426, logit=15.8125000000, token_id=40195
#   20. Drive: prob=0.0006599426, logit=15.8125000000, token_id=34407
# Probability mass for top 20: 0.968750 (96.875%)
# Processed 384x384.jpg:
#   Straight: 0.3652343750
#   Left: 0.3652343750
#   Right: 0.1718750000
# Removed 384x384.jpg from input folder
