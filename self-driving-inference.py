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

# Steering prompt
# steering_prompt = "You are driving a car. Look at the image of the road ahead and decide if you should steer left, right or straight. Reply with only one word."

def process_image(image_path, steering_prompt, debug=False):
    """Process a single image and return the steering prediction."""
    prompt = "<image_placeholder>" + steering_prompt
    conversation = [
        {"role": "User", "content": prompt, "images": [image_path]},
        {"role": "Assistant", "content": ""}
    ]

    # Load and prepare inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    # Generate embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # Run inference with timing if debug is enabled
    if debug:
        start_time = time.time()
    
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

    if debug:
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"Inference time for {os.path.basename(image_path)}: {inference_time:.3f} seconds")

    # Decode the output
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer.strip()  # Ensure it's a single word

# Main loop
print("Starting image monitoring loop...")
while True:
    # Get list of images in the input folder
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.png'))]
    
    if image_files:
        # Process the first image found
        image_file = image_files[0]
        image_path = os.path.join(INPUT_FOLDER, image_file)
        
        try:
            # Make prediction with debug option
            import os
            # open the prompt file
            with open("prompts/prompt.txt", "r") as f:
                steering_prompt = f.read()
            prediction = process_image(image_path, steering_prompt, debug=args.debug)
            print(f"Processed {image_file}: {prediction}")
            
            # Save prediction to output folder
            output_file = os.path.splitext(image_file)[0] + ".txt"
            output_path = os.path.join(OUTPUT_FOLDER, output_file)
            with open(output_path, "w") as f:
                f.write(prediction)
            
            # Remove the processed image
            os.remove(image_path)
            print(f"Removed {image_file} from input folder")
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    else:
        # No images found, sleep briefly
        time.sleep(0.1)
