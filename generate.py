import argparse
import torch
from PIL import Image
from peft import PeftModel

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text from an image with nanoVLM")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a local checkpoint (directory or safetensors/pth). If omitted, we pull from HF."
    )
    parser.add_argument(
        "--hf_model", type=str, default="lusxvr/nanoVLM-450M",
        help="HuggingFace repo ID to download from incase --checkpoint isnt set."
    )
    
    parser.add_argument(
        "--lora_adapter", type=str, default=None,
        help="Path to LoRA adapter checkpoint. If provided, will load LoRA weights on top of base model."
    )
    parser.add_argument(
        "--merge_lora", action="store_true",
        help="Merge LoRA weights into base model for faster inference (requires --lora_adapter)."
    )
    parser.add_argument("--image", type=str, default="assets/image.png",
                        help="Path to input image")
    parser.add_argument("--prompt", type=str, default="What is this?",
                        help="Text prompt to feed the model")
    parser.add_argument("--generations", type=int, default=5,
                        help="Num. of outputs to generate")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="Maximum number of tokens per output")
    return parser.parse_args()


def load_model_with_lora(source, lora_adapter_path=None, merge_lora=False, device="cuda"):
    print(f"Loading base model from: {source}")
    base_model = VisionLanguageModel.from_pretrained(source)
    
    if lora_adapter_path:
        print(f"Loading LoRA adapter from: {lora_adapter_path}")
        try:
            model = PeftModel.from_pretrained(base_model, lora_adapter_path)

            if merge_lora:
                print("Merging LoRA weights into base model...")
                model = model.merge_and_unload()
                print("LoRA weights merged successfully")
            else:
                print("LoRA adapter loaded (weights not merged)")
                
        except Exception as e:
            print(f"Warning: Failed to load LoRA adapter: {e}")
            print("Falling back to base model only")
            model = base_model
    else:
        model = base_model
        print("Using base model without LoRA")
    
    return model.to(device)


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if args.merge_lora and not args.lora_adapter:
        print("Warning: --merge_lora specified but no --lora_adapter provided. Ignoring merge_lora.")
        args.merge_lora = False

    source = args.checkpoint if args.checkpoint else args.hf_model
    
    model = load_model_with_lora(
        source=source,
        lora_adapter_path=args.lora_adapter,
        merge_lora=args.merge_lora,
        device=device
    )
    model.eval()

    if hasattr(model, 'base_model'):
        model_cfg = model.base_model.cfg
    else:
        model_cfg = model.cfg

    tokenizer = get_tokenizer(model_cfg.lm_tokenizer, model_cfg.vlm_extra_tokens)
    image_processor = get_image_processor(model_cfg.vit_img_size)

    messages = [{"role": "user", "content": tokenizer.image_token * model_cfg.mp_image_token_length + args.prompt}]
    encoded_prompt = tokenizer.apply_chat_template([messages], tokenize=True, add_generation_prompt=True)
    tokens = torch.tensor(encoded_prompt).to(device)

    img = Image.open(args.image).convert("RGB")
    img_t = image_processor(img).unsqueeze(0).to(device)

    if args.lora_adapter:
        if hasattr(model, 'print_trainable_parameters'):
            print("\nLoRA Model Info:")
            model.print_trainable_parameters()
        lora_status = "merged" if args.merge_lora else "active"
        print(f"LoRA status: {lora_status}")

    print("\nInput:\n ", args.prompt, "\n\nOutputs:")
    
    with torch.no_grad():
        for i in range(args.generations):
            gen = model.generate(tokens, img_t, max_new_tokens=args.max_new_tokens)
            out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
            print(f"  >> Generation {i+1}: {out}")


if __name__ == "__main__":
    main()
