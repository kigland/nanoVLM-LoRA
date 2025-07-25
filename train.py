import math
import time
import torch
import wandb
import numpy
import random
import argparse
import contextlib
import torch.optim as optim
from statistics import mean
from dataclasses import asdict
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from data.collators import VQACollator, MMStarCollator
from data.datasets import MMStarDataset, VQADataset
from data.processors import get_image_processor, get_tokenizer
from models.vision_language_model import VisionLanguageModel
import models.config as config
import models.utils as utils
from evaluation import evaluate

#Otherwise, the tokenizer will throw a warning
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def init_dist():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

def destroy_dist():
    dist.destroy_process_group()

def is_dist():
    return dist.is_available() and dist.is_initialized()

def is_master():
    return dist.get_rank() == 0 if is_dist() else True

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def get_rank():
    return dist.get_rank() if is_dist() else 0

def dist_gather(o):
    o_all = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(o_all, o)
    return o_all

def wrap_model(model):
    return DistributedDataParallel(model, device_ids=[dist.get_rank()])

def get_run_name(train_cfg, vlm_cfg):
    dataset_size = "full_ds" if train_cfg.data_cutoff_idx is None else f"{train_cfg.data_cutoff_idx}samples"
    batch_size = f"bs{int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}"
    epochs = f"ep{train_cfg.epochs}"
    learning_rate = f"lr{train_cfg.lr_backbones}-{train_cfg.lr_mp}"
    num_gpus = f"{get_world_size()}xGPU"
    date = time.strftime("%m%d-%H%M%S")
    vit = f"{vlm_cfg.vit_model_type.split('/')[-1]}"
    mp = f"mp{vlm_cfg.mp_pixel_shuffle_factor}"
    llm = f"{vlm_cfg.lm_model_type.split('/')[-1]}"
    
    lora_suffix = ""
    if train_cfg.use_lora:
        lora_suffix = f"_lora_r{train_cfg.lora_r}_alpha{train_cfg.lora_alpha}"

    return f"nanoVLM_{vit}_{mp}_{llm}_{num_gpus}_{dataset_size}_{batch_size}_{epochs}_{learning_rate}{lora_suffix}_{date}"

def get_dataloaders(train_cfg, vlm_cfg):
    # Create datasets
    image_processor = get_image_processor(vlm_cfg.vit_img_size)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)

    # Load and combine all training datasets
    combined_train_data = []
    for dataset_name in train_cfg.train_dataset_name:
        train_ds = load_dataset(train_cfg.train_dataset_path, dataset_name)
        combined_train_data.append(train_ds['train'])
    train_ds = concatenate_datasets(combined_train_data)
    
    test_ds = load_dataset(train_cfg.test_dataset_path)
    train_ds = train_ds.shuffle(seed=0) # Shuffle the training dataset, so train and val get equal contributions from all concatenated datasets

    # Apply cutoff if specified
    if train_cfg.data_cutoff_idx is None:
        total_samples = len(train_ds)  # Use the entire dataset
    else:
        total_samples = min(len(train_ds), train_cfg.data_cutoff_idx)

    val_size = int(total_samples * train_cfg.val_ratio)
    train_size = total_samples - val_size

    train_dataset = VQADataset(train_ds.select(range(train_size)), tokenizer, image_processor, vlm_cfg.mp_image_token_length)
    val_dataset = VQADataset(train_ds.select(range(train_size, total_samples)), tokenizer, image_processor, vlm_cfg.mp_image_token_length)
    test_dataset = MMStarDataset(test_ds['val'], tokenizer, image_processor, vlm_cfg.mp_image_token_length)

    # Create collators
    vqa_collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)
    mmstar_collator = MMStarCollator(tokenizer)

    g = torch.Generator()
    g.manual_seed(0)

    # Create dataloaders
    train_sampler = DistributedSampler(
        train_dataset, 
        rank=get_rank(),
        num_replicas=get_world_size(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,    # =per device BS in DDP
        sampler=train_sampler,
        collate_fn=vqa_collator,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        rank=get_rank(),
        num_replicas=get_world_size(),
        shuffle=False  # Usually False for validation
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        sampler=val_sampler,
        collate_fn=vqa_collator,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=train_cfg.mmstar_batch_size, 
        shuffle=False, 
        collate_fn=mmstar_collator,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        )

    return train_loader, val_loader, test_loader

def apply_lora_to_model(model, train_cfg):
    if not train_cfg.use_lora:
        return model
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=train_cfg.lora_r,
        lora_alpha=train_cfg.lora_alpha,
        lora_dropout=train_cfg.lora_dropout,
        target_modules=train_cfg.lora_target_modules,
        bias="none",
        use_rslora=train_cfg.use_rslora,
    )
    
    model = get_peft_model(model, lora_config)
    
    if is_master():
        model.print_trainable_parameters()
    
    return model

def test_mmstar(model, tokenizer, test_loader, device):
    total_examples = 0
    correct_predictions = 0
    with torch.no_grad():
        for batch in test_loader:
            image = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            correct_answer = tokenizer.batch_decode(labels, skip_special_tokens=True)
            gen = model.generate(input_ids, image, attention_mask, greedy=True, max_new_tokens=10)
            model_output = tokenizer.batch_decode(gen, skip_special_tokens=True)
            
            is_correct = utils.check_multiple_choice_with_regex(model_output, correct_answer)
            
            total_examples += len(is_correct)
            if is_correct:
                correct_predictions += sum(is_correct)
    accuracy = correct_predictions / total_examples if total_examples > 0 else 0
    return accuracy

# Cosine learning rate schedule with warmup (from Karpathy)
# https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L353
def get_lr(it, max_lr, max_steps):
    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.03
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def train(train_cfg, vlm_cfg):
    train_loader, val_loader, test_loader = get_dataloaders(train_cfg, vlm_cfg)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)

    run_name = get_run_name(train_cfg, vlm_cfg)
    total_dataset_size = len(train_loader.dataset)
    if train_cfg.data_cutoff_idx is None:
            run_name = run_name.replace("full_ds", f"{total_dataset_size}samples")
    if train_cfg.log_wandb and is_master():
        run = wandb.init(
            entity=train_cfg.wandb_entity,
            project="nanoVLM",
            config={
                "VLMConfig": asdict(vlm_cfg),
                "TrainConfig": asdict(train_cfg)
            },
            name=run_name,
        )

    # Initialize model
    if train_cfg.resume_from_vlm_checkpoint:
        if train_cfg.use_lora and train_cfg.resume_from_lora_checkpoint:
            base_model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
            model = PeftModel.from_pretrained(base_model, train_cfg.lora_checkpoint_path)
        else:
            model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
            if train_cfg.use_lora:
                model = apply_lora_to_model(model, train_cfg)
    else:
        model = VisionLanguageModel(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)
        if train_cfg.use_lora:
            model = apply_lora_to_model(model, train_cfg)
    
    if is_master():
        if train_cfg.use_lora:
            print(f"nanoVLM with LoRA initialized")
            if hasattr(model, 'print_trainable_parameters'):
                model.print_trainable_parameters()
        else:
            print(f"nanoVLM initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        print(f"Training summary{' (global)' if is_dist() else ''}: {len(train_loader.dataset)} samples, {int(len(train_loader)*get_world_size())} batches/epoch, batch size {int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}")
        if is_dist():
            print(f"Training summary per GPU: {len(train_loader)} batches/epoch, batch size {train_loader.batch_size}")
        print(f"Validation summary{' (global)' if is_dist() else ''}: {len(val_loader.dataset)} samples, {int(len(val_loader)*get_world_size())} batches/epoch, batch size {int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}")
        if is_dist():
            print(f"Validation summary per GPU: {len(val_loader)} batches/epoch, batch size {val_loader.batch_size}")

    # Define optimizer groups
    # Since we have pretrained vision and language backbones, but a newly initialized modality projection layer, it doesn't make sense to train them with the same learning rate
    # You could opt to fully freeze the backbones and only train the MP layer, but finetuning them with a lower learning rate makes the training as a whole easier
    if train_cfg.use_lora:
        lora_params = [p for n, p in model.named_parameters() if 'lora_' in n]
        mp_params = [p for n, p in model.named_parameters() if 'MP' in n and 'lora_' not in n]
        
        param_groups = []
        if mp_params:
            param_groups.append({'params': mp_params, 'lr': train_cfg.lr_mp})
        if lora_params:
            param_groups.append({'params': lora_params, 'lr': train_cfg.lora_lr})
        
        if not param_groups:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            param_groups = [{'params': trainable_params, 'lr': train_cfg.lora_lr}]
    else:
        param_groups = [{'params': list(model.MP.parameters()), 'lr': train_cfg.lr_mp},
                        {'params': list(model.decoder.parameters()) + list(model.vision_encoder.parameters()), 'lr': train_cfg.lr_backbones}]
    
    optimizer = optim.AdamW(param_groups)
    all_params = [p for group in optimizer.param_groups for p in group['params']]

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    if device.type == "mps":
        torch.backends.mps.enable_fallback_to_cpu = True
        torch.mps.empty_cache()
    
    print(f"Using device: {device}")
    model.to(device)
    
    if train_cfg.compile:
        model = torch.compile(model)
    if is_dist():
        model = wrap_model(model)

    epoch_times = []
    best_accuracy = 0
    best_val_loss = float('inf')
    global_step = 0
    for epoch in range(train_cfg.epochs):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        total_tokens_processed = 0
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            is_update_step = (i + 1) % train_cfg.gradient_accumulation_steps == 0 or i + 1 == len(train_loader)
            batch_start_time = time.time()
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # When using DDP with gradient accumulation,
            # skip gradient synchronization on intermediate steps to save time.
            # Gradients only need to be synced at the end of each accumulation cycle.
            if (is_dist()
                and train_cfg.gradient_accumulation_steps > 1
                and not is_update_step):
                context = model.no_sync()
            else:
                context = contextlib.nullcontext()

            autocast_context = torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type in ['cuda', 'cpu'] else torch.float16
            )
            with autocast_context:

                with context:
                    _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)

            if train_cfg.gradient_accumulation_steps > 1:
                loss = loss / train_cfg.gradient_accumulation_steps

            loss.backward()

            if is_update_step:
                if train_cfg.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=train_cfg.max_grad_norm)

                if train_cfg.use_lora:
                    for i, group in enumerate(optimizer.param_groups):
                        if i == 0 and len(optimizer.param_groups) > 1:
                            group['lr'] = get_lr(global_step, train_cfg.lr_mp, len(train_loader) * train_cfg.epochs // train_cfg.gradient_accumulation_steps)
                        else:  # LoRA
                            group['lr'] = get_lr(global_step, train_cfg.lora_lr, len(train_loader) * train_cfg.epochs // train_cfg.gradient_accumulation_steps)
                else:
                    adj_lr_mp = get_lr(global_step, train_cfg.lr_mp, len(train_loader) * train_cfg.epochs // train_cfg.gradient_accumulation_steps)
                    adj_lr_backbones = get_lr(global_step, train_cfg.lr_backbones, len(train_loader) * train_cfg.epochs // train_cfg.gradient_accumulation_steps)
                    optimizer.param_groups[0]['lr'] = adj_lr_mp
                    optimizer.param_groups[1]['lr'] = adj_lr_backbones

                optimizer.step()
                optimizer.zero_grad()

            batch_loss = loss.item()
            if train_cfg.gradient_accumulation_steps > 1:
                batch_loss = batch_loss * train_cfg.gradient_accumulation_steps
            total_train_loss += batch_loss

            num_tokens = torch.sum(attention_mask).item() # Sum of attention mask gives number of tokens
            total_tokens_processed += num_tokens

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            tokens_per_second = num_tokens / batch_duration 

            # gather loss and t/s from all ranks if DDP
            batch_loss = mean(dist_gather(batch_loss)) if is_dist() else batch_loss  
            tokens_per_second = sum(dist_gather(tokens_per_second)) if is_dist() else tokens_per_second  

            if train_cfg.eval_in_epochs and global_step % train_cfg.eval_interval == 0 and is_update_step:
                model.eval()
                if device == "cuda":
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    save = False
                    total_val_loss = 0
                    for batch in val_loader:
                        images = batch["image"].to(device)
                        input_ids = batch["input_ids"].to(device)
                        labels = batch["labels"].to(device)
                        attention_mask = batch["attention_mask"].to(device)

                        with autocast_context:
                            _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)

                        total_val_loss += loss.item()
                    avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
                    avg_val_loss = mean(dist_gather(avg_val_loss)) if is_dist() else avg_val_loss
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        save = True
                    if train_cfg.log_wandb and is_master():
                        run.log({"val_loss": avg_val_loss}, step=global_step)

                    lmms_results = {}
                    if train_cfg.use_lmms_eval and global_step % (train_cfg.eval_interval*2) == 0:
                        eval_results = evaluate(
                            model=model.module if is_dist() else model,
                            tasks=train_cfg.lmms_eval_tasks,
                            limit=train_cfg.lmms_eval_limit,
                            batch_size=train_cfg.lmms_eval_batch_size,
                            process_with_media=True,
                            device=device,
                        )

                        if is_master() and eval_results and "results" in eval_results[0]:
                            for task_name, task_results in eval_results[0]["results"].items():
                                for metric_name, metric_value in task_results.items():
                                    if isinstance(metric_value, (int, float)):
                                        lmms_results[f"lmms_{task_name}_{metric_name.split(',')[0]}"] = metric_value
                        
                    if is_master() and global_step % (train_cfg.eval_interval*2) == 0:
                        eval_model = model.module if is_dist() else model  # unwrap the model for eval if DDP
                        epoch_accuracy = test_mmstar(eval_model, tokenizer, test_loader, device)
                        if epoch_accuracy > best_accuracy:
                            best_accuracy = epoch_accuracy
                            save = True
                        if save:
                            if train_cfg.use_lora:
                                eval_model.save_pretrained(os.path.join(vlm_cfg.vlm_checkpoint_path, run_name))
                            else:
                                eval_model.save_pretrained(save_directory=os.path.join(vlm_cfg.vlm_checkpoint_path, run_name))
                        
                        if train_cfg.log_wandb and is_master():    
                            run.log({"accuracy": epoch_accuracy, **lmms_results}, step=global_step)
                        print(f"Step: {global_step}, Loss: {batch_loss:.4f}, Tokens/s: {tokens_per_second:.2f}, Accuracy: {epoch_accuracy:.4f}")
                    elif is_master() and not global_step % (train_cfg.eval_interval*4) == 0:
                        print(f"Step: {global_step}, Loss: {batch_loss:.4f}, Tokens/s: {tokens_per_second:.2f}")

                model.train()          

            if train_cfg.log_wandb and is_master():
                run.log({
                    "batch_loss": batch_loss,
                    "tokens_per_second": tokens_per_second,
                    **({"grad_norm": grad_norm} if train_cfg.max_grad_norm is not None and is_update_step else {})
                }, step=global_step)
                
            if is_update_step:
                global_step += 1

        avg_train_loss = total_train_loss / len(train_loader)
        # gather average batch loss from all ranks if DDP
        avg_train_loss = mean(dist_gather(avg_train_loss)) if is_dist() else avg_train_loss  

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        # gather and sum total_tokens_processed across all ranks if DDP
        total_tokens_processed = sum(dist_gather(total_tokens_processed)) if is_dist() else total_tokens_processed  
        epoch_tokens_per_second = total_tokens_processed / epoch_duration

        if is_master():
            if train_cfg.log_wandb:
                run.log({"epoch_loss": avg_train_loss,
                         "epoch_duration": epoch_duration,
                         "epoch_tokens_per_second": epoch_tokens_per_second})

            print(f"Epoch {epoch+1}/{train_cfg.epochs}, Train Loss: {avg_train_loss:.4f} | Time: {epoch_duration:.2f}s | T/s: {epoch_tokens_per_second:.2f}")

    # Summary Statistics
    if is_master():
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        total_training_time = sum(epoch_times)
        total_samples_processed = len(train_loader.dataset) * train_cfg.epochs
        avg_time_per_sample = total_training_time / total_samples_processed
        print(f"Average time per epoch: {avg_epoch_time:.2f}s")
        print(f"Average time per sample: {avg_time_per_sample:.4f}s")

        # Push the best model to the hub (Please set your user name in the config!)
        if vlm_cfg.hf_repo_name is not None:
            print("Training complete. Pushing model to Hugging Face Hub...")
            if train_cfg.use_lora:
                base_model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
                hf_model = PeftModel.from_pretrained(base_model, os.path.join(vlm_cfg.vlm_checkpoint_path, run_name))
                hf_model.push_to_hub(vlm_cfg.hf_repo_name)
            else:
                hf_model = VisionLanguageModel.from_pretrained(os.path.join(vlm_cfg.vlm_checkpoint_path, run_name))
                hf_model.push_to_hub(vlm_cfg.hf_repo_name)

        if train_cfg.log_wandb:
            run.summary["avg_epoch_time"] = avg_epoch_time
            run.summary["avg_time_per_sample"] = avg_time_per_sample
            run.summary["mmstar_acc"] = best_accuracy
            run.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_mp', type=float, help='Learning rate for the mapping network')
    parser.add_argument('--lr_backbones', type=float, help='Learning rate for the backbones')
    parser.add_argument('--vlm_checkpoint_path', type=str, help='Path to the VLM checkpoint for loading or saving')
    parser.add_argument('--compile', type=bool, help='Use torch.compile to optimize the model')
    parser.add_argument('--log_wandb', type=bool, help='Log to wandb')
    parser.add_argument('--resume_from_vlm_checkpoint', type=bool, default=False, help='Resume training from VLM checkpoint specified by vlm_checkpoint_path (or default if not provided)')
    
    parser.add_argument('--use_lora', type=bool, default=False, help='Use LoRA for parameter-efficient fine-tuning')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    parser.add_argument('--lora_lr', type=float, default=1e-4, help='Learning rate for LoRA parameters')
    parser.add_argument('--resume_from_lora_checkpoint', type=bool, default=False, help='Resume from LoRA checkpoint')
    parser.add_argument('--lora_checkpoint_path', type=str, help='Path to LoRA checkpoint')

    args = parser.parse_args()

    vlm_cfg = config.VLMConfig()
    train_cfg = config.TrainConfig()

    if args.lr_mp is not None:
        train_cfg.lr_mp = args.lr_mp
    if args.lr_backbones is not None:
        train_cfg.lr_backbones = args.lr_backbones
    if args.vlm_checkpoint_path is not None:
        vlm_cfg.vlm_checkpoint_path = args.vlm_checkpoint_path
    if args.compile is not None:
        train_cfg.compile = args.compile
    if args.log_wandb is not None:
        train_cfg.log_wandb = args.log_wandb
    
    if args.use_lora is not None:
        train_cfg.use_lora = args.use_lora
    if args.lora_r is not None:
        train_cfg.lora_r = args.lora_r
    if args.lora_alpha is not None:
        train_cfg.lora_alpha = args.lora_alpha
    if args.lora_dropout is not None:
        train_cfg.lora_dropout = args.lora_dropout
    if args.lora_lr is not None:
        train_cfg.lora_lr = args.lora_lr
    if args.resume_from_lora_checkpoint is not None:
        train_cfg.resume_from_lora_checkpoint = args.resume_from_lora_checkpoint
    if args.lora_checkpoint_path is not None:
        train_cfg.lora_checkpoint_path = args.lora_checkpoint_path

    if args.resume_from_vlm_checkpoint and args.vlm_checkpoint_path is not None:
        train_cfg.resume_from_vlm_checkpoint = True
        # When resuming a full VLM, we don't need to load individual backbone weights from original sources
        vlm_cfg.vlm_load_backbone_weights = False

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist()

    if is_master():
        print("--- VLM Config ---")
        print(vlm_cfg)
        print("--- Train Config ---")
        print(train_cfg)

    train(train_cfg, vlm_cfg)

    if is_dist():
        destroy_dist()

if __name__ == "__main__":
    main()
