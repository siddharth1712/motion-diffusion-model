# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args, lora_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
import wandb
from peft import LoraModel, LoraConfig
from diffusion import logger
from utils.model_util import load_model_wo_clip

def main():
    args = lora_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')
    
    if args.wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project="motion-flux",
            # Track hyperparameters and run metadata
            config={"learning_rate": args.lr,"steps": args.num_steps,"batch_size":args.batch_size},
            name="flux_2_2_4_lora"
            )

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames , hml_mode='lora')

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()
    
    pre_trained_checkpoint = args.pretrained_model

    logger.log(f"loading model from checkpoint: {pre_trained_checkpoint}...")
    load_model_wo_clip(model,dist_util.load_state_dict(pre_trained_checkpoint, map_location=dist_util.dev()))
    
    print(model)
    
    if (args.arch == "flux"):
        # We only train the additional adapter LoRA layers
        model.transformer.requires_grad_(False)
        model.text_encoders[0].requires_grad_(False)
        model.text_encoders[1].requires_grad_(False)
        
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
            ]
        
        # now we will add new LoRA weights the transformer layers
        transformer_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
            )
        
        model.transformer.add_adapter(transformer_lora_config)
    else:
        for param in model.parameters():
            param.requires_grad = False
        
        target_modules = [
            # Transformer Encoder Layers
            "seqTransEncoder.layers.*.self_attn.out_proj",
            "seqTransEncoder.layers.*.linear1",
            "seqTransEncoder.layers.*.linear2",
            
            # Embedding Layers
            "poseEmbedding",
            "embed_text",
            
            # Timestep Embedder Layers
            "embed_timestep.time_embed.0",  # First Linear layer in time embedding
            "embed_timestep.time_embed.2",  # Second Linear layer in time embedding
            
            # Output Process
            "output_process.poseFinal"
        ]
        
        # now we will add new LoRA weights the transformer layers
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
            )
        
        model = LoraModel(model,lora_config,"default")
        
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print('Trainable params: %.2f' % (sum(p.numel() for p in model.parameters_wo_clip() if p.requires_grad)))
    print("Fine-tuning with LoRA...")
    TrainLoop(args, train_platform, model, diffusion, data).lora()
    train_platform.close()

if __name__ == "__main__":
    main()
