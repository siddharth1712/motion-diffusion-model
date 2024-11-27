import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz
from flux.transformer_flux import FluxTransformer2DModel, FluxSingleTransformerBlock, FluxTransformerBlock
from flux.pipeline_flux import FluxPipeline
from diffusers.loaders import FluxLoraLoaderMixin
from transformers import  CLIPTokenizer, T5TokenizerFast, AutoTokenizer, T5Tokenizer, SiglipTokenizer
from transformers import CLIPTextModel, T5EncoderModel, SiglipTextModel
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class MDM(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset
        self.weight_dtype = torch.float32

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim,self.arch,dtype=self.weight_dtype)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'gru':
            print("GRU init")
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        elif self.arch == 'flux': 
            print("FLUX Init")
            self.transformer = FluxTransformer2DModel(in_channels=self.input_feats,guidance_embeds=True,num_layers=2,num_single_layers=2,num_attention_heads=4,joint_attention_dim=1024)
            self.transformer.to(dtype=self.weight_dtype)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru, flux]')

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)
            elif 'flux' in self.cond_mode:
                print("EMBED FLUX TEXT")
                pretrained_model_name_or_path = ['google/siglip-base-patch16-224','google/t5-v1_1-large']
                self.text_encoders = self.load_text_encoders(pretrained_model_name_or_path)
                self.tokenizers = self.load_text_tokenizers(pretrained_model_name_or_path)
            
                self.text_encoders[0].requires_grad_(False)
                self.text_encoders[1].requires_grad_(False)
                
                self.text_encoders[0].to(device="cuda:0")
                self.text_encoders[1].to(device="cuda:0")
            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print('EMBED ACTION')

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats,self.arch,dtype=self.weight_dtype)

        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]
    
    def load_text_encoders(self,pretrained_model_name_or_path):
        text_encoder_one = SiglipTextModel.from_pretrained(pretrained_model_name_or_path[0])
        text_encoder_two = T5EncoderModel.from_pretrained(pretrained_model_name_or_path[1])
        # text_encoder_two = T5EncoderModel.from_pretrained(pretrained_model_name_or_path[1],subfolder='text_encoder_2')
        
        text_encoder_one.to(dtype=self.weight_dtype)
        text_encoder_two.to(dtype=self.weight_dtype)
        
        return [text_encoder_one,text_encoder_two]
    
    def load_text_tokenizers(self,pretrained_model_name_or_path):
        tokenizer_one = SiglipTokenizer.from_pretrained(pretrained_model_name_or_path[0])
        tokenizer_two = T5TokenizerFast.from_pretrained(pretrained_model_name_or_path[1])
        # tokenizer_two = T5TokenizerFast.from_pretrained(pretrained_model_name_or_path[1], subfolder='tokenizer_2')
        
        return [tokenizer_one,tokenizer_two]
    
    def _encode_prompt_with_t5(self,max_sequence_length=512,prompt=None,num_images_per_prompt=1,device=None,text_input_ids=None,):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.tokenizers[1] is not None:
            text_inputs = self.tokenizers[1](
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
        else:
            if text_input_ids is None:
                raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

        prompt_embeds = self.text_encoders[1](text_input_ids.to(device))[0]

        dtype = self.text_encoders[1].dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds


    def _encode_prompt_with_clip(self,prompt: str,device=None,text_input_ids=None,num_images_per_prompt: int = 1,):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.tokenizers[0] is not None:
            text_inputs = self.tokenizers[0](
                prompt,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=False,
                return_length=False,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
        else:
            if text_input_ids is None:
                raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

        prompt_embeds = self.text_encoders[0](text_input_ids.to(device),output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoders[0].dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds


    def encode_prompt(self,
        prompt: str,
        max_sequence_length,
        device=None,
        num_images_per_prompt: int = 1,
        text_input_ids_list=None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        dtype = self.text_encoders[0].dtype

        pooled_prompt_embeds = self._encode_prompt_with_clip(
            prompt=prompt,
            device=device if device is not None else self.text_encoders[0].device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
        )

        prompt_embeds = self._encode_prompt_with_t5(
            max_sequence_length=max_sequence_length,
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device if device is not None else self.text_encoders[1].device,
            text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
        )

        text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
        text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        device = next(self.parameters()).device
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        prompt_embeds = None
        pooled_prompt_embeds = None
        text_ids = None
        
        x = x.to(dtype=self.weight_dtype)

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            if 'text_embed' in y.keys():  # caching option
                enc_text = y['text_embed']
            else:
                enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        elif 'flux' in self.cond_mode:
            prompt = y['text']
            mask = torch.bernoulli(torch.ones(bs, device=device) * self.cond_mask_prob)  # 1-> use null_cond, 0-> use real cond
            prompt = ['' if mask[i].item() == 1 else prompt[i] for i in range(bs)]
            prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(prompt,device=device,max_sequence_length=256)
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)

        if self.arch == 'gru':
            x_reshaped = x.reshape(bs, njoints*nfeats, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)     #[#frames, bs, d]
            emb_gru = emb_gru.permute(1, 2, 0)      #[bs, d, #frames]
            emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)  #[bs, d, 1, #frames]
            x = torch.cat((x_reshaped, emb_gru), axis=1)  #[bs, d+joints*feat, 1, #frames]

        x = self.input_process(x)

        if self.arch == 'trans_enc':
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch == 'trans_dec':
            if self.emb_trans_dec:
                xseq = torch.cat((emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            if self.emb_trans_dec:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)[1:] # [seqlen, bs, d] # FIXME - maybe add a causal mask
            else:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)
        elif self.arch == 'gru':
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            output, _ = self.gru(xseq)
        elif self.arch == 'flux':
            self.latent_img_ids = FluxPipeline._prepare_latent_image_ids(batch_size=bs,n_frames=nframes,device=device,dtype=self.weight_dtype)
            output = self.transformer(
                hidden_states=x,
                timestep = timesteps/1000,
                guidance = torch.tensor([2.5],device=device),
                pooled_projections = pooled_prompt_embeds,
                encoder_hidden_states = prompt_embeds,
                txt_ids = text_ids,
                img_ids=self.latent_img_ids,
                return_dict = False,
                )[0]

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output


    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim,arch,dtype):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.arch = arch
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim,dtype=dtype)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim,dtype=dtype)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)
        
        if self.arch == 'flux':
            x = x.permute((1,0,2))
            return x
        
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats,arch,dtype):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.arch = arch
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats,dtype=dtype)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats,dtype=dtype)

    def forward(self, output):
        if self.arch == 'flux':
            output = output.permute((1,0,2))
            nframes, bs, d = output.shape
            output = output.reshape(nframes, bs, self.njoints, self.nfeats)
            output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
            return output
        
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output