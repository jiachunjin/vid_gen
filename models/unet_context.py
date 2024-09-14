import copy
import types
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D, DownBlock2D, UpBlock2D
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel, UNet2DConditionOutput
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, logging, scale_lora_layers, unscale_lora_layers

from .conditioner import get_conditioner

logger = logging.get_logger(__name__)

class UNet_context(UNet2DConditionModel):
    def return_extra_parameters(self):
        parameters = []
        parameters.extend(list(self.conditioner.parameters()))
        for down_block in self.down_blocks:
            if isinstance(down_block, DownBlock2D):
                continue
            parameters.extend(list(down_block.extra_attn_1.parameters()))
            parameters.extend(list(down_block.extra_attn_2.parameters()))

        parameters.extend(list(self.mid_block.extra_attn.parameters()))

        for up_block in self.up_blocks:
            if isinstance(up_block, UpBlock2D):
                continue
            parameters.extend(list(up_block.extra_attn_1.parameters()))
            parameters.extend(list(up_block.extra_attn_2.parameters()))
            parameters.extend(list(up_block.extra_attn_3.parameters()))

        return parameters

    def hack(
        self,
        con_type,
        con_dim,
        con_depth,
        con_numq,
        con_nframes,
        con_patch_size,
        con_heads,
        con_dim_head,
    ):
        self.conditioner = get_conditioner(
            con_type = con_type,
            con_dim = con_dim,
            con_depth = con_depth,
            con_numq = con_numq,
            con_nframes = con_nframes,
            con_patch_size = con_patch_size,
            con_dim_head = con_dim_head,
            con_heads = con_heads,
        )
        ##############################################################################################################
        ################################################# Down block #################################################
        ##############################################################################################################
        def down_cross_attn_fwd(
            self,
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            additional_residuals: Optional[torch.Tensor] = None,
            context_tokens: Optional[torch.Tensor] = None,
        ):
            use_context = context_tokens is not None
            output_states = ()
            blocks = list(zip(self.resnets, self.attentions))

            for i, (resnet, attn) in enumerate(blocks):
                if self.training and self.gradient_checkpointing:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                    if i == 0 and use_context:
                        hidden_states = self.extra_attn_1(
                            hidden_states,
                            encoder_hidden_states=context_tokens,
                            cross_attention_kwargs=cross_attention_kwargs,
                            attention_mask=attention_mask,
                            encoder_attention_mask=encoder_attention_mask,
                            return_dict=False,
                        )[0]
                    elif i == 1 and use_context:
                        hidden_states = self.extra_attn_2(
                            hidden_states,
                            encoder_hidden_states=context_tokens,
                            cross_attention_kwargs=cross_attention_kwargs,
                            attention_mask=attention_mask,
                            encoder_attention_mask=encoder_attention_mask,
                            return_dict=False,
                        )[0]
                    else:
                        pass
                        # raise ValueError(f"No extra_attn added for block id {i} in down_cross_attn_fwd().")

                else:
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                    if i == 0 and use_context:
                        hidden_states = self.extra_attn_1(
                            hidden_states,
                            encoder_hidden_states=context_tokens,
                            cross_attention_kwargs=cross_attention_kwargs,
                            attention_mask=attention_mask,
                            encoder_attention_mask=encoder_attention_mask,
                            return_dict=False,
                        )[0]
                    elif i == 1 and use_context:
                        hidden_states = self.extra_attn_2(
                            hidden_states,
                            encoder_hidden_states=context_tokens,
                            cross_attention_kwargs=cross_attention_kwargs,
                            attention_mask=attention_mask,
                            encoder_attention_mask=encoder_attention_mask,
                            return_dict=False,
                        )[0]
                    else:
                        pass
                        # raise ValueError(f"No extra_attn added for block id {i} in down_cross_attn_fwd().")

                # apply additional residuals to the output of the last pair of resnet and attention blocks
                if i == len(blocks) - 1 and additional_residuals is not None:
                    hidden_states = hidden_states + additional_residuals

                output_states = output_states + (hidden_states,)

            if self.downsamplers is not None:
                for downsampler in self.downsamplers:
                    hidden_states = downsampler(hidden_states)

                output_states = output_states + (hidden_states,)

            return hidden_states, output_states

        for down_block in self.down_blocks:
            if isinstance(down_block, DownBlock2D):
                continue
            extra_attn_1 = copy.deepcopy(down_block.attentions[0])
            extra_attn_2 = copy.deepcopy(down_block.attentions[1])
            down_block.extra_attn_1 = extra_attn_1
            down_block.extra_attn_2 = extra_attn_2
            down_block.down_cross_attn_fwd = types.MethodType(down_cross_attn_fwd, down_block)

        ##############################################################################################################
        ################################################## Mid block #################################################
        ##############################################################################################################
        def mid_cross_attn_fwd(
            self,
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            context_tokens: Optional[torch.Tensor] = None,
        ):
            use_context = context_tokens is not None

            hidden_states = self.resnets[0](hidden_states, temb)
            for attn, resnet in zip(self.attentions, self.resnets[1:]):
                if self.training and self.gradient_checkpointing:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                    if use_context:
                        hidden_states = self.extra_attn(
                            hidden_states,
                            encoder_hidden_states=context_tokens,
                            cross_attention_kwargs=cross_attention_kwargs,
                            attention_mask=attention_mask,
                            encoder_attention_mask=encoder_attention_mask,
                            return_dict=False,
                        )[0]
                else:
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                    hidden_states = resnet(hidden_states, temb)
                    if use_context:
                        hidden_states = self.extra_attn(
                            hidden_states,
                            encoder_hidden_states=context_tokens,
                            cross_attention_kwargs=cross_attention_kwargs,
                            attention_mask=attention_mask,
                            encoder_attention_mask=encoder_attention_mask,
                            return_dict=False,
                        )[0]

            return hidden_states

        extra_attn = copy.deepcopy(self.mid_block.attentions[0])
        self.mid_block.extra_attn = extra_attn
        self.mid_block.mid_cross_attn_fwd = types.MethodType(mid_cross_attn_fwd, self.mid_block)

        ##############################################################################################################
        ################################################## Up block ##################################################
        ##############################################################################################################
        def up_cross_attn_fwd(
            self,
            hidden_states: torch.Tensor,
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],
            temb: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            upsample_size: Optional[int] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            context_tokens: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            use_context = context_tokens is not None
            i = 0
            for resnet, attn in zip(self.resnets, self.attentions):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                if self.training and self.gradient_checkpointing:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                    if i == 0 and use_context:
                        hidden_states = self.extra_attn_1(
                            hidden_states,
                            encoder_hidden_states=context_tokens,
                            cross_attention_kwargs=cross_attention_kwargs,
                            attention_mask=attention_mask,
                            encoder_attention_mask=encoder_attention_mask,
                            return_dict=False,
                        )[0]
                    elif i == 1 and use_context:
                        hidden_states = self.extra_attn_2(
                            hidden_states,
                            encoder_hidden_states=context_tokens,
                            cross_attention_kwargs=cross_attention_kwargs,
                            attention_mask=attention_mask,
                            encoder_attention_mask=encoder_attention_mask,
                            return_dict=False,
                        )[0]
                    elif i == 2 and use_context:
                        hidden_states = self.extra_attn_3(
                            hidden_states,
                            encoder_hidden_states=context_tokens,
                            cross_attention_kwargs=cross_attention_kwargs,
                            attention_mask=attention_mask,
                            encoder_attention_mask=encoder_attention_mask,
                            return_dict=False,
                        )[0]
                    else:
                        pass
                        # raise ValueError(f"No extra_attn added for block id {i} in up_cross_attn_fwd().")
                else:
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                    if i == 0 and use_context:
                        hidden_states = self.extra_attn_1(
                            hidden_states,
                            encoder_hidden_states=context_tokens,
                            cross_attention_kwargs=cross_attention_kwargs,
                            attention_mask=attention_mask,
                            encoder_attention_mask=encoder_attention_mask,
                            return_dict=False,
                        )[0]
                    elif i == 1 and use_context:
                        hidden_states = self.extra_attn_2(
                            hidden_states,
                            encoder_hidden_states=context_tokens,
                            cross_attention_kwargs=cross_attention_kwargs,
                            attention_mask=attention_mask,
                            encoder_attention_mask=encoder_attention_mask,
                            return_dict=False,
                        )[0]
                    elif i == 2 and use_context:
                        hidden_states = self.extra_attn_3(
                            hidden_states,
                            encoder_hidden_states=context_tokens,
                            cross_attention_kwargs=cross_attention_kwargs,
                            attention_mask=attention_mask,
                            encoder_attention_mask=encoder_attention_mask,
                            return_dict=False,
                        )[0]
                    else:
                        pass
                        # raise ValueError(f"No extra_attn added for block id {i} in up_cross_attn_fwd().")
                
                i += 1

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states
            

        for up_block in self.up_blocks:
            if isinstance(up_block, UpBlock2D):
                continue
            extra_attn_1 = copy.deepcopy(up_block.attentions[0])
            extra_attn_2 = copy.deepcopy(up_block.attentions[1])
            extra_attn_3 = copy.deepcopy(up_block.attentions[2])
            up_block.extra_attn_1 = extra_attn_1
            up_block.extra_attn_2 = extra_attn_2
            up_block.extra_attn_3 = extra_attn_3
            up_block.up_cross_attn_fwd = types.MethodType(up_cross_attn_fwd, up_block)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        contexts: Optional[torch.Tensor] = None,
        random_frame_indices: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        The [`UNet2DConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
                Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                through the `self.time_embedding` layer to obtain the timestep embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            added_cond_kwargs: (`dict`, *optional*):
                A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.
            down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                A tuple of tensors that if specified are added to the residuals of down unet blocks.
            mid_block_additional_residual: (`torch.Tensor`, *optional*):
                A tensor that if specified is added to the residual of the middle unet block.
            down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
                additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)
            encoder_attention_mask (`torch.Tensor`):
                A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                which adds large negative values to the attention scores corresponding to "discard" tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] is returned,
                otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if self.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        # 2. pre-process
        sample = self.conv_in(sample)

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        if contexts is not None:
            if random_frame_indices is None:
                # inference mode
                context_batch = 4
                context_tokens = []
                b, f, c, h, w = contexts.shape
                for i in range(0, f, context_batch):
                    context_token = self.conditioner(contexts.repeat(context_batch, 1, 1, 1, 1), timestep[0].unsqueeze(0).repeat(context_batch), torch.as_tensor(range(i, i+context_batch), device=contexts.device)).squeeze()
                    context_tokens.append(context_token)
                context_tokens = torch.cat(context_tokens, dim=0)
            else:
                context_tokens = self.conditioner(contexts, timestep, random_frame_indices) # (b, l, d)
                # context_tokens_down = self.conditioner(contexts, timestep, random_frame_indices) # (b, l, d)
                # context_tokens_mid = self.conditioner_mid(contexts, timestep, random_frame_indices) # (b, l, d)
                # context_tokens_up = self.conditioner_up(contexts, timestep, random_frame_indices) # (b, l, d)
        else:
            context_tokens = None
        # 3. down
        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
        is_adapter = down_intrablock_additional_residuals is not None
        # maintain backward compatibility for legacy usage, where
        #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
        #       but can only use one or the other
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            deprecate(
                "T2I should not use down_block_additional_residuals",
                "1.3.0",
                "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                       and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                       for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
                standard_warn=False,
            )
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                sample, res_samples = downsample_block.down_cross_attn_fwd(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    context_tokens=context_tokens,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block.mid_cross_attn_fwd(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    context_tokens=context_tokens,
                )
            else:
                sample = self.mid_block(sample, emb)

            # To support T2I-Adapter-XL
            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0
                and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample += down_intrablock_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block.up_cross_attn_fwd(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    context_tokens=context_tokens,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)