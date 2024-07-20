import torch
from functools import partial
from . import image_encoder, prompt_encoder, mask_decoder, sam3D, mask_decoder_use_penn


def build_sam3D_vit_b_ori(args=None, checkpoint=None):
    return _build_sam3D_ori(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        args=args,
    )

sam_model_registry3D = {
    "vit_b_ori": build_sam3D_vit_b_ori,
}


def _build_sam3D_ori(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    args=None,
):
    prompt_embed_dim = 384
    image_size = args.image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = sam3D.Sam3D(
        image_encoder=image_encoder.ImageEncoderViT(
            args,
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),

        prompt_encoder=prompt_encoder.PromptEncoder3D(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size, image_size),
            mask_in_chans=16,
            num_multiple_outputs=args.num_multiple_outputs,
            multiple_outputs=args.multiple_outputs,
        ),

        mask_decoder=mask_decoder_use_penn.MaskDecoder3D(
            args,
            transformer_dim=prompt_embed_dim,
            num_multiple_outputs=args.num_multiple_outputs,
            multiple_outputs=args.multiple_outputs,
        ) if args.use_penn
        else mask_decoder.MaskDecoder3D(
            args,
            transformer_dim=prompt_embed_dim,
            num_multiple_outputs=args.num_multiple_outputs,
            multiple_outputs=args.multiple_outputs,
        )
        ,
    )

    # FIXME
    #  PRISM mask decoder  217208669184
    # a = torch.rand(1, 2967, 384)
    # b = torch.rand(1, 384, 8, 8, 8)
    # c = torch.rand(1, 32, 128, 128, 128)
    # d = torch.rand(1, 32, 64, 64, 64)
    # e = torch.rand(1, 64, 32, 32, 32)
    # f = torch.rand(1, 128, 16, 16, 16)
    # from fvcore.nn import FlopCountAnalysis
    # flop_counter = FlopCountAnalysis(sam.mask_decoder.cpu(), inputs=(a, b, [c, d, e, f]))
    # print(flop_counter.total())
    # print(1)

    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location=args.device)
        if args.use_sam3d_turbo and args.split == 'train':
            # Initialize a new state dictionary for the image_encoder
            encoder_state_dict = {}
            for key in state_dict['model_state_dict']:
                if key.startswith(
                        'image_encoder.'):  # Adjust 'image_encoder.' based on how the keys are named in your state_dict
                    # Remove the 'image_encoder.' prefix and save the modified key
                    new_key = key[len('image_encoder.'):]
                    encoder_state_dict[new_key] = state_dict['model_state_dict'][key]
            # Now load the adjusted state dict into the image_encoder part of your model
            sam.image_encoder.load_state_dict(encoder_state_dict, strict=False)
        else:
            sam.load_state_dict(state_dict['model_state_dict'])

    return sam

