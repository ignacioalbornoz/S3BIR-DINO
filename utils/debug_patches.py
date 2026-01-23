#!/usr/bin/env python3
"""
Script de depuración para verificar cuántos patches genera cada imagen.
"""

import torch
from PIL import Image
from torchvision.transforms import v2
from s3bir_dinov3_model import S3birDinov3

def make_transform():
    to_tensor = v2.ToImage()
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, to_float, normalize])

if __name__ == "__main__":
    ckpt_path = "/home/shared_data/s3bir/saved_models/skDinoV3_sketchy.ckpt"
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = S3birDinov3().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    img = Image.open("extracted_frames/Untitled design (2)_frame_0000_t0.00s_square_224.jpg").convert("RGB")
    sketch = Image.open("gato_square_224.jpg").convert("RGB")

    print(f"Imagen tamaño: {img.size}")
    print(f"Sketch tamaño: {sketch.size}")

    transform = make_transform()
    img_tensor = transform(img).unsqueeze(0).to(device)
    sketch_tensor = transform(sketch).unsqueeze(0).to(device)

    print(f"\nTensor imagen shape: {img_tensor.shape}")
    print(f"Tensor sketch shape: {sketch_tensor.shape}")

    with torch.no_grad():
        img_embedding = model(img_tensor, dtype="image", is_training=True)
        sketch_embedding = model(sketch_tensor, dtype="image", is_training=True)
        
    patch_img = img_embedding['x_norm_patchtokens']
    patch_sketch = sketch_embedding['x_norm_patchtokens']

    print(f"\nPatch tokens imagen shape: {patch_img.shape}")
    print(f"Patch tokens sketch shape: {patch_sketch.shape}")
    print(f"\nNúmero de patches imagen: {patch_img.shape[1]}")
    print(f"Número de patches sketch: {patch_sketch.shape[1]}")

    # Verificar si hay otros campos en el embedding
    print(f"\nCampos en img_embedding: {img_embedding.keys()}")
    if 'x_storage_tokens' in img_embedding:
        print(f"Storage tokens imagen: {img_embedding['x_storage_tokens'].shape}")
    if 'x_norm_prompts' in img_embedding:
        print(f"Prompts imagen: {img_embedding['x_norm_prompts'].shape}")

