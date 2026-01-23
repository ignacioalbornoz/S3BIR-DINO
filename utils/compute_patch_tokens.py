import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import v2
from s3bir_dinov3_model import S3birDinov3
import os
from pathlib import Path
from tqdm import tqdm

def make_transform():
    to_tensor = v2.ToImage()
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, to_float, normalize])

def infer_grid(n_tokens: int):
    """Infer grid HxW from number of tokens."""
    h = int(np.sqrt(n_tokens))
    if h * h == n_tokens:
        return h, h
    # fallback: busca factores
    for hh in range(h, 1, -1):
        if n_tokens % hh == 0:
            return hh, n_tokens // hh
    raise ValueError(f"No se pudo inferir un grid HxW válido para {n_tokens} tokens.")

def main():
    # Paths
    base_dir = "/home/ialbornoz/tesis/S3BIR-DINOv2/efilo_T1_2025-08-04_test"
    images_dir = os.path.join(base_dir, "images")
    output_dir = os.path.join(base_dir, "patch_tokens")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load white finding class token
    cls_token_path = os.path.join(base_dir, "white_finding_extraction", "white_finding_cls_token.npy")
    if not os.path.exists(cls_token_path):
        print(f"Error: Class token not found at {cls_token_path}")
        print("Please run extract_white_finding.py first!")
        return
    
    white_finding_cls_token = torch.from_numpy(np.load(cls_token_path)).to(torch.float32)
    print(f"Loaded white finding class token: shape {white_finding_cls_token.shape}")
    
    # Model checkpoint
    ckpt_path = "/home/shared_data/s3bir/saved_models/skDinoV3_sketchy.ckpt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "/home/ialbornoz/tesis/S3BIR-DINOv2/s3bir_dinov3.ckpt"
    
    # Load model
    print("\nLoading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = S3birDinov3().to(device)
    
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"Model loaded from {ckpt_path}")
    else:
        print(f"Warning: Checkpoint not found at {ckpt_path}, using untrained model")
    
    model.eval()
    transform = make_transform()
    
    # Move class token to device
    white_finding_cls_token = white_finding_cls_token.to(device)
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(images_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"\nFound {len(image_files)} images to process")
    
    # Storage for results
    all_similarities = []
    all_image_names = []
    all_patch_tokens = {}
    
    # Process each image
    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(images_dir, img_file)
        
        try:
            # Load and preprocess image
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Extract patch tokens
            with torch.no_grad():
                img_embedding = model(img_tensor, dtype="image", is_training=True)
                patch_tokens = img_embedding['x_norm_patchtokens']  # (1, N, D)
            
            # Compute similarity: CLS(white finding) -> patches(image)
            # Expand class token to match patch tokens shape
            cls_expanded = white_finding_cls_token.unsqueeze(0).unsqueeze(1).expand(-1, patch_tokens.shape[1], -1)
            similarities = F.cosine_similarity(cls_expanded, patch_tokens, dim=-1)  # (1, N)
            similarities = similarities.squeeze(0).cpu().numpy()  # (N,)
            
            # Get grid dimensions
            N_patches = patch_tokens.shape[1]
            H, W = infer_grid(N_patches)
            similarity_map = similarities.reshape(H, W)
            
            # Store results
            all_similarities.append(similarity_map)
            all_image_names.append(img_file)
            
            # Save patch tokens for this image
            patch_tokens_np = patch_tokens.squeeze(0).cpu().numpy()  # (N, D)
            patch_tokens_path = os.path.join(output_dir, f"{Path(img_file).stem}_patch_tokens.npy")
            np.save(patch_tokens_path, patch_tokens_np)
            
            # Also save similarity map
            similarity_path = os.path.join(output_dir, f"{Path(img_file).stem}_similarity.npy")
            np.save(similarity_path, similarity_map)
            
        except Exception as e:
            print(f"\nError processing {img_file}: {e}")
            continue
    
    # Save summary
    summary = {
        'image_names': all_image_names,
        'num_images': len(all_image_names),
        'white_finding_cls_token_path': cls_token_path
    }
    
    # Compute statistics
    avg_similarities = [np.mean(sim) for sim in all_similarities]
    max_similarities = [np.max(sim) for sim in all_similarities]
    min_similarities = [np.min(sim) for sim in all_similarities]
    
    summary['avg_similarities'] = avg_similarities
    summary['max_similarities'] = max_similarities
    summary['min_similarities'] = min_similarities
    
    # Save summary
    summary_path = os.path.join(output_dir, "summary.npz")
    np.savez(summary_path, **summary)
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"Processed {len(all_image_names)} images")
    print(f"Patch tokens saved in: {output_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")
    
    # Print top matches
    if len(avg_similarities) > 0:
        top_indices = np.argsort(avg_similarities)[-10:][::-1]
        print("\nTop 10 images by average similarity:")
        for i, idx in enumerate(top_indices):
            print(f"{i+1:2d}. {all_image_names[idx]:60s} - Avg: {avg_similarities[idx]:.4f}, Max: {max_similarities[idx]:.4f}")

if __name__ == "__main__":
    main()





