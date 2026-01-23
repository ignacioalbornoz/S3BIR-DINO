import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from s3bir_dinov3_model import S3birDinov3
import xml.etree.ElementTree as ET
import os
from pathlib import Path
from typing import List, Tuple
import cv2

def make_transform():
    to_tensor = v2.ToImage()
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, to_float, normalize])

def parse_polygon_points(points_str: str) -> List[Tuple[float, float]]:
    """Parse polygon points string into list of (x, y) tuples."""
    points = []
    for point_str in points_str.split(';'):
        if point_str.strip():
            x, y = map(float, point_str.strip().split(','))
            points.append((x, y))
    return points

def extract_white_finding_mask(annotations_path: str, image_name: str, img_width: int, img_height: int) -> np.ndarray:
    """
    Extract mask for 'hallazgo blanco' from annotations.xml for a specific image.
    Returns a binary mask (1 for white finding, 0 otherwise).
    """
    tree = ET.parse(annotations_path)
    root = tree.getroot()
    
    # Find the image element
    image_element = None
    for image in root.findall('image'):
        if image.get('name') == image_name:
            image_element = image
            break
    
    if image_element is None:
        print(f"Image {image_name} not found in annotations")
        return np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Create mask
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Find all polygons labeled "hallazgo blanco"
    for polygon in image_element.findall('polygon'):
        if polygon.get('label') == 'hallazgo blanco':
            points_str = polygon.get('points', '')
            points = parse_polygon_points(points_str)
            
            # Convert to numpy array for cv2.fillPoly
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
    
    return mask

def create_patch_mask_from_image_mask(image_mask: np.ndarray, patch_size: int = 16) -> torch.Tensor:
    """
    Convert image-level mask to patch-level mask.
    A patch is masked (True) if more than 50% of its area is covered by the image mask.
    """
    h, w = image_mask.shape
    patch_h = h // patch_size
    patch_w = w // patch_size
    
    # Resize mask to patch grid size
    mask_resized = cv2.resize(image_mask.astype(np.float32), (patch_w, patch_h), interpolation=cv2.INTER_AREA)
    
    # Threshold: patch is masked if > 50% covered
    patch_mask = (mask_resized > 127.5).astype(bool)
    
    # Flatten to match patch token dimensions
    patch_mask_flat = patch_mask.flatten()
    
    return torch.from_numpy(patch_mask_flat).bool()

def infer_grid(n_tokens: int, h_hint: int = None, w_hint: int = None):
    """Infiera grid HxW; si no hay hints, asume cuadrado."""
    if h_hint is not None and w_hint is not None:
        assert h_hint * w_hint == n_tokens, "Hints de grid no coinciden con N tokens."
        return h_hint, w_hint
    h = int(np.sqrt(n_tokens))
    if h * h == n_tokens:
        return h, h
    # fallback: busca factores
    for hh in range(h, 1, -1):
        if n_tokens % hh == 0:
            return hh, n_tokens // hh
    raise ValueError("No se pudo inferir un grid HxW válido para N tokens.")

def visualize_cls_patch_comparison(img_path: str, img: Image.Image, 
                                  cls_mask_token: torch.Tensor,
                                  patch_tokens: torch.Tensor,
                                  save_path: str = None):
    """
    Visualize similarity between class token from masked white finding 
    and patch tokens from an image.
    """
    # Compute similarity: CLS(mask) -> patches(image)
    sim_cls_mask_patches = F.cosine_similarity(
        cls_mask_token.unsqueeze(0).unsqueeze(1).expand(-1, patch_tokens.shape[1], -1), 
        patch_tokens, 
        dim=-1
    )
    
    N_patches = patch_tokens.shape[1]
    H, W = infer_grid(N_patches)
    similarity_map = sim_cls_mask_patches.view(H, W).detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(img)
    axes[0].set_title(f"Image: {os.path.basename(img_path)}")
    axes[0].axis("off")
    
    im = axes[1].imshow(similarity_map, cmap="viridis")
    axes[1].set_title("Similarity: CLS(white finding) -> patches(image)")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return similarity_map

def main():
    # Paths
    base_dir = "/home/ialbornoz/tesis/S3BIR-DINOv2/efilo_T1_2025-08-04_test"
    annotations_path = os.path.join(base_dir, "annotations.xml")
    images_dir = os.path.join(base_dir, "images")
    output_dir = os.path.join(base_dir, "white_finding_comparisons")
    os.makedirs(output_dir, exist_ok=True)
    
    # Model checkpoint
    ckpt_path = "/home/shared_data/s3bir/saved_models/skDinoV3_sketchy.ckpt"
    if not os.path.exists(ckpt_path):
        # Try alternative path
        ckpt_path = "/home/ialbornoz/tesis/S3BIR-DINOv2/s3bir_dinov3.ckpt"
    
    # Load model
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    # Find an image with "hallazgo blanco" annotation of reasonable size
    print("Finding image with 'hallazgo blanco' annotation of reasonable size...")
    tree = ET.parse(annotations_path)
    root = tree.getroot()
    
    # Find all images with hallazgo blanco and calculate their sizes
    candidates = []
    for image in root.findall('image'):
        img_name = image.get('name')
        img_width = int(image.get('width'))
        img_height = int(image.get('height'))
        
        # Extract mask for this image
        mask = extract_white_finding_mask(annotations_path, img_name, img_width, img_height)
        
        # Calculate area of white finding (in pixels)
        white_finding_area = np.sum(mask > 0)
        total_image_area = img_width * img_height
        percentage = (white_finding_area / total_image_area) * 100
        
        if white_finding_area > 0:
            candidates.append({
                'name': img_name,
                'width': img_width,
                'height': img_height,
                'area': white_finding_area,
                'percentage': percentage
            })
            print(f"  Found: {img_name} - Area: {white_finding_area} pixels ({percentage:.2f}% of image)")
    
    if len(candidates) == 0:
        print("Error: No image with 'hallazgo blanco' annotation found")
        return
    
    # Select a candidate with reasonable size (at least 0.5% of image area and at least 1000 pixels)
    min_area = 1000
    min_percentage = 0.5
    
    suitable_candidates = [c for c in candidates if c['area'] >= min_area and c['percentage'] >= min_percentage]
    
    if len(suitable_candidates) == 0:
        print(f"Warning: No image with hallazgo blanco >= {min_area} pixels and >= {min_percentage}% found")
        print("Using the largest available...")
        suitable_candidates = candidates
    
    # Select the one with largest area
    selected = max(suitable_candidates, key=lambda x: x['area'])
    
    reference_image_name = selected['name']
    reference_image_width = selected['width']
    reference_image_height = selected['height']
    
    print(f"\nSelected reference image: {reference_image_name}")
    print(f"  White finding area: {selected['area']} pixels ({selected['percentage']:.2f}% of image)")
    
    # Load reference image and extract mask
    reference_image_path = os.path.join(images_dir, reference_image_name)
    if not os.path.exists(reference_image_path):
        print(f"Error: Reference image not found at {reference_image_path}")
        return
    
    reference_img = Image.open(reference_image_path).convert("RGB")
    mask = extract_white_finding_mask(annotations_path, reference_image_name, 
                                     reference_image_width, reference_image_height)
    
    # Resize mask to match image if needed
    if mask.shape[:2] != reference_img.size[::-1]:  # PIL size is (W, H)
        mask = cv2.resize(mask, reference_img.size, interpolation=cv2.INTER_NEAREST)
    
    # Create patch-level mask
    patch_mask = create_patch_mask_from_image_mask(mask, patch_size=16)
    
    # Prepare reference image tensor
    reference_tensor = transform(reference_img).unsqueeze(0).to(device)
    
    # Extract class token from white finding region
    print("Extracting class token from white finding region...")
    
    # Find bounding box of white finding
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        print("Error: No white finding pixels found in mask")
        return
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    # Add some padding
    padding = 10
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(reference_img.size[0], x_max + padding)
    y_max = min(reference_img.size[1], y_max + padding)
    
    # Crop the white finding region
    white_finding_crop = reference_img.crop((x_min, y_min, x_max, y_max))
    
    # Resize to model input size (224x224)
    white_finding_crop = white_finding_crop.resize((224, 224), Image.Resampling.LANCZOS)
    
    # Extract class token from cropped region
    with torch.no_grad():
        crop_tensor = transform(white_finding_crop).unsqueeze(0).to(device)
        crop_embedding = model(crop_tensor, dtype="image", is_training=True)
        cls_mask_token = crop_embedding['x_norm_clstoken']  # (1, D)
    
    print(f"Class token shape: {cls_mask_token.shape}")
    
    # Save visualization of extracted white finding
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(reference_img)
    axes[0].set_title(f"Reference Image: {reference_image_name}")
    axes[0].axis("off")
    
    # Show mask overlay
    mask_overlay = np.array(reference_img).copy()
    mask_colored = np.zeros_like(mask_overlay)
    mask_colored[mask > 0] = [255, 255, 0]  # Yellow overlay
    mask_overlay = cv2.addWeighted(mask_overlay, 0.7, mask_colored, 0.3, 0)
    axes[1].imshow(mask_overlay)
    axes[1].set_title("White Finding Mask Overlay")
    axes[1].axis("off")
    
    axes[2].imshow(white_finding_crop)
    axes[2].set_title("Extracted White Finding Region")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "white_finding_extraction.png"), dpi=150, bbox_inches='tight')
    print(f"White finding extraction saved to {os.path.join(output_dir, 'white_finding_extraction.png')}")
    plt.close()
    
    # Process all images in the folder
    print(f"\nProcessing all images in {images_dir}...")
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    all_similarities = []
    all_image_names = []
    
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(images_dir, img_file)
        print(f"Processing {idx+1}/{len(image_files)}: {img_file}")
        
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                img_embedding = model(img_tensor, dtype="image", is_training=True)
                patch_tokens = img_embedding['x_norm_patchtokens']  # (1, N, D)
            
            # Compute similarity map
            similarity_map = visualize_cls_patch_comparison(
                img_path, img, cls_mask_token, patch_tokens,
                save_path=os.path.join(output_dir, f"{Path(img_file).stem}_similarity.png")
            )
            
            all_similarities.append(similarity_map)
            all_image_names.append(img_file)
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    # Create summary visualization
    print("\nCreating summary visualization...")
    n_images = len(all_similarities)
    if n_images > 0:
        # Compute average similarity per image
        avg_similarities = [np.mean(sim) for sim in all_similarities]
        max_similarities = [np.max(sim) for sim in all_similarities]
        
        # Plot summary
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Average similarities
        axes[0].plot(avg_similarities, marker='o', linestyle='-')
        axes[0].set_title("Average Similarity: CLS(white finding) -> patches(image)")
        axes[0].set_xlabel("Image Index")
        axes[0].set_ylabel("Average Similarity")
        axes[0].grid(True)
        
        # Max similarities
        axes[1].plot(max_similarities, marker='s', linestyle='-', color='orange')
        axes[1].set_title("Max Similarity: CLS(white finding) -> patches(image)")
        axes[1].set_xlabel("Image Index")
        axes[1].set_ylabel("Max Similarity")
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "summary_similarities.png"), dpi=150, bbox_inches='tight')
        print(f"Summary saved to {os.path.join(output_dir, 'summary_similarities.png')}")
        plt.close()
        
        # Save top matches
        top_indices = np.argsort(avg_similarities)[-10:][::-1]
        print("\nTop 10 images by average similarity:")
        for i, idx in enumerate(top_indices):
            print(f"{i+1}. {all_image_names[idx]}: {avg_similarities[idx]:.4f}")
    
    print(f"\nDone! Results saved in {output_dir}")

if __name__ == "__main__":
    main()

