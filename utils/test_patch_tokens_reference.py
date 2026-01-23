import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from s3bir_dinov3_model import S3birDinov3
import os
import cv2
import xml.etree.ElementTree as ET

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
    for hh in range(h, 1, -1):
        if n_tokens % hh == 0:
            return hh, n_tokens // hh
    raise ValueError(f"No se pudo inferir un grid HxW válido para {n_tokens} tokens.")

def parse_polygon_points(points_str: str):
    """Parse polygon points string into list of (x, y) tuples."""
    points = []
    for point_str in points_str.split(';'):
        if point_str.strip():
            x, y = map(float, point_str.strip().split(','))
            points.append((x, y))
    return points

def extract_white_finding_mask(annotations_path: str, image_name: str, img_width: int, img_height: int):
    """
    Extract mask for 'hallazgo blanco' from annotations.xml for a specific image.
    Returns a binary mask (255 for white finding, 0 otherwise).
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

def main():
    # Paths
    base_dir = "/home/ialbornoz/tesis/S3BIR-DINOv2/efilo_T1_2025-08-04_test"
    images_dir = os.path.join(base_dir, "images")
    output_dir = os.path.join(base_dir, "test_reference")
    os.makedirs(output_dir, exist_ok=True)
    
    # Reference image (the one from which we extracted the white finding)
    reference_image_name = "cropped_2025-08-04_14-14-04_EFILOME_T1_HD_frame_0036_t108s.png"
    reference_image_path = os.path.join(images_dir, reference_image_name)
    
    # Load white finding class token
    cls_token_path = os.path.join(base_dir, "white_finding_extraction", "white_finding_cls_token.npy")
    if not os.path.exists(cls_token_path):
        print(f"Error: Class token not found at {cls_token_path}")
        print("Please run extract_white_finding.py first!")
        return
    
    cls_token_np = np.load(cls_token_path)
    # Handle different shapes: could be (1, 768) or (768,)
    if cls_token_np.ndim > 1:
        cls_token_np = cls_token_np.squeeze()
    white_finding_cls_token = torch.from_numpy(cls_token_np).to(torch.float32)
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
    
    # Move class token to device and ensure correct shape
    white_finding_cls_token = white_finding_cls_token.to(device)
    if white_finding_cls_token.dim() > 1:
        white_finding_cls_token = white_finding_cls_token.squeeze()
    
    # Load reference image
    print(f"\nProcessing reference image: {reference_image_name}")
    img = Image.open(reference_image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Load white finding mask for this image
    annotations_path = os.path.join(base_dir, "annotations.xml")
    img_width, img_height = img.size
    white_finding_mask = extract_white_finding_mask(annotations_path, reference_image_name, img_width, img_height)
    
    # Resize mask to match image if needed
    if white_finding_mask.shape[:2] != (img_height, img_width):
        white_finding_mask = cv2.resize(white_finding_mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
    
    print(f"White finding mask: {np.sum(white_finding_mask > 0)} pixels ({100 * np.sum(white_finding_mask > 0) / (img_width * img_height):.2f}% of image)")
    
    # Extract patch tokens
    print("Extracting patch tokens...")
    with torch.no_grad():
        # Use model.forward() with is_training=True like in visualization.py
        # This gives us the full features dict with x_norm_clstoken and x_norm_patchtokens
        feat = model(img_tensor, dtype="image", is_training=True)
        patch_tokens = feat['x_norm_patchtokens']  # (1, N, D)
    
    print(f"Patch tokens shape: {patch_tokens.shape}")
    
    # Compute similarity: CLS(white finding) -> patches(image)
    cls_expanded = white_finding_cls_token.unsqueeze(0).unsqueeze(1).expand(-1, patch_tokens.shape[1], -1)
    similarities = F.cosine_similarity(cls_expanded, patch_tokens, dim=-1)  # (1, N)
    similarities = similarities.squeeze(0).cpu().numpy()  # (N,)
    
    # Get grid dimensions
    N_patches = patch_tokens.shape[1]
    H, W = infer_grid(N_patches)
    similarity_map = similarities.reshape(H, W)
    
    print(f"Similarity map shape: {similarity_map.shape}")
    print(f"Similarity stats: min={similarity_map.min():.4f}, max={similarity_map.max():.4f}, mean={similarity_map.mean():.4f}")
    
    # Save patch tokens
    patch_tokens_np = patch_tokens.squeeze(0).cpu().numpy()
    patch_tokens_path = os.path.join(output_dir, f"{os.path.splitext(reference_image_name)[0]}_patch_tokens.npy")
    np.save(patch_tokens_path, patch_tokens_np)
    print(f"Patch tokens saved to: {patch_tokens_path}")
    
    # Save similarity map
    similarity_path = os.path.join(output_dir, f"{os.path.splitext(reference_image_name)[0]}_similarity.npy")
    np.save(similarity_path, similarity_map)
    print(f"Similarity map saved to: {similarity_path}")
    
    # Create visualization - Compare heatmap with ground truth mask
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Resize similarity map to match original image size
    img_w, img_h = img.size
    similarity_map_resized = cv2.resize(similarity_map, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    
    # Panel 1: Original image
    axes[0].imshow(img)
    axes[0].set_title(f"Original Image\n{reference_image_name}")
    axes[0].axis("off")
    
    # Panel 2: Ground truth mask (white finding)
    axes[1].imshow(white_finding_mask, cmap="gray")
    axes[1].set_title(f"Ground Truth: White Finding Mask\n{np.sum(white_finding_mask > 0)} pixels")
    axes[1].axis("off")
    
    # Panel 3: Similarity heatmap
    im = axes[2].imshow(similarity_map_resized, cmap="viridis", vmin=similarity_map.min(), vmax=similarity_map.max())
    axes[2].set_title(f"Similarity Heatmap\nPatches vs CLS(white finding)\nMean: {similarity_map.mean():.4f}, Max: {similarity_map.max():.4f}")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="Cosine Similarity")
    
    # Also create overlay comparison
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
    
    # Show original image
    ax2.imshow(img, alpha=0.5)
    
    # Overlay similarity map
    im2 = ax2.imshow(similarity_map_resized, cmap="hot", alpha=0.6, vmin=similarity_map.min(), vmax=similarity_map.max())
    
    # Overlay ground truth mask contour
    contours, _ = cv2.findContours(white_finding_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        ax2.plot(contour[:, 0, 0], contour[:, 0, 1], 'cyan', linewidth=2, label='Ground Truth')
    
    ax2.set_title(f"Comparison: Similarity Heatmap + Ground Truth Mask\n{reference_image_name}")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Cosine Similarity")
    
    # Create masks from cosine similarity with different thresholds
    print(f"\n{'='*60}")
    print("Creating masks from cosine similarity...")
    
    # Try different threshold strategies
    thresholds = {
        '70%_max': similarity_map.max() * 0.7,
        'mean': similarity_map.mean(),
        'mean_plus_std': similarity_map.mean() + similarity_map.std(),
        'median': np.median(similarity_map),
        'otsu': None  # Will use Otsu's method
    }
    
    # Calculate Otsu threshold
    similarity_map_uint8 = ((similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min()) * 255).astype(np.uint8)
    otsu_threshold_value, _ = cv2.threshold(similarity_map_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholds['otsu'] = similarity_map.min() + (otsu_threshold_value / 255.0) * (similarity_map.max() - similarity_map.min())
    
    # Create visualization with different threshold masks
    fig3, axes3 = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes3[0, 0].imshow(img)
    axes3[0, 0].set_title("Original Image")
    axes3[0, 0].axis("off")
    
    # Ground truth mask
    axes3[0, 1].imshow(img)
    axes3[0, 1].imshow(white_finding_mask, cmap='Reds', alpha=0.5)
    axes3[0, 1].set_title("Ground Truth: White Finding")
    axes3[0, 1].axis("off")
    
    # Similarity heatmap
    im3 = axes3[0, 2].imshow(similarity_map_resized, cmap="viridis", vmin=similarity_map.min(), vmax=similarity_map.max())
    axes3[0, 2].set_title("Cosine Similarity Heatmap")
    axes3[0, 2].axis("off")
    plt.colorbar(im3, ax=axes3[0, 2], fraction=0.046, pad=0.04)
    
    # For each threshold, create mask and find contours
    threshold_results = {}
    for idx, (threshold_name, threshold_value) in enumerate(thresholds.items()):
        if threshold_value is None:
            continue
            
        # Create binary mask
        binary_mask = (similarity_map_resized > threshold_value).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours (noise)
        min_area = 100  # pixels
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # Store results
        threshold_results[threshold_name] = {
            'threshold': threshold_value,
            'mask': binary_mask,
            'contours': filtered_contours,
            'num_regions': len(filtered_contours),
            'total_area': np.sum(binary_mask > 0)
        }
        
        # Visualize on image
        row = 1
        col = idx % 3
        if idx < 3:
            ax = axes3[row, col]
        else:
            # Create additional figure if needed
            continue
            
        ax.imshow(img)
        
        # Draw contours as polygons
        for contour in filtered_contours:
            # Simplify contour to reduce points
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            ax.plot(approx[:, 0, 0], approx[:, 0, 1], 'lime', linewidth=2)
            # Fill polygon
            ax.fill(approx[:, 0, 0], approx[:, 0, 1], color='lime', alpha=0.3)
        
        ax.set_title(f"Detected Regions: {threshold_name}\nThreshold: {threshold_value:.4f}\nRegions: {len(filtered_contours)}, Area: {np.sum(binary_mask > 0)} px")
        ax.axis("off")
        
        print(f"  {threshold_name}: threshold={threshold_value:.4f}, regions={len(filtered_contours)}, area={np.sum(binary_mask > 0)} pixels")
    
    plt.tight_layout()
    output_path3 = os.path.join(output_dir, "detected_regions_comparison.png")
    plt.savefig(output_path3, dpi=150, bbox_inches='tight')
    print(f"Detected regions comparison saved to: {output_path3}")
    plt.close()
    
    # Create detailed visualization for best threshold (mean + std)
    best_threshold_name = 'mean_plus_std'
    if best_threshold_name in threshold_results:
        best_result = threshold_results[best_threshold_name]
        
        fig4, axes4 = plt.subplots(1, 2, figsize=(16, 8))
        
        # Side by side: Ground truth vs Detected
        axes4[0].imshow(img)
        axes4[0].imshow(white_finding_mask, cmap='Reds', alpha=0.5)
        axes4[0].set_title("Ground Truth: White Finding")
        axes4[0].axis("off")
        
        axes4[1].imshow(img)
        for contour in best_result['contours']:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            axes4[1].plot(approx[:, 0, 0], approx[:, 0, 1], 'lime', linewidth=3)
            axes4[1].fill(approx[:, 0, 0], approx[:, 0, 1], color='lime', alpha=0.4)
        axes4[1].set_title(f"Detected Regions (threshold: {best_result['threshold']:.4f})\n{best_result['num_regions']} regions, {best_result['total_area']} pixels")
        axes4[1].axis("off")
        
        plt.tight_layout()
        output_path4 = os.path.join(output_dir, "gt_vs_detected.png")
        plt.savefig(output_path4, dpi=150, bbox_inches='tight')
        print(f"Ground truth vs Detected saved to: {output_path4}")
        plt.close()
    
    # Calculate overlap statistics for best threshold
    similarity_threshold = thresholds['mean_plus_std']
    high_similarity_mask = (similarity_map_resized > similarity_threshold).astype(np.uint8) * 255
    
    # Calculate overlap
    overlap = np.logical_and(white_finding_mask > 0, high_similarity_mask > 0)
    overlap_pixels = np.sum(overlap)
    gt_pixels = np.sum(white_finding_mask > 0)
    high_sim_pixels = np.sum(high_similarity_mask > 0)
    
    if gt_pixels > 0:
        recall = overlap_pixels / gt_pixels  # How much of GT is covered
    else:
        recall = 0.0
    
    if high_sim_pixels > 0:
        precision = overlap_pixels / high_sim_pixels  # How much of high sim is in GT
    else:
        precision = 0.0
    
    print(f"\n{'='*60}")
    print("Comparison Statistics (using mean+std threshold):")
    print(f"  Ground truth pixels: {gt_pixels}")
    print(f"  High similarity pixels (>{similarity_threshold:.4f}): {high_sim_pixels}")
    print(f"  Overlap pixels: {overlap_pixels}")
    print(f"  Recall (GT coverage): {recall:.4f} ({recall*100:.2f}%)")
    print(f"  Precision (high sim in GT): {precision:.4f} ({precision*100:.2f}%)")
    print(f"{'='*60}")
    
    plt.tight_layout()
    
    output_path1 = os.path.join(output_dir, "reference_comparison.png")
    plt.savefig(output_path1, dpi=150, bbox_inches='tight')
    print(f"Comparison visualization saved to: {output_path1}")
    plt.close()
    
    output_path2 = os.path.join(output_dir, "reference_overlay.png")
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Overlay visualization saved to: {output_path2}")
    plt.close()
    
    print(f"\n{'='*60}")
    print("Test complete!")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

