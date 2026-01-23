import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
import cv2
import torch
from torchvision.transforms import v2
from s3bir_dinov3_model import S3birDinov3

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

def create_patch_mask_from_image_mask(image_mask: np.ndarray, patch_size: int = 16, img_size: int = 224):
    """
    Convert image-level mask to patch-level mask.
    A patch is masked (True) if it should be IGNORED (i.e., if it's NOT in the white finding).
    Returns: torch.Tensor of shape (num_patches,) where True means patch should be masked out.
    """
    # Resize mask to img_size x img_size
    mask_resized = cv2.resize(image_mask.astype(np.float32), (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    
    # Calculate number of patches
    num_patches_h = img_size // patch_size
    num_patches_w = img_size // patch_size
    num_patches = num_patches_h * num_patches_w
    
    # Create patch-level mask
    patch_mask = np.zeros((num_patches_h, num_patches_w), dtype=bool)
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            y_start = i * patch_size
            y_end = (i + 1) * patch_size
            x_start = j * patch_size
            x_end = (j + 1) * patch_size
            
            # Check if patch contains white finding
            patch_region = mask_resized[y_start:y_end, x_start:x_end]
            # Patch is masked (True) if LESS than 50% is white finding
            white_finding_ratio = np.sum(patch_region > 127.5) / (patch_size * patch_size)
            patch_mask[i, j] = white_finding_ratio < 0.5  # True = mask out this patch
    
    # Flatten to match patch token dimensions
    patch_mask_flat = patch_mask.flatten()
    
    return torch.from_numpy(patch_mask_flat).bool()

def make_transform():
    to_tensor = v2.ToImage()
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, to_float, normalize])

def main():
    # Paths
    base_dir = "/home/ialbornoz/tesis/S3BIR-DINOv2/efilo_T1_2025-08-04_test"
    annotations_path = os.path.join(base_dir, "annotations.xml")
    images_dir = os.path.join(base_dir, "images")
    output_dir = os.path.join(base_dir, "white_finding_extraction")
    os.makedirs(output_dir, exist_ok=True)
    
    # Model checkpoint
    ckpt_path = "/home/shared_data/s3bir/saved_models/skDinoV3_sketchy.ckpt"
    if not os.path.exists(ckpt_path):
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
    
    # Find all images with "hallazgo blanco" annotation and calculate their sizes
    print("\nFinding images with 'hallazgo blanco' annotation...")
    tree = ET.parse(annotations_path)
    root = tree.getroot()
    
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
    
    # Select a candidate with reasonable size
    min_area = 1000
    min_percentage = 0.5
    
    suitable_candidates = [c for c in candidates if c['area'] >= min_area and c['percentage'] >= min_percentage]
    
    if len(suitable_candidates) == 0:
        print(f"\nWarning: No image with hallazgo blanco >= {min_area} pixels and >= {min_percentage}% found")
        print("Using the largest available...")
        suitable_candidates = candidates
    
    # Select the one with largest area
    selected = max(suitable_candidates, key=lambda x: x['area'])
    
    reference_image_name = selected['name']
    reference_image_width = selected['width']
    reference_image_height = selected['height']
    
    print(f"\n{'='*60}")
    print(f"SELECTED IMAGE: {reference_image_name}")
    print(f"  White finding area: {selected['area']} pixels ({selected['percentage']:.2f}% of image)")
    print(f"  Image dimensions: {reference_image_width} x {reference_image_height}")
    print(f"{'='*60}\n")
    
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
    
    # Find connected components to get individual white findings
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels < 2:
        print("Error: No white finding pixels found in mask")
        return
    
    # Find the largest component
    component_areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    component_areas.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nFound {num_labels - 1} white finding components:")
    for idx, (label_id, area) in enumerate(component_areas[:5]):
        print(f"  Component {label_id}: {area} pixels")
    
    # Select the largest component
    selected_label = component_areas[0][0]
    selected_area = component_areas[0][1]
    print(f"\nSelected component {selected_label} with {selected_area} pixels")
    
    # Create mask with only the selected component
    single_component_mask = np.zeros_like(mask)
    single_component_mask[labels == selected_label] = 255
    
    # Use the single component mask
    mask = single_component_mask
    
    # Resize image to 224x224 for model input
    img_resized = reference_img.resize((224, 224), Image.Resampling.LANCZOS)
    
    # Create patch-level mask (True = mask out, False = keep)
    patch_mask = create_patch_mask_from_image_mask(mask, patch_size=16, img_size=224)
    
    # Prepare image tensor
    img_tensor = transform(img_resized).unsqueeze(0).to(device)
    patch_mask_tensor = patch_mask.unsqueeze(0).to(device)
    
    print(f"\nPatch mask: {patch_mask_tensor.sum().item()} patches will be masked out (ignored)")
    print(f"          : {(patch_mask_tensor.numel() - patch_mask_tensor.sum()).item()} patches will be used (white finding)")
    
    # Extract class token using the mask
    print("\nExtracting class token with patch mask...")
    with torch.no_grad():
        # Access the encoder directly to pass the mask
        feat = model.encoder.model.forward_features(
            img_tensor, 
            masks=patch_mask_tensor,
            prompt=model.img_prompt.expand(img_tensor.shape[0], -1, -1)
        )
        cls_token = feat['x_norm_clstoken']
    
    print(f"Class token shape: {cls_token.shape}")
    print(f"Class token extracted successfully using patch mask!")
    
    # Save the class token
    cls_token_np = cls_token.cpu().numpy()
    np.save(os.path.join(output_dir, "white_finding_cls_token.npy"), cls_token_np)
    print(f"Class token saved to: {os.path.join(output_dir, 'white_finding_cls_token.npy')}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(reference_img)
    axes[0, 0].set_title(f"Original Image: {reference_image_name}\nSize: {reference_image_width}x{reference_image_height}")
    axes[0, 0].axis("off")
    
    # Mask overlay
    mask_overlay = np.array(reference_img).copy()
    mask_colored = np.zeros_like(mask_overlay)
    
    for i in range(1, num_labels):
        if i == selected_label:
            mask_colored[labels == i] = [255, 255, 0]
        else:
            mask_colored[labels == i] = [128, 128, 128]
    
    mask_overlay = cv2.addWeighted(mask_overlay, 0.7, mask_colored, 0.3, 0)
    axes[0, 1].imshow(mask_overlay)
    axes[0, 1].set_title(f"White Finding Mask Overlay\nSelected (yellow): {selected_area} pixels\nTotal components: {num_labels - 1}")
    axes[0, 1].axis("off")
    
    # Resized image with patch grid
    img_resized_np = np.array(img_resized)
    patch_size = 16
    num_patches_h = 224 // patch_size
    num_patches_w = 224 // patch_size
    
    # Draw patch grid
    img_with_grid = img_resized_np.copy()
    for i in range(num_patches_h + 1):
        y = i * patch_size
        cv2.line(img_with_grid, (0, y), (224, y), (255, 0, 0), 1)
    for j in range(num_patches_w + 1):
        x = j * patch_size
        cv2.line(img_with_grid, (x, 0), (x, 224), (255, 0, 0), 1)
    
    # Highlight masked patches in red
    patch_mask_2d = patch_mask.numpy().reshape(num_patches_h, num_patches_w)
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            if patch_mask_2d[i, j]:  # This patch is masked out
                y_start = i * patch_size
                y_end = (i + 1) * patch_size
                x_start = j * patch_size
                x_end = (j + 1) * patch_size
                img_with_grid[y_start:y_end, x_start:x_end] = (
                    img_with_grid[y_start:y_end, x_start:x_end] * 0.5 + 
                    np.array([255, 0, 0]) * 0.5
                ).astype(np.uint8)
    
    axes[1, 0].imshow(img_with_grid)
    axes[1, 0].set_title(f"Resized Image (224x224) with Patch Grid\nRed patches = masked out (ignored)\nYellow patches = used (white finding)")
    axes[1, 0].axis("off")
    
    # Patch mask visualization
    patch_mask_vis = (~patch_mask_2d).astype(float)  # Invert: True = used, False = masked
    im = axes[1, 1].imshow(patch_mask_vis, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 1].set_title(f"Patch Mask Visualization\nGreen = used patches ({np.sum(~patch_mask_2d)})\nRed = masked patches ({np.sum(patch_mask_2d)})")
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "white_finding_extraction_with_mask.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Save mask
    mask_path = os.path.join(output_dir, "white_finding_mask.png")
    Image.fromarray(mask).save(mask_path)
    print(f"Mask saved to: {mask_path}")
    
    plt.show()
    
    print(f"\n{'='*60}")
    print("White finding extraction with patch mask complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()






