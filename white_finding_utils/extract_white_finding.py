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

def find_largest_square_in_mask(mask):
    """
    Find the largest square that fits completely inside the mask.
    Returns: (x, y, size) where (x, y) is top-left corner and size is side length.
    """
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None, None, 0
    
    # Get bounding box
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    # Maximum possible square size is limited by the smaller dimension of bbox
    max_possible_size = min(y_max - y_min + 1, x_max - x_min + 1)
    
    # Search for the largest square that fits completely in the mask
    best_size = 0
    best_x = None
    best_y = None
    
    # Try different square sizes (from largest to smallest)
    for size in range(max_possible_size, 0, -1):
        # Try different positions within the bounding box
        for y in range(y_min, y_max - size + 2):
            for x in range(x_min, x_max - size + 2):
                # Check if the square is completely inside the mask
                square_region = mask[y:y+size, x:x+size]
                if np.all(square_region > 0):  # All pixels in square are white finding
                    best_size = size
                    best_x = x
                    best_y = y
                    return best_x, best_y, best_size
    
    # If no square found, return None
    return None, None, 0

def create_patch_mask_from_image_mask(image_mask, patch_size=16, img_size=224):
    """
    Convert image-level mask to patch-level mask.
    True = mask out (ignore), False = use
    """
    # Resize mask to img_size x img_size
    mask_resized = cv2.resize(image_mask.astype(np.float32), (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    
    num_patches_h = img_size // patch_size
    num_patches_w = img_size // patch_size
    
    patch_mask = np.zeros((num_patches_h, num_patches_w), dtype=bool)
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            y_start = i * patch_size
            y_end = (i + 1) * patch_size
            x_start = j * patch_size
            x_end = (j + 1) * patch_size
            
            patch_region = mask_resized[y_start:y_end, x_start:x_end]
            # True = mask out if LESS than 50% is white finding
            white_finding_ratio = np.sum(patch_region > 127.5) / (patch_size * patch_size)
            patch_mask[i, j] = white_finding_ratio < 0.5
    
    return torch.from_numpy(patch_mask.flatten()).bool()

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

def main():
    # Paths
    base_dir = "/home/ialbornoz/tesis/S3BIR-DINOv2/efilo_T1_2025-08-04_test"
    annotations_path = os.path.join(base_dir, "annotations.xml")
    images_dir = os.path.join(base_dir, "images")
    output_dir = os.path.join(base_dir, "white_finding_extraction")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images with "hallazgo blanco" annotation and calculate their sizes
    print("Finding images with 'hallazgo blanco' annotation...")
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
            print(f"  Found: {img_name}")
            print(f"    Area: {white_finding_area} pixels ({percentage:.2f}% of image)")
    
    if len(candidates) == 0:
        print("Error: No image with 'hallazgo blanco' annotation found")
        return
    
    # Select a candidate with reasonable size (at least 0.5% of image area and at least 1000 pixels)
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
    
    if num_labels < 2:  # 0 is background, so we need at least 2 (background + 1 component)
        print("Error: No white finding pixels found in mask")
        return
    
    # Find the largest component (excluding background which is label 0)
    component_areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    component_areas.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nFound {num_labels - 1} white finding components:")
    for idx, (label_id, area) in enumerate(component_areas[:5]):  # Show top 5
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
    
    # 1. Find the largest square inside the mask
    print("\nFinding largest square inside white finding mask...")
    square_x, square_y, square_size = find_largest_square_in_mask(mask)
    
    if square_size == 0:
        print("Error: Could not find a square inside the mask")
        return
    
    print(f"Largest square found:")
    print(f"  Top-left corner: ({square_x}, {square_y})")
    print(f"  Size: {square_size} x {square_size} pixels")
    print(f"  Area: {square_size * square_size} pixels")
    
    # 2. Crop the square from original image
    square_crop = reference_img.crop((square_x, square_y, square_x + square_size, square_y + square_size))
    
    # 3. Crop the mask to the same square size
    mask_square = mask[square_y:square_y+square_size, square_x:square_x+square_size]
    
    # 4. Apply mask: set pixels outside white finding to black
    img_array = np.array(square_crop)
    mask_3d = (mask_square > 0)[:, :, np.newaxis]
    masked_img_array = img_array * mask_3d  # Pixels outside mask = black
    masked_img = Image.fromarray(masked_img_array.astype(np.uint8))
    
    # 6. Resize to 224x224 (model input size)
    img_resized = masked_img.resize((224, 224), Image.Resampling.LANCZOS)
    
    print(f"Cropped and masked image resized to: {img_resized.size[0]}x{img_resized.size[1]}")
    print(f"White finding pixels in square: {np.sum(mask_square > 0)} ({100 * np.sum(mask_square > 0) / mask_square.size:.2f}% of square)")
    
    # Load model
    print("\nLoading model...")
    
    # Model checkpoint
    ckpt_path = "/home/shared_data/s3bir/saved_models/skDinoV3_sketchy.ckpt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "/home/ialbornoz/tesis/S3BIR-DINOv2/s3bir_dinov3.ckpt"
    
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
    
    # Transform
    to_tensor = v2.ToImage()
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    transform = v2.Compose([to_tensor, to_float, normalize])
    
    # Prepare image tensor (already masked, no need for patch mask)
    img_tensor = transform(img_resized).unsqueeze(0).to(device)
    
    # Extract class token using dtype="image" with is_training=True
    print("\nExtracting class token as 'image' from cropped and masked image...")
    with torch.no_grad():
        # Use model.forward() with is_training=True like in visualization.py
        # This gives us the full features dict with x_norm_clstoken and x_norm_patchtokens
        feat = model(img_tensor, dtype="image", is_training=True)
        cls_token = feat['x_norm_clstoken']
    
    print(f"Class token shape: {cls_token.shape}")
    
    # Save the class token
    cls_token_np = cls_token.cpu().numpy()
    np.save(os.path.join(output_dir, "white_finding_cls_token.npy"), cls_token_np)
    print(f"Class token saved to: {os.path.join(output_dir, 'white_finding_cls_token.npy')}")
    
    print(f"\nWhite finding area: {np.sum(mask > 0)} pixels")
    print(f"Final image for model: cropped bbox, masked, resized to 224x224")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(reference_img)
    axes[0, 0].set_title(f"Original Image: {reference_image_name}\nSize: {reference_image_width}x{reference_image_height}")
    axes[0, 0].axis("off")
    
    # Mask overlay (show all components in different colors, highlight selected)
    mask_overlay = np.array(reference_img).copy()
    mask_colored = np.zeros_like(mask_overlay)
    
    # Show all components in different colors
    for i in range(1, num_labels):
        if i == selected_label:
            # Highlight selected component in yellow
            mask_colored[labels == i] = [255, 255, 0]
        else:
            # Other components in light gray
            mask_colored[labels == i] = [128, 128, 128]
    
    mask_overlay = cv2.addWeighted(mask_overlay, 0.7, mask_colored, 0.3, 0)
    
    # Draw largest square on overlay
    cv2.rectangle(mask_overlay, (square_x, square_y), (square_x + square_size, square_y + square_size), (0, 255, 0), 3)
    
    axes[0, 1].imshow(mask_overlay)
    axes[0, 1].set_title(f"White Finding Mask Overlay + Largest Square\nSelected (yellow): {selected_area} pixels\nSquare (green): {square_size}x{square_size} at ({square_x}, {square_y})")
    axes[0, 1].axis("off")
    
    # Original with square
    img_with_square = np.array(reference_img).copy()
    cv2.rectangle(img_with_square, (square_x, square_y), (square_x + square_size, square_y + square_size), (0, 255, 0), 3)
    axes[0, 2].imshow(img_with_square)
    axes[0, 2].set_title(f"Largest Square on Original\nSize: {square_size}x{square_size}")
    axes[0, 2].axis("off")
    
    # Show square crop
    axes[1, 0].imshow(square_crop)
    axes[1, 0].set_title(f"Step 1: Square Crop\nSize: {square_crop.size[0]}x{square_crop.size[1]}")
    axes[1, 0].axis("off")
    
    # Show masked crop (before resize)
    axes[1, 1].imshow(masked_img)
    axes[1, 1].set_title(f"Step 2: Masked Crop\n(pixels outside mask = black)\nSize: {masked_img.size[0]}x{masked_img.size[1]}")
    axes[1, 1].axis("off")
    
    # Show final resized image (what goes to model)
    axes[1, 2].imshow(img_resized)
    axes[1, 2].set_title(f"Step 3: Final Image for Model\nResized to 224x224\n(Only white finding, rest is black)")
    axes[1, 2].axis("off")
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "white_finding_extraction.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Also save the resized image (for reference)
    img_resized_path = os.path.join(output_dir, "white_finding_resized.png")
    img_resized.save(img_resized_path)
    print(f"Resized image saved to: {img_resized_path}")
    
    # Save mask
    mask_path = os.path.join(output_dir, "white_finding_mask.png")
    Image.fromarray(mask).save(mask_path)
    print(f"Mask saved to: {mask_path}")
    
    plt.show()
    
    print(f"\n{'='*60}")
    print("White finding extraction complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

