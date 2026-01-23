import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from s3bir_dinov3_model import S3birDinov3
import os

def make_transform():
    to_tensor = v2.ToImage()
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, to_float, normalize])

def visualize_cls_patch(img1, img2, cls_img1, cls_img2, patch_img1, patch_img2, save_path=None):
    # CLS-to-patches y patches-to-CLS
    sim_cls_img1_patch_img2 = F.cosine_similarity(cls_img1.unsqueeze(1).expand(-1, patch_img2.shape[1], -1), patch_img2, dim=-1)
    sim_cls_img2_patch_img1 = F.cosine_similarity(cls_img2.unsqueeze(1).expand(-1, patch_img1.shape[1], -1), patch_img1, dim=-1)

    # Calcular grids para cada imagen por separado
    N_img1 = patch_img1.shape[1]
    N_img2 = patch_img2.shape[1]
    H_img1, W_img1 = infer_grid(N_img1)
    H_img2, W_img2 = infer_grid(N_img2)
    
    map12 = sim_cls_img1_patch_img2.view(H_img2, W_img2).detach().cpu().numpy()
    map21 = sim_cls_img2_patch_img1.view(H_img1, W_img1).detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))

    axes[0, 0].imshow(img1)
    axes[0, 0].set_title("image 1")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img2)
    axes[0, 1].set_title("image 2")
    axes[0, 1].axis("off")

    im2 = axes[1, 0].imshow(map21, cmap="viridis")
    axes[1, 0].set_title("Similarity: CLS(image2) -> patches(image1)")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im1 = axes[1, 1].imshow(map12, cmap="viridis")
    axes[1, 1].set_title("Similarity: CLS(image1) -> patches(image2)")
    plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualización guardada en: {save_path}")
    
    plt.show()
    plt.close()

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

def idx_to_rc(idx: int, grid_h: int, grid_w: int):
    """Convierte índice lineal a (row, col)."""
    assert 0 <= idx < grid_h * grid_w, "Índice fuera de rango."
    r = idx // grid_w
    c = idx % grid_w
    return r, c

def plot_with_red_dot(ax, pil_img, idx: int, grid_h: int, grid_w: int, title: str):
    """Muestra imagen y dibuja punto en el centro del patch idx."""
    ax.imshow(pil_img)
    ax.set_title(title)
    ax.axis("off")
    w, h = pil_img.size
    cell_w = w / grid_w
    cell_h = h / grid_h
    r, c = idx_to_rc(idx, grid_h, grid_w)
    x = (c + 0.5) * cell_w
    y = (r + 0.5) * cell_h
    ax.scatter([x], [y], s=60, c="red", marker="o", edgecolors="white", linewidths=1.5)

def patch_to_patches_similarity(query_patch_vec: torch.Tensor,
                                target_patches: torch.Tensor) -> torch.Tensor:
    """
    query_patch_vec: (D,) o (1,D)
    target_patches: (1, N, D)
    retorna: (N,) similitudes coseno
    """
    if query_patch_vec.dim() == 1:
        query_patch_vec = query_patch_vec.unsqueeze(0)  # (1, D)
    query_expand = query_patch_vec.unsqueeze(1).expand(-1, target_patches.shape[1], -1)  # (1, N, D)
    sim = F.cosine_similarity(target_patches, query_expand, dim=-1)  # (1, N)
    return sim.squeeze(0)  # (N,)

def visualize_patch_matching(img1, img2,
                             patch_img1, patch_img2,
                             img1_patch_idx: int,
                             img2_patch_idx: int,
                             grid_img1: tuple = None,
                             grid_img2: tuple = None,
                             cmap="viridis",
                             save_path=None):
    """
      [0,0] Imagen 1 con punto (img1_patch_idx)
      [0,1] Imagen 2 con punto (img2_patch_idx)
      [1,0] fmap: patch(Imagen1) -> patches(Imagen2)
      [1,1] fmap: patch(Imagen2) -> patches(Imagen1)
    """
    # Grids
    N_img1 = patch_img1.shape[1]
    N_img2 = patch_img2.shape[1]
    H_img1, W_img1 = infer_grid(N_img1, *(grid_img1 or (None, None)))
    H_img2, W_img2 = infer_grid(N_img2, *(grid_img2 or (None, None)))

    # Similitud: patch(Imagen1) -> patches(Imagen2)
    q_vec_img1 = patch_img1[0, img1_patch_idx, :]  # (D,)
    sim_img1_to_img2 = patch_to_patches_similarity(q_vec_img1, patch_img2)  # (N_img2,)
    fmap_img1_to_img2 = sim_img1_to_img2.view(H_img2, W_img2).detach().cpu().numpy()

    # Similitud: patch(Imagen2) -> patches(Imagen1)
    q_vec_img2 = patch_img2[0, img2_patch_idx, :]  # (D,)
    sim_img2_to_img1 = patch_to_patches_similarity(q_vec_img2, patch_img1)  # (N_img1,)
    fmap_img2_to_img1 = sim_img2_to_img1.view(H_img1, W_img1).detach().cpu().numpy()

    # Figura
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))

    plot_with_red_dot(axes[0, 0], img1, img1_patch_idx, H_img1, W_img1, "image 1 (selected patch)")
    plot_with_red_dot(axes[0, 1], img2, img2_patch_idx, H_img2, W_img2, "image 2 (selected patch)")

    im_right = axes[1, 0].imshow(fmap_img2_to_img1, cmap=cmap)
    axes[1, 0].set_title("Similarity: patch(image2) -> patches(image1)")
    plt.colorbar(im_right, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im_left = axes[1, 1].imshow(fmap_img1_to_img2, cmap=cmap)
    axes[1, 1].set_title("Similarity: patch(image1) -> patches(image2)")
    plt.colorbar(im_left, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualización guardada en: {save_path}")
    
    plt.show()
    plt.close()

def visualize_patch_tokens_vs_cls(frame_img, gato_cls_token, frame_patch_tokens, save_path=None, cmap="viridis", alpha=0.6):
    """
    Calcula la similitud entre cada patch token del frame y el CLS token de gato.jpg
    y crea un mapa de calor del tamaño de la imagen original del frame.
    
    Args:
        frame_img: PIL Image del frame
        gato_cls_token: CLS token de gato.jpg (tensor de shape [1, D])
        frame_patch_tokens: Patch tokens del frame (tensor de shape [1, N, D])
        save_path: Ruta para guardar la visualización
        cmap: Colormap para el mapa de calor
        alpha: Transparencia del mapa de calor sobre la imagen
    """
    # Asegurar que estén en el mismo dispositivo
    device = gato_cls_token.device
    frame_patch_tokens = frame_patch_tokens.to(device)
    gato_cls_token = gato_cls_token.to(device)
    
    # Calcular similitud entre cada patch token del frame y el CLS token de gato
    # gato_cls_token: [1, D]
    # frame_patch_tokens: [1, N, D]
    
    # Expandir CLS token para comparar con todos los patches
    cls_expanded = gato_cls_token.unsqueeze(1).expand(-1, frame_patch_tokens.shape[1], -1)  # [1, N, D]
    
    # Calcular similitud coseno
    similarities = F.cosine_similarity(frame_patch_tokens, cls_expanded, dim=-1)  # [1, N]
    similarities = similarities.squeeze(0)  # [N,]
    
    # Obtener el grid de patches
    N_patches = frame_patch_tokens.shape[1]
    H_patch, W_patch = infer_grid(N_patches)
    
    # Reshape a grid de patches
    heatmap_patch_grid = similarities.view(H_patch, W_patch).detach().cpu()  # Mantener como tensor
    
    # Redimensionar el mapa de calor al tamaño de la imagen original
    img_height, img_width = frame_img.size[1], frame_img.size[0]  # PIL usa (width, height)
    
    # Convertir a tensor con dimensiones [1, 1, H, W] para usar interpolate
    heatmap_tensor = heatmap_patch_grid.unsqueeze(0).unsqueeze(0)  # [1, 1, H_patch, W_patch]
    
    # Redimensionar usando interpolación bilineal de torch
    heatmap_fullsize_tensor = F.interpolate(
        heatmap_tensor, 
        size=(img_height, img_width), 
        mode='bilinear', 
        align_corners=False
    )
    
    # Convertir a numpy
    heatmap_fullsize = heatmap_fullsize_tensor.squeeze().numpy()
    
    # Crear visualización
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Imagen original del frame
    axes[0].imshow(frame_img)
    axes[0].set_title("Frame Original")
    axes[0].axis("off")
    
    # Mapa de calor a tamaño completo
    im1 = axes[1].imshow(heatmap_fullsize, cmap=cmap, interpolation='bilinear')
    axes[1].set_title("Heatmap: Patch tokens (frame) vs CLS token (gato.jpg)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Superposición del mapa de calor sobre la imagen
    axes[2].imshow(frame_img)
    im2 = axes[2].imshow(heatmap_fullsize, cmap=cmap, alpha=alpha, interpolation='bilinear')
    axes[2].set_title("Overlay: Heatmap sobre Frame")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualización guardada en: {save_path}")
    
    plt.show()
    plt.close()

if __name__ == "__main__":
    ckpt_path = "/home/shared_data/s3bir/saved_models/skDinoV3_sketchy.ckpt"
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = S3birDinov3().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Cargar las imágenes
    img1 = Image.open("extracted_frames/Untitled design (2)_frame_0000_t0.00s.jpg").convert("RGB")
    img2 = Image.open("gato.jpg").convert("RGB")

    transform = make_transform()
    img1_tensor = transform(img1).unsqueeze(0).to(device)     # [1, 3, 224, 224]
    img2_tensor = transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        img1_embedding = model(img1_tensor, dtype="image", is_training=True)
        cls_img2 = model(img2_tensor, dtype="image", is_training=False)  # False devuelve directamente el CLS token
        
    cls_img1 = img1_embedding['x_norm_clstoken']
    patch_img1 = img1_embedding['x_norm_patchtokens']
    
    # Cuando is_training=False, el modelo devuelve directamente el CLS token
    # Asegurar que cls_img2 tenga shape [1, D]
    if cls_img2.dim() == 1:
        cls_img2 = cls_img2.unsqueeze(0)

    # Crear directorio para guardar las visualizaciones
    output_dir = "visualization_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Visualización principal: patch tokens del frame vs CLS token de gato.jpg
    visualize_patch_tokens_vs_cls(
        frame_img=img1,
        gato_cls_token=cls_img2,  # CLS token de gato.jpg
        frame_patch_tokens=patch_img1,  # Patch tokens del frame
        save_path=os.path.join(output_dir, "frame_patches_vs_gato_cls.png"),
        cmap="viridis",
        alpha=0.6
    )
    
    print(f"\nVisualización guardada en el directorio: {output_dir}/")

