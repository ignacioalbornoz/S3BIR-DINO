#!/usr/bin/env python3
"""
Script para redimensionar ambas imágenes cuadradas al mismo tamaño.
Esto asegura que ambas generen el mismo número de patches en el modelo.
"""

from PIL import Image
import os

def resize_to_size(img_path, output_path, target_size=224):
    """
    Redimensiona una imagen cuadrada a un tamaño específico.
    
    Args:
        img_path: Ruta de la imagen cuadrada original
        output_path: Ruta donde guardar la imagen redimensionada
        target_size: Tamaño objetivo (ancho y alto, debe ser múltiplo de 16)
    """
    img = Image.open(img_path).convert("RGB")
    width, height = img.size
    
    # Verificar que la imagen sea cuadrada
    if width != height:
        print(f"Advertencia: {img_path} no es cuadrada ({width}x{height}), redimensionando de todos modos...")
    
    # Redimensionar usando LANCZOS para mejor calidad
    resized = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Guardar
    resized.save(output_path, quality=95)
    print(f"✓ {img_path} ({width}x{height}) -> {output_path} ({target_size}x{target_size})")
    
    return resized

if __name__ == "__main__":
    # Tamaño objetivo (debe ser múltiplo de 16 para el patch_size del modelo)
    TARGET_SIZE = 224  # Tamaño estándar del modelo
    
    # Rutas de las imágenes cuadradas
    sketch_square = "gato_square.jpg"
    frame_square = "extracted_frames/Untitled design (2)_frame_0000_t0.00s_square.jpg"
    
    # Rutas de salida (mismo tamaño)
    sketch_resized = "gato_square_224.jpg"
    frame_resized = "extracted_frames/Untitled design (2)_frame_0000_t0.00s_square_224.jpg"
    
    # Verificar que existan las imágenes cuadradas
    if not os.path.exists(sketch_square):
        print(f"Error: No se encuentra {sketch_square}")
        print("Ejecuta primero crop_images_to_square.py")
        exit(1)
    
    if not os.path.exists(frame_square):
        print(f"Error: No se encuentra {frame_square}")
        print("Ejecuta primero crop_images_to_square.py")
        exit(1)
    
    print(f"Redimensionando imágenes cuadradas a {TARGET_SIZE}x{TARGET_SIZE}...\n")
    
    # Redimensionar sketch
    resize_to_size(sketch_square, sketch_resized, TARGET_SIZE)
    
    # Redimensionar frame
    resize_to_size(frame_square, frame_resized, TARGET_SIZE)
    
    print("\n✓ Proceso completado!")
    print(f"\nImágenes redimensionadas guardadas en:")
    print(f"  - {sketch_resized}")
    print(f"  - {frame_resized}")
    print(f"\nAmbas imágenes ahora tienen el mismo tamaño: {TARGET_SIZE}x{TARGET_SIZE}")
    print("Esto asegura que generen el mismo número de patches en el modelo.")

