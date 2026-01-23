#!/usr/bin/env python3
"""
Script para cropear el frame a cuadrado (centrado).
"""

from PIL import Image
import os

def crop_to_square(img_path, output_path):
    """
    Cropea una imagen a cuadrada centrada.
    
    Args:
        img_path: Ruta de la imagen original
        output_path: Ruta donde guardar la imagen cropeada
    """
    img = Image.open(img_path).convert("RGB")
    width, height = img.size
    
    # Determinar el tamaño del lado del cuadrado (el menor)
    size = min(width, height)
    
    # Calcular el recorte centrado
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    
    # Cropear
    cropped = img.crop((left, top, right, bottom))
    
    # Guardar
    cropped.save(output_path, quality=95)
    print(f"✓ {img_path} ({width}x{height}) -> {output_path} ({size}x{size})")
    
    return cropped

if __name__ == "__main__":
    # Ruta del frame original
    frame_path = "efilo_T1_2025-08-04_test/images/cropped_2025-08-04_14-14-04_EFILOME_T1_HD_frame_0036_t108s.png"
    
    # Ruta de salida
    frame_cropped = "efilo_T1_2025-08-04_test/images/cropped_2025-08-04_14-14-04_EFILOME_T1_HD_frame_0036_t108s_square.png"
    
    # Verificar que exista la imagen
    if not os.path.exists(frame_path):
        print(f"Error: No se encuentra {frame_path}")
        exit(1)
    
    print("Cropeando frame a cuadrado...\n")
    
    # Cropear frame
    crop_to_square(frame_path, frame_cropped)
    
    print("\n✓ Proceso completado!")
    print(f"\nFrame cropeado guardado en:")
    print(f"  - {frame_cropped}")

