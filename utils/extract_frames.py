import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_folder, interval_seconds=1):
    """
    Extrae frames de un video cada N segundos y los guarda en una carpeta.
    
    Args:
        video_path: Ruta al archivo de video
        output_folder: Carpeta donde guardar los frames
        interval_seconds: Intervalo en segundos entre frames (default: 1)
    """
    # Crear carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)
    
    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return
    
    # Obtener información del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duración: {duration:.2f} segundos")
    print(f"Extrayendo frames cada {interval_seconds} segundo(s)...")
    
    # Calcular el intervalo en frames
    frame_interval = int(fps * interval_seconds)
    
    frame_count = 0
    saved_count = 0
    
    # Obtener el nombre del video sin extensión para nombrar los frames
    video_name = Path(video_path).stem
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Guardar frame si corresponde al intervalo
        if frame_count % frame_interval == 0:
            # Calcular el tiempo en segundos
            time_seconds = frame_count / fps
            
            # Nombre del archivo: nombre_video_tiempo_segundos.jpg
            frame_filename = f"{video_name}_frame_{saved_count:04d}_t{time_seconds:.2f}s.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            print(f"Guardado: {frame_filename} (t={time_seconds:.2f}s)")
        
        frame_count += 1
    
    cap.release()
    print(f"\nExtracción completada!")
    print(f"Total de frames guardados: {saved_count}")
    print(f"Frames guardados en: {output_folder}")

if __name__ == "__main__":
    video_path = "Untitled design (2).mp4"
    output_folder = "extracted_frames"
    
    extract_frames(video_path, output_folder, interval_seconds=1)

