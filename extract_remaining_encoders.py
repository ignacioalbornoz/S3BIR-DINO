#!/usr/bin/env python3
"""
Script to extract encoders from S3BIR-DINOv3 model for REMAINING images in quickdraw_jpg directory.
Only processes classes that don't have output directories or have missing encoders.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import logging
from tqdm import tqdm
import argparse
import time

# Import the model
from s3bir_dinov3_model import S3birDinov3

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IncrementalEncoderExtractor:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the encoder extractor with the trained S3BIR-DINOv3 model.
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on
        """
        self.device = device
        self.model_path = model_path
        self.model = None
        self.transform = None
        
        # Initialize transforms for image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained S3BIR-DINOv3 model from checkpoint."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Initialize the model
            self.model = S3birDinov3(n_prompts=3, prompt_dim=768)
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load state dict
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def extract_encoder(self, image_path):
        """
        Extract encoder from a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            numpy array containing the encoder features
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract encoder features
            with torch.no_grad():
                # Get features from the model
                features = self.model(image_tensor, dtype='image')
                
                # Extract the encoder output (class token)
                if isinstance(features, dict):
                    encoder = features['x_norm_clstoken'].cpu().numpy()
                else:
                    encoder = features.cpu().numpy()
            
            return encoder
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None
    
    def get_missing_classes(self, input_dir, output_dir):
        """Get list of classes that are missing or incomplete."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        missing_classes = []
        incomplete_classes = []
        
        # Get all class directories from input
        class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            class_output_dir = output_path / class_name
            
            if not class_output_dir.exists():
                missing_classes.append(class_name)
                logger.info(f"Missing class: {class_name}")
            else:
                # Check if class is incomplete
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
                
                # Count images in input
                input_images = [f for f in class_dir.iterdir() if f.suffix.lower() in image_extensions]
                
                # Count encoders in output
                output_encoders = [f for f in class_output_dir.iterdir() if f.suffix == '.npy']
                
                if len(output_encoders) < len(input_images):
                    incomplete_classes.append((class_name, len(input_images), len(output_encoders)))
                    logger.info(f"Incomplete class: {class_name} ({len(output_encoders)}/{len(input_images)} encoders)")
        
        return missing_classes, incomplete_classes
    
    def process_remaining_classes(self, input_dir, output_dir):
        """
        Process only the classes that are missing or incomplete.
        
        Args:
            input_dir: Path to quickdraw_jpg directory
            output_dir: Path to save encoder .npy files
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            logger.error(f"Input directory {input_dir} does not exist")
            return
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get missing and incomplete classes
        missing_classes, incomplete_classes = self.get_missing_classes(input_dir, output_dir)
        
        logger.info(f"Found {len(missing_classes)} missing classes")
        logger.info(f"Found {len(incomplete_classes)} incomplete classes")
        
        total_processed = 0
        total_failed = 0
        start_time = time.time()
        
        # Process missing classes (complete processing)
        for class_name in missing_classes:
            logger.info(f"Processing missing class: {class_name}")
            class_dir = input_path / class_name
            class_output_dir = output_path / class_name
            class_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all images in this class
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = [f for f in class_dir.iterdir() if f.suffix.lower() in image_extensions]
            
            logger.info(f"  Found {len(image_files)} images in class {class_name}")
            
            class_processed = 0
            class_failed = 0
            
            for image_path in tqdm(image_files, desc=f"Processing {class_name}", leave=False):
                try:
                    # Create corresponding output path
                    output_file_path = class_output_dir / image_path.with_suffix('.npy').name
                    
                    # Skip if encoder already exists
                    if output_file_path.exists():
                        continue
                    
                    # Extract encoder
                    encoder = self.extract_encoder(image_path)
                    
                    if encoder is not None:
                        # Save encoder
                        np.save(output_file_path, encoder)
                        class_processed += 1
                        total_processed += 1
                    else:
                        class_failed += 1
                        total_failed += 1
                        
                except Exception as e:
                    class_failed += 1
                    total_failed += 1
                    logger.error(f"Error processing {image_path}: {e}")
            
            logger.info(f"  Class {class_name}: Processed {class_processed}, Failed {class_failed}")
        
        # Process incomplete classes (only missing encoders)
        for class_name, total_images, existing_encoders in incomplete_classes:
            logger.info(f"Processing incomplete class: {class_name} ({existing_encoders}/{total_images} done)")
            class_dir = input_path / class_name
            class_output_dir = output_path / class_name
            
            # Get all images in this class
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = [f for f in class_dir.iterdir() if f.suffix.lower() in image_extensions]
            
            class_processed = 0
            class_failed = 0
            
            for image_path in tqdm(image_files, desc=f"Completing {class_name}", leave=False):
                try:
                    # Create corresponding output path
                    output_file_path = class_output_dir / image_path.with_suffix('.npy').name
                    
                    # Skip if encoder already exists
                    if output_file_path.exists():
                        continue
                    
                    # Extract encoder
                    encoder = self.extract_encoder(image_path)
                    
                    if encoder is not None:
                        # Save encoder
                        np.save(output_file_path, encoder)
                        class_processed += 1
                        total_processed += 1
                    else:
                        class_failed += 1
                        total_failed += 1
                        
                except Exception as e:
                    class_failed += 1
                    total_failed += 1
                    logger.error(f"Error processing {image_path}: {e}")
            
            logger.info(f"  Class {class_name}: Processed {class_processed}, Failed {class_failed}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Processing complete!")
        logger.info(f"Total processed: {total_processed}")
        logger.info(f"Total failed: {total_failed}")
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        if total_processed > 0:
            logger.info(f"Average time per image: {elapsed_time/total_processed:.4f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Extract remaining encoders from S3BIR-DINOv3 model')
    parser.add_argument('--model_path', type=str, default='s3bir_dinov3.ckpt',
                       help='Path to the trained model checkpoint')
    parser.add_argument('--input_dir', type=str, default='/data/quickdraw_jpg',
                       help='Path to quickdraw_jpg directory')
    parser.add_argument('--output_dir', type=str, default='/data/quickdraw_encoders',
                       help='Path to save encoder .npy files')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Initialize extractor
    extractor = IncrementalEncoderExtractor(args.model_path, device)
    
    # Process remaining classes
    extractor.process_remaining_classes(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
