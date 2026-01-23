#!/usr/bin/env python3
"""
Script to extract encoders from S3BIR-DINOv3 model for all images in quickdraw_jpg directory.
Saves encoders as .npy files maintaining the same folder structure.
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

# Import the model
from s3bir_dinov3_model import S3birDinov3

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EncoderExtractor:
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
    
    def process_directory(self, input_dir, output_dir):
        """
        Process all images in the input directory and save encoders maintaining folder structure.
        
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
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_files.append(Path(root) / file)
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        processed = 0
        failed = 0
        
        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                # Calculate relative path from input directory
                rel_path = image_path.relative_to(input_path)
                
                # Create corresponding output path
                output_file_path = output_path / rel_path.with_suffix('.npy')
                
                # Create output directory if it doesn't exist
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Skip if encoder already exists
                if output_file_path.exists():
                    logger.debug(f"Skipping {image_path} - encoder already exists")
                    continue
                
                # Extract encoder
                encoder = self.extract_encoder(image_path)
                
                if encoder is not None:
                    # Save encoder
                    np.save(output_file_path, encoder)
                    processed += 1
                    logger.debug(f"Saved encoder for {image_path} -> {output_file_path}")
                else:
                    failed += 1
                    logger.warning(f"Failed to extract encoder for {image_path}")
                    
            except Exception as e:
                failed += 1
                logger.error(f"Error processing {image_path}: {e}")
        
        logger.info(f"Processing complete. Processed: {processed}, Failed: {failed}")

def main():
    parser = argparse.ArgumentParser(description='Extract encoders from S3BIR-DINOv3 model')
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
    
    # Initialize extractor
    extractor = EncoderExtractor(args.model_path, device)
    
    # Process directory
    extractor.process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()


