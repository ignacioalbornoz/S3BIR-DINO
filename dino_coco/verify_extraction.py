#!/usr/bin/env python3
"""
Script to verify that all images were processed and encoders were extracted correctly.
Compares input images with output encoders and reports any missing files.
"""

import os
import sys
from pathlib import Path
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_extraction(input_dir, output_dir):
    """
    Verify that all images have corresponding encoder files.
    
    Args:
        input_dir: Path to quickdraw_jpg directory
        output_dir: Path to quickdraw_encoders directory
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        return False
    
    if not output_path.exists():
        logger.error(f"Output directory {output_dir} does not exist")
        return False
    
    # Image extensions to look for
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Statistics
    stats = {
        'total_images': 0,
        'total_encoders': 0,
        'missing_encoders': 0,
        'extra_encoders': 0,
        'classes_processed': 0,
        'classes_missing': 0,
        'corrupted_encoders': 0
    }
    
    missing_files = []
    extra_files = []
    corrupted_files = []
    
    # Get all class directories
    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    logger.info(f"Found {len(class_dirs)} classes in input directory")
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        logger.info(f"Verifying class: {class_name}")
        
        # Get all images in this class
        image_files = []
        for file in class_dir.iterdir():
            if file.suffix.lower() in image_extensions:
                image_files.append(file)
        
        stats['total_images'] += len(image_files)
        logger.info(f"  Found {len(image_files)} images in class {class_name}")
        
        # Check corresponding output directory
        class_output_dir = output_path / class_name
        if not class_output_dir.exists():
            logger.error(f"  Output directory for class {class_name} does not exist!")
            stats['classes_missing'] += 1
            # All images in this class are missing
            for img_file in image_files:
                missing_files.append((class_name, img_file.name))
                stats['missing_encoders'] += 1
            continue
        
        stats['classes_processed'] += 1
        
        # Get all encoder files in this class
        encoder_files = []
        for file in class_output_dir.iterdir():
            if file.suffix == '.npy':
                encoder_files.append(file)
        
        stats['total_encoders'] += len(encoder_files)
        logger.info(f"  Found {len(encoder_files)} encoder files in class {class_name}")
        
        # Check each image has corresponding encoder
        for img_file in image_files:
            expected_encoder = class_output_dir / img_file.with_suffix('.npy').name
            
            if expected_encoder.exists():
                # Verify encoder file is not corrupted
                try:
                    import numpy as np
                    encoder = np.load(expected_encoder)
                    if encoder.shape != (1, 768):
                        logger.warning(f"  Corrupted encoder: {expected_encoder} (shape: {encoder.shape})")
                        corrupted_files.append((class_name, expected_encoder.name))
                        stats['corrupted_encoders'] += 1
                except Exception as e:
                    logger.warning(f"  Corrupted encoder: {expected_encoder} (error: {e})")
                    corrupted_files.append((class_name, expected_encoder.name))
                    stats['corrupted_encoders'] += 1
            else:
                missing_files.append((class_name, img_file.name))
                stats['missing_encoders'] += 1
        
        # Check for extra encoder files (no corresponding image)
        for encoder_file in encoder_files:
            expected_image = class_dir / encoder_file.with_suffix('.jpg').name
            if not expected_image.exists():
                # Try other extensions
                found = False
                for ext in image_extensions:
                    expected_image = class_dir / encoder_file.with_suffix(ext).name
                    if expected_image.exists():
                        found = True
                        break
                
                if not found:
                    extra_files.append((class_name, encoder_file.name))
                    stats['extra_encoders'] += 1
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EXTRACTION VERIFICATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total images found: {stats['total_images']}")
    logger.info(f"Total encoders found: {stats['total_encoders']}")
    logger.info(f"Classes processed: {stats['classes_processed']}")
    logger.info(f"Classes missing: {stats['classes_missing']}")
    logger.info(f"Missing encoders: {stats['missing_encoders']}")
    logger.info(f"Extra encoders: {stats['extra_encoders']}")
    logger.info(f"Corrupted encoders: {stats['corrupted_encoders']}")
    
    # Calculate success rate
    if stats['total_images'] > 0:
        success_rate = ((stats['total_images'] - stats['missing_encoders'] - stats['corrupted_encoders']) / stats['total_images']) * 100
        logger.info(f"Success rate: {success_rate:.2f}%")
    
    # Report missing files
    if missing_files:
        logger.warning(f"\nMISSING ENCODERS ({len(missing_files)} files):")
        for class_name, filename in missing_files[:10]:  # Show first 10
            logger.warning(f"  {class_name}/{filename}")
        if len(missing_files) > 10:
            logger.warning(f"  ... and {len(missing_files) - 10} more")
    
    # Report extra files
    if extra_files:
        logger.warning(f"\nEXTRA ENCODERS ({len(extra_files)} files):")
        for class_name, filename in extra_files[:10]:  # Show first 10
            logger.warning(f"  {class_name}/{filename}")
        if len(extra_files) > 10:
            logger.warning(f"  ... and {len(extra_files) - 10} more")
    
    # Report corrupted files
    if corrupted_files:
        logger.warning(f"\nCORRUPTED ENCODERS ({len(corrupted_files)} files):")
        for class_name, filename in corrupted_files[:10]:  # Show first 10
            logger.warning(f"  {class_name}/{filename}")
        if len(corrupted_files) > 10:
            logger.warning(f"  ... and {len(corrupted_files) - 10} more")
    
    # Final status
    if stats['missing_encoders'] == 0 and stats['corrupted_encoders'] == 0:
        logger.info("\n✅ ALL IMAGES PROCESSED SUCCESSFULLY!")
        return True
    else:
        logger.warning(f"\n❌ EXTRACTION INCOMPLETE: {stats['missing_encoders']} missing, {stats['corrupted_encoders']} corrupted")
        return False

def get_directory_stats(directory):
    """Get statistics about a directory structure."""
    path = Path(directory)
    if not path.exists():
        return None
    
    stats = {
        'total_files': 0,
        'total_dirs': 0,
        'file_extensions': defaultdict(int),
        'subdirs': []
    }
    
    for item in path.rglob('*'):
        if item.is_file():
            stats['total_files'] += 1
            stats['file_extensions'][item.suffix] += 1
        elif item.is_dir() and item != path:
            stats['total_dirs'] += 1
            stats['subdirs'].append(item.name)
    
    return stats

def main():
    input_dir = "/data/quickdraw_jpg"
    output_dir = "/data/quickdraw_encoders"
    
    logger.info("Starting extraction verification...")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Get input directory stats
    logger.info("\nINPUT DIRECTORY STATISTICS:")
    input_stats = get_directory_stats(input_dir)
    if input_stats:
        logger.info(f"  Total files: {input_stats['total_files']}")
        logger.info(f"  Total directories: {input_stats['total_dirs']}")
        logger.info(f"  File extensions: {dict(input_stats['file_extensions'])}")
        logger.info(f"  Classes: {len(input_stats['subdirs'])}")
    
    # Get output directory stats
    logger.info("\nOUTPUT DIRECTORY STATISTICS:")
    output_stats = get_directory_stats(output_dir)
    if output_stats:
        logger.info(f"  Total files: {output_stats['total_files']}")
        logger.info(f"  Total directories: {output_stats['total_dirs']}")
        logger.info(f"  File extensions: {dict(output_stats['file_extensions'])}")
        logger.info(f"  Classes: {len(output_stats['subdirs'])}")
    else:
        logger.error("Output directory does not exist!")
        return
    
    # Verify extraction
    logger.info("\nVERIFYING EXTRACTION...")
    success = verify_extraction(input_dir, output_dir)
    
    if success:
        logger.info("\n🎉 VERIFICATION PASSED: All images processed successfully!")
    else:
        logger.warning("\n⚠️  VERIFICATION FAILED: Some images were not processed correctly.")

if __name__ == "__main__":
    main()

