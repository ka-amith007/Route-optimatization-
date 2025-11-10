"""
Satellite Image Collection Module
Handles loading and preprocessing of satellite imagery
"""

import numpy as np
from PIL import Image
import os

class SatelliteImageCollector:
    """Handle satellite image loading and preprocessing"""
    
    def __init__(self):
        """Initialize the image collector"""
        pass
    
    def load_image(self, image_path):
        """
        Load an image from path
        
        Args:
            image_path: Path to the image file
            
        Returns:
            numpy array of the image
        """
        img = Image.open(image_path)
        return np.array(img)
    
    def preprocess_image(self, image, target_size=None):
        """
        Preprocess image for analysis
        
        Args:
            image: Input image as numpy array
            target_size: Optional tuple (width, height) to resize to
            
        Returns:
            Preprocessed image as numpy array
        """
        if target_size:
            img = Image.fromarray(image)
            img = img.resize(target_size, Image.LANCZOS)
            image = np.array(img)
        
        return image
    
    def save_image(self, image, output_path):
        """Save image to file"""
        img = Image.fromarray(image)
        img.save(output_path)
