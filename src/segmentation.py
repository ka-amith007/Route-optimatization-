"""
Land Cover Segmentation Module
Implements various segmentation methods for satellite imagery
"""

import numpy as np
from PIL import Image
import cv2

class LandCoverSegmenter:
    """
    Segment satellite images into land cover types:
    - Water (0)
    - Forest (1)
    - Urban (2)
    - Barren (3)
    - Road (4)
    """
    
    def __init__(self, method='rule_based', model_path=None):
        """
        Initialize segmenter
        
        Args:
            method: Segmentation method ('rule_based', 'unet', 'deeplabv3')
            model_path: Path to pre-trained model (for ML methods)
        """
        self.method = method
        self.model_path = model_path
        
        # Color mapping for visualization
        self.color_map = {
            0: [0, 0, 255],      # Water - Blue
            1: [34, 139, 34],    # Forest - Green
            2: [128, 128, 128],  # Urban - Gray
            3: [210, 180, 140],  # Barren - Tan
            4: [0, 0, 0]         # Road - Black
        }
    
    def segment_image(self, image):
        """
        Segment the input image
        
        Args:
            image: Input RGB image as numpy array
            
        Returns:
            mask: Segmentation mask (H x W) with class labels
            colored_mask: Colored visualization of segmentation (H x W x 3)
        """
        if self.method == 'rule_based':
            mask = self._rule_based_segmentation(image)
        elif self.method == 'unet':
            mask = self._unet_segmentation(image)
        elif self.method == 'deeplabv3':
            mask = self._deeplab_segmentation(image)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        colored_mask = self._colorize_mask(mask)
        return mask, colored_mask
    
    def _rule_based_segmentation(self, image):
        """
        Simple rule-based segmentation using color thresholds
        
        Args:
            image: RGB image
            
        Returns:
            Segmentation mask
        """
        # Convert to float for processing
        img = image.astype(np.float32) / 255.0
        
        # Initialize mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # Extract RGB channels
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        
        # Water: High blue, low red and green
        water_mask = (b > 0.5) & (r < 0.3) & (g < 0.5)
        mask[water_mask] = 0
        
        # Forest: High green
        forest_mask = (g > 0.4) & (g > r) & (g > b) & ~water_mask
        mask[forest_mask] = 1
        
        # Urban: Similar RGB values (gray)
        urban_mask = (np.abs(r - g) < 0.1) & (np.abs(g - b) < 0.1) & (np.abs(r - b) < 0.1)
        urban_mask = urban_mask & (r > 0.3) & ~water_mask & ~forest_mask
        mask[urban_mask] = 2
        
        # Barren: High red and green, lower blue
        barren_mask = (r > 0.4) & (g > 0.4) & (b < 0.4) & ~water_mask & ~forest_mask & ~urban_mask
        mask[barren_mask] = 3
        
        # Road: Very dark or very uniform
        road_mask = ((r < 0.2) & (g < 0.2) & (b < 0.2)) & ~water_mask
        mask[road_mask] = 4
        
        return mask
    
    def _unet_segmentation(self, image):
        """
        U-Net based segmentation (placeholder - requires trained model)
        """
        print("U-Net segmentation not yet implemented, using rule-based")
        return self._rule_based_segmentation(image)
    
    def _deeplab_segmentation(self, image):
        """
        DeepLabV3+ segmentation (placeholder - requires trained model)
        """
        print("DeepLabV3+ segmentation not yet implemented, using rule-based")
        return self._rule_based_segmentation(image)
    
    def _colorize_mask(self, mask):
        """
        Convert class mask to colored visualization
        
        Args:
            mask: Segmentation mask with class labels
            
        Returns:
            RGB colored mask
        """
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in self.color_map.items():
            colored[mask == class_id] = color
        
        return colored
    
    def get_class_statistics(self, mask):
        """
        Calculate statistics for each land cover class
        
        Args:
            mask: Segmentation mask
            
        Returns:
            Dictionary with pixel counts and percentages per class
        """
        stats = {}
        total_pixels = mask.size
        
        class_names = {
            0: 'Water',
            1: 'Forest',
            2: 'Urban',
            3: 'Barren',
            4: 'Road'
        }
        
        for class_id, name in class_names.items():
            count = np.sum(mask == class_id)
            percentage = (count / total_pixels) * 100
            stats[name] = {
                'pixels': int(count),
                'percentage': float(percentage)
            }
        
        return stats
