"""
Route Visualization Module
Creates visualizations of planned routes on satellite imagery
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple

class RouteVisualizer:
    """Visualize routes on satellite imagery"""
    
    def __init__(self):
        """Initialize the visualizer"""
        self.route_color = (0, 255, 255)  # Cyan for route
        self.start_color = (0, 255, 0)    # Green for start
        self.end_color = (255, 0, 0)      # Red for end
        self.line_thickness = 3
        self.marker_size = 10
    
    def visualize_route(self, image: np.ndarray, path: List[Tuple[int, int]], 
                       start: Tuple[int, int], end: Tuple[int, int]) -> np.ndarray:
        """
        Draw route on the image
        
        Args:
            image: Original satellite image
            path: List of (row, col) coordinates forming the route
            start: Start point coordinates
            end: End point coordinates
            
        Returns:
            Image with route drawn on it
        """
        # Make a copy to avoid modifying original
        result = image.copy()
        
        # Draw the path
        for i in range(len(path) - 1):
            pt1 = (path[i][1], path[i][0])  # (col, row) for cv2
            pt2 = (path[i+1][1], path[i+1][0])
            cv2.line(result, pt1, pt2, self.route_color, self.line_thickness)
        
        # Draw start marker
        start_pt = (start[1], start[0])
        cv2.circle(result, start_pt, self.marker_size, self.start_color, -1)
        cv2.circle(result, start_pt, self.marker_size + 2, (255, 255, 255), 2)
        
        # Draw end marker
        end_pt = (end[1], end[0])
        cv2.circle(result, end_pt, self.marker_size, self.end_color, -1)
        cv2.circle(result, end_pt, self.marker_size + 2, (255, 255, 255), 2)
        
        return result
    
    def create_comparison_grid(self, images: List[np.ndarray], 
                               titles: List[str] = None) -> np.ndarray:
        """
        Create a grid comparison of multiple images
        
        Args:
            images: List of images to compare
            titles: Optional list of titles for each image
            
        Returns:
            Combined grid image
        """
        n_images = len(images)
        
        if n_images == 0:
            return None
        
        # Determine grid size
        cols = min(2, n_images)
        rows = (n_images + cols - 1) // cols
        
        # Get max dimensions
        max_h = max(img.shape[0] for img in images)
        max_w = max(img.shape[1] for img in images)
        
        # Create white background
        grid_h = max_h * rows + 50 * rows  # Extra space for titles
        grid_w = max_w * cols
        
        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
        
        # Place images in grid
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            
            y_offset = row * (max_h + 50)
            x_offset = col * max_w
            
            h, w = img.shape[:2]
            grid[y_offset:y_offset+h, x_offset:x_offset+w] = img
            
            # Add title if provided
            if titles and idx < len(titles):
                title_y = y_offset + h + 30
                cv2.putText(grid, titles[idx], (x_offset + 10, title_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        return grid
    
    def create_result_summary(self, original_image: np.ndarray,
                             segmentation: np.ndarray,
                             cost_map: np.ndarray,
                             route_image: np.ndarray,
                             stats: dict = None) -> np.ndarray:
        """
        Create a comprehensive result summary visualization
        
        Args:
            original_image: Original satellite image
            segmentation: Segmented land cover image
            cost_map: Cost map visualization
            route_image: Image with route drawn
            stats: Optional statistics dictionary
            
        Returns:
            Summary visualization
        """
        # Normalize cost map for visualization
        cost_vis = ((cost_map - cost_map.min()) / 
                   (cost_map.max() - cost_map.min()) * 255).astype(np.uint8)
        cost_vis = cv2.applyColorMap(cost_vis, cv2.COLORMAP_HOT)
        
        images = [original_image, segmentation, cost_vis, route_image]
        titles = ['Original Image', 'Land Segmentation', 'Cost Map', 'Planned Route']
        
        return self.create_comparison_grid(images, titles)
    
    def save_visualization(self, image: np.ndarray, output_path: str):
        """
        Save visualization to file
        
        Args:
            image: Image to save
            output_path: Output file path
        """
        img = Image.fromarray(image)
        img.save(output_path)
    
    def add_legend(self, image: np.ndarray, legend_items: List[Tuple[str, Tuple[int, int, int]]]) -> np.ndarray:
        """
        Add a legend to the image
        
        Args:
            image: Input image
            legend_items: List of (label, color) tuples
            
        Returns:
            Image with legend
        """
        result = image.copy()
        h, w = result.shape[:2]
        
        # Legend dimensions
        legend_width = 200
        legend_height = len(legend_items) * 30 + 20
        
        # Create semi-transparent legend box
        legend_box = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
        
        # Add legend items
        for idx, (label, color) in enumerate(legend_items):
            y_pos = 20 + idx * 30
            
            # Draw color box
            cv2.rectangle(legend_box, (10, y_pos), (30, y_pos + 20), color, -1)
            
            # Draw label
            cv2.putText(legend_box, label, (40, y_pos + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Overlay legend on image (top-right corner)
        x_pos = w - legend_width - 10
        y_pos = 10
        
        # Blend legend with image
        roi = result[y_pos:y_pos+legend_height, x_pos:x_pos+legend_width]
        blended = cv2.addWeighted(roi, 0.3, legend_box, 0.7, 0)
        result[y_pos:y_pos+legend_height, x_pos:x_pos+legend_width] = blended
        
        return result
