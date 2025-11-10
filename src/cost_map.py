"""
Cost Map Generation Module
Calculates construction costs based on terrain types
"""

import numpy as np

class CostMapGenerator:
    """Generate cost maps from land cover segmentation"""
    
    def __init__(self, terrain_costs=None):
        """
        Initialize cost map generator
        
        Args:
            terrain_costs: Dictionary mapping class IDs to cost values
        """
        if terrain_costs is None:
            # Default terrain costs
            self.terrain_costs = {
                0: 1000,  # Water - very expensive
                1: 500,   # Forest - expensive
                2: 200,   # Urban - moderate
                3: 100,   # Barren - cheap
                4: 50     # Road - very cheap
            }
        else:
            self.terrain_costs = terrain_costs
    
    def generate_cost_map(self, segmentation_mask):
        """
        Generate cost map from segmentation
        
        Args:
            segmentation_mask: Land cover segmentation mask
            
        Returns:
            Cost map as numpy array (same size as mask)
        """
        cost_map = np.zeros_like(segmentation_mask, dtype=np.float32)
        
        for class_id, cost in self.terrain_costs.items():
            cost_map[segmentation_mask == class_id] = cost
        
        return cost_map
    
    def update_costs(self, new_costs):
        """
        Update terrain cost values
        
        Args:
            new_costs: Dictionary with updated cost values
        """
        self.terrain_costs.update(new_costs)
    
    def get_terrain_statistics(self, segmentation_mask, cost_map):
        """
        Calculate cost statistics for each terrain type
        
        Args:
            segmentation_mask: Land cover mask
            cost_map: Generated cost map
            
        Returns:
            Dictionary with statistics per terrain type
        """
        stats = {}
        
        terrain_names = {
            0: 'Water',
            1: 'Forest',
            2: 'Urban',
            3: 'Barren',
            4: 'Road'
        }
        
        for class_id, name in terrain_names.items():
            mask_area = segmentation_mask == class_id
            if np.any(mask_area):
                stats[name] = {
                    'cost_value': self.terrain_costs[class_id],
                    'pixel_count': int(np.sum(mask_area)),
                    'total_cost': float(np.sum(cost_map[mask_area]))
                }
        
        return stats
