"""
A* Pathfinding Algorithm Module
Finds optimal routes considering terrain costs
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional

class AStarPathfinder:
    """A* pathfinding algorithm for route optimization"""
    
    def __init__(self):
        """Initialize the pathfinder"""
        self.path = None
        self.cost = None
    
    def find_path(self, cost_map: np.ndarray, start: Tuple[int, int], 
                  end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find optimal path using A* algorithm
        
        Args:
            cost_map: 2D array of terrain costs
            start: Starting coordinates (row, col)
            end: Ending coordinates (row, col)
            
        Returns:
            List of coordinates representing the path, or None if no path found
        """
        rows, cols = cost_map.shape
        
        # Validate start and end points
        if not (0 <= start[0] < rows and 0 <= start[1] < cols):
            raise ValueError("Start point out of bounds")
        if not (0 <= end[0] < rows and 0 <= end[1] < cols):
            raise ValueError("End point out of bounds")
        
        # Initialize data structures
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, end)}
        
        closed_set = set()
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == end:
                # Reconstruct path
                path = self._reconstruct_path(came_from, current)
                self.path = path
                self.cost = g_score[end]
                return path
            
            closed_set.add(current)
            
            # Check all neighbors (8-directional movement)
            for neighbor in self._get_neighbors(current, rows, cols):
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                move_cost = cost_map[neighbor[0], neighbor[1]]
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)
                    f_score[neighbor] = f
                    
                    # Add to open set if not already there
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f, neighbor))
        
        # No path found
        return None
    
    def _heuristic(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """
        Calculate heuristic distance (Euclidean distance)
        
        Args:
            point1: First point
            point2: Second point
            
        Returns:
            Heuristic distance
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _get_neighbors(self, point: Tuple[int, int], rows: int, cols: int) -> List[Tuple[int, int]]:
        """
        Get valid neighboring points (8-directional)
        
        Args:
            point: Current point
            rows: Number of rows in grid
            cols: Number of columns in grid
            
        Returns:
            List of valid neighbor coordinates
        """
        neighbors = []
        row, col = point
        
        # 8 directions: N, NE, E, SE, S, SW, W, NW
        directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols:
                neighbors.append((new_row, new_col))
        
        return neighbors
    
    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct path from start to end
        
        Args:
            came_from: Dictionary mapping points to their predecessors
            current: End point
            
        Returns:
            List of points forming the path
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def get_path_statistics(self, cost_map: np.ndarray) -> dict:
        """
        Get statistics about the found path
        
        Args:
            cost_map: The cost map used for pathfinding
            
        Returns:
            Dictionary with path statistics
        """
        if self.path is None:
            return None
        
        total_cost = sum(cost_map[p[0], p[1]] for p in self.path)
        avg_cost = total_cost / len(self.path)
        
        return {
            'length': len(self.path),
            'total_cost': float(total_cost),
            'average_cost': float(avg_cost),
            'start': self.path[0],
            'end': self.path[-1]
        }
