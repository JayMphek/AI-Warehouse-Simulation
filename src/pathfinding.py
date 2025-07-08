import math
import heapq

class Pathfinding:
    def __init__(self, warehouse):
        self.warehouse =warehouse

    def distance_between(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def _find_nearest_navigable_cell(self, grid_pos):
        x, y = grid_pos
        max_radius = max(self.warehouse.grid_width, self.warehouse.grid_height)
        
        for radius in range(1, max_radius):
            for dx in range(-radius, radius+1):
                for dy in [-radius, radius]: 
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.warehouse.grid_width and 0 <= ny < self.warehouse.grid_height and self.warehouse.navigation_grid[ny, nx]):
                        return (nx, ny)
            
            for dx in [-radius, radius]:  
                for dy in range(-radius+1, radius): 
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.warehouse.grid_width and 0 <= ny < self.warehouse.grid_height and self.warehouse.navigation_grid[ny, nx]):
                        return (nx, ny)
        return grid_pos
    
    def find_path(self, start, end, robot_id=None, robots=None, avoid_robots=True):
        start_grid = (int(start[0] // self.warehouse.grid_size), int(start[1] // self.warehouse.grid_size))
        end_grid = (int(end[0] // self.warehouse.grid_size), int(end[1] // self.warehouse.grid_size))
        if not (0 <= start_grid[0] < self.warehouse.grid_width and 0 <= start_grid[1] < self.warehouse.grid_height):
            start_grid = (max(0, min(start_grid[0], self.warehouse.grid_width-1)), 
                        max(0, min(start_grid[1], self.warehouse.grid_height-1)))
        if not (0 <= end_grid[0] < self.warehouse.grid_width and 0 <= end_grid[1] < self.warehouse.grid_height):
            end_grid = (max(0, min(end_grid[0], self.warehouse.grid_width-1)), 
                        max(0, min(end_grid[1], self.warehouse.grid_height-1)))
        
        if not self.warehouse.navigation_grid[start_grid[1], start_grid[0]]:
            start_grid = self._find_nearest_navigable_cell(start_grid)
        if not self.warehouse.navigation_grid[end_grid[1], end_grid[0]]:
            end_grid = self._find_nearest_navigable_cell(end_grid)
        
        temp_grid = self.warehouse.navigation_grid.copy()
        if avoid_robots and robot_id is not None and robots is not None:
            for other_robot in robots:
                if other_robot.id != robot_id:
                    rx, ry = other_robot.position
                    rgx, rgy = int(rx // self.warehouse.grid_size), int(ry // self.warehouse.grid_size)
                    
                    radius = 2  # Size of the area to avoid
                    for dx in range(-radius, radius+1):
                        for dy in range(-radius, radius+1):
                            nx, ny = rgx + dx, rgy + dy
                            if 0 <= nx < self.warehouse.grid_width and 0 <= ny < self.warehouse.grid_height:
                                temp_grid[ny, nx] = False
        
        # A* algorithm
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.distance_between(start_grid, end_grid)}
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
        
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == end_grid:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return [(x * self.warehouse.grid_size + self.warehouse.grid_size // 2, 
                        y * self.warehouse.grid_size + self.warehouse.grid_size // 2) 
                        for x, y in reversed(path)]
            
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if (0 <= neighbor[0] < self.warehouse.grid_width and 0 <= neighbor[1] < self.warehouse.grid_height and temp_grid[neighbor[1], neighbor[0]]):
                    move_cost = 1.4 if abs(dx) + abs(dy) == 2 else 1.0
                    tentative_g = g_score.get(current, float('inf')) + move_cost
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self.distance_between(neighbor, end_grid)
                        if neighbor not in [i[1] for i in open_set]:
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return self._generate_fallback_path(start, end, robot_id, robots)
    
    def _try_a_star_path(self, start, end, robot_id, robots, avoid_robots=True):
        start_grid = (int(start[0] // self.warehouse.grid_size), int(start[1] // self.warehouse.grid_size))
        end_grid = (int(end[0] // self.warehouse.grid_size), int(end[1] // self.warehouse.grid_size))
        
        if not (0 <= start_grid[0] < self.warehouse.grid_width and 0 <= start_grid[1] < self.warehouse.grid_height):
            return None
        if not (0 <= end_grid[0] < self.warehouse.grid_width and 0 <= end_grid[1] < self.warehouse.grid_height):
            return None
        
        if not self.warehouse.navigation_grid[start_grid[1], start_grid[0]] or not self.warehouse.navigation_grid[end_grid[1], end_grid[0]]:
            return None
        
        temp_grid = self.warehouse.navigation_grid.copy()
        if avoid_robots and robot_id is not None and robots is not None:
            for other_robot in robots:
                if other_robot.id != robot_id:
                    rx, ry = other_robot.position
                    rgx, rgy = int(rx // self.warehouse.grid_size), int(ry // self.warehouse.grid_size)
                    radius = 2
                    for dx in range(-radius, radius+1):
                        for dy in range(-radius, radius+1):
                            nx, ny = rgx + dx, rgy + dy
                            if 0 <= nx < self.warehouse.grid_width and 0 <= ny < self.warehouse.grid_height:
                                temp_grid[ny, nx] = False
        
        # A* algorithm implementation
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: math.sqrt((start_grid[0] - end_grid[0])**2 + (start_grid[1] - end_grid[1])**2)}
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
        
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == end_grid:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                
                return [(x * self.warehouse.grid_size + self.warehouse.grid_size // 2, 
                        y * self.warehouse.grid_size + self.warehouse.grid_size // 2) 
                        for x, y in reversed(path)]
            
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if (0 <= neighbor[0] < self.warehouse.grid_width and 0 <= neighbor[1] < self.warehouse.grid_height and  temp_grid[neighbor[1], neighbor[0]]):
                    move_cost = 1.4 if abs(dx) + abs(dy) == 2 else 1.0
                    tentative_g = g_score.get(current, float('inf')) + move_cost
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + math.sqrt((neighbor[0] - end_grid[0])**2 + (neighbor[1] - end_grid[1])**2)
                        
                        if neighbor not in [i[1] for i in open_set]:
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None
    
    def _generate_fallback_path(self, start, end, robot_id, robots, aisles=None):
        if aisles is None:
            margin = 50
            waypoints = [
                (margin, margin),
                (start[0], margin),
                (start[0], end[1]),
                (end[0], end[1])
            ]
            return waypoints
        
        waypoints = []
        for aisle in aisles:
            aisle_center_x = aisle.centerx
            num_waypoints = 5
            for i in range(1, num_waypoints + 1):
                y_pos = aisle.top + (aisle.height * i) // (num_waypoints + 1)
                waypoint = (aisle_center_x, y_pos)
                waypoint_grid = (int(waypoint[0] // self.warehouse.grid_size), int(waypoint[1] // self.warehouse.grid_size))
                
                if (0 <= waypoint_grid[0] < self.warehouse.grid_width and 0 <= waypoint_grid[1] < self.warehouse.grid_height and self.warehouse.navigation_grid[waypoint_grid[1], waypoint_grid[0]]):
                    waypoints.append(waypoint)

        if not waypoints:
            margin = 50
            waypoints = [
                (margin, margin),
                (start[0], margin),
                (start[0], end[1]),
                (end[0], end[1])
            ]
            return waypoints
        
        best_path = None
        shortest_length = float('inf')
        
        for w1 in waypoints:
            path1 = self._try_a_star_path(start, w1, robot_id, robots, avoid_robots=False)
            if path1:
                for w2 in waypoints:
                    if w1 != w2:
                        path2 = self._try_a_star_path(w1, w2, robot_id, robots, avoid_robots=False)
                        if path2:
                            path3 = self._try_a_star_path(w2, end, robot_id, robots, avoid_robots=False)
                            if path3:
                                full_path = path1[:-1] + path2[:-1] + path3
                                path_length = sum(self.distance_between(full_path[i], full_path[i+1]) 
                                                for i in range(len(full_path)-1))
                                
                                if path_length < shortest_length:
                                    shortest_length = path_length
                                    best_path = full_path
        
        if best_path:
            return best_path
        
        edge_margin = 40
        corner_path = [
            start,
            (edge_margin, start[1]),
            (edge_margin, edge_margin),
            (self.warehouse.grid_width * self.warehouse.grid_size - edge_margin, edge_margin),
            (self.warehouse.grid_width * self.warehouse.grid_size - edge_margin, self.warehouse.grid_height * self.warehouse.grid_size - edge_margin),
            (edge_margin, self.warehouse.grid_height * self.warehouse.grid_size - edge_margin),
            (edge_margin, end[1]),
            end
        ]
        
        return corner_path