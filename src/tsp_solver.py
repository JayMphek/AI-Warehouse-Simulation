import numpy as np

class TSP_Solver:
    def __init__(self, pathfinder):
        self.pathfinder = pathfinder
    
    def solve_tsp(self, locations, start_pos, robot_id, robots):
        if not locations:
            return [], 0
    
        all_locations = [start_pos] + locations
        n = len(all_locations)
        if n <= 2:
            return all_locations, self.pathfinder.distance_between(start_pos, locations[0])
        
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n): 
                path = self.pathfinder.find_path(all_locations[i], all_locations[j], robot_id, robots, False)
                path_length = sum(self.pathfinder.distance_between(path[k], path[k+1]) for k in range(len(path)-1))
                dist_matrix[i, j] = path_length
                dist_matrix[j, i] = path_length  
        
        nearest = min(range(1, n), key=lambda i: dist_matrix[0, i])
        route = [0, nearest]
        unvisited = set(range(1, n))
        unvisited.remove(nearest)
        while unvisited:
            best_insertion = None
            best_cost_increase = float('inf')
            
            for loc in unvisited:
                for i in range(1, len(route)):
                    prev = route[i-1]
                    curr = route[i]
                    cost_increase = (dist_matrix[prev, loc] + dist_matrix[loc, curr] - dist_matrix[prev, curr])
                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_insertion = (loc, i)
            
            loc_to_insert, position = best_insertion
            route.insert(position, loc_to_insert)
            unvisited.remove(loc_to_insert)
        
        total_distance = sum(dist_matrix[route[i], route[i+1]] for i in range(len(route)-1))
        final_route = [all_locations[i] for i in route]
        return final_route, total_distance