import math

def distance_between(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

class Robot:
    def __init__(self, robot, warehouse):
        self.robot = robot
        self.warehouse = warehouse

        self.id = robot['id']
        self.position = robot['position']
        self.color = robot['color']
        self.order_queue = robot['order_queue']
        self.current_path = robot['current_path']
        self.current_order = robot['current_order']
        self.target_index = robot['target_index']
        self.items_collected = robot['items_collected']
        self.state = robot['state']
        self.radius = robot['radius']
        self.assigned_checkout = robot['assigned_checkout']
        self.reward = robot['reward']
        self.rewarded_items = robot['rewarded_items']

    def reset(self, position):
        self.position = position
        self.order_queue.clear()
        self.current_path = []
        self.target_index = 0
        self.items_collected = []
        self.state = 'idle'
        self.current_order = None
        self.reward = 0
        self.rewarded_items = []

    def process_robot_actions(self):
        if self.state == 'idle' and self.order_queue:
            self.current_order = self.order_queue[0]
            self.state = 'collecting'
            self.items_collected = []
            self.rewarded_items = []
            
            item_locations = []
            for item in self.current_order['items']:
                aisle, shelf = self.warehouse.products[item]
                if (aisle, shelf) in self.warehouse.shelf_to_coord:
                    item_locations.append(self.warehouse.shelf_to_coord[(aisle, shelf)])
            
            route, _ = self.warehouse.tsp_solver.solve_tsp(item_locations, self.position, self.id, self.warehouse.robots)
            full_path = []

            for i in range(len(route) - 1):
                segment = self.warehouse.pathfinding.find_path(route[i], route[i+1], self.id)
                full_path.extend(segment[:-1])  
            
            if full_path:
                full_path.append(route[-1]) 
                self.current_path = full_path
                self.target_index = 0
        elif self.state == 'collecting':
            if self.current_path and self.target_index < len(self.current_path):
                target = self.current_path[self.target_index]
                dx = target[0] - self.position[0]
                dy = target[1] - self.position[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < 2:  
                    self.position = target
                    self.target_index += 1
                    for item in self.current_order['items']:
                        if item not in self.items_collected: 
                            aisle, shelf = self.warehouse.products[item]
                            if (aisle, shelf) in self.warehouse.shelf_to_coord:
                                item_pos = self.warehouse.shelf_to_coord[(aisle, shelf)]
                                if distance_between(self.position, item_pos) < 20:  
                                    self.items_collected.append(item)
                                    print(f"Robot {self.id} collected {item}. Total: {len(self.items_collected)}/{len(self.current_order['items'])}")
                else:
                    speed = 2
                    move_distance = min(speed, distance)
                    angle = math.atan2(dy, dx)
                    new_x = self.position[0] + move_distance * math.cos(angle)
                    new_y = self.position[1] + move_distance * math.sin(angle)
                    collision = False
                    for other_robot in self.warehouse.robots:
                        if other_robot.id != self.id:
                            if distance_between((new_x, new_y), other_robot.position) < 2 * self.radius:
                                collision = True
                                self.warehouse.collision_count += 1
                                if not self.robot.get('collision_repath_timer', 0):
                                    self.robot['collision_repath_timer'] = 10
                                    current_pos = self.position
                                    remaining_path = self.current_path[self.target_index:]
                                    new_path = self.warehouse.pathfinding.find_path(current_pos, remaining_path[-1], self.id)
                                    
                                    # Replace remaining path with new path
                                    self.current_path = self.current_path[:self.target_index] + new_path
                                break

                    if self.robot.get('collision_repath_timer', 0) > 0:
                        self.robot['collision_repath_timer'] -= 1
                    
                    if not collision:
                        self.position = (new_x, new_y)
                    else:
                        for angle_offset in [0.2, -0.2, 0.4, -0.4, 0.6, -0.6]:
                            new_angle = angle + angle_offset
                            test_x = self.position[0] + move_distance * math.cos(new_angle)
                            test_y = self.position[1] + move_distance * math.sin(new_angle)
                            
                            alt_collision = False
                            for other_robot in self.warehouse.robots:
                                if other_robot.id != self.id:
                                    if distance_between((test_x, test_y), other_robot.position) < 2 * self.radius:
                                        alt_collision = True
                                        break
                            
                            if not alt_collision:
                                self.position = (test_x, test_y)
                                break
                
                for item in self.current_order['items']:
                    if item not in self.items_collected:
                        aisle, shelf = self.warehouse.products[item]
                        if (aisle, shelf) in self.warehouse.shelf_to_coord:
                            item_pos = self.warehouse.shelf_to_coord[(aisle, shelf)]
                            if distance_between(self.position, item_pos) < 20:
                                self.items_collected.append(item)
                                print(f"Robot {self.id} collected {item}. Total: {len(self.items_collected)}/{len(self.current_order['items'])}")
            
            if len(self.items_collected) == len(self.current_order['items']):
                print(f"Robot {self.id} collected all items for order {self.current_order['id']}. Heading to checkout.")
                assigned_checkout = self.assigned_checkout
                checkout_rect = self.warehouse.checkouts[assigned_checkout]
                checkout_pos = (checkout_rect.centerx, checkout_rect.centery - 40)  # Position in front of checkout
                self.current_path = self.warehouse.pathfinding.find_path(self.position, checkout_pos, self.id)
                self.target_index = 0
                self.state = 'checkout'
            elif self.target_index >= len(self.current_path):
                remaining_items = [item for item in self.current_order['items'] 
                                if item not in self.items_collected]
                if remaining_items:
                    remaining_locations = []
                    for item in remaining_items:
                        aisle, shelf = self.warehouse.products[item]
                        if (aisle, shelf) in self.warehouse.shelf_to_coord:
                            remaining_locations.append(self.warehouse.shelf_to_coord[(aisle, shelf)])
                    
                    if remaining_locations:
                        route, _ = self.warehouse.tsp_solver.solve_tsp(remaining_locations, self.position, self.id, self.warehouse.robots)
                        full_path = []
                        for i in range(len(route) - 1):
                            segment = self.warehouse.pathfinding.find_path(route[i], route[i+1], self.id)
                            full_path.extend(segment[:-1])
                        
                        if full_path:
                            full_path.append(route[-1])
                            self.current_path = full_path
                            self.target_index = 0
                            print(f"Robot {self.id} regenerating path to {len(remaining_items)} remaining items")
        elif self.state == 'checkout':
            if self.current_path and self.target_index < len(self.current_path):
                target = self.current_path[self.target_index]
                dx = target[0] - self.position[0]
                dy = target[1] - self.position[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < 2: 
                    self.position = target
                    self.target_index += 1
                else:
                    speed = 2
                    move_distance = min(speed, distance)
                    angle = math.atan2(dy, dx)
                    new_x = self.position[0] + move_distance * math.cos(angle)
                    new_y = self.position[1] + move_distance * math.sin(angle)
                    collision = False
                    for other_robot in self.warehouse.robots:
                        if other_robot.id != self.id:
                            if distance_between((new_x, new_y), other_robot.position) < 2 * self.radius:
                                collision = True
                                self.warehouse.collision_count += 1
                                if not self.robot.get('collision_repath_timer', 0):
                                    self.robot['collision_repath_timer'] = 10 
                                    current_pos = self.position
                                    remaining_path = self.current_path[self.target_index:]
                                    new_path = self.warehouse.pathfinding.find_path(current_pos, remaining_path[-1], self.id)
                                    self.current_path = self.current_path[:self.target_index] + new_path
                                break
                    
                    if self.robot.get('collision_repath_timer', 0) > 0:
                        self.robot['collision_repath_timer'] -= 1
                    
                    if not collision:
                        self.position = (new_x, new_y)
                    else:
                        for angle_offset in [0.2, -0.2, 0.4, -0.4, 0.6, -0.6]:
                            new_angle = angle + angle_offset
                            test_x = self.position[0] + move_distance * math.cos(new_angle)
                            test_y = self.position[1] + move_distance * math.sin(new_angle)
                            alt_collision = False
                            for other_robot in self.warehouse.robots:
                                if other_robot.id != self.id:
                                    if distance_between((test_x, test_y), other_robot.position) < 2 * self.radius:
                                        alt_collision = True
                                        break
                            if not alt_collision:
                                self.position = (test_x, test_y)
                                break
            
            if self.target_index >= len(self.current_path):
                print(f"Robot {self.id} completed order {self.current_order['id']} at checkout {self.assigned_checkout+1}")
                self.current_order['status'] = 'completed'
                self.order_queue.popleft()
                self.state = 'idle'
                self.current_path = []