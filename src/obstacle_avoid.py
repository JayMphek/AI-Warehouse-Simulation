import math
import random
from collections import defaultdict
import ast
import time

class Obstacle_Avoidance:
    def __init__(self, warehouse_env, robot, order_alloc, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3):
        self.warehouse = warehouse_env
        self.robot = robot
        self.order_alloc = order_alloc
        self.alpha = learning_rate      
        self.gamma = discount_factor     
        self.epsilon = exploration_rate  
        self.q_table = defaultdict(lambda: defaultdict(float))  
        self.grid_size = 10  
        
        self.actions = [
            (0, -1),    # North
            (1, -1),    # Northeast
            (1, 0),     # East
            (1, 1),     # Southeast
            (0, 1),     # South
            (-1, 1),    # Southwest
            (-1, 0),    # West
            (-1, -1)    # Northwest
        ]
        
        self.rewards = {
            'collect_item': 100.0,      # Reward for collecting an item
            'complete_order': 200.0,    # Reward for completing an order
            'collision': -100.0,        # Penalty for collision with obstacle or robot
            'idle': 0.0,               # No reward for being idle
            'approaching_target': 15.0,  # Reward for getting closer to target
            'away_from_target': -10.0,   # Penalty for moving away from target
            'checkout': 150.0,          # Reward for reaching checkout
        }
        self.robot_rewards = {i+1: 0 for i in range(len(self.warehouse.robots))}

        self.episode_rewards = []
        self.collision_counts = []
        self.items_collected = []
        self.orders_completed = []
        self.robot_q_tables = {i+1: defaultdict(lambda: defaultdict(float)) for i in range(len(self.warehouse.robots))}
            
    def discretize_state(self, robot):
        x, y = robot.position
        grid_x = int(x / self.warehouse.width * self.grid_size)
        grid_y = int(y / self.warehouse.height * self.grid_size)
        
        nearest_obstacles = []
        for obstacle in self.warehouse.obstacles:
            obs_x, obs_y = obstacle.center
            dist = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            if dist < 100:  
                obs_grid_x = int(obs_x / self.warehouse.width * self.grid_size)
                obs_grid_y = int(obs_y / self.warehouse.height * self.grid_size)
                nearest_obstacles.append((obs_grid_x - grid_x, obs_grid_y - grid_y))
        
        nearest_robots = []
        for other_robot in self.warehouse.robots:
            if other_robot.id != robot.id:
                other_x, other_y = other_robot.position
                dist = math.sqrt((x - other_x)**2 + (y - other_y)**2)
                if dist < 100:  
                    other_grid_x = int(other_x / self.warehouse.width * self.grid_size)
                    other_grid_y = int(other_y / self.warehouse.height * self.grid_size)
                    nearest_robots.append((other_grid_x - grid_x, other_grid_y - grid_y))
        
        if robot.current_path and robot.target_index < len(robot.current_path):
            target = robot.current_path[robot.target_index]
            target_x, target_y = target
            target_grid_x = int(target_x / self.warehouse.width * self.grid_size)
            target_grid_y = int(target_y / self.warehouse.height * self.grid_size)
            target_pos = (target_grid_x - grid_x, target_grid_y - grid_y)
        else:
            target_pos = (0, 0)  

        robot_state = robot.state
        state = (
            (grid_x, grid_y),
            target_pos,
            robot_state,
            tuple(sorted(nearest_obstacles[:3])),  
            tuple(sorted(nearest_robots[:2]))     
        )        
        return state
    
    def get_valid_actions(self, robot):
        valid_actions = []
        x, y = robot.position
        
        for i, (dx, dy) in enumerate(self.actions):
            speed = 2
            new_x = x + dx * speed
            new_y = y + dy * speed
            
            if (0 < new_x < self.warehouse.width and 0 < new_y < self.warehouse.height):
                collision = False
                for obstacle in self.warehouse.obstacles:
                    if obstacle.collidepoint(new_x, new_y):
                        collision = True
                        break
                
                if not collision:
                    for other_robot in self.warehouse.robots:
                        if other_robot.id != robot.id:
                            if math.dist((new_x, new_y), other_robot.position) < 2 * robot.radius:
                                collision = True
                                break
                    
                    if not collision:
                        valid_actions.append(i)
        
        if not valid_actions:
            valid_actions = list(range(len(self.actions)))
        
        return valid_actions
    
    def choose_action(self, robot):
        state = self.discretize_state(robot)
        valid_actions = self.get_valid_actions(robot)
        
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_table = self.robot_q_tables[robot.id]
            
            if state not in q_table or not any(q_table[state][a] != 0 for a in valid_actions):
                return random.choice(valid_actions)
            
            q_values = [q_table[state][a] for a in valid_actions]
            max_q = max(q_values)
            best_actions = [action for i, action in enumerate(valid_actions) if q_values[i] == max_q]
            return random.choice(best_actions)
    
    def get_reward(self, robot, old_state, new_state, action, old_position, new_position):
        reward = 0
        
        if old_position == new_position:
            reward += self.rewards['idle']
        
        collision = False
        for obstacle in self.warehouse.obstacles:
            if obstacle.collidepoint(new_position[0], new_position[1]):
                reward += self.rewards['collision']
                collision = True
                break
        if collision:
            return reward  
        
        for other_robot in self.warehouse.robots:
            if other_robot.id != robot.id:
                if math.dist(new_position, other_robot.position) < 2 * robot.radius:
                    reward += self.rewards['collision']
                    return reward  # Early return for collision
        
        if robot.current_order:
            for item in robot.current_order['items']:
                if item not in robot.rewarded_items:  
                    aisle, shelf = self.warehouse.products[item]
                    if (aisle, shelf) in self.warehouse.shelf_to_coord:
                        item_pos = self.warehouse.shelf_to_coord[(aisle, shelf)]
                        if math.dist(new_position, item_pos) < 20:
                            reward += self.rewards['collect_item']
                            robot.rewarded_items.append(item)

        for item in robot.rewarded_items:
            aisle, shelf = self.warehouse.products[item]
            if (aisle, shelf) in self.warehouse.shelf_to_coord:
                item_pos = self.warehouse.shelf_to_coord[(aisle, shelf)]
                if math.dist(new_position, item_pos) < 20:
                    reward -= 2
        
        if robot.state == 'checkout' and robot.target_index >= len(robot.current_path) - 1:
            if robot.current_order:
                checkout_idx = robot.current_order['checkout']
                checkout_pos = (self.warehouse.checkouts[checkout_idx].centerx, 
                              self.warehouse.checkouts[checkout_idx].centery - 20)
                if math.dist(new_position, checkout_pos) < 30:
                    reward += self.rewards['checkout']
                    
                    if set(robot.items_collected) == set(robot.current_order['items']):
                        reward += self.rewards['complete_order']
        
        if robot.current_path and robot.target_index < len(robot.current_path):
            target = robot.current_path[robot.target_index]
            old_dist = math.dist(old_position, target)
            new_dist = math.dist(new_position, target)
            
            if new_dist < old_dist:
                reward += self.rewards['approaching_target']
            else:
                reward += self.rewards['away_from_target']
        
        return reward
    
    def update_q_value(self, robot, state, action, next_state, reward):
        q_table = self.robot_q_tables[robot.id]
        current_q = q_table[state][action]
        next_max_q = max([q_table[next_state][a] for a in range(len(self.actions))], default=0)
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        q_table[state][action] = new_q
    
    def act(self, robot):
        state = self.discretize_state(robot)
        action_idx = self.choose_action(robot)
        dx, dy = self.actions[action_idx]
        old_position = robot.position
        
        speed = 2
        new_x = old_position[0] + dx * speed
        new_y = old_position[1] + dy * speed
        new_position = (new_x, new_y)
        
        new_x = max(robot.radius, min(new_x, self.warehouse.width - robot.radius))
        new_y = max(robot.radius, min(new_y, self.warehouse.height - robot.radius))
        new_position = (new_x, new_y)
        robot.position = new_position
        
        next_state = self.discretize_state(robot)
        reward = self.get_reward(robot, state, next_state, action_idx, old_position, new_position)
        robot.reward += reward
        self.update_q_value(robot, state, action_idx, next_state, reward)
        
        return reward
    
    def update_learning_parameters(self, episode, total_episodes):
        self.epsilon = max(0.05, 0.9 * (1 - episode / total_episodes))
        self.alpha = max(0.01, 0.1 * (1 - episode / (2 * total_episodes)))
    
    def train(self, episodes=1000, max_steps=500, num_orders_per_episode=6):
        print(f"Starting training for {episodes} episodes with {num_orders_per_episode} orders per episode...")
        start_time = time.time()
        
        if len(self.warehouse.robots) < 2:
            print("Warning: Adding additional robots for training")
            current_robots = len(self.warehouse.robots)
            for i in range(current_robots, 2):
                self.warehouse.create_robot()
            self.robot_q_tables = {i+1: defaultdict(lambda: defaultdict(float)) for i in range(len(self.warehouse.robots))}
        
        for episode in range(episodes):
            self.warehouse.reset_for_rl_training()
            self.warehouse.order_queue = []
            for _ in range(num_orders_per_episode):
                new_order = self.warehouse.order_allocator.generate_order()
                new_order['checkout'] = random.randint(0, len(self.warehouse.checkouts) - 1)
                self.warehouse.order_queue.append(new_order)
            
            orders_completed = 0
            total_reward = 0
            collisions = 0
            items_collected_count = 0
            
            for robot in self.warehouse.robots:
                robot.reward = 0
                robot.rewarded_items = []  # Track items already rewarded
            
            for step in range(max_steps):
                self.order_alloc.assign_orders_to_robots(self.warehouse.robots,self.warehouse.pathfinding,self.warehouse.tsp_solver)

                for robot in self.warehouse.robots:
                    if robot.state != 'idle':
                        reward = self.act(robot)
                        total_reward += reward
                        if reward <= self.rewards['collision']:
                            collisions += 1
                
                for robot in self.warehouse.robots:
                    prev_state = robot.state
                    prev_items = len(robot.items_collected)
                    robot.process_robot_actions()
                    new_items = len(robot.items_collected) - prev_items
                    items_collected_count += new_items
                    if prev_state == 'checkout' and robot.state == 'idle':
                        orders_completed += 1
                
                if orders_completed >= num_orders_per_episode:
                    print(f"Episode {episode + 1}: All {num_orders_per_episode} orders completed in {step+1} steps!")
                    break
                
                if (len(self.warehouse.order_queue) == 0 and 
                    all(robot.state == 'idle' and not robot.order_queue for robot in self.warehouse.robots)):
                    print(f"Episode {episode + 1}: Processed all available orders in {step+1} steps.")
                    break
            
            for robot in self.warehouse.robots:
                self.robot_rewards[robot.id] = robot.reward
            
            self.episode_rewards.append(total_reward)
            self.collision_counts.append(collisions)
            self.items_collected.append(items_collected_count)
            self.orders_completed.append(orders_completed)
            self.update_learning_parameters(episode, episodes)
            
            if (episode + 1) % 10 == 0 or episode == 0:
                print(f"Episode {episode + 1}/{episodes}, "
                    f"Reward: {total_reward:.2f}, "
                    f"Collisions: {collisions}, "
                    f"Items: {items_collected_count}, "
                    f"Orders: {orders_completed}/{num_orders_per_episode}, "
                    f"Epsilon: {self.epsilon:.4f}")
                
                for robot_id, reward in self.robot_rewards.items():
                    print(f"  Robot {robot_id} reward: {reward:.2f}")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Final exploration rate: {self.epsilon:.4f}")
        
        avg_reward = sum(self.episode_rewards[-10:]) / 10
        avg_collisions = sum(self.collision_counts[-10:]) / 10
        avg_orders = sum(self.orders_completed[-10:]) / 10
        print(f"Final 10 episodes - Avg reward: {avg_reward:.2f}, "
            f"Avg collisions: {avg_collisions:.2f}, "
            f"Avg orders completed: {avg_orders:.2f}/{num_orders_per_episode}")
        
        return {
            'rewards': self.episode_rewards,
            'collisions': self.collision_counts,
            'items_collected': self.items_collected,
            'orders_completed': self.orders_completed
        }
    
    def save_q_tables(self, filename="robot_q_tables.txt"):
        try:
            with open(filename, 'w') as f:
                f.write("# Robot Q-Tables\n")
                f.write("# Format: robot_id,state,action,q_value\n\n")
                for robot_id, table in self.robot_q_tables.items():
                    f.write(f"# Robot {robot_id}\n")
                    for state, actions in table.items():
                        for action, value in actions.items():
                            if value != 0:  
                                state_str = str(state).replace(',', ';')
                                f.write(f"{robot_id},{state_str},{action},{value}\n")
                    f.write("\n")  
            print(f"Q-tables successfully saved to {filename}")
        except Exception as e:
            print(f"Error saving Q-tables: {e}")
    
    def load_q_tables(self, filename="robot_q_tables.txt"):
        try:
            self.robot_q_tables = {i+1: defaultdict(lambda: defaultdict(float)) for i in range(len(self.warehouse.robots))}
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        parts = line.split(',', 3)  
                        if len(parts) != 4:
                            continue
                        robot_id = int(parts[0])
                        state_str = parts[1].replace(';', ',')
                        action = int(parts[2])
                        value = float(parts[3])
                        state = ast.literal_eval(state_str)
                        self.robot_q_tables[robot_id][state][action] = value
                    except (ValueError, SyntaxError) as e:
                        print(f"Warning: Could not parse line: {line}")
                        print(f"Error: {e}")
                        continue
            print(f"Q-tables successfully loaded from {filename}")
        except Exception as e:
            print(f"Error loading Q-tables: {e}")
