import pygame
import random
import numpy as np
from collections import deque
from core.order import Order_Allocator
from core.pathfinding import Pathfinding
from core.tsp_solver import TSP_Solver
from core.robot import Robot

class WarehouseGenerator:
    def __init__(self, width=800, height=600, num_aisles=8, shelves_per_aisle=6):
        self.width = width
        self.height = height
        self.num_aisles = num_aisles
        self.shelves_per_aisle = shelves_per_aisle
        
        self.FLOOR = (240, 240, 240)
        self.SHELF = (160, 82, 45)  # Brown for shelves
        self.AISLE = (220, 220, 220) #Light gray for aisles
        self.ROBOT = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # Blue, Red, Green robots
        self.CHECKOUT = (255, 215, 0)  # Gold for checkout points
        self.TEXT_COLOR = (0, 0, 0)
        self.OBSTACLE = (128, 128, 128)  # Gray for obstacles
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("AI-POWERED RETAIL WAREHOUSE ROBOTIC SIMULATION")
        self.font = pygame.font.SysFont('Arial', 12)
        self.clock = pygame.time.Clock()
        
        self.create_warehouse()
        self.robots = self.create_robots()
        self.obstacles = self.create_obstacles(10)  
        self.products = self.create_product_database()
        self.order_queue = []
        self.collision_count = 0
        self.pathfinding = Pathfinding(self)
        self.tsp_solver = TSP_Solver(self.pathfinding)
        self.order_allocator = Order_Allocator(self)
        
    def create_warehouse(self):
        aisle_width = 40
        shelf_width = 20
        shelf_length = 60
        margin = 40
        
        total_aisle_space = self.width - 2 * margin
        aisle_spacing = total_aisle_space / (self.num_aisles + 1)
        
        self.shelves = []
        self.aisles = []
        self.shelf_to_coord = {}  
        
        shelf_id = 1
        for aisle in range(self.num_aisles):
            aisle_x = margin + aisle_spacing * (aisle + 1)
            aisle_rect = pygame.Rect(aisle_x - aisle_width // 2, margin, 
                                    aisle_width, self.height - 2 * margin)
            self.aisles.append(aisle_rect)
            
            shelf_spacing = (self.height - 2 * margin) / (self.shelves_per_aisle + 1)
            for shelf in range(self.shelves_per_aisle):
                shelf_y = margin + shelf_spacing * (shelf + 1)
                shelf_rect = pygame.Rect(aisle_x - aisle_width // 2 - shelf_width, 
                                    shelf_y - shelf_length // 2,
                                    shelf_width, shelf_length)
                self.shelves.append(shelf_rect)
                self.shelf_to_coord[(aisle+1, shelf*2+1)] = (aisle_x - aisle_width // 4, shelf_y)
                shelf_id += 1
            for shelf in range(self.shelves_per_aisle):
                shelf_y = margin + shelf_spacing * (shelf + 1)
                shelf_rect = pygame.Rect(aisle_x + aisle_width // 2, 
                                    shelf_y - shelf_length // 2,
                                    shelf_width, shelf_length)
                self.shelves.append(shelf_rect)
                self.shelf_to_coord[(aisle+1, shelf*2+2)] = (aisle_x + aisle_width // 4, shelf_y)
                shelf_id += 1
        
        self.checkouts = []
        checkout_width = 40
        checkout_height = 30
        checkout_spacing = self.width / 4
        for i in range(3):
            x = checkout_spacing * (i + 1) - checkout_width // 2
            y = self.height - margin // 2 - checkout_height // 2
            self.checkouts.append(pygame.Rect(x, y, checkout_width, checkout_height))

        self.grid_size = 10  
        self.grid_width = self.width // self.grid_size
        self.grid_height = self.height // self.grid_size
        self.navigation_grid = np.ones((self.grid_height, self.grid_width), dtype=bool)
        
        for shelf in self.shelves:
            x1, y1 = shelf.topleft[0] // self.grid_size, shelf.topleft[1] // self.grid_size
            x2, y2 = shelf.bottomright[0] // self.grid_size, shelf.bottomright[1] // self.grid_size
            for x in range(max(0, x1), min(self.grid_width, x2 + 1)):
                for y in range(max(0, y1), min(self.grid_height, y2 + 1)):
                    self.navigation_grid[y, x] = False
        for checkout in self.checkouts:
            x1, y1 = checkout.topleft[0] // self.grid_size, checkout.topleft[1] // self.grid_size
            x2, y2 = checkout.bottomright[0] // self.grid_size, checkout.bottomright[1] // self.grid_size
            for x in range(max(0, x1), min(self.grid_width, x2 + 1)):
                for y in range(max(0, y1), min(self.grid_height, y2 + 1)):
                    self.navigation_grid[y, x] = False
                
    def create_robots(self):
        robots = []
        checkout_positions = [
            (self.checkouts[0].centerx, self.checkouts[0].centery - 50),  # Robot 1 near Checkout 1
            (self.checkouts[1].centerx, self.checkouts[1].centery - 50),  # Robot 2 near Checkout 2
            (self.checkouts[2].centerx, self.checkouts[2].centery - 50)   # Robot 3 near Checkout 3
        ]

        for i in range(3): 
            robot_data = {
                'id': i + 1,
                'position': checkout_positions[i],  
                'color': self.ROBOT[i],
                'order_queue': deque(maxlen=3),
                'current_path': [],
                'current_order': None,
                'target_index': 0,
                'items_collected': [],
                'state': 'idle',  # idle, collecting, checkout
                'radius': 10,
                'assigned_checkout': i, # Assign each robot to a their checkout
                'reward': 0,
                'rewarded_items': []
            }
            robot = Robot(robot_data, self)
            robots.append(robot)
        return robots
    
    def reset_for_rl_training(self):
        checkout_positions = [
            (self.checkouts[0].centerx, self.checkouts[0].centery - 50),
            (self.checkouts[1].centerx, self.checkouts[1].centery - 50),
            (self.checkouts[2].centerx, self.checkouts[2].centery - 50)
        ]
        for i, robot in enumerate(self.robots):
            start_pos = checkout_positions[i % len(checkout_positions)]
            robot.reset(start_pos)
        
        self.order_queue = []
        for i in range(len(self.robots)):
            new_order = self.order_allocator.generate_order()
            new_order['checkout'] = i % 3 
            self.order_queue.append(new_order)
    
    def create_obstacles(self, num_obstacles):
        obstacles = []
        for _ in range(num_obstacles):
            valid_position = False
            while not valid_position:
                x = random.randint(50, self.width - 50)
                y = random.randint(50, self.height - 50)
                obstacle_rect = pygame.Rect(x - 15, y - 15, 30, 30)
                valid_position = True
                for shelf in self.shelves:
                    if shelf.colliderect(obstacle_rect):
                        valid_position = False
                        break
                
                for checkout in self.checkouts:
                    if checkout.colliderect(obstacle_rect):
                        valid_position = False
                        break
                
                for aisle in self.aisles:
                    if aisle.colliderect(obstacle_rect):
                        valid_position = False
                        break
            obstacles.append(obstacle_rect)
            
            x1, y1 = obstacle_rect.topleft[0] // self.grid_size, obstacle_rect.topleft[1] // self.grid_size
            x2, y2 = obstacle_rect.bottomright[0] // self.grid_size, obstacle_rect.bottomright[1] // self.grid_size
            for x in range(max(0, x1), min(self.grid_width, x2 + 1)):
                for y in range(max(0, y1), min(self.grid_height, y2 + 1)):
                    self.navigation_grid[y, x] = False
                    
        return obstacles
    
    def create_product_database(self):
        product_categories = {
            "Dairy & Bakery":   ["Milk", "Cheese", "Yogurt", "Butter", "Cream", "Custard", "Bread", "Buns", "Muffins", "Scones","Cupcakes", "Cake"],
            "Fruits & Veg":     ["Apples", "Bananas", "Oranges", "Grapes", "Strawberries", "Blueberries", "Lettuce", "Peppers", "Tomatoes", "Cucumber", "Carrots", "Onions"],
            "Butchery":             ["Chicken", "Beef", "Pork", "Fish", "Shrimp", "Tofu","Crab","Eggs","Viennas", "Polony", "Russian","Mince Meat"],
            "Pastries":           ["Rice", "Pasta", "Cereal", "Flour", "Sugar", "Salt","Tumeric", "Paprika", "Masala", "Parsley", "Oil", "Sauce"],
            "Beverages":        ["Soda", "Water", "Juice", "Coffee", "Tea", "Beer", "Ice", "Wine", "Champagne", "Cider", "Vodka", "Milkshake"],
            "Snacks":           ["Chips", "Cookies", "Crackers", "Sweets", "Chocolate", "Nuts", "Popcorns", "Energy bars", "Pretzel", "Biscuits", "Granola", "Muesli"],
            "Toiletries":       ["Soap", "Shampoo", "Toothpaste", "Toilet Paper", "Paper Towels", "Detergent", "Face Cloth", "Spray", "Lotion", "Roll-on", "Loafer", "Toothbrush"],
            "Cleaning":         ["Broom", "Mop", "Floor cleaner", "Pine gel", "Dustpan", "Brush", "Dishwasher", "Splunger", "Bucket", "Vaccumm", "Cloth", "Rack"]
        }

        product_mapping = {}
        aisle_names = {} 
        aisle_number = 1  
        for category, items in product_categories.items():
            aisle_names[aisle_number] = category  
            for shelf in range(1, 13):  
                product_mapping[items[shelf - 1]] = (aisle_number, shelf) 
            aisle_number += 1 

        self.aisle_names = aisle_names 
        return product_mapping
    
    def draw(self):
        # Fill background
        self.screen.fill(self.FLOOR)
        
        # Draw aisles
        for aisle in self.aisles:
            pygame.draw.rect(self.screen, self.AISLE, aisle)
        
        # Draw shelves
        for shelf in self.shelves:
            pygame.draw.rect(self.screen, self.SHELF, shelf)
        
        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, self.OBSTACLE, obstacle)
        
        # Draw checkout points
        for i, checkout in enumerate(self.checkouts):
            pygame.draw.rect(self.screen, self.CHECKOUT, checkout)
            checkout_text = self.font.render(f"Checkout {i+1}", True, self.TEXT_COLOR)
            self.screen.blit(checkout_text, (checkout.x, checkout.y - 15))
        
        # Draw shelf labels, aisle group labels above each aisle
        for aisle, group in self.aisle_names.items():
            aisle_x = self.aisles[aisle - 1].centerx
            label_surface = self.font.render(group, True, self.TEXT_COLOR)
            self.screen.blit(label_surface, (aisle_x - len(group) * 3, 20))

        # Draw vertical product labels on shelves
        for product, (aisle, shelf) in self.products.items():
            if (aisle, shelf) in self.shelf_to_coord:
                pos = self.shelf_to_coord[(aisle, shelf)]
                label_surface = self.font.render(product, True, self.TEXT_COLOR)
                label_surface = pygame.transform.rotate(label_surface, 90)
                self.screen.blit(label_surface, (pos[0] - 10, pos[1] - len(product) * 3))
  
        # Draw robots and their paths
        for robot in self.robots:
            if robot.current_path and robot.target_index < len(robot.current_path):
                path_color = robot.color
                for i in range(robot.target_index, len(robot.current_path) - 1):
                    pygame.draw.line(self.screen, path_color, 
                                robot.current_path[i],
                                robot.current_path[i+1], 2)
            pygame.draw.circle(self.screen, robot.color, robot.position, robot.radius)

            id_text = self.font.render(f"R{robot.id}", True, (255, 255, 255))
            self.screen.blit(id_text, (robot.position[0] - 5, robot.position[1] - 5))
            if robot.current_order:
                order_text = self.font.render(f"Order: {robot.current_order['id']}", 
                                        True, self.TEXT_COLOR)
                self.screen.blit(order_text, (robot.position[0] - 60, robot.position[1] - 40))
                items_text = self.font.render(f"Items: {len(robot.items_collected)}/{len(robot.current_order['items'])}", 
                                        True, self.TEXT_COLOR)
                self.screen.blit(items_text, (robot.position[0] - 50, robot.position[1] - 25))     
                reward_text = self.font.render(f"Reward: {int(robot.reward)}", True, self.TEXT_COLOR)
                self.screen.blit(reward_text, (robot.position[0] - 50, robot.position[1] + 35))


        # Draw product location markers
        for (aisle, shelf), pos in self.shelf_to_coord.items():
            pygame.draw.circle(self.screen, (255, 0, 125), pos, 3)                                 
        
        # Display instructions
        instructions = self.font.render("Press 'O' to add new order | 'R' to reset | ESC to quit", True, self.TEXT_COLOR)
        self.screen.blit(instructions, (self.width - 380, 10))

        # Display number of collisions
        collision_text = self.font.render(f"Collisions: {self.collision_count}", True, self.TEXT_COLOR)
        self.screen.blit(collision_text, (10, 10))
        pygame.display.flip()
        
    def run(self, use_rl=False, rl_agent=None):
        running = True
        order_timer = 0
        
        while running:
            self.clock.tick(60) 
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_o:
                        new_order = self.order_allocator.generate_order()
                        if new_order:
                            self.order_queue.append(new_order)
                            print(f"New order #{new_order['id']} generated: {new_order['items']}")
                    elif event.key == pygame.K_r:
                        self.__init__(self.width, self.height, self.num_aisles, self.shelves_per_aisle)

            order_timer += 1
            if order_timer >= 120:
                if len(self.order_queue) < 9:
                    new_order = self.order_allocator.generate_order()
                    if new_order:
                        self.order_queue.append(new_order)
                        print(f"Auto-generated order #{new_order['id']}: {new_order['items']}")
                order_timer = 0

            self.order_allocator.assign_orders_to_robots(self.robots, self.pathfinding, self.tsp_solver)

            if use_rl and rl_agent:
                for robot in self.robots:
                    if robot.state != 'idle':
                        rl_agent.act(robot)
            
            for robot in self.robots:
                robot.process_robot_actions()
            self.draw()
            pygame.display.flip()
        pygame.quit()
