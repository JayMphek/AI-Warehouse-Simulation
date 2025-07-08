import random

class Order_Allocator:
    def __init__(self, warehouse):
        self.warehouse = warehouse
    
    def generate_order(self):
        num_items = random.randint(1, 10)
        available_products = list(self.warehouse.products.keys())
        order_items = random.sample(available_products, num_items)
        checkout_point = random.randint(0, 2)  
        order = {
            'id': len(self.warehouse.order_queue) + 1,
            'items': order_items,
            'checkout': checkout_point,
            'status': 'pending'  # pending, assigned, completed
        }
        return order
    
    def assign_orders_to_robots(self, robots, pathfinder, tsp_solver):
        for order in self.warehouse.order_queue:
            if order['status'] == 'pending':
                for robot in robots:
                    if len(robot.order_queue) < 1:
                        robot.order_queue.append(order)
                        order['status'] = 'assigned'
                        if robot.state == 'idle':
                            robot.current_order = order
                            robot.state = 'collecting'
                            robot.items_collected = []
                            
                            item_locations = []
                            for item in order['items']:
                                aisle, shelf = self.warehouse.products[item]
                                if (aisle, shelf) in self.warehouse.shelf_to_coord:
                                    item_locations.append(self.warehouse.shelf_to_coord[(aisle, shelf)])
                            
                            route, _ = tsp_solver.solve_tsp(item_locations, robot.position, robot.id, robots)
                            full_path = []
                            for i in range(len(route) - 1):
                                segment = pathfinder.find_path(route[i], route[i+1], robot.id, robots)
                                full_path.extend(segment[:-1])  
                            if full_path:
                                full_path.append(route[-1])  
                                robot.current_path = full_path
                                robot.target_index = 0
                        break