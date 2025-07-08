from core.warehouse import WarehouseGenerator
from core.obstacle_avoid import Obstacle_Avoidance

def main():
    warehouse_simulation = WarehouseGenerator()

    rl_agent = Obstacle_Avoidance(warehouse_simulation,warehouse_simulation.robots,warehouse_simulation.order_allocator)
    rl_agent.train(episodes=100, max_steps=1000, num_orders_per_episode=2)
    rl_agent.save_q_tables()
    
    warehouse_simulation.run(use_rl=True, rl_agent=rl_agent)

if __name__ == "__main__":
    main()