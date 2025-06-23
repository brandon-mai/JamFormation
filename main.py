from simulation import TrafficSimulation, create_two_slowdown_config
import matplotlib.pyplot as plt

def main():
    """Main function to test the traffic simulation."""
    print("Traffic Jam Formation Simulation")
    print("=" * 40)
    
    config = create_two_slowdown_config()
    sim = TrafficSimulation(config)
    
    print(f"Initial headway: {config['initial_headway']:.2f}")
    print(f"Number of vehicles: {config['number_of_vehicles']}")
    print(f"Initial density: {sim.highway.density:.4f}")
    print(f"Sensitivity parameter: {config['sensitivity']}")
    
    sim.run_until_steady_state(max_time=500.0)
    
    print("\nAnalyzing jam formation...")
    results = sim.analyze_jam_formation()
    
    print(f"Total jam length: {results['total_jam_length']:.2f}")
    print(f"Jam ratio: {results['jam_ratio']:.4f}")
    print(f"Number of jams: {results['number_of_jams']}")
    print(f"Jammed vehicles: {results['jammed_vehicles']}")
    
    print("\nGenerating plots...")
    
    sim.plot_headway_profile()
    sim.plot_velocity_profile()
    
    plt.figure(figsize=(12, 8))
    for i in range(0, len(sim.velocity_history), len(sim.velocity_history)//10):
        velocities = sim.velocity_history[i]
        positions = sim.position_history[i]
        time = sim.time_history[i]
        plt.plot(positions, velocities, alpha=0.7, label=f't={time:.1f}')
    
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Velocity Profile Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("Simulation completed successfully!")

if __name__ == "__main__":
    main()