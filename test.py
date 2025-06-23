from highway import Highway
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool


density = 0.2
number_of_vehicles = 500

road_length = number_of_vehicles / density

one_slowdown = Highway(
    density=density,
    number_of_vehicles=number_of_vehicles
)

one_slowdown.set_sections([
    ('normal', 0, road_length * 0.5, 2.0),
    ('slowdown', road_length * 0.5, road_length, 1.0)
])

if __name__ == "__main__":
    def simulate_density_mp(density):
        highway = one_slowdown.fresh_copy()
        highway.density = density
        highway.simulate_till_steady_state()
        return density, highway.get_current()
    
    with Pool() as pool:
        density_values = [np.random.uniform(0.01, 0.8) for _ in range(200)]
        results = pool.map(simulate_density_mp, density_values)

    densities = []
    currents = []

    for density, current in results:
        densities.append(density)
        currents.append(current)

    plt.figure(figsize=(10, 6))
    plt.scatter(densities, currents, alpha=0.7)
    plt.title('Fundamental Diagram: Current vs. Density')
    plt.xlabel('Density')
    plt.ylabel('Current (Flow)')
    plt.grid(True)
    plt.show()