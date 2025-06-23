import numpy as np
import math
from typing import List, Tuple
from tqdm.auto import trange

class Highway:
    """
    Manages the traffic simulation environment including cars and road characteristics.
    """
    
    def __init__(self, density: float, number_of_vehicles: int, 
                 sensitivity: float = 2.5, vf_max: float = 2.0,
                 time_step: float = 1/128, xf_c: float = 2.0, xs_c: float = 2.0):
        """
        Initialize the highway simulation.
        
        Args:
            density: Initial density of vehicles (1 / headway)
            number_of_vehicles: Total number of cars
            sensitivity: Driver sensitivity parameter (a)
            vf_max: Maximum velocity in normal sections
            time_step: Discrete time interval for numerical integration
            xf_c: Turning point parameter for normal sections
            xs_c: Turning point parameter for slowdown sections
        """
        self.sensitivity = sensitivity
        self.vf_max = vf_max
        self.time_step = time_step
        self.xf_c = xf_c
        self.xs_c = xs_c
        self.current_time = 0.0
        self.density = density
        self.number_of_vehicles = number_of_vehicles
        self.road_length = (1 / density) * number_of_vehicles
        
        # State vector: [x1, v1, x2, v2, ..., xN, vN]
        self.state = np.zeros(2 * number_of_vehicles)
        for i in range(number_of_vehicles):
            self.state[2*i] = i * (1 / density)  # position
            perturbation = 0.1 * (np.random.random() - 0.5)
            self.state[2*i + 1] = vf_max * (0.8) # initial velocity
        
        self.slowdowns = []
    
    def fresh_copy(self):
        """Create a fresh copy of the highway object."""
        new_highway = Highway(
            density=self.density,
            number_of_vehicles=self.number_of_vehicles,
            sensitivity=self.sensitivity,
            vf_max=self.vf_max,
            time_step=self.time_step,
            xf_c=self.xf_c,
            xs_c=self.xs_c
        )
        new_highway.sections = self.sections.copy()
        return new_highway

    def set_slowdowns(self, slowdowns: List[Tuple[float, float, float]]):
        """
        Set the highway slowdown sections.
        
        Args:
            sections: List of (start_pos, end_pos, speed_limit) tuples, 
                      where 0.0 <= start_pos < end_pos <= 1.0
                      and speed_limit is the maximum speed in that section.
        """
        for slowdown in slowdowns:
            start_pos, end_pos, speed_limit = slowdown
            if not (0.0 <= start_pos < end_pos <= 1.0):
                raise ValueError("Positions must be in the range [0.0, 1.0) and start_pos < end_pos")
            if speed_limit <= 0:
                raise ValueError("Speed limit must be positive")
        self.slowdowns = slowdowns


    def get_speed_limits(self) -> np.ndarray:
        """Get speed limits for each vehicle based on their current section."""
        positions = self.get_positions()
        speed_limits = np.full(self.number_of_vehicles, self.vf_max, dtype=float)
        
        for i in range(self.number_of_vehicles):
            pos_norm = positions[i] / self.road_length # position normalized to [0, 1)
            for start_pos, end_pos, speed_limit in self.slowdowns:
                if start_pos <= pos_norm < end_pos:
                    speed_limits[i] = speed_limit
                    break
        
        return speed_limits
        
    
    def get_positions(self) -> np.ndarray:
        """Get array of all vehicle positions."""
        return self.state[::2]
    

    def get_velocities(self) -> np.ndarray:
        """Get array of all vehicle velocities."""
        return self.state[1::2]
    

    def get_headways(self) -> np.ndarray:
        """Calculate headways (distance to vehicle in front) for all vehicles."""
        positions = self.get_positions()
        headways = np.roll(positions, -1) - positions
        headways = np.where(headways < 0, self.road_length + headways, headways)

        return headways
    

    def get_optimal_velocities(self) -> np.ndarray:
        """Calculate optimal velocities for all vehicles based on headway and section properties.
        Returns:
            Array of optimal velocities for each vehicle.
        """
        headways = self.get_headways()
        speed_limits = self.get_speed_limits()
        xf_c = 2.0
        velocities = np.zeros(headways.size, dtype=float)
        
        for i in range(headways.size):
            speed_limit = speed_limits[i]
            tanh_arg1 = self.sensitivity * (headways[i] - xf_c)
            tanh_arg2 = self.sensitivity * xf_c
            
            velocities[i] = speed_limit / 2 * (math.tanh(tanh_arg1) + math.tanh(tanh_arg2))
        
        return velocities
    

    def get_current(self, average_headway: float = None) -> float:
        """Calculate the current based on average headway.
        Args:
            average_headway: Average headway.
        Returns:
            Current value.
        """
        pass
    

    def get_derivatives(self) -> np.ndarray:
        """Calculate the derivatives of the state vector.
        Returns:
            Array of derivatives [v1, a1, v2, a2, ..., vN, aN]
        """
        velocities = self.get_velocities()
        optimal_velocities = self.get_optimal_velocities()
        derivatives = np.zeros_like(self.state)
        
        for i in range(self.number_of_vehicles):
            derivatives[2*i] = velocities[i]
            derivatives[2*i + 1] = self.sensitivity * (optimal_velocities[i] - velocities[i])
        
        return derivatives
    

    def rk4_step(self):
        """Perform a single Runge-Kutta 4th order step to update the state."""
        h = self.time_step
        original_state = self.state.copy()
        
        # k1: slope at the beginning
        k1 = h * self.get_derivatives()
        
        # k2: slope at midpoint using k1
        self.state = original_state + k1/2
        k2 = h * self.get_derivatives()
        
        # k3: slope at midpoint using k2
        self.state = original_state + k2/2
        k3 = h * self.get_derivatives()
        
        # k4: slope at end using k3
        self.state = original_state + k3
        k4 = h * self.get_derivatives()
        
        # Final update
        self.state = original_state + (k1 + 2*k2 + 2*k3 + k4)/6
        self.state[::2] %= self.road_length  # wrap positions
        self.current_time += h
    

    def simulate_till_steady_state(self, max_iterations: int = 50000, tolerance: float = 1e-4):
        """Run the simulation until steady state is reached or max iterations."""
        check_interval = 100  # Check every 100 steps
        
        for step in trange(max_iterations):
            self.rk4_step()
            
            if step % check_interval == 0:
                # STEADY MOTHER FUCKING STATE CHECK
                if np.allclose(self.get_velocities(), self.get_optimal_velocities(), atol=tolerance):
                    print(f"Steady state reached at step {step}.")
                    break


if __name__ == "__main__":
    print("puriketsu")