# Quantum Gravity Cellular Automaton Simulation

This project implements a **quantum gravity simulation** using a **cellular automaton** model based on 4-simplices, representing local spacetime geometries. The model propagates quantum fields, such as the electromagnetic and fermion fields, stochastically. At larger scales, classical fields and spacetime curvature emerge naturally. Gravitational effects are simulated probabilistically by conjoining simplices where energy-momentum is high (representing gravitational contraction) and splitting them where energy-momentum is low (simulating cosmic expansion).

## Features

- **Quantum Field Propagation**: Fields (electromagnetic, fermion, Higgs, etc.) propagate stochastically, with quantum interference, superposition, and probabilistic behavior at each node.
- **Emergent Classical Physics**: At macroscopic scales, quantum fluctuations average out, leading to the emergence of classical physics (such as Maxwell's equations and general relativity).
- **Probabilistic Gravity**: Gravity is modeled probabilistically. Simplices are more likely to conjoin where energy-momentum is high, simulating gravitational contraction, and more likely to split where energy-momentum is low, simulating spacetime expansion.
- **Spacetime Evolution**: Spacetime dynamically evolves by adding and connecting 4-simplices over time.

## How It Works

1. **4-Simplex Representation**: The simulation models spacetime using 4-simplices (a 5-vertex structure in 4 dimensions). Each simplex is connected to its neighbors by shared 3D faces (tetrahedra).
   
2. **Quantum Field Propagation**: Fields such as electromagnetic, fermion, and Higgs propagate stochastically, with quantum superposition and interference built into the propagation rules.

3. **Gravity and Spacetime Dynamics**: The simplicial structure evolves probabilistically based on local energy-momentum. Higher energy densities make neighboring simplices more likely to conjoin, simulating gravitational contraction, while lower energy densities make simplices more likely to split, simulating cosmic expansion.

4. **Emergent Classical Physics**: As quantum effects propagate, classical behavior emerges naturally at larger scales due to averaging over many simplices.

## Simulation Steps

- Initialize a set of 4-simplices, representing the initial spacetime.
- Evolve spacetime by adding new simplices and propagating quantum fields stochastically.
- Modify the simplicial structure probabilistically based on local energy-momentum, simulating gravitational effects.

## How to Run the Simulation

1. Clone the repository or download the script.
   
2. Ensure you have all required dependencies installed (numpy, networkx, matplotlib).

3. Run the simulation:


The simulation will initialize the spacetime structure, propagate quantum fields, and evolve the spacetime according to probabilistic gravity dynamics.

## Code Structure

- **Simplex4D Class**: Represents a 4-simplex with vertices and quantum fields (electromagnetic, fermion, Higgs, etc.).
- **Quantum Field Propagation**: Simulates the propagation of quantum fields with stochastic effects.
- **Probabilistic Gravity**: Simulates gravitational effects by probabilistically modifying the simplicial structure based on energy-momentum density.
- **Spacetime Evolution**: Adds new simplices over time and evolves the spacetime.

## Future Work

- **Visualization**: Adding visualizations of the simplicial structure and field propagation.
- **Refining Energy Density Calculations**: Improve the accuracy of energy-momentum density calculations based on quantum fields.
- **Extending the Model**: Explore different spacetime topologies and field interactions.

## License

This project is licensed under the MIT License.
