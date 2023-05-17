import numpy as np

from vivarium.core.process import Process
from vivarium.core.engine import Engine

from tqdm import tqdm
import readdy
from pint import UnitRegistry

from ..util import monomer_ports_schema, create_monomer_update

NAME = "READDY"


class ReaddyProcess(Process):
    """
    This process uses ReaDDy to model arbitrary
    reaction diffusion systems
    """

    name = NAME

    defaults = {
        "internal_timestep": 0.1,
        "box_size": 100.0,
        "periodic_boundary": False,
        "temperature_C": 22.0,
        "viscosity": 1.0,  # cP
        "force_constant": 250.0,
        "n_cpu": 4,
        "particle_radii": {},
        "topology_types": [],
        "topology_particles": [],
        "bond_pairs": [],
        "reactions": [],
        "time_units": "s",
        "spatial_units": "m",
    }

    def __init__(self, parameters=None):
        super(ReaddyProcess, self).__init__(parameters)
        self.create_readdy_system()

    def ports_schema(self):
        return monomer_ports_schema

    def next_update(self, timestep, states):
        self.create_readdy_simulation()
        self.add_particle_instances(states["monomers"])
        self.simulate_readdy(timestep)
        new_monomers = self.get_current_monomers()
        return create_monomer_update(states["monomers"], new_monomers)

    def create_readdy_system(self):
        """
        Create the ReaDDy system
        including particle species, constraints, and reactions
        """
        self.system = readdy.ReactionDiffusionSystem(
            box_size=[self.parameters["box_size"]] * 3,
            periodic_boundary_conditions=3
            * [bool(self.parameters["periodic_boundary"])],
        )
        self.parameters["temperature_K"] = self.parameters["temperature_C"] + 273.15
        self.system.temperature = self.parameters["temperature_K"]
        self.add_particle_species()
        self.add_topology_types()
        all_particle_types = self.parameters["particle_radii"].keys()
        self.check_add_global_box_potential(all_particle_types)
        self.add_repulsions(all_particle_types)
        self.add_bonds()
        self.add_reactions()

    @staticmethod
    def calculate_diffusionCoefficient(radius, viscosity, temperature, spatial_units):
        """
        calculate the theoretical diffusion constant of a spherical particle
            with radius [spatial_units]
            in a media with viscosity [cP]
            at temperature [Kelvin]

            returns [spatial_units^2/s]
        """
        ureg = UnitRegistry()
        convert_to_nm = ureg(spatial_units).to("m").magnitude
        return (
            (1.38065 * 10 ** (-23) * temperature)
            / (6 * np.pi * viscosity * 10 ** (-3) * radius * convert_to_nm)
            / 10**9
            / (convert_to_nm * convert_to_nm)
        )

    def add_particle_species(self):
        """
        Add all particle species
        """
        added_particle_types = []
        for particle_name in self.parameters["particle_radii"]:
            if particle_name in added_particle_types:
                continue
            diffCoeff = ReaddyProcess.calculate_diffusionCoefficient(
                self.parameters["particle_radii"][particle_name],
                self.parameters["viscosity"],
                self.parameters["temperature_K"],
                self.parameters["spatial_units"],
            )
            if particle_name in self.parameters["topology_particles"]:
                self.system.add_topology_species(particle_name, diffCoeff)
            else:
                self.system.add_species(particle_name, diffCoeff)
            added_particle_types.append(particle_name)

    def add_topology_types(self):
        """
        Add all topology types
        """
        for topology_type in self.parameters["topology_types"]:
            self.system.topologies.add_type(topology_type)

    def check_add_global_box_potential(self, all_particle_types):
        """
        If the boundaries are not periodic,
        add a box potential for all particles
        to keep them in the box volume
        (margin of 1.0 units on each side)
        """
        if bool(self.parameters["periodic_boundary"]):
            return
        box_potential_size = np.array([self.parameters["box_size"] - 2.0] * 3)
        for particle_type in all_particle_types:
            self.system.potentials.add_box(
                particle_type=particle_type,
                force_constant=self.parameters["force_constant"],
                origin=-0.5 * box_potential_size,
                extent=box_potential_size,
            )

    def add_repulsions(self, all_particle_types):
        """
        Add pairwise harmonic repulsions for all particle pairs
        to enforce volume exclusion
        """
        for type1 in all_particle_types:
            for type2 in all_particle_types:
                self.system.potentials.add_harmonic_repulsion(
                    type1,
                    type2,
                    force_constant=self.parameters["force_constant"],
                    interaction_distance=(
                        self.parameters["particle_radii"].get(type1, 1.0)
                        + self.parameters["particle_radii"].get(type2, 1.0)
                    ),
                )

    def add_bonds(self):
        """
        Add harmonic bonds
        """
        for bond_pair in self.parameters["bond_pairs"]:
            self.system.topologies.configure_harmonic_bond(
                bond_pair[0],
                bond_pair[1],
                force_constant=self.parameters["force_constant"],
                length=(
                    self.parameters["particle_radii"].get(bond_pair[0], 1.0)
                    + self.parameters["particle_radii"].get(bond_pair[1], 1.0)
                ),
            )

    def add_reactions(self):
        """
        Add reactions for non-topology particles
        """
        for reaction in self.parameters["reactions"]:
            self.system.reactions.add(reaction["descriptor"], rate=reaction["rate"])

    def create_readdy_simulation(self):
        """
        Create the ReaDDy simulation
        """
        self.simulation = self.system.simulation("CPU")
        self.simulation.kernel_configuration.n_threads = self.parameters["n_cpu"]

    def add_particle_instances(self, monomers):
        """
        Add particle instances to the simulation
        """
        # add topology particles
        topology_particle_ids = []
        for topology_id in monomers["topologies"]:
            topology = monomers["topologies"][topology_id]
            topology_particle_ids += topology["particle_ids"]
            types = []
            positions = []
            for particle_id in topology["particle_ids"]:
                particle = monomers["particles"][particle_id]
                types.append(particle["type_name"])
                positions.append(particle["position"])
            top = self.simulation.add_topology(
                topology["type_name"], types, np.array(positions)
            )
            added_edges = []
            for index, particle_id in enumerate(topology["particle_ids"]):
                for neighbor_id in monomers["particles"][particle_id]["neighbor_ids"]:
                    neighbor_index = topology["particle_ids"].index(neighbor_id)
                    if (index, neighbor_index) not in added_edges and (
                        neighbor_index,
                        index,
                    ) in added_edges:
                        continue
                    top.get_graph().add_edge(index, neighbor_index)
                    added_edges.append((index, neighbor_index))
                    added_edges.append((neighbor_index, index))
        # add non-topology particles
        for particle_id in monomers["particles"]:
            if particle_id in topology_particle_ids:
                continue
            particle = monomers["particles"][particle_id]
            self.simulation.add_particle(
                type=particle["type_name"], position=particle["position"]
            )

    def simulate_readdy(self, timestep):
        """
        Simulate in ReaDDy for the given timestep
        """

        def loop():
            readdy_actions = self.simulation._actions
            init = readdy_actions.initialize_kernel()
            diffuse = readdy_actions.integrator_euler_brownian_dynamics(
                self.parameters["internal_timestep"]
            )
            calculate_forces = readdy_actions.calculate_forces()
            create_nl = readdy_actions.create_neighbor_list(
                self.system.calculate_max_cutoff().magnitude
            )
            update_nl = readdy_actions.update_neighbor_list()
            react = readdy_actions.reaction_handler_uncontrolled_approximation(
                self.parameters["internal_timestep"]
            )
            observe = readdy_actions.evaluate_observables()
            init()
            create_nl()
            calculate_forces()
            update_nl()
            observe(0)
            n_steps = int(timestep / self.parameters["internal_timestep"])
            for t in tqdm(range(1, n_steps + 1)):
                diffuse()
                update_nl()
                react()
                update_nl()
                calculate_forces()
                observe(t)

        self.simulation._run_custom_loop(loop, show_summary=False)

    def current_particle_edges(self):
        """
        Get all the edges in the ReaDDy topologies
        as (particle1 id, particle2 id)
        from readdy.simulation.current_topologies
        """
        result = []
        for top in self.simulation.current_topologies:
            for v1, v2 in top.graph.edges:
                p1_id = top.particle_id_of_vertex(v1)
                p2_id = top.particle_id_of_vertex(v2)
                if p1_id <= p2_id:
                    result.append((p1_id, p2_id))
        return result

    def get_current_monomers(self):
        """
        Get data for topologies of particles
        from readdy.simulation.current_topologies
        and non-topology particles from readdy.simulation.current_particles
        """
        result = {
            "topologies": {},
            "particles": {},
        }
        # topologies
        edges = self.current_particle_edges()
        for index, topology in enumerate(self.simulation.current_topologies):
            particle_ids = []
            for p_ix, particle in enumerate(topology.particles):
                particle_ids.append(particle.id)
                neighbor_ids = []
                for edge in edges:
                    if particle.id == edge[0]:
                        neighbor_ids.append(edge[1])
                    elif particle.id == edge[1]:
                        neighbor_ids.append(edge[0])
                result["particles"][p_ix] = {
                    "type_name": particle.type,
                    "position": particle.pos,
                    "neighbor_ids": neighbor_ids,
                    "radius": self.parameters["particle_radii"].get(particle.type, 1.0),
                }
            result["topologies"][index] = {
                "type_name": topology.type,
                "particle_ids": particle_ids,
            }
        # non-topology particles
        for index, particle in enumerate(self.simulation.current_particles):
            result["particles"][index] = {
                "type_name": particle.type,
                "position": particle.pos,
                "neighbor_ids": [],
                "radius": self.parameters["particle_radii"].get(particle.type, 1.0),
            }
        return result

    def initial_state(self, config=None):
        box_size = 100.0 * np.ones(3)
        n_particles_substrate = 100
        n_particles_enzyme = 5
        n_particles_chain = 20
        chain_particle_radius = 2.0
        particles = {}
        last_id = 0
        # substrate particles
        substrate_positions = (
            np.random.uniform(size=(n_particles_substrate, 3)) - 0.5
        ) * box_size
        for position in substrate_positions:
            particles[last_id] = {
                "type_name": "A",
                "position": position,
                "neighbor_ids": [],
            }
            last_id += 1
        # enzyme particles
        enzyme_positions = (
            np.random.uniform(size=(n_particles_enzyme, 3)) - 0.5
        ) * box_size
        for position in enzyme_positions:
            particles[last_id] = {
                "type_name": "C",
                "position": position,
                "neighbor_ids": [],
                "radius": 2.0,
            }
            last_id += 1
        # inert chain particles
        chain_position = (np.random.uniform(3) - 0.5) * box_size
        chain_particle_ids = []
        for index in range(n_particles_chain):
            particle_id = last_id + index
            neighbor_ids = []
            if index > 0:
                neighbor_ids.append(particle_id - 1)
            if index < n_particles_chain - 1:
                neighbor_ids.append(particle_id + 1)
            particles[particle_id] = {
                "type_name": "D",
                "position": chain_position,
                "neighbor_ids": neighbor_ids,
                "radius": 4.0,
            }
            chain_particle_ids.append(particle_id)
            chain_position += 2 * chain_particle_radius * np.random.uniform(3)
        return {
            "monomers": {
                "box_size": box_size,
                "topologies": {
                    0: {
                        "type_name": "Chain",
                        "particle_ids": chain_particle_ids,
                    }
                },
                "particles": particles,
            },
        }


def run_readdy_process():
    readdy_process = ReaddyProcess(
        {
            "particle_radii": {
                "A": 1.0,
                "B": 1.0,
                "C": 2.0,
                "D": 4.0,
            },
            "topology_types": [
                "Chain",
            ],
            "topology_particles": [
                "D",
            ],
            "bond_pairs": [
                ["D", "D"],
            ],
            "reactions": [
                {
                    "descriptor": "enz: A +(3) C -> B + C",
                    "rate": 0.1,
                }
            ],
        }
    )
    composite = readdy_process.generate()
    engine = Engine(
        composite=composite,
        initial_state=readdy_process.initial_state(),
        emitter="simularium",
    )
    engine.update(1.0)  # 10 steps
    engine.emitter.get_data()


if __name__ == "__main__":
    run_readdy_process()
