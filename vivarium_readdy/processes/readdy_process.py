import numpy as np

from vivarium.core.process import Process
from vivarium.core.engine import Engine, pf
from vivarium.core.composition import simulate_process

from tqdm import tqdm
import readdy

NAME = "READDY"


class ReaddyProcess(Process):
    """
    This process uses ReaDDy to model arbitrary 
    reaction diffusion systems
    """

    name = NAME

    defaults = {
        "internal_timestep": 0.1,  # s
        "box_size": 100.0,  # m
        "periodic_boundary": False,
        "temperature_C": 22.0,
        "viscosity": 8.1,  # cP, viscosity in cytoplasm
        "force_constant": 250.0,
        "n_cpu": 4,
        "particle_radii": {},
        "topology_particles": [],
        "reactions": [],
    }

    def __init__(self, parameters=None):
        super(ReaddyProcess, self).__init__(parameters)

    def ports_schema(self):
        return {
            "box_size": {
                "_default": 100.,
                "_updater": "set",
                "_emit": True,
            },
            "topologies": {
                "*": {
                    "type_name": {
                        "_default": "",
                        "_updater": "set",
                        "_emit": True,
                    },
                    "particle_ids": {
                        "_default": [],
                        "_updater": "set",
                        "_emit": True,
                    },
                }
            },
            "particles": {
                "*": {
                    "type_name": {
                        "_default": "",
                        "_updater": "set",
                        "_emit": True,
                    },
                    "position": {
                        "_default": np.zeros(3),
                        "_updater": "set",
                        "_emit": True,
                    },
                    "neighbor_ids": {
                        "_default": [],
                        "_updater": "set",
                        "_emit": True,
                    },
                }
            }
        }

    def next_update(self, timestep, states):
        self.create_readdy_system(states)
        self.create_readdy_simulation()
        self.add_particle_instances(states)
        self.simulate_readdy(timestep)
        current_state = self.get_current_state()
        # update topologies
        topologies_update = {"_add": [], "_delete": []}
        for id, state in current_state["topologies"].items():
            if id in states["topologies"]:
                topologies_update[id] = state
            else:
                topologies_update["_add"].append({"key": id, "state": state})
        for existing_id in states["topologies"].keys():
            if existing_id not in current_state["topologies"]:
                topologies_update["_delete"].append(existing_id)
        # update particles
        particles_update = {"_add": [], "_delete": []}
        for id, state in current_state["particles"].items():
            if id in states["particles"]:
                particles_update[id] = state
            else:
                particles_update["_add"].append({"key": id, "state": state})
        for existing_id in states["particles"].keys():
            if existing_id not in current_state["particles"]:
                particles_update["_delete"].append(existing_id)
        return {
            "box_size": states["box_size"],
            "topologies": topologies_update, 
            "particles": particles_update,
        }

    def create_readdy_system(self, states):
        """
        Create the ReaDDy system
        including particle species, constraints, and reactions
        """
        self.system = readdy.ReactionDiffusionSystem(
            box_size=[self.parameters["box_size"]] * 3,
            periodic_boundary_conditions=3 * [bool(self.parameters["periodic_boundary"])],
        )
        self.parameters["temperature_K"] = self.parameters["temperature_C"] + 273.15
        self.system.temperature = self.parameters["temperature_K"]
        self.add_particle_species()
        self.add_topology_types(states)
        all_particle_types = set()
        for particle_id in states["particles"]:
            particle = states["particles"][particle_id]
            all_particle_types.add(particle["type_name"])
        self.check_add_global_box_potential(states, all_particle_types)
        self.add_repulsions(all_particle_types)
        self.add_bonds(states)
        self.add_reactions()

    @staticmethod
    def calculate_diffusionCoefficient(radius, eta, T):
        """
        calculate the theoretical diffusion constant of a spherical particle
            with radius [m]
            in a media with viscosity eta [cP]
            at temperature T [Kelvin]

            returns m^2/s
        """
        return (
            (1.38065 * 10 ** (-23) * T)
            / (6 * np.pi * eta * 10 ** (-3) * radius * 10 ** (-9))
            / 10 ** 9
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
            )
            if particle_name in self.parameters["topology_particles"]:
                self.system.add_topology_species(particle_name, diffCoeff)
            else:
                self.system.add_species(particle_name, diffCoeff)
            added_particle_types.append(particle_name)

    def add_topology_types(self, states):
        """
        Add all topology types
        """
        topology_types = set()
        for topology_id in states["topologies"]:
            topology_types.add(states["topologies"][topology_id]["type_name"])
        for topology_type in topology_types:
            self.system.topologies.add_type(topology_type)

    def check_add_global_box_potential(self, states, all_particle_types):
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
            if type1 not in self.parameters["particle_radii"]:
                raise Exception(
                    "Please provide a radius for particle type "
                    f"{type1} in parameters['particle_radii']"
                )
            for type2 in all_particle_types:
                if type2 not in self.parameters["particle_radii"]:
                    raise Exception(
                        "Please provide a radius for particle type "
                        f"{type1} in parameters['particle_radii']"
                    )
                self.system.potentials.add_harmonic_repulsion(
                    type1,
                    type2,
                    force_constant=self.parameters["force_constant"], 
                    interaction_distance=(
                        self.parameters["particle_radii"][type1] 
                        + self.parameters["particle_radii"][type2]
                    )
                )

    def add_bonds(self, states):
        """
        Add harmonic bonds
        """
        added_bonds = []
        for particle_id in states["particles"]:
            particle = states["particles"][particle_id]
            for neighbor_id in particle["neighbor_ids"]:
                neighbor = states["particles"][neighbor_id]
                type1 = particle["type_name"]
                type2 = neighbor["type_name"]
                if (type1, type2) in added_bonds or (type2, type1) in added_bonds:
                    continue
                self.system.topologies.configure_harmonic_bond(
                    type1, 
                    type2, 
                    force_constant=self.parameters["force_constant"], 
                    length=(
                        self.parameters["particle_radii"][type1] 
                        + self.parameters["particle_radii"][type2]
                    ),
                )
                added_bonds.append((type1, type2))
                added_bonds.append((type2, type1))

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

    def add_particle_instances(self, states):
        """
        Add particle instances to the simulation
        """
        # add topology particles
        topology_particle_ids = []
        for topology_id in states["topologies"]:
            topology = states["topologies"][topology_id]
            topology_particle_ids += topology["particle_ids"]
            types = []
            positions = []
            for particle_id in topology["particle_ids"]:
                particle = states["particles"][particle_id]
                types.append(particle["type_name"])
                positions.append(particle["position"])
            top = self.simulation.add_topology(
                topology["type_name"], types, np.array(positions)
            )
            added_edges = []
            for index, particle_id in enumerate(topology["particle_ids"]):
                for neighbor_id in states["particles"][particle_id][
                    "neighbor_ids"
                ]:
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
        for particle_id in states["particles"]:
            if particle_id in topology_particle_ids:
                continue
            particle = states["particles"][particle_id]
            self.simulation.add_particle(
                type=particle["type_name"], 
                position=particle["position"]
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
        self.simulation._run_custom_loop(loop)

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

    def get_current_state(self):
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
            for p in topology.particles:
                particle_ids.append(p.id)
                neighbor_ids = []
                for edge in edges:
                    if p.id == edge[0]:
                        neighbor_ids.append(edge[1])
                    elif p.id == edge[1]:
                        neighbor_ids.append(edge[0])
                result["particles"][p.id] = {
                    "type_name": p.type,
                    "position": p.pos,
                    "neighbor_ids": neighbor_ids,
                }
            result["topologies"][index] = {
                "type_name": topology.type,
                "particle_ids": particle_ids,
            }
        # non-topology particles
        for index, particle in enumerate(self.simulation.current_particles):
            result["particles"][p.id] = {
                "type_name": p.type,
                "position": p.pos,
                "neighbor_ids": [],
            }
        return result

    def initial_state(self, config=None):
        box_size = 100. * np.ones(3)
        n_particles_substrate = 100
        n_particles_enzyme = 5
        n_particles_chain = 20
        chain_particle_radius = 2.
        particles = {}
        last_id = 0
        # substrate particles
        substrate_positions = (np.random.uniform(size=(n_particles_substrate, 3)) - 0.5) * box_size
        for position in substrate_positions:
            particles[last_id] = {
                "type_name": "A",
                "position": position,
                "neighbor_ids": [],
            }
            last_id += 1
        # enzyme particles
        enzyme_positions = (np.random.uniform(size=(n_particles_enzyme, 3)) - 0.5) * box_size
        for position in enzyme_positions:
            particles[last_id] = {
                "type_name": "C",
                "position": position,
                "neighbor_ids": [],
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
            }
            chain_particle_ids.append(particle_id)
            chain_position += 2 * chain_particle_radius * np.random.uniform(3)
        return {
            "box_size": box_size,
            "topologies": {
                0: {
                    "type_name": "Chain",
                    "particle_ids": chain_particle_ids,
                }
            },
            "particles": particles
        }


def test_readdy_process():
    readdy_process = ReaddyProcess({
        "particle_radii": {
            "A": 1.,
            "B": 1.,
            "C": 2.,
            "D": 4.,
        },
        "topology_particles": [
            "D",
        ],
        "reactions": [
            {
                "descriptor": "enz: A +(3) C -> B + C",
                "rate": 0.1,
            }
        ],
    })
    engine = Engine(
        processes={'readdy': readdy_process},
        topology={
            'readdy': {
                'box_size': ('box_size',),
                'topologies': ('topologies',),
                'particles': ('particles',)}},
        initial_state=readdy_process.initial_state(),
        emitter='simularium',
    )
    engine.update(1.0)  # 10 steps
    output = engine.emitter.get_data()
    print(pf(output))


if __name__ == "__main__":
    test_readdy_process()
