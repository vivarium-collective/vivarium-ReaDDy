from typing import Any, Dict, Tuple

from vivarium.core.emitter import Emitter

import numpy as np
import pandas as pd
from simulariumio import (
    TrajectoryConverter,
    TrajectoryData,
    AgentData,
    MetaData,
    UnitData,
)

# TODO add viz type to state?
VIZ_FOR_CHOICE = {
    "medyan_active": "fibers",
    "cytosim_active": "fibers",
    "readdy_active": "monomers",
    "init": "fibers",
}

class SimulariumEmitter(Emitter):
    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.configuration_data = None
        self.saved_data: Dict[float, Dict[str, Any]] = {}

    def emit(self, data: Dict[str, Any]) -> None:
        """
        Emit the timeseries history portion of ``data``, which is
        ``data['data'] if data['table'] == 'history'`` and put it at
        ``data['data']['time']`` in the history.
        """
        if data["table"] == "configuration":
            self.configuration_data = data["data"]
            assert "processes" in self.configuration_data, "please emit processes"
        if data["table"] == "history":
            emit_data = data["data"]
            time = emit_data["time"]
            self.saved_data[time] = {
                key: value for key, value in emit_data.items() if key not in ["time"]
            }

    def get_simularium_fibers(
        self, time, fibers, actin_radius, position_offset, trajectory
    ):
        """
        Shape fiber state data into Simularium fiber agents
        """
        print(f"      {position_offset}")
        n_agents = 0
        time_index = len(trajectory["times"])
        trajectory["times"].append(time)
        trajectory["unique_ids"].append([])
        trajectory["type_names"].append([])
        trajectory["n_subpoints"].append([])
        trajectory["subpoints"].append([])
        for fiber_id in fibers:
            fiber = fibers[fiber_id]
            fiber_points = [
                np.array(point) + position_offset for point in fiber["points"]
            ]
            n_agents += 1
            trajectory["unique_ids"][time_index].append(int(fiber_id))
            trajectory["type_names"][time_index].append(fiber["type_name"])
            trajectory["n_subpoints"][time_index].append(len(fiber_points))
            trajectory["subpoints"][time_index].append(fiber_points)
        trajectory["n_agents"].append(n_agents)
        trajectory["viz_types"].append(n_agents * [1001.0])
        trajectory["positions"].append(n_agents * [[0.0, 0.0, 0.0]])
        trajectory["radii"].append(n_agents * [actin_radius])
        return trajectory

    def get_simularium_monomers(
        self, time, monomers, actin_radius, position_offset, trajectory
    ):
        """
        Shape monomer state data into Simularium agents
        """
        time_index = len(trajectory["times"])
        trajectory["times"].append(time)
        trajectory["unique_ids"].append([])
        trajectory["type_names"].append([])
        trajectory["positions"].append([])
        edge_ids = []
        edge_positions = []
        for particle_id in monomers.get("particles", {}):
            particle = monomers["particles"][particle_id]
            trajectory["unique_ids"][time_index].append(int(particle_id))
            trajectory["type_names"][time_index].append(particle["type_name"])
            trajectory["positions"][time_index].append(
                np.array(particle["position"]) + position_offset
            )
            # visualize edges between particles
            for neighbor_id in particle["neighbor_ids"]:
                neighbor_id_str = str(neighbor_id)
                edge = (particle_id, neighbor_id_str)
                reverse_edge = (
                    neighbor_id_str,
                    particle_id,
                )
                if edge not in edge_ids and reverse_edge not in edge_ids:
                    edge_ids.append(edge)
                    edge_positions.append(
                        [
                            np.array(particle["position"]) + position_offset,
                            np.array(monomers["particles"][neighbor_id_str]["position"])
                            + position_offset,
                        ]
                    )
        n_agents = len(trajectory["unique_ids"][time_index])
        n_edges = len(edge_ids)
        trajectory["n_agents"].append(n_agents + n_edges)
        trajectory["viz_types"].append(n_agents * [1000.0])
        trajectory["viz_types"][time_index] += n_edges * [1001.0]
        trajectory["unique_ids"][time_index] += [1000 + i for i in range(n_edges)]
        trajectory["type_names"][time_index] += ["edge" for edge in range(n_edges)]
        trajectory["positions"][time_index] += n_edges * [[0.0, 0.0, 0.0]]
        trajectory["radii"].append(n_agents * [actin_radius])
        trajectory["radii"][time_index] += n_edges * [1.0]
        trajectory["n_subpoints"].append(n_agents * [0])
        trajectory["n_subpoints"][time_index] += n_edges * [2]
        trajectory["subpoints"].append(n_agents * [2 * [[0.0, 0.0, 0.0]]])
        trajectory["subpoints"][time_index] += edge_positions
        return trajectory

    @staticmethod
    def fill_df(df, fill):
        """
        Fill Nones in a DataFrame with a fill value
        """
        # Create a dataframe of fill values
        fill_array = [[fill] * df.shape[1]] * df.shape[0]
        fill_df = pd.DataFrame(fill_array)
        # Replace all entries with None with the fill
        df[df.isna()] = fill_df
        return df

    @staticmethod
    def jagged_3d_list_to_numpy_array(jagged_3d_list):
        """
        Shape a jagged list with 3 dimensions to a numpy array
        """
        df = SimulariumEmitter.fill_df(pd.DataFrame(jagged_3d_list), [0.0, 0.0, 0.0])
        df_t = df.transpose()
        exploded = [df_t[col].explode() for col in list(df_t.columns)]
        return np.array(exploded).reshape((df.shape[0], df.shape[1], 3))

    @staticmethod
    def get_subpoints_numpy_array(trajectory) -> np.ndarray:
        """
        Shape a 4 dimensional jagged list for subpoints into a numpy array
        """
        frame_arrays = []
        max_agents = 0
        max_subpoints = 0
        total_steps = len(trajectory["subpoints"])
        for time_index in range(total_steps):
            frame_array = SimulariumEmitter.jagged_3d_list_to_numpy_array(
                trajectory["subpoints"][time_index]
            )
            if frame_array.shape[0] > max_agents:
                max_agents = frame_array.shape[0]
            if frame_array.shape[1] > max_subpoints:
                max_subpoints = frame_array.shape[1]
            frame_arrays.append(frame_array)
        values_per_frame = max_agents * max_subpoints * 3
        result = np.zeros(total_steps * values_per_frame)
        for time_index, frame_array in enumerate(frame_arrays):
            if frame_array.shape[1] < max_subpoints:
                new_frame_array = np.zeros((frame_array.shape[0], max_subpoints, 3))
                new_frame_array[:, : frame_array.shape[1]] = frame_array
                frame_array = new_frame_array
            flat_array = frame_array.flatten()
            start_index = time_index * values_per_frame
            result[start_index : start_index + flat_array.shape[0]] = flat_array
        return result.reshape(total_steps, max_agents, max_subpoints, 3)

    @staticmethod
    def get_agent_data_from_jagged_lists(trajectory, scale_factor) -> AgentData:
        """
        Shape a dictionary of jagged lists into a Simularium AgentData object
        """
        return AgentData(
            times=np.arange(len(trajectory["times"])),
            n_agents=np.array(trajectory["n_agents"]),
            viz_types=SimulariumEmitter.fill_df(
                pd.DataFrame(trajectory["viz_types"]), 1000.0
            ).to_numpy(),
            unique_ids=SimulariumEmitter.fill_df(
                pd.DataFrame(trajectory["unique_ids"]), 0
            ).to_numpy(dtype=int),
            types=trajectory["type_names"],
            positions=scale_factor
            * SimulariumEmitter.jagged_3d_list_to_numpy_array(trajectory["positions"]),
            radii=scale_factor
            * SimulariumEmitter.fill_df(
                pd.DataFrame(trajectory["radii"]), 0.0
            ).to_numpy(),
            n_subpoints=SimulariumEmitter.fill_df(
                pd.DataFrame(trajectory["n_subpoints"]), 0
            ).to_numpy(dtype=int),
            subpoints=scale_factor
            * SimulariumEmitter.get_subpoints_numpy_array(trajectory),
        )

    @staticmethod
    def get_simularium_converter(
        trajectory, box_dimensions, scale_factor
    ) -> TrajectoryConverter:
        """
        Shape a dictionary of jagged lists into a Simularium TrajectoryData object
        and provide it to a TrajectoryConverter for conversion
        """
        spatial_units = UnitData("nm")
        spatial_units.multiply(1 / scale_factor)
        return TrajectoryConverter(
            TrajectoryData(
                meta_data=MetaData(
                    box_size=scale_factor * box_dimensions,
                ),
                agent_data=SimulariumEmitter.get_agent_data_from_jagged_lists(
                    trajectory, scale_factor
                ),
                time_units=UnitData("count"),
                spatial_units=spatial_units,
            )
        )

    @staticmethod
    def get_active_choice(choices) -> Tuple[str, bool]:
        """
        Determine the active simulator
        """
        current_choice = None
        for choice in choices:
            if choices[choice]:
                return choice

    def get_data(self) -> dict:
        """
        Save the accumulated timeseries history of "emitted" data to file
        """
        if "readdy_actin" in self.configuration_data:
            actin_radius = self.configuration_data["readdy_actin"]["actin_radius"]
        else:
            actin_radius = 3.0  # TODO add to MEDYAN/Cytosim configs
        box_dimensions = None
        trajectory = {
            "times": [],
            "n_agents": [],
            "viz_types": [],
            "unique_ids": [],
            "type_names": [],
            "positions": [],
            "radii": [],
            "n_subpoints": [],
            "subpoints": [],
        }
        times = list(self.saved_data.keys())
        times.sort()
        vizualize_time_index = 0
        for time, state in self.saved_data.items():
            index = times.index(time)
            prev_choice = SimulariumEmitter.get_active_choice(
                self.saved_data[times[index - 1]]["choices"]
            ) if index > 0 else "init"
            current_choice = SimulariumEmitter.get_active_choice(state["choices"])
            if box_dimensions is None and "fibers_box_extent" in state:
                box_dimensions = np.array(state["fibers_box_extent"])
            if VIZ_FOR_CHOICE[current_choice] == "fibers":
                center_fibers = index == 0 or "readdy_active" in state["choices"] #TODO generalize
                trajectory = self.get_simularium_fibers(
                    vizualize_time_index,
                    state["fibers"],
                    actin_radius,
                    - 0.5 * box_dimensions if center_fibers else np.zeros(3),
                    trajectory,
                )
                vizualize_time_index += 1
            if VIZ_FOR_CHOICE[current_choice] == "monomers":
                trajectory = self.get_simularium_monomers(
                    vizualize_time_index,
                    state["monomers"],
                    actin_radius,
                    (
                        np.array(state["monomers"]["box_center"])
                        - 0.5 * box_dimensions
                    ),
                    trajectory,
                )
                vizualize_time_index += 1
        simularium_converter = SimulariumEmitter.get_simularium_converter(
            trajectory, box_dimensions, 0.1
        )
        simularium_converter.write_JSON("out/actin_test")
