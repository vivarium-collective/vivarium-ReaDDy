from typing import Any, Dict, List

from vivarium.core.emitter import Emitter

import numpy as np
import pandas as pd
from simulariumio import (
    TrajectoryConverter,
    TrajectoryData,
    AgentData,
    MetaData,
    UnitData,
    DisplayData,
    DISPLAY_TYPE,
)


class SimulariumMonomerEmitter(Emitter):
    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.configuration_data = None
        self.saved_data: List[Dict[str, Any]] = []

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
            self.saved_data.append(data["data"])

    def get_simularium_monomers(self, time, monomers, trajectory):
        """
        Shape monomer state data into Simularium agents
        """
        time_index = len(trajectory["times"])
        trajectory["times"].append(time)
        trajectory["unique_ids"].append([])
        trajectory["type_names"].append([])
        trajectory["positions"].append([])
        trajectory["radii"].append([])
        edge_ids = []
        edge_positions = []
        for particle_id in monomers["particles"]:
            particle = monomers["particles"][particle_id]
            trajectory["unique_ids"][time_index].append(int(particle_id))
            trajectory["type_names"][time_index].append(particle["type_name"])
            # HACK needed until simulariumio default display data fix
            if particle["type_name"] not in trajectory["display_data"]:
                trajectory["display_data"][particle["type_name"]] = DisplayData(
                    name=particle["type_name"],
                    display_type=DISPLAY_TYPE.SPHERE,
                )
            trajectory["positions"][time_index].append(np.array(particle["position"]))
            trajectory["radii"][time_index].append(particle["radius"])
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
                            np.array(particle["position"]),
                            np.array(
                                monomers["particles"][neighbor_id_str]["position"]
                            ),
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
        df = SimulariumMonomerEmitter.fill_df(
            pd.DataFrame(jagged_3d_list), [0.0, 0.0, 0.0]
        )
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
            frame_array = SimulariumMonomerEmitter.jagged_3d_list_to_numpy_array(
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
            times=trajectory["times"],
            n_agents=np.array(trajectory["n_agents"]),
            viz_types=SimulariumMonomerEmitter.fill_df(
                pd.DataFrame(trajectory["viz_types"]), 1000.0
            ).to_numpy(),
            unique_ids=SimulariumMonomerEmitter.fill_df(
                pd.DataFrame(trajectory["unique_ids"]), 0
            ).to_numpy(dtype=int),
            types=trajectory["type_names"],
            positions=scale_factor
            * SimulariumMonomerEmitter.jagged_3d_list_to_numpy_array(
                trajectory["positions"]
            ),
            radii=scale_factor
            * SimulariumMonomerEmitter.fill_df(
                pd.DataFrame(trajectory["radii"]), 0.0
            ).to_numpy(),
            n_subpoints=SimulariumMonomerEmitter.fill_df(
                pd.DataFrame(trajectory["n_subpoints"]), 0
            ).to_numpy(dtype=int),
            subpoints=scale_factor
            * SimulariumMonomerEmitter.get_subpoints_numpy_array(trajectory),
            display_data=trajectory["display_data"],
        )

    @staticmethod
    def get_simularium_converter(
        trajectory, box_dimensions, scale_factor, time_units, spatial_units
    ) -> TrajectoryConverter:
        """
        Shape a dictionary of jagged lists into a Simularium TrajectoryData object
        and provide it to a TrajectoryConverter for conversion
        """
        spatial_units.multiply(1 / scale_factor)
        return TrajectoryConverter(
            TrajectoryData(
                meta_data=MetaData(
                    box_size=scale_factor * box_dimensions,
                ),
                agent_data=SimulariumMonomerEmitter.get_agent_data_from_jagged_lists(
                    trajectory, scale_factor
                ),
                time_units=time_units,
                spatial_units=spatial_units,
            )
        )

    def get_data(self) -> dict:
        """
        Save the accumulated timeseries history of "emitted" data to file
        """
        box_size = None
        time_units = None
        spatial_units = None
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
            "display_data": {},
        }
        for state in self.saved_data:
            if box_size is None:
                box_size = state["monomers"]["box_size"]
            if time_units is None:
                time_units = state["time_units"]
            if spatial_units is None:
                spatial_units = state["spatial_units"]
            trajectory = self.get_simularium_monomers(
                state["time"],
                state["monomers"],
                trajectory,
            )
        simularium_converter = SimulariumMonomerEmitter.get_simularium_converter(
            trajectory, box_size, 0.1, UnitData(time_units), UnitData(spatial_units)
        )
        simularium_converter.save("out/test")
        return {"simularium_converter": simularium_converter}
