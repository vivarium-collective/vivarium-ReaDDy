import numpy as np


monomer_ports_schema = {
    "time_units": {
        "_default": "s",
        "_updater": "set",
        "_emit": True,
    },
    "spatial_units": {
        "_default": "m",
        "_updater": "set",
        "_emit": True,
    },
    "monomers": {
        "box_center": {
            "_default": np.array([0.0, 0.0, 0.0]),
            "_updater": "set",
            "_emit": True,
        },
        "box_size": {
            "_default": 500.0,
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
                "radius": {
                    "_default": 1.0,
                    "_updater": "set",
                    "_emit": True,
                },
            }
        },
    },
}


def agents_update(existing, projected):
    update = {"_add": [], "_delete": []}

    for id, state in projected.items():
        if id in existing:
            update[id] = state
        else:
            update["_add"].append({"key": id, "state": state})

    for existing_id in existing.keys():
        if existing_id not in projected:
            update["_delete"].append(existing_id)

    return update


def create_monomer_update(previous_monomers, new_monomers):
    return {
        "monomers": {
            "topologies": agents_update(
                previous_monomers["topologies"], new_monomers["topologies"]
            ),
            "particles": agents_update(
                previous_monomers["particles"], new_monomers["particles"]
            ),
        }
    }
