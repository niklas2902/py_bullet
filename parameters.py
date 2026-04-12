from dataclasses import dataclass


@dataclass
class SceneParameters:
    rotation_parts:tuple = ()
    rotation_fidelity: float = 45.
    velocity_range:tuple = ((0,0), (0,0), (0,0))
    position_range: tuple = (0,0)
    random_rotation:bool = False
    offset:int = 0