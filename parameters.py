from dataclasses import dataclass, field
from typing import NamedTuple


class Angles(NamedTuple):
    theta: float = 0
    phi: float = 0

@dataclass
class SceneParameters:
    rotation_parts:tuple = ()
    rotation_fidelity: float = 45.
    velocity_range:tuple = (0,0)
    position_range: tuple = (0,0)
    random_rotation:bool = False
    offset:int = 0
    spawn_radius:float = 2
    spawn_angles: Angles = field(default_factory = lambda:Angles())