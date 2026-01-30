from .go2 import go2
from .robot_data import RobotData

__all__ = ["go2"]


def get_robot_data(robot_name: str) -> RobotData:
    if robot_name == "go2":
        return go2
    else:
        raise ValueError(f"Robot {robot_name} not found")
