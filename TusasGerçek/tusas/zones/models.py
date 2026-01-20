from typing import List, Dict, Any


class Zone:
    """Zone (katman) temsil eden sınıf"""

    def __init__(self, zone_id: int, name: str, sequence: List[int], ply_count: int):
        self.zone_id = zone_id
        self.name = name
        self.sequence = sequence
        self.ply_count = ply_count
        self.fitness_score = 0.0
        self.source_zones = []  # type: List[int]
        self.transition_type = "drop_off"  # "drop_off" | "merge" | "angle_drop_off"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "sequence": self.sequence,
            "ply_count": self.ply_count,
            "fitness_score": self.fitness_score,
            "source_zones": self.source_zones,
            "transition_type": self.transition_type,
        }

