from typing import Dict, List, Optional, Any

from ..core.dropoff_optimizer import DropOffOptimizer
from ..core.laminate_optimizer import LaminateOptimizer
from .models import Zone


class ZoneManager:
    """Zone'ları ve geçişleri yöneten sınıf (in-memory)."""

    def __init__(self):
        self.zones = {}  # type: Dict[int, Zone]
        self.transitions = []  # type: List[Dict[str, Any]]
        self.next_zone_id = 1

    def create_zone_from_dropoff(
        self,
        source_zone_id: int,
        target_ply: int,
        optimizer: LaminateOptimizer,
        drop_optimizer: DropOffOptimizer,
    ) -> Zone:
        """Kaynak zone'dan drop-off yaparak yeni zone oluştur"""
        source_zone = self.zones[source_zone_id]
        drop_optimizer.master_sequence = source_zone.sequence
        drop_optimizer.total_plies = source_zone.ply_count

        new_sequence, score, dropped = drop_optimizer.optimize_drop(target_ply)

        zone_id = self.next_zone_id
        self.next_zone_id += 1

        new_zone = Zone(zone_id=zone_id, name="Zone {}".format(zone_id), sequence=new_sequence, ply_count=target_ply)
        new_zone.fitness_score = score
        new_zone.source_zones = [source_zone_id]
        new_zone.transition_type = "drop_off"

        self.zones[zone_id] = new_zone

        self.transitions.append(
            {
                "from": source_zone_id,
                "to": zone_id,
                "type": "drop_off",
                "dropped_indices": dropped,
                "target_ply": target_ply,
            }
        )

        return new_zone

    def create_zone_from_merge(
        self, source_zone_ids: List[int], target_ply: int, optimizer: LaminateOptimizer
    ) -> Zone:
        """Birden fazla zone'u birleştirerek yeni zone oluştur"""
        merged_sequences = []
        for zone_id in source_zone_ids:
            if zone_id in self.zones:
                merged_sequences.append(self.zones[zone_id].sequence)

        if not merged_sequences:
            raise ValueError("Geçerli kaynak zone bulunamadı")

        longest_seq = max(merged_sequences, key=len)

        if target_ply is not None and target_ply < len(longest_seq):
            drop_optimizer = DropOffOptimizer(longest_seq, optimizer)
            new_sequence, score, _dropped = drop_optimizer.optimize_drop(target_ply)
        else:
            new_sequence = longest_seq
            score, _ = optimizer.calculate_fitness(new_sequence)

        zone_id = self.next_zone_id
        self.next_zone_id += 1

        new_zone = Zone(
            zone_id=zone_id,
            name="Merge Zone {}".format(zone_id),
            sequence=new_sequence,
            ply_count=len(new_sequence),
        )
        new_zone.fitness_score = score
        new_zone.source_zones = source_zone_ids
        new_zone.transition_type = "merge"

        self.zones[zone_id] = new_zone

        self.transitions.append({"from": source_zone_ids, "to": zone_id, "type": "merge", "target_ply": target_ply})

        return new_zone

    def create_zone_from_angle_dropoff(
        self,
        source_zone_id: int,
        target_ply_counts: Dict[int, int],
        optimizer: LaminateOptimizer,
        drop_optimizer: DropOffOptimizer,
    ) -> Zone:
        """Kaynak zone'dan açıya özel drop-off yaparak yeni zone oluştur"""
        source_zone = self.zones[source_zone_id]
        drop_optimizer.master_sequence = source_zone.sequence
        drop_optimizer.total_plies = source_zone.ply_count

        new_sequence, score, dropped_by_angle = drop_optimizer.optimize_drop_with_angle_targets(target_ply_counts)

        zone_id = self.next_zone_id
        self.next_zone_id += 1

        new_zone = Zone(zone_id=zone_id, name="Zone {}".format(zone_id), sequence=new_sequence, ply_count=len(new_sequence))
        new_zone.fitness_score = score
        new_zone.source_zones = [source_zone_id]
        new_zone.transition_type = "angle_drop_off"

        self.zones[zone_id] = new_zone

        self.transitions.append(
            {
                "from": source_zone_id,
                "to": zone_id,
                "type": "angle_drop_off",
                "target_ply_counts": target_ply_counts,
                "dropped_by_angle": dropped_by_angle,
            }
        )

        return new_zone

    def get_zone(self, zone_id: int) -> Optional[Zone]:
        return self.zones.get(zone_id)

    def get_all_zones(self) -> List[Dict[str, Any]]:
        return [zone.to_dict() for zone in self.zones.values()]

    def get_transitions(self) -> List[Dict[str, Any]]:
        return self.transitions

    def add_root_zone(self, sequence: List[int], optimizer: LaminateOptimizer) -> None:
        """Root zone'u ekle (master sequence)"""
        score, _ = optimizer.calculate_fitness(sequence)
        root_zone = Zone(zone_id=0, name="Root", sequence=sequence, ply_count=len(sequence))
        root_zone.fitness_score = score
        self.zones[0] = root_zone

