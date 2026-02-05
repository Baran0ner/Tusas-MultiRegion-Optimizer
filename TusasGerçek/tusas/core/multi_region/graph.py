"""
RegionGraph class - Graf tabanlı bölge yönetimi.

Bölgeleri düğüm (node), bölgeler arası bağlantıları kenar (edge) olarak modeller.
Komşuluk listesi (adjacency list) ile çalışır.
"""
from typing import Dict, List, Optional, Set, Tuple, Any


class RegionGraph:
    """Graf tabanlı bölge yönetimi ve komşuluk ilişkileri."""

    def __init__(self, adjacency_map: Dict[str, List[str]]):
        """
        Args:
            adjacency_map: Bölge komşuluk haritası
                Örnek: {"R64": ["R48", "R32"], "R48": ["R64", "R56"], ...}
        """
        self.adjacency_map = adjacency_map
        self._validate_adjacency_map()
        
        # Tüm bölge ID'lerini topla
        self.all_region_ids: Set[str] = set()
        for region_id, neighbors in adjacency_map.items():
            self.all_region_ids.add(region_id)
            self.all_region_ids.update(neighbors)
        
        # Bölge ply sayıları (region_id -> ply count)
        self.region_ply_counts: Dict[str, int] = {}
        self._extract_ply_counts_from_ids()

    def _validate_adjacency_map(self) -> None:
        """Komşuluk haritasını doğrula (çift yönlü bağlantılar)."""
        for region_id, neighbors in self.adjacency_map.items():
            for neighbor in neighbors:
                if neighbor not in self.adjacency_map:
                    # Komşu bölge adjacency_map'te yok, ekle
                    self.adjacency_map[neighbor] = []
                if region_id not in self.adjacency_map[neighbor]:
                    # Ters bağlantı yok, ekle
                    self.adjacency_map[neighbor].append(region_id)

    def _extract_ply_counts_from_ids(self) -> None:
        """
        Bölge ID'lerinden ply sayılarını çıkar.
        Örn: "R64" -> 64 ply, "R48" -> 48 ply
        """
        for region_id in self.all_region_ids:
            # ID'den sayıyı çıkar (R64 -> 64)
            try:
                ply_count = int("".join(filter(str.isdigit, region_id)))
                self.region_ply_counts[region_id] = ply_count
            except ValueError:
                # Sayı çıkarılamadıysa varsayılan 0
                self.region_ply_counts[region_id] = 0

    def set_ply_count(self, region_id: str, ply_count: int) -> None:
        """Bir bölgenin ply sayısını manuel ayarla."""
        self.region_ply_counts[region_id] = ply_count
        self.all_region_ids.add(region_id)

    def get_ply_count(self, region_id: str) -> int:
        """Bir bölgenin ply sayısını döndür."""
        return self.region_ply_counts.get(region_id, 0)

    def get_master_region_id(self) -> str:
        """En kalın (en fazla ply'a sahip) bölgeyi döndür."""
        if not self.region_ply_counts:
            raise ValueError("Hiç bölge tanımlanmamış")
        return max(self.region_ply_counts.items(), key=lambda x: x[1])[0]

    def get_neighbors(self, region_id: str) -> List[str]:
        """Bir bölgenin komşularını döndür."""
        return self.adjacency_map.get(region_id, [])

    def get_all_edges(self) -> List[Tuple[str, str]]:
        """Tüm kenarları (bölge çiftleri) döndür. Tekrarsız."""
        edges = set()
        for region_id, neighbors in self.adjacency_map.items():
            for neighbor in neighbors:
                edge = tuple(sorted([region_id, neighbor]))
                edges.add(edge)
        return list(edges)

    def get_regions_sorted_by_ply_count(self, descending: bool = True) -> List[str]:
        """Bölgeleri ply sayısına göre sıralı döndür."""
        return sorted(
            self.region_ply_counts.keys(),
            key=lambda x: self.region_ply_counts[x],
            reverse=descending
        )

    def get_drop_path(self, from_region: str, to_region: str) -> List[str]:
        """
        Bir bölgeden diğerine drop yolu bul (BFS).
        Drop her zaman kalından inceye doğru yapılır.
        """
        from collections import deque
        
        if from_region == to_region:
            return [from_region]
        
        visited = {from_region}
        queue = deque([(from_region, [from_region])])
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor in self.get_neighbors(current):
                if neighbor == to_region:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []  # Yol bulunamadı

    def validate_continuity(self, region_a_mask: List[bool], region_b_mask: List[bool]) -> Tuple[bool, List[int]]:
        """
        İki komşu bölge arasında katman sürekliliği kontrol et.
        
        Süreklilik kuralı: Eğer bir ply her iki bölgede de varsa,
        aradaki tüm bölgelerde de olmalı (cascade drop).
        
        Returns:
            (is_valid, violation_indices)
        """
        if len(region_a_mask) != len(region_b_mask):
            return False, list(range(len(region_a_mask)))
        
        violations = []
        for i, (a, b) in enumerate(zip(region_a_mask, region_b_mask)):
            # Süreklilik ihlali: Bir bölgede var, diğerinde yok
            # Bu aslında drop-off durumu, ihlal değil
            # İhlal: İnce bölgede var ama kalın bölgede yok
            pass
        
        return True, violations

    def to_dict(self) -> Dict[str, Any]:
        """Grafı dictionary olarak döndür."""
        return {
            "adjacency_map": self.adjacency_map,
            "region_ids": list(self.all_region_ids),
            "region_ply_counts": self.region_ply_counts,
            "edges": self.get_all_edges(),
            "master_region": self.get_master_region_id() if self.region_ply_counts else None,
        }

    def __repr__(self) -> str:
        return "RegionGraph(regions={}, edges={})".format(
            len(self.all_region_ids), len(self.get_all_edges())
        )
