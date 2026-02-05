"""
Region class - Tek bir bölgeyi temsil eder.

Her bölge:
- Benzersiz bir ID (örn: "R64", "R48")
- Hedef ply sayısı
- Optimize edilmiş sequence
- MPT referanslı boolean mask
- Komşu bölge listesi
"""
from typing import List, Dict, Any, Optional
from collections import Counter


class Region:
    """Kompozit yapıdaki tek bir bölgeyi temsil eder."""

    def __init__(self, region_id: str, target_ply_count: int):
        """
        Args:
            region_id: Bölge tanımlayıcı (örn: "R64", "R48")
            target_ply_count: Bu bölgedeki hedef ply sayısı
        """
        self.region_id = region_id
        self.target_ply_count = target_ply_count
        
        # Optimize edilmiş sequence (açı listesi)
        self.sequence: List[int] = []
        
        # MPT referanslı boolean mask
        # True = bu ply bu bölgede var, False = drop edilmiş
        self.ply_mask: List[bool] = []
        
        # Komşu bölge ID'leri
        self.neighbors: List[str] = []
        
        # Fitness skoru detayları (R1, R2, vb.)
        self.fitness_details: Dict[str, float] = {}
        
        # Bu bölgenin master bölge olup olmadığı
        self.is_master: bool = False
        
        # Drop edilen ply indeksleri (MPT referanslı)
        self.dropped_indices: List[int] = []

    def set_neighbors(self, neighbor_ids: List[str]) -> None:
        """Komşu bölgeleri ayarla."""
        self.neighbors = neighbor_ids[:]

    def set_sequence_from_mask(self, master_sequence: List[int], mask: List[bool]) -> None:
        """
        MPT ve mask'tan bu bölgenin sequence'ını oluştur.
        
        Args:
            master_sequence: Master bölgenin tam sequence'ı
            mask: Boolean mask (True = ply var, False = drop)
        """
        if len(master_sequence) != len(mask):
            raise ValueError(
                "Master sequence ({}) ve mask ({}) uzunlukları eşleşmiyor".format(
                    len(master_sequence), len(mask)
                )
            )
        
        self.ply_mask = mask[:]
        self.sequence = [ply for ply, keep in zip(master_sequence, mask) if keep]
        self.dropped_indices = [i for i, keep in enumerate(mask) if not keep]

    def get_angle_counts(self) -> Dict[int, int]:
        """Bu bölgedeki açı dağılımını döndür."""
        return dict(Counter(self.sequence))

    def validate_symmetry(self) -> bool:
        """Sequence simetrik mi kontrol et."""
        n = len(self.sequence)
        for i in range(n // 2):
            if self.sequence[i] != self.sequence[n - 1 - i]:
                return False
        return True

    def validate_balance(self) -> bool:
        """±45 dengesi kontrol et."""
        counts = self.get_angle_counts()
        return counts.get(45, 0) == counts.get(-45, 0)

    def validate_external_plies(self) -> bool:
        """External ply'lar ±45 mi kontrol et."""
        if len(self.sequence) < 2:
            return True
        return (
            abs(self.sequence[0]) == 45 and
            abs(self.sequence[-1]) == 45
        )

    def to_dict(self) -> Dict[str, Any]:
        """Bölgeyi dictionary olarak döndür (API response için)."""
        return {
            "region_id": self.region_id,
            "target_ply_count": self.target_ply_count,
            "actual_ply_count": len(self.sequence),
            "sequence": self.sequence,
            "ply_mask": self.ply_mask,
            "neighbors": self.neighbors,
            "fitness_score": round(self.fitness_score, 2),
            "fitness_details": self.fitness_details,
            "is_master": self.is_master,
            "dropped_indices": self.dropped_indices,
            "angle_counts": self.get_angle_counts(),
            "validations": {
                "symmetric": self.validate_symmetry(),
                "balanced": self.validate_balance(),
                "external_plies_ok": self.validate_external_plies(),
            }
        }

    def __repr__(self) -> str:
        return "Region({}, plies={}, neighbors={})".format(
            self.region_id, len(self.sequence), self.neighbors
        )
