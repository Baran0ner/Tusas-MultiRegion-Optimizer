"""
MasterPlyTable class - Master Ply Table yönetimi.

MPT, en kalın bölgenin (Master Region) sequence'ını referans olarak kullanır.
Diğer tüm bölgeler, bu tablodaki ply'ların varlık/yokluk durumuna göre
boolean mask ile tanımlanır.
"""
from typing import Dict, List, Any, Optional, Tuple


class MasterPlyTable:
    """
    Master Ply Table - tüm bölgeler için referans tablo.
    
    Her satır bir ply'ı temsil eder:
    - Ply indeksi (0-based)
    - Açı değeri (0, 90, 45, -45)
    - Her bölge için boolean (var/yok)
    """

    def __init__(self, master_sequence: List[int], master_region_id: str):
        """
        Args:
            master_sequence: Master bölgenin optimize edilmiş sequence'ı
            master_region_id: Master bölge ID'si (örn: "R64")
        """
        self.master_sequence = master_sequence[:]
        self.master_region_id = master_region_id
        self.ply_count = len(master_sequence)
        
        # Her bölge için boolean mask: region_id -> List[bool]
        # True = ply bu bölgede var, False = drop edilmiş
        self.region_masks: Dict[str, List[bool]] = {}
        
        # Master bölge için tüm ply'lar var
        self.region_masks[master_region_id] = [True] * self.ply_count

    def add_region_mask(self, region_id: str, mask: List[bool]) -> None:
        """
        Bir bölge için mask ekle.
        
        Args:
            region_id: Bölge ID'si
            mask: Boolean mask (uzunluk = master sequence uzunluğu)
        """
        if len(mask) != self.ply_count:
            raise ValueError(
                "Mask uzunluğu ({}) master sequence uzunluğu ({}) ile eşleşmiyor".format(
                    len(mask), self.ply_count
                )
            )
        self.region_masks[region_id] = mask[:]

    def derive_sequence(self, region_id: str) -> List[int]:
        """
        Bir bölgenin sequence'ını mask'tan türet.
        
        Args:
            region_id: Bölge ID'si
            
        Returns:
            Bu bölgenin ply sequence'ı
        """
        if region_id not in self.region_masks:
            raise KeyError("Bölge '{}' için mask tanımlanmamış".format(region_id))
        
        mask = self.region_masks[region_id]
        return [ply for ply, keep in zip(self.master_sequence, mask) if keep]

    def get_region_ply_count(self, region_id: str) -> int:
        """Bir bölgenin ply sayısını döndür."""
        if region_id not in self.region_masks:
            return 0
        return sum(self.region_masks[region_id])

    def get_dropped_indices(self, region_id: str) -> List[int]:
        """Bir bölgede drop edilen ply indekslerini döndür."""
        if region_id not in self.region_masks:
            return []
        return [i for i, keep in enumerate(self.region_masks[region_id]) if not keep]

    def create_mask_for_target_count(
        self, 
        target_ply_count: int, 
        drop_strategy: str = "center_first"
    ) -> List[bool]:
        """
        Belirli bir ply sayısı için mask oluştur.
        
        Args:
            target_ply_count: Hedef ply sayısı
            drop_strategy: Drop stratejisi
                - "center_first": Ortadan başla (havacılık standardı)
                - "edges_first": Kenarlardan başla
                
        Returns:
            Boolean mask
        """
        if target_ply_count >= self.ply_count:
            return [True] * self.ply_count
        
        drops_needed = self.ply_count - target_ply_count
        mask = [True] * self.ply_count
        
        # Simetrik drop için çiftler halinde drop et
        # External ply'ları (ilk 2 ve son 2) koru
        mid = self.ply_count // 2
        
        if drop_strategy == "center_first":
            # Ortadan başlayarak drop et (havacılık standardı - Rule 10)
            dropped = 0
            offset = 0
            
            while dropped < drops_needed:
                # Orta düzleme yakın pozisyonlardan drop et
                # Simetrik: i ve (n-1-i) birlikte
                left_idx = mid - 1 - offset
                right_idx = mid + offset
                
                # Tek ply sayısı için orta ply
                if self.ply_count % 2 == 1 and offset == 0:
                    if mask[mid] and dropped < drops_needed:
                        mask[mid] = False
                        dropped += 1
                    offset += 1
                    continue
                
                # External ply'ları koru (ilk 2 ve son 2)
                if left_idx >= 2 and right_idx <= self.ply_count - 3:
                    if mask[left_idx] and dropped < drops_needed:
                        mask[left_idx] = False
                        dropped += 1
                    if mask[right_idx] and dropped < drops_needed:
                        mask[right_idx] = False
                        dropped += 1
                
                offset += 1
                if offset > mid:
                    break
        
        return mask

    def generate_dropoff_map(self) -> Dict[str, Any]:
        """
        Görsel drop-off haritası için veri üret.
        
        Returns:
            Drop-off map verisi (tabloya dönüştürülebilir format)
        """
        rows = []
        
        for i, angle in enumerate(self.master_sequence):
            row = {
                "ply_index": i + 1,  # 1-based görüntüleme
                "angle": angle,
                "regions": {}
            }
            
            for region_id, mask in self.region_masks.items():
                row["regions"][region_id] = mask[i]
            
            rows.append(row)
        
        # Bölgeleri ply sayısına göre sırala (çoktan aza)
        sorted_regions = sorted(
            self.region_masks.keys(),
            key=lambda x: sum(self.region_masks[x]),
            reverse=True
        )
        
        return {
            "rows": rows,
            "region_order": sorted_regions,
            "master_region_id": self.master_region_id,
            "total_plies": self.ply_count,
            "region_ply_counts": {
                rid: sum(mask) for rid, mask in self.region_masks.items()
            }
        }

    def validate_continuity_between(self, region_a: str, region_b: str) -> Tuple[bool, List[int]]:
        """
        İki bölge arasında süreklilik kontrolü.
        
        Kural: İnce bölgede var olan bir ply, kalın bölgede de olmalı.
        
        Returns:
            (is_valid, violation_indices)
        """
        if region_a not in self.region_masks or region_b not in self.region_masks:
            return False, []
        
        mask_a = self.region_masks[region_a]
        mask_b = self.region_masks[region_b]
        
        count_a = sum(mask_a)
        count_b = sum(mask_b)
        
        # İnce ve kalın bölgeyi belirle
        if count_a >= count_b:
            thick_mask, thin_mask = mask_a, mask_b
        else:
            thick_mask, thin_mask = mask_b, mask_a
        
        violations = []
        for i, (thick, thin) in enumerate(zip(thick_mask, thin_mask)):
            # İhlal: İnce bölgede var ama kalın bölgede yok
            if thin and not thick:
                violations.append(i)
        
        return len(violations) == 0, violations

    def to_dict(self) -> Dict[str, Any]:
        """MPT'yi dictionary olarak döndür."""
        return {
            "master_region_id": self.master_region_id,
            "master_sequence": self.master_sequence,
            "ply_count": self.ply_count,
            "region_masks": {
                rid: mask for rid, mask in self.region_masks.items()
            },
            "region_ply_counts": {
                rid: sum(mask) for rid, mask in self.region_masks.items()
            },
            "dropoff_map": self.generate_dropoff_map(),
        }

    def to_table_string(self) -> str:
        """MPT'yi tablo formatında string olarak döndür."""
        sorted_regions = sorted(
            self.region_masks.keys(),
            key=lambda x: sum(self.region_masks[x]),
            reverse=True
        )
        
        # Header
        header = "Ply# | Angle | " + " | ".join(sorted_regions)
        separator = "-" * len(header)
        
        lines = [header, separator]
        
        for i, angle in enumerate(self.master_sequence):
            row_values = []
            for region_id in sorted_regions:
                mask = self.region_masks.get(region_id, [])
                if i < len(mask):
                    row_values.append("✓" if mask[i] else "✗")
                else:
                    row_values.append("-")
            
            line = "{:4d} | {:5d} | {}".format(
                i + 1, angle, " | ".join("{:^{}}".format(v, len(r)) for v, r in zip(row_values, sorted_regions))
            )
            lines.append(line)
        
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "MasterPlyTable(master={}, plies={}, regions={})".format(
            self.master_region_id, self.ply_count, len(self.region_masks)
        )
