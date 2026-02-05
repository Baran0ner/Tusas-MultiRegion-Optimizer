"""
MultiRegionOptimizer class - Çoklu bölge optimizasyonu.

Havacılık kurallarına ve katman sürekliliğine uygun şekilde
çoklu bölgeleri optimize eder.

Strateji:
1. Master bölge için tam optimizasyon (LaminateOptimizer kullanarak)
2. Diğer bölgeler için continuity-aware drop-off
3. Multi-directional blending (çoklu komşu sınırları)
"""
from typing import Dict, List, Any, Optional, Tuple
import time

from .region import Region
from .graph import RegionGraph
from .mpt import MasterPlyTable
from ..laminate_optimizer import LaminateOptimizer
from ..dropoff_optimizer import DropOffOptimizer


class MultiRegionOptimizer:
    """
    Çoklu bölge optimizasyonu için ana sınıf.
    
    Graf tabanlı yaklaşımla:
    - Bölgeler arası komşuluk ilişkilerini yönetir
    - Master Ply Table ile referans sequence tutar
    - Continuity kurallarına uygun drop-off yapar
    """

    def __init__(self, graph: RegionGraph, ply_counts: Dict[int, int], fast_mode: bool = True):
        """
        Args:
            graph: Bölge grafiği (komşuluk ilişkileri)
            ply_counts: Master bölge için açı dağılımı
                Örnek: {0: 18, 90: 18, 45: 18, -45: 18}
            fast_mode: True ise hızlı optimizasyon (skeleton-only), False ise tam optimizasyon
        """
        self.graph = graph
        self.ply_counts = ply_counts
        self.fast_mode = fast_mode
        self.base_optimizer = LaminateOptimizer(ply_counts)
        
        # Sonuç yapıları
        self.regions: Dict[str, Region] = {}
        self.mpt: Optional[MasterPlyTable] = None
        
        # İstatistikler
        self.stats: Dict[str, Any] = {}

    def optimize(self, master_region_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Çoklu bölge optimizasyonunu çalıştır.
        
        Args:
            master_region_id: Master bölge ID'si (None ise otomatik seç)
            
        Returns:
            Optimizasyon sonucu (mpt, regions, stats)
        """
        start_time = time.time()
        
        # 1. Master bölgeyi belirle
        if master_region_id is None:
            master_region_id = self.graph.get_master_region_id()
        
        print("=" * 60)
        print("MULTI-REGION OPTIMIZATION")
        print("=" * 60)
        print("Master Region: {}".format(master_region_id))
        print("Total Regions: {}".format(len(self.graph.all_region_ids)))
        print("Mode: {}".format("FAST" if self.fast_mode else "FULL"))
        
        # 2. Master bölge için optimizasyon
        print("\nPhase 1: Master Region Optimization")
        
        if self.fast_mode:
            # Hızlı mod: Sadece skeleton + quick local search
            skeleton = self.base_optimizer._build_smart_skeleton()
            master_sequence, master_score = self.base_optimizer._local_search(skeleton, max_iter=30)
            # Calculate details
            _, master_details = self.base_optimizer.calculate_fitness(master_sequence)
        else:
            # Tam mod: Hybrid optimization (yavaş)
            master_sequence, master_score, master_details, _ = self.base_optimizer.run_hybrid_optimization()
        
        # Master bölge Region objesi oluştur
        master_region = Region(master_region_id, len(master_sequence))
        master_region.sequence = master_sequence
        master_region.fitness_score = master_score
        master_region.fitness_details = master_details
        master_region.is_master = True
        master_region.ply_mask = [True] * len(master_sequence)
        master_region.set_neighbors(self.graph.get_neighbors(master_region_id))
        self.regions[master_region_id] = master_region
        
        # 3. MPT oluştur
        self.mpt = MasterPlyTable(master_sequence, master_region_id)
        
        # 4. Diğer bölgeleri ply sayısına göre sıralı işle (kalından inceye)
        print("\nPhase 2: Region Drop-offs (Continuity-Aware)")
        sorted_regions = self.graph.get_regions_sorted_by_ply_count(descending=True)
        
        for region_id in sorted_regions:
            if region_id == master_region_id:
                continue
            
            target_ply_count = self.graph.get_ply_count(region_id)
            
            print("  Processing {} (target: {} plies)...".format(region_id, target_ply_count))
            
            # Bu bölge için drop-off yap
            region = self._create_region_with_dropoff(
                region_id, 
                target_ply_count,
                master_sequence
            )
            
            self.regions[region_id] = region
            
            # MPT'ye mask ekle
            self.mpt.add_region_mask(region_id, region.ply_mask)
            
            print("    Result: {} plies, fitness: {:.2f}".format(
                len(region.sequence), region.fitness_score
            ))
        
        # 5. Continuity doğrulama
        print("\nPhase 3: Continuity Validation")
        continuity_results = self._validate_all_continuity()
        
        # 6. İstatistikler
        total_time = time.time() - start_time
        self.stats = {
            "total_time_seconds": round(total_time, 2),
            "master_region_id": master_region_id,
            "total_regions": len(self.regions),
            "master_fitness": master_score,
            "continuity_valid": all(r["valid"] for r in continuity_results),
            "continuity_details": continuity_results,
        }
        
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("Total time: {:.2f}s".format(total_time))
        print("=" * 60)
        
        return self.get_result()

    def _create_region_with_dropoff(
        self, 
        region_id: str, 
        target_ply_count: int,
        master_sequence: List[int]
    ) -> Region:
        """
        Bir bölge için continuity-aware drop-off yap.
        
        Args:
            region_id: Bölge ID'si
            target_ply_count: Hedef ply sayısı
            master_sequence: Master sequence
            
        Returns:
            Optimize edilmiş Region objesi
        """
        region = Region(region_id, target_ply_count)
        region.set_neighbors(self.graph.get_neighbors(region_id))
        
        # Komşu bölgeleri kontrol et (continuity için)
        neighbor_constraints = self._get_neighbor_constraints(region_id)
        
        # Collect forced drops from constraints (thicker neighbors)
        # If a thicker neighbor dropped a ply (False), shorter region MUST also drop it
        forced_drop_set = set()
        for mask in neighbor_constraints.values():
            for i, keep in enumerate(mask):
                if not keep:
                    forced_drop_set.add(i)
        
        fixed_drop_indices = sorted(list(forced_drop_set))
        
        # Drop-off optimizer kullan
        drop_optimizer = DropOffOptimizer(master_sequence, self.base_optimizer)
        new_sequence, score, dropped_indices = drop_optimizer.optimize_drop(target_ply_count, fixed_drop_indices=fixed_drop_indices)
        
        # Detayları hesapla
        final_score, final_details = self.base_optimizer.calculate_fitness(new_sequence)
        
        # Mask oluştur
        mask = [True] * len(master_sequence)
        for idx in dropped_indices:
            mask[idx] = False
        
        # Region'ı güncelle
        region.set_sequence_from_mask(master_sequence, mask)
        region.fitness_score = final_score
        region.fitness_details = final_details
        
        return region

    def _get_neighbor_constraints(self, region_id: str) -> Dict[str, List[bool]]:
        """
        Bir bölgenin komşularından gelen kısıtları topla.
        
        Daha kalın komşuların mask'ları constraint olarak kullanılır.
        """
        constraints = {}
        region_ply_count = self.graph.get_ply_count(region_id)
        
        for neighbor_id in self.graph.get_neighbors(region_id):
            if neighbor_id in self.regions:
                neighbor = self.regions[neighbor_id]
                neighbor_ply_count = len(neighbor.sequence)
                
                # Sadece daha kalın komşulardan constraint al
                if neighbor_ply_count > region_ply_count:
                    constraints[neighbor_id] = neighbor.ply_mask
        
        return constraints

    def _validate_all_continuity(self) -> List[Dict[str, Any]]:
        """Tüm komşu çiftleri arasında süreklilik kontrolü."""
        results = []
        
        for edge in self.graph.get_all_edges():
            region_a, region_b = edge
            
            if region_a not in self.regions or region_b not in self.regions:
                continue
            
            is_valid, violations = self.mpt.validate_continuity_between(region_a, region_b)
            
            results.append({
                "edge": [region_a, region_b],
                "valid": is_valid,
                "violations": violations,
            })
            
            if not is_valid:
                print("  WARNING: Continuity violation between {} and {} at indices {}".format(
                    region_a, region_b, violations
                ))
            else:
                print("  OK: {} <-> {}".format(region_a, region_b))
        
        return results

    def get_result(self) -> Dict[str, Any]:
        """Optimizasyon sonucunu döndür."""
        return {
            "mpt": self.mpt.to_dict() if self.mpt else None,
            "regions": {rid: r.to_dict() for rid, r in self.regions.items()},
            "graph": self.graph.to_dict(),
            "stats": self.stats,
            "dropoff_map": self.mpt.generate_dropoff_map() if self.mpt else None,
        }

    def get_mpt_table(self) -> str:
        """MPT'yi tablo formatında döndür."""
        if self.mpt is None:
            return "MPT not initialized"
        return self.mpt.to_table_string()

    def __repr__(self) -> str:
        return "MultiRegionOptimizer(regions={}, mpt={})".format(
            len(self.regions), self.mpt is not None
        )
