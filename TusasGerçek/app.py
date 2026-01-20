from tusas import create_app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

from tusas import create_app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

from tusas import create_app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

import random
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

# -----------------------------------------------------------------------------
# Genetic Algorithm Core
# -----------------------------------------------------------------------------


class LaminateOptimizer:
    """
    Composite laminate stacking optimizer.
    Rules implemented: 1–8 (including lateral bending stiffness – Rule 8)
    """
    
    # Class-level weight constants for consistent scoring
    WEIGHTS = {
        "R1": 20.0,    # Symmetry
        "R2": 12.0,    # Balance
        "R3": 13.0,    # Percentage
        "R4": 12.0,    # External plies
        "R5": 8.0,     # Distribution
        "R6": 21.0,    # Grouping
        "R7": 7.0,     # Buckling
        "R8": 7.0,     # Lateral bending
    }
    
    # Thresholds for various rules
    LATERAL_BENDING_THRESHOLD = 0.35
    DISTRIBUTION_STD_RATIO = 0.7
    DROP_OFF_ATTEMPTS = 3000
    ANGLE_TARGET_DROP_ATTEMPTS = 3000

    def __init__(self, ply_counts: Dict[int, int]):
        self.ply_counts = ply_counts
        self.initial_pool: List[int] = []
        for angle, count in ply_counts.items():
            self.initial_pool.extend([angle] * int(count))

        self.total_plies = len(self.initial_pool)
        self._fitness_cache: Dict[tuple, Tuple[float, Dict]] = {}  # Fitness cache

    def calculate_clt_proxy(self, sequence: List[int]) -> float:
        return abs(sequence.count(45) - sequence.count(-45)) * 2.5

    def _is_symmetric(self, sequence: List[int]) -> bool:
        """Sequence simetrik mi kontrol et."""
        n = len(sequence)
        for i in range(n // 2):
            if sequence[i] != sequence[n - 1 - i]:
                return False
        return True

    def _create_symmetric_individual(self) -> List[int]:
        """
        Baştan simetrik birey oluştur.
        Her açı için pool'dan yarısını al, sol yarıya koy, aynı ply'ları reverse yaparak sağ yarıya koy.
        Bu sayede sequence simetrik olur ve her açının sayısı korunur.
        
        Tek sayıda toplam ply için:
        - Sol yarı + Orta ply + Sağ yarı (sol yarının mirror'ı)
        - Orta ply, tek sayıda olan açılardan birinden seçilir
        """
        total = len(self.initial_pool)
        half = total // 2
        is_odd_total = total % 2 == 1
        
        # Her açının sayısını hesapla
        angle_total_counts = {}
        for angle in set(self.initial_pool):
            angle_total_counts[angle] = self.initial_pool.count(angle)
        
        # Tek sayıda olan açıları bul (orta ply için aday)
        odd_angles = [ang for ang, cnt in angle_total_counts.items() if cnt % 2 == 1]
        
        # Orta ply'ı belirle (tek sayılı toplam için)
        middle_ply = None
        if is_odd_total:
            if odd_angles:
                # Tek sayıda olan açılardan birini orta ply yap
                middle_ply = random.choice(odd_angles)
            else:
                # Hepsi çift - bu teorik olarak olmamalı (tek toplam = en az bir tek açı)
                # Ama yine de handle edelim
                middle_ply = random.choice(list(angle_total_counts.keys()))
        
        # Her açı için sol yarıya koyulacak sayıyı hesapla
        angle_counts_for_left = {}
        for angle, total_count in angle_total_counts.items():
            if is_odd_total and angle == middle_ply:
                # Orta ply olarak seçilen açıdan 1 eksik al
                angle_counts_for_left[angle] = (total_count - 1) // 2
            else:
                angle_counts_for_left[angle] = total_count // 2
        
        # Pool'u karıştır
        pool_copy = self.initial_pool[:]
        random.shuffle(pool_copy)
        
        # Sol yarıyı oluştur
        left_half = []
        angle_counts_left = {angle: 0 for angle in angle_total_counts.keys()}
        
        for ply in pool_copy:
            target_count = angle_counts_for_left.get(ply, 0)
            if angle_counts_left[ply] < target_count and len(left_half) < half:
                left_half.append(ply)
                angle_counts_left[ply] += 1
        
        # Eğer sol yarı yeterli değilse, kalan ply'ları ekle
        while len(left_half) < half:
            for ply in pool_copy:
                if len(left_half) >= half:
                    break
                target = angle_counts_for_left.get(ply, 0)
                if angle_counts_left[ply] < target:
                    left_half.append(ply)
                    angle_counts_left[ply] += 1
            # Sonsuz döngüyü önle
            if len(left_half) < half:
                # Herhangi birinden ekle
                for ply in pool_copy:
                    if len(left_half) >= half:
                        break
                    if left_half.count(ply) < angle_total_counts[ply] // 2 + 1:
                        left_half.append(ply)
                break
        
        # Sol yarıyı karıştır (daha iyi dağılım için)
        random.shuffle(left_half)
        
        # Sağ yarı = Sol yarının aynası (reverse) - simetrik olması için
        right_half = left_half[::-1]
        
        # Sequence oluştur
        if is_odd_total:
            sequence = left_half + [middle_ply] + right_half
        else:
            sequence = left_half + right_half
        
        # Validation: Ply sayılarını kontrol et
        assert len(sequence) == total, f"Sequence length mismatch: {len(sequence)} != {total}"
        for angle in set(self.initial_pool):
            expected = self.initial_pool.count(angle)
            actual = sequence.count(angle)
            assert expected == actual, f"Angle {angle} count mismatch: expected {expected}, got {actual}"
        
        return sequence

    def _check_symmetry_distance_weighted(self, sequence: List[int]) -> float:
        """Rule 1: Distance-weighted symmetry penalty."""
        penalty = 0.0
        n = len(sequence)
        mid = (n - 1) / 2
        max_penalty = self.WEIGHTS["R1"]
        
        for i in range(n // 2):
            if sequence[i] != sequence[-1 - i]:
                # Middle plane'e yakınsa daha az penalty
                dist_from_mid = abs(i - mid) / max(1, mid)
                penalty += max_penalty * dist_from_mid
        
        return min(penalty, max_penalty)

    def _check_balance_45(self, sequence: List[int]) -> float:
        """Rule 2: ±45 balance check."""
        diff = abs(sequence.count(45) - sequence.count(-45))
        total_45_count = sequence.count(45) + sequence.count(-45)
        max_penalty = self.WEIGHTS["R2"]
        
        if total_45_count > 0:
            normalized_diff = min(1.0, diff / max(1, total_45_count // 2))
            penalty = max_penalty * normalized_diff
        else:
            penalty = 0.0
        
        return penalty

    def _check_percentage_rule(self, sequence: List[int]) -> float:
        """Rule 3: Percentage rule - her yönde %8-67 kontrolü."""
        penalty = 0.0
        n = len(sequence)
        max_penalty = self.WEIGHTS["R3"]
        per_violation_penalty = max_penalty / 4  # 4 açı için eşit dağılım
        
        for angle in [0, 45, -45, 90]:
            count = sequence.count(angle)
            ratio = count / n if n > 0 else 0.0
            
            if ratio < 0.08 or ratio > 0.67:
                penalty += per_violation_penalty
        
        return min(penalty, max_penalty)

    def _check_external_plies(self, sequence: List[int]) -> float:
        """Rule 4: External plies - ilk 2 ve son 2 katman kontrolü."""
        n = len(sequence)
        max_score = self.WEIGHTS["R4"]
        
        if n < 2:
            return max_score
        
        score = max_score
        penalty = 0.0
        quarter_penalty = max_score * 0.25  # Her ihlal için %25 penalty
        
        # SOFT: 90° başlangıç/bitiş AVOID
        if sequence[0] == 90 or sequence[-1] == 90:
            penalty += quarter_penalty
        
        # SOFT: İlk 2 ve son 2 katman ±45 bonus
        outer_4 = [sequence[0], sequence[1], sequence[-2], sequence[-1]]
        ideal_count = sum(abs(ply) == 45 for ply in outer_4)
        
        # 4'ünden biri ideal değilse penalty
        if ideal_count < 4:
            penalty += (4 - ideal_count) * (quarter_penalty / 1.5)  # Her eksik için penalty
        
        score = max(0, max_score - penalty)
        return score

    def _check_distribution_variance(self, sequence: List[int]) -> float:
        """Rule 5: Standard deviation based distribution kontrolü."""
        penalty = 0.0
        n = len(sequence)
        max_penalty = self.WEIGHTS["R5"]
        per_angle_penalty = max_penalty / 4  # 4 açı için eşit dağılım
        
        for angle in [0, 45, -45, 90]:
            indices = [i for i, x in enumerate(sequence) if x == angle]
            
            if len(indices) > 1:
                # İdeal: Eşit aralıklı dağılım
                ideal_spacing = n / len(indices)
                actual_spacings = np.diff(indices) if len(indices) > 1 else []
                
                # Standart sapma yüksekse dağılım kötü
                if len(actual_spacings) > 0:
                    std_dev = np.std(actual_spacings)
                    normalized_std = min(1.0, std_dev / max(ideal_spacing, 1.0))
                    penalty += normalized_std * per_angle_penalty
        
        return min(penalty, max_penalty)

    def _count_groupings(self, sequence: List[int]) -> int:
        """Sequence'deki toplam grouping sayısını döndür (adjacent pairs)."""
        count = 0
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i - 1]:
                count += 1
        return count

    def _find_groups_of_size(self, sequence: List[int], target_size: int) -> int:
        """Belirli boyutta grupları say (örn: 3'lü gruplar)."""
        if len(sequence) < target_size:
            return 0
        count = 0
        curr = 1
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i - 1]:
                curr += 1
            else:
                if curr == target_size:
                    count += 1
                curr = 1
        if curr == target_size:
            count += 1
        return count

    def _check_grouping(self, sequence: List[int], max_group: int = 3) -> float:
        """Rule 6: Grouping kontrolü - max 3 ply üst üste + toplam grouping minimize + 3'lü grup penalty."""
        penalty = 0.0
        max_group_found = 1
        curr = 1
        total_adjacent_pairs = 0  # Toplam yan yana aynı açı sayısı
        max_penalty = self.WEIGHTS["R6"]
        
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i - 1]:
                curr += 1
                total_adjacent_pairs += 1  # Her yan yana aynı açı bir grouping
            else:
                curr = 1
            max_group_found = max(max_group_found, curr)
        
        # Penalty 1: Max group > 3 ise büyük penalty (max_penalty'nin %25'i her fazla için)
        if max_group_found > max_group:
            excess = max_group_found - max_group
            penalty += excess * (max_penalty * 0.25)
        
        # Penalty 2: 3'lü gruplar için küçük penalty (ideal: 2'li veya daha az)
        groups_of_3 = self._find_groups_of_size(sequence, 3)
        penalty += groups_of_3 * 1.0  # Her 3'lü grup için 1 puan penalty
        
        # Penalty 3: Toplam adjacent pairs'e göre küçük penalty
        n = len(sequence)
        if n > 1:
            adjacent_ratio = total_adjacent_pairs / (n - 1)
            adjacent_penalty = adjacent_ratio * (max_penalty * 0.5)  # Max %50 penalty
            penalty += adjacent_penalty
        
        return min(penalty, max_penalty)

    def _check_buckling(self, sequence: List[int]) -> float:
        """Rule 7: Buckling - ±45 katmanlar dış yüzeylerde olmalı (orta düzlemden uzak).
        
        Buckling direnci için ±45° katmanlar sequence'in dış taraflarında olmalı.
        Ortaya yakın ±45° varsa penalty verilir.
        """
        max_penalty = self.WEIGHTS["R7"]
        n = len(sequence)
        mid = (n - 1) / 2
        
        # ±45 katmanların pozisyonlarını bul
        positions_45 = [i for i, ang in enumerate(sequence) if abs(ang) == 45]
        
        if not positions_45:
            return 0.0  # ±45 yoksa penalty yok
        
        # Orta düzleme yakınlık penalty'si hesapla
        # Sadece ortaya yakın olanlar için penalty
        center_zone = 0.4  # Orta %40'lık bölge
        penalty_sum = 0.0
        
        for pos in positions_45:
            dist = abs(pos - mid) / max(1, mid)  # 0 = ortada, 1 = uçta
            
            # Sadece ortaya yakınsa (dist < center_zone) penalty ver
            if dist < center_zone:
                # Ortaya ne kadar yakınsa o kadar penalty
                proximity_penalty = (center_zone - dist) / center_zone
                penalty_sum += proximity_penalty
        
        # Normalize: penalty_sum'u max_penalty'ye scale et
        # Eğer tüm ±45'ler ortadaysa max penalty
        total_45_count = len(positions_45)
        if total_45_count > 0:
            normalized_penalty = (penalty_sum / total_45_count) * max_penalty
        else:
            normalized_penalty = 0.0
        
        return min(normalized_penalty, max_penalty)

    def _check_lateral_bending(self, sequence: List[int]) -> float:
        """Rule 8: Lateral bending - 90° katmanlar dış yüzeylerde olmalı (orta düzlemden uzak).
        
        Lateral bending sertliği için 90° katmanlar dış yüzeylerde olmalı.
        Ortaya yakın 90° varsa penalty verilir.
        """
        max_penalty = self.WEIGHTS["R8"]
        threshold = self.LATERAL_BENDING_THRESHOLD
        n = len(sequence)
        mid = (n - 1) / 2
        
        # 90° katmanların pozisyonlarını bul
        positions_90 = [i for i, ang in enumerate(sequence) if ang == 90]
        
        if not positions_90:
            return 0.0  # 90° yoksa penalty yok
        
        penalty_sum = 0.0
        for pos in positions_90:
            dist = abs(pos - mid) / max(1, mid)
            if dist < threshold:
                # Ortaya ne kadar yakınsa o kadar penalty
                penalty_sum += (threshold - dist) / threshold
        
        # Normalize: penalty_sum'u max_penalty'ye scale et
        total_90_count = len(positions_90)
        if total_90_count > 0:
            normalized_penalty = (penalty_sum / total_90_count) * max_penalty
        else:
            normalized_penalty = 0.0
        
        return min(normalized_penalty, max_penalty)

    def _symmetry_preserving_swap(self, sequence: List[int]) -> None:
        """Simetriyi koruyarak swap yap - sol yarıda swap, sağ yarıda mirror."""
        n = len(sequence)
        half = n // 2
        
        if half < 2:
            return
        
        # Sol yarıdan iki index seç
        i = random.randint(0, half - 1)
        j = random.randint(0, half - 1)
        
        if i == j:
            return
        
        # Sol yarıda swap
        sequence[i], sequence[j] = sequence[j], sequence[i]
        
        # Aynı swap'i sağ yarıda da yap (simetrik)
        i_mirror = n - 1 - i
        j_mirror = n - 1 - j
        sequence[i_mirror], sequence[j_mirror] = sequence[j_mirror], sequence[i_mirror]

    def _grouping_aware_mutation(self, sequence: List[int]) -> bool:
        """Grouping'i azaltan symmetry-preserving swap yap. Başarılı olursa True döner."""
        n = len(sequence)
        half = n // 2
        
        if half < 2:
            return False
        
        current_groupings = self._count_groupings(sequence)
        
        # Grouping azaltan swap'leri bul
        good_swaps = []
        
        for i in range(half):
            for j in range(i + 1, half):
                candidate = sequence[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                mirror_i = n - 1 - i
                mirror_j = n - 1 - j
                candidate[mirror_i], candidate[mirror_j] = candidate[mirror_j], candidate[mirror_i]
                
                candidate_groupings = self._count_groupings(candidate)
                
                if candidate_groupings < current_groupings:
                    good_swaps.append((i, j))
        
        if good_swaps:
            # Random bir grouping-azaltan swap seç
            i, j = random.choice(good_swaps)
            sequence[i], sequence[j] = sequence[j], sequence[i]
            mirror_i = n - 1 - i
            mirror_j = n - 1 - j
            sequence[mirror_i], sequence[mirror_j] = sequence[mirror_j], sequence[mirror_i]
            return True
        
        return False  # Grouping azaltan swap bulunamadı

    def _balance_aware_mutation(self, sequence: List[int]) -> None:
        """Balance'ı koruyarak mutasyon yap - +45 ile -45 swap et (simetrik)."""
        n = len(sequence)
        half = n // 2
        
        # Sol yarıda +45 ve -45 bul
        pos_45_left = [i for i in range(half) if sequence[i] == 45]
        neg_45_left = [i for i in range(half) if sequence[i] == -45]
        
        if pos_45_left and neg_45_left:
            i1 = random.choice(pos_45_left)
            i2 = random.choice(neg_45_left)
            
            # Sol yarıda swap
            sequence[i1], sequence[i2] = sequence[i2], sequence[i1]
            
            # Sağ yarıda da simetrik swap
            i1_mirror = n - 1 - i1
            i2_mirror = n - 1 - i2
            sequence[i1_mirror], sequence[i2_mirror] = sequence[i2_mirror], sequence[i1_mirror]

    def _get_balanced_candidates(self, pool: List[int], sequence: List[int]) -> List[int]:
        """Balance'ı koruyacak candidate'leri döndür."""
        count_45 = sequence.count(45)
        count_neg45 = sequence.count(-45)
        
        if count_45 > count_neg45:
            # -45 öncelikli
            candidates = [p for p in pool if p == -45]
            return candidates if candidates else pool[:]
        elif count_neg45 > count_45:
            # +45 öncelikli
            candidates = [p for p in pool if p == 45]
            return candidates if candidates else pool[:]
        else:
            # Dengeli, herhangi biri
            return pool[:]

    def _violates_adjacency(self, sequence: List[int], candidate: int) -> bool:
        """candidate eklenirse 0-90 adjacency oluşur mu?"""
        if len(sequence) == 0:
            return False
        
        last_ply = sequence[-1]
        
        # 0-90 veya 90-0 yasak
        if {last_ply, candidate} == {0, 90}:
            return True
        
        return False

    def _violates_grouping(self, sequence: List[int], candidate: int, max_group: int = 3) -> bool:
        """candidate eklenirse grouping limiti aşılır mı?"""
        if len(sequence) == 0:
            return False
        
        # Son kaç ply candidate ile aynı?
        count = 0
        for i in range(len(sequence) - 1, -1, -1):
            if sequence[i] == candidate:
                count += 1
            else:
                break
        
        # +1 eklersek limit aşılır mı?
        return (count + 1) > max_group

    def _distribution_score_incremental(self, sequence: List[int], candidate: int) -> float:
        """candidate eklendikten sonra distribution score'u tahmin et."""
        temp_seq = sequence + [candidate]
        
        # candidate'in aynı açılı ply'larının index'lerini bul
        indices = [i for i, x in enumerate(temp_seq) if x == candidate]
        
        if len(indices) <= 1:
            return 1.0  # Tek ply, perfect distribution
        
        # Spacing variance hesapla
        spacings = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
        
        if len(spacings) == 0:
            return 1.0
        
        variance = np.var(spacings) if len(spacings) > 0 else 0.0
        
        # Düşük variance → iyi distribution
        return 1.0 / (1.0 + variance) if variance > 0 else 1.0

    def _build_smart_skeleton(self) -> List[int]:
        """Kuralları sırayla tatmin eden başlangıç sequence oluştur (simetrik)."""
        # Simetrik skeleton oluşturmak için _create_symmetric_individual metodunu kullan
        # Bu metod her açının sayısını koruyarak simetrik bir sequence oluşturur
        return self._create_symmetric_individual()

    def _multi_start_ga(self, skeleton: List[int], n_runs: int = 7) -> Tuple[List[int], float]:
        """Multi-start GA: Skeleton'dan başlayarak farklı local optima'lara bakar."""
        print("Phase 2: Multi-Start GA")
        
        skeleton_score, _ = self.calculate_fitness(skeleton)
        print(f"  Skeleton score: {skeleton_score:.2f}/100")
        
        # Ply sayısına göre adaptive parametreler
        population_size = 120
        generations = 200
        
        # Büyük ply sayıları için ölçekleme
        if self.total_plies > 40:
            population_size = min(150, int(120 * (self.total_plies / 40.0)))
            generations = min(250, int(200 * (self.total_plies / 40.0)))
        
        best_global = skeleton[:]
        best_score = skeleton_score
        
        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}...", end=" ")
            
            # Skeleton'dan population türet
            population = []
            for i in range(population_size):
                mutated = skeleton[:]
                
                # Progressively stronger mutations
                n_mutations = (run + 1) + (i // 20)
                for _ in range(n_mutations):
                    self._symmetry_preserving_swap(mutated)
                
                population.append(mutated)
            
            # GA refine
            best_seq = None
            best_fit = -1
            
            for gen in range(generations):
                scored = []
                for ind in population:
                    fit, _ = self.calculate_fitness(ind)
                    scored.append((fit, ind))
                
                scored.sort(reverse=True, key=lambda x: x[0])
                
                if scored[0][0] > best_fit:
                    best_fit = scored[0][0]
                    best_seq = scored[0][1][:]
                
                # Elite + mutation (top 10% elite)
                elite_size = max(10, int(population_size * 0.1))
                elite = [x[1][:] for x in scored[:elite_size]]
                next_gen = elite[:]
                
                while len(next_gen) < population_size:
                    parent = random.choice(elite)[:]
                    # Önce grouping-aware mutation dene
                    if random.random() < 0.4:
                        if not self._grouping_aware_mutation(parent):
                            # Başarısız olursa normal swap
                            if random.random() < 0.5:
                                self._symmetry_preserving_swap(parent)
                    next_gen.append(parent)
                
                population = next_gen
            
            print(f"Score: {best_fit:.2f}")
            
            if best_fit > best_score:
                best_score = best_fit
                best_global = best_seq[:]
        
        print(f"  Best across runs: {best_score:.2f}/100")
        return best_global, best_score

    def _local_search(self, sequence: List[int], max_iter: int = 100) -> Tuple[List[int], float]:
        """Hill climbing: 3'lü grupları 2'liye düşürmeyi önceliklendir.
        Normal zamanlarda Rule 4'ü korur, ama 3'lü grup azaltma için Rule 4'ü bozabilir."""
        print("Phase 3: Local Search")
        
        current = sequence[:]
        current_score, _ = self.calculate_fitness(current)
        current_groupings = self._count_groupings(current)
        current_groups_of_3 = self._find_groups_of_size(current, 3)
        print(f"  Initial score: {current_score:.2f}, Groupings: {current_groupings}, Groups of 3: {current_groups_of_3}")
        
        iteration = 0
        improvements = 0
        
        while iteration < max_iter:
            improved = False
            n = len(current)
            half = n // 2
            
            # AŞAMA 1: Rule 4'ü koruyan swap'ler
            candidates_rule4_preserved = []
            # AŞAMA 2: Rule 4'ü bozan ama 3'lü grup azaltan swap'ler
            candidates_rule4_violated = []
            
            for i in range(half):
                for j in range(i + 1, half):
                    candidate = current[:]
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    mirror_i = n - 1 - i
                    mirror_j = n - 1 - j
                    candidate[mirror_i], candidate[mirror_j] = candidate[mirror_j], candidate[mirror_i]
                    
                    candidate_score, _ = self.calculate_fitness(candidate)
                    candidate_groupings = self._count_groupings(candidate)
                    candidate_groups_of_3 = self._find_groups_of_size(candidate, 3)
                    
                    grouping_change = current_groupings - candidate_groupings
                    groups_of_3_change = current_groups_of_3 - candidate_groups_of_3
                    
                    # Rule 4 kontrolü: Başlangıç ve bitiş ±45 olmalı
                    rule4_preserved = (candidate[0] in [45, -45] and candidate[-1] in [45, -45])
                    
                    # Öncelik sırası
                    priority = (
                        groups_of_3_change > 0,  # 3'lü grup azaltanlar en öncelikli
                        grouping_change > 0,
                        groups_of_3_change,  # Ne kadar 3'lü grup azalttı
                        grouping_change,
                        candidate_score
                    )
                    
                    candidate_data = (
                        candidate,
                        candidate_score,
                        priority,
                        grouping_change,
                        groups_of_3_change
                    )
                    
                    if rule4_preserved:
                        # Rule 4 korunuyor - normal aday
                        candidates_rule4_preserved.append(candidate_data)
                    elif groups_of_3_change > 0:
                        # Rule 4 bozuluyor AMA 3'lü grup azaltıyor - özel durum
                        candidates_rule4_violated.append(candidate_data)
                    # Diğer durumlar: Rule 4 bozuluyor ve 3'lü grup azaltmıyor - ATLA
            
            # AŞAMA 1: Önce Rule 4'ü koruyan swap'leri dene
            candidates_rule4_preserved.sort(key=lambda x: x[2], reverse=True)
            
            for candidate, candidate_score, priority, grouping_change, groups_of_3_change in candidates_rule4_preserved:
                if candidate_score > current_score:
                    current = candidate
                    current_score = candidate_score
                    current_groupings = self._count_groupings(candidate)
                    current_groups_of_3 = self._find_groups_of_size(candidate, 3)
                    improved = True
                    improvements += 1
                    print(f"  Iteration {iteration}: Improved to {current_score:.2f} (Rule4 preserved), Groupings: {current_groupings} ({grouping_change:+d}), Groups of 3: {current_groups_of_3} ({groups_of_3_change:+d})")
                    break
            
            # AŞAMA 2: Eğer Rule 4 korunarak 3'lü grup azaltılamadıysa, Rule 4'ü bozan swap'leri dene
            if not improved and candidates_rule4_violated:
                candidates_rule4_violated.sort(key=lambda x: x[2], reverse=True)
                
                for candidate, candidate_score, priority, grouping_change, groups_of_3_change in candidates_rule4_violated:
                    # Rule 4 bozuluyor ama 3'lü grup azaltıyor - kabul et (fitness fonksiyonu trade-off yapacak)
                    if candidate_score > current_score:
                        current = candidate
                        current_score = candidate_score
                        current_groupings = self._count_groupings(candidate)
                        current_groups_of_3 = self._find_groups_of_size(candidate, 3)
                        improved = True
                        improvements += 1
                        print(f"  Iteration {iteration}: Improved to {current_score:.2f} (Rule4 violated for grouping reduction), Groupings: {current_groupings} ({grouping_change:+d}), Groups of 3: {current_groups_of_3} ({groups_of_3_change:+d})")
                        break
            
            if not improved:
                print(f"  Converged after {iteration} iterations ({improvements} improvements)")
                break
            
            iteration += 1
        
        final_groups_of_3 = self._find_groups_of_size(current, 3)
        print(f"  Final score: {current_score:.2f}/100, Final groupings: {self._count_groupings(current)}, Final groups of 3: {final_groups_of_3}")
        return current, current_score

    def run_hybrid_optimization(self) -> Tuple[List[int], float, Dict[str, Any], List[float]]:
        """3-phase hybrid optimization pipeline."""
        print("="*60)
        print("3-PHASE HYBRID OPTIMIZATION")
        print("="*60)
        
        start_time = time.time()
        
        # Phase 1
        print("\nPhase 1: Smart Skeleton Construction")
        skeleton = self._build_smart_skeleton()
        phase1_score, phase1_details = self.calculate_fitness(skeleton)
        print(f"  Score: {phase1_score:.2f}/100")
        print(f"  Time: {time.time() - start_time:.2f}s")
        
        # Phase 2
        phase2_start = time.time()
        # Ply sayısına göre run sayısını ayarla
        n_runs = 7 if self.total_plies <= 40 else 10
        best_seq, phase2_score = self._multi_start_ga(skeleton, n_runs=n_runs)
        print(f"  Time: {time.time() - phase2_start:.2f}s")
        
        # Phase 3
        phase3_start = time.time()
        final_seq, final_score = self._local_search(best_seq, max_iter=100)
        print(f"  Time: {time.time() - phase3_start:.2f}s")
        
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print(f"FINAL RESULT: {final_score:.2f}/100 (in {total_time:.2f}s)")
        print("="*60)
        
        # Final details
        _, final_details = self.calculate_fitness(final_seq)
        
        # History (dummy - hybrid optimization'da history yok)
        history = [final_score]
        
        return final_seq, final_score, final_details, history

    def calculate_fitness(self, sequence: List[int]):
        """
        PDF kurallarına göre fitness hesapla.
        Max score = 100 (tüm rule weights toplamı)
        """
        # Use class-level WEIGHTS for consistency
        WEIGHTS = self.WEIGHTS
        
        n = len(sequence)
        rules_result = {}

        # ========== HARD CONSTRAINTS ==========
        
        # HARD 0: Symmetry check (critical constraint)
        # NOT: Bu opsiyonel - şu an soft constraint olarak Rule 1'de var
        # Eğer hard constraint istenirse uncomment edin:
        # if not self._is_symmetric(sequence):
        #     return 0.0, {
        #         "total_score": 0.0,
        #         "max_score": 100.0,
        #         "rules": {
        #             "SYMMETRY_VIOLATION": {
        #                 "weight": 999.0,
        #                 "score": 0,
        #                 "penalty": 999.0,
        #                 "reason": "Sequence asimetrik (HARD CONSTRAINT)"
        #             }
        #         }
        #     }
        
        # HARD 1: 0°-90° adjacency - REMOVED (now allowed)
        # 0° and 90° can now be adjacent to each other

        # HARD 2: 0° başlangıç/bitiş YASAK
        if sequence[0] == 0 or sequence[-1] == 0:
            return 0.0, {
                "total_score": 0.0,
                "max_score": 100.0,
                "rules": {
                    "EXTERNAL_0": {
                        "weight": 999.0,
                        "score": 0,
                        "penalty": 999.0,
                        "reason": "0° başlangıç veya bitiş katmanı (YASAK)"
                    }
                }
            }

        # ========== SOFT CONSTRAINTS ==========
        
        # Rule 1: Symmetry (distance-weighted)
        penalty_r1 = self._check_symmetry_distance_weighted(sequence)
        score_r1 = max(0, WEIGHTS["R1"] - penalty_r1)
        rules_result["R1"] = {
            "weight": WEIGHTS["R1"],
            "score": round(score_r1, 2),
            "penalty": round(penalty_r1, 2),
            "reason": "Asimetri var" if penalty_r1 > 0 else ""
        }

        # Rule 2: Balance (sadece ±45 için)
        penalty_r2 = self._check_balance_45(sequence)
        score_r2 = max(0, WEIGHTS["R2"] - penalty_r2)
        rules_result["R2"] = {
            "weight": WEIGHTS["R2"],
            "score": round(score_r2, 2),
            "penalty": round(penalty_r2, 2),
            "reason": "+45/-45 sayıları eşit değil" if penalty_r2 > 0 else ""
        }

        # Rule 3: Percentage (8-67%)
        penalty_r3 = self._check_percentage_rule(sequence)
        score_r3 = max(0, WEIGHTS["R3"] - penalty_r3)
        rules_result["R3"] = {
            "weight": WEIGHTS["R3"],
            "score": round(score_r3, 2),
            "penalty": round(penalty_r3, 2),
            "reason": "Bazı açılar %8-67 dışında" if penalty_r3 > 0 else ""
        }

        # Rule 4: External plies (ilk/son 2 katman)
        score_r4 = self._check_external_plies(sequence)
        penalty_r4 = WEIGHTS["R4"] - score_r4
        rules_result["R4"] = {
            "weight": WEIGHTS["R4"],
            "score": round(score_r4, 2),
            "penalty": round(penalty_r4, 2),
            "reason": "Dış katmanlar ideal değil" if penalty_r4 > 0 else ""
        }

        # Rule 5: Distribution (variance-based)
        penalty_r5 = self._check_distribution_variance(sequence)
        score_r5 = max(0, WEIGHTS["R5"] - penalty_r5)
        rules_result["R5"] = {
            "weight": WEIGHTS["R5"],
            "score": round(score_r5, 2),
            "penalty": round(penalty_r5, 2),
            "reason": "Dağılım uniform değil" if penalty_r5 > 0 else ""
        }

        # Rule 6: Grouping (max 3)
        penalty_r6 = self._check_grouping(sequence, max_group=3)
        score_r6 = max(0, WEIGHTS["R6"] - penalty_r6)
        rules_result["R6"] = {
            "weight": WEIGHTS["R6"],
            "score": round(score_r6, 2),
            "penalty": round(penalty_r6, 2),
            "reason": "Grouping limiti aşıldı" if penalty_r6 > 0 else ""
        }

        # Rule 7: Buckling (±45 uzakta)
        penalty_r7 = self._check_buckling(sequence)
        score_r7 = max(0, WEIGHTS["R7"] - penalty_r7)
        rules_result["R7"] = {
            "weight": WEIGHTS["R7"],
            "score": round(score_r7, 2),
            "penalty": round(penalty_r7, 2),
            "reason": "±45 middle plane'e yakın" if penalty_r7 > 0 else ""
        }

        # Rule 8: Lateral bending (90° uzakta)
        penalty_r8 = self._check_lateral_bending(sequence)
        score_r8 = max(0, WEIGHTS["R8"] - penalty_r8)
        rules_result["R8"] = {
            "weight": WEIGHTS["R8"],
            "score": round(score_r8, 2),
            "penalty": round(penalty_r8, 2),
            "reason": "90° middle plane'e yakın" if penalty_r8 > 0 else ""
        }

        # FINAL SCORE
        total_score = sum(r["score"] for r in rules_result.values())

        return total_score, {
            "total_score": round(total_score, 2),
            "max_score": 100.0,
            "rules": rules_result
        }

    def run_genetic_algorithm(
        self, population_size: int = 120, generations: int = 600
    ) -> Tuple[List[int], float, Dict[str, float], List[float]]:
        # Ply sayısına göre otomatik ayarlama (eğer varsayılan değerler kullanılıyorsa)
        if population_size <= 120:
            # Yüksek ply sayıları için daha büyük popülasyon
            base_pop = 120
            ply_factor = max(1.0, self.total_plies / 72.0)  # 72 ply için 1x
            population_size = int(base_pop * ply_factor)
            population_size = min(population_size, 400)  # Max 400
        
        if generations <= 600:
            # Yüksek ply sayıları için daha fazla jenerasyon
            base_gen = 600
            ply_factor = max(1.0, self.total_plies / 72.0)
            generations = int(base_gen * ply_factor)
            generations = min(generations, 1500)  # Max 1500
        
        # Symmetry-aware population initialization
        population: List[List[int]] = []
        for _ in range(population_size):
            ind = self._create_symmetric_individual()
            population.append(ind)

        best_sol = None  # type: Optional[List[int]]
        best_fit = -1.0
        best_det: Dict[str, float] = {}
        history: List[float] = []

        for gen in range(generations):
            scored_pop = []
            for ind in population:
                fit, det = self.calculate_fitness(ind)
                scored_pop.append((fit, ind))
                if fit > best_fit:
                    best_fit = fit
                    best_sol = ind[:]
                    best_det = det

            history.append(best_fit)
            scored_pop.sort(key=lambda x: x[0], reverse=True)
            elite_idx = max(1, int(population_size * 0.1))
            next_gen = [x[1][:] for x in scored_pop[:elite_idx]]

            # Adaptive mutation rate
            if gen > 50:
                recent_improvement = history[-1] - history[-50] if len(history) >= 50 else 1.0
                mutation_rate = 0.4 if recent_improvement < 0.1 else 0.2
            else:
                mutation_rate = 0.2

            while len(next_gen) < population_size:
                parent = max(random.sample(scored_pop, 3), key=lambda x: x[0])[1][:]
                
                # Symmetry-preserving swap mutation
                if random.random() < mutation_rate:
                    self._symmetry_preserving_swap(parent)
                
                # Balance-aware mutation
                if random.random() < 0.3:
                    self._balance_aware_mutation(parent)
                
                next_gen.append(parent)
            population = next_gen

        return best_sol or [], best_fit, best_det, history

    def auto_optimize(
        self,
        runs: int = 10,
        population_size: int = 180,
        generations: int = 800,
        stagnation_window: int = 150,
    ) -> Dict[str, Any]:
        """
        Automatic multi-run optimization system.
        
        Runs the genetic algorithm multiple times and tracks the best solution
        across all runs. Detects early convergence using fitness stagnation.
        
        Args:
            runs: Number of independent GA runs to execute
            population_size: Population size for each run
            generations: Maximum generations per run
            stagnation_window: Number of generations to check for stagnation
            
        Returns:
            Dictionary containing:
                - best_sequence: Best stacking sequence found across all runs
                - best_fitness: Fitness score of the best sequence
                - penalties: Penalty details for the best sequence
                - history: Combined history from all runs (best fitness per generation)
        """
        # Track the best solution across all runs
        global_best_sequence = None  # type: Optional[List[int]]
        global_best_fitness = -1.0
        global_best_penalties: Dict[str, float] = {}
        all_histories: List[List[float]] = []
        
        print(f"Starting auto-optimization: {runs} runs, pop={population_size}, gen={generations}")
        
        # Run the genetic algorithm multiple times
        for run_num in range(1, runs + 1):
            print(f"Run {run_num}/{runs}...")
            
            # Run a single GA execution
            sequence, fitness, penalties, history = self.run_genetic_algorithm(
                population_size=population_size,
                generations=generations
            )
            
            # Track history for this run
            all_histories.append(history)
            
            # Check for early convergence using fitness stagnation
            if len(history) >= stagnation_window:
                # Get the last N generations
                recent_fitness = history[-stagnation_window:]
                max_recent = max(recent_fitness)
                min_recent = min(recent_fitness)
                fitness_range = max_recent - min_recent
                
                # If fitness has stagnated (range < 0.01), print convergence message
                if fitness_range < 0.01:
                    print(f"  Run {run_num}: Converged early (fitness range: {fitness_range:.6f})")
            
            # Update global best if this run found a better solution
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_sequence = sequence[:]
                global_best_penalties = penalties.copy()
                print(f"  Run {run_num}: New best fitness = {fitness:.2f}")
        
        # Combine histories: take the best fitness at each generation across all runs
        combined_history: List[float] = []
        max_gen_length = max(len(h) for h in all_histories) if all_histories else 0
        
        for gen_idx in range(max_gen_length):
            # Get the best fitness at this generation across all runs
            gen_best = -1.0
            for history in all_histories:
                if gen_idx < len(history):
                    gen_best = max(gen_best, history[gen_idx])
            combined_history.append(gen_best)
        
        print(f"Auto-optimization complete. Best fitness: {global_best_fitness:.2f}")
        
        return {
            "best_sequence": global_best_sequence or [],
            "best_fitness": round(global_best_fitness, 2),
            "penalties": global_best_penalties,
            "history": combined_history,
        }


class DropOffOptimizer:
    """
    Tapering optimizer for ply drop-off.
    """

    def __init__(self, master_sequence: List[int], base_optimizer: LaminateOptimizer):
        self.master_sequence = master_sequence
        self.base_opt = base_optimizer
        self.total_plies = len(master_sequence)

    def optimize_drop(self, target_ply: int) -> Tuple[List[int], float, List[int]]:
        """
        Drop-off optimization with odd/even ply support.
        
        Supports:
        - Even → Even: Normal symmetric drop (pairs)
        - Odd → Odd: Symmetric drop (pairs), middle ply preserved
        - Odd → Even: Drop middle ply + symmetric pairs
        - Even → Odd: Break one pair - keep one as middle, drop its mirror
        """
        remove_cnt = self.total_plies - target_ply
        if remove_cnt <= 0:
            return self.master_sequence, 0.0, []
        
        master_is_odd = self.total_plies % 2 == 1
        target_is_odd = target_ply % 2 == 1
        half_len = self.total_plies // 2
        middle_idx = half_len if master_is_odd else None  # Ortadaki ply'ın index'i
        
        # Drop stratejisini belirle
        drop_middle = False
        break_pair_for_middle = False  # Çift → Tek için: bir çifti kır
        break_pair_idx = None  # Kırılacak çiftin sol yarıdaki pozisyonu
        
        if master_is_odd and not target_is_odd:
            # Tek → Çift: Ortadaki ply'ı da drop et
            drop_middle = True
            pairs_to_remove = (remove_cnt - 1) // 2
        elif not master_is_odd and target_is_odd:
            # Çift → Tek: Bir çifti kır - soldaki ortaya geçer, sağdaki drop edilir
            break_pair_for_middle = True
            # remove_cnt tek olmalı (örn: 36→35 = 1, 36→33 = 3)
            # Bir ply ortaya geçecek, geri kalan çiftler halinde drop edilecek
            pairs_to_remove = (remove_cnt - 1) // 2  # Örn: 1→0, 3→1, 5→2
        else:
            # Tek → Tek veya Çift → Çift: Normal çift drop
            if remove_cnt % 2 != 0:
                remove_cnt += 1  # Çift sayıya yuvarla
            pairs_to_remove = remove_cnt // 2
        
        # External plies koruması: ilk 2 katmanı koru (pozisyon 0 ve 1)
        # Rule 4'e göre ilk 2 ve son 2 katman korunmalı
        search_indices = list(range(2, half_len))  # Pozisyon 0 ve 1 hariç

        best_candidate = None
        best_key = None
        best_dropped = []

        attempts = self.base_opt.DROP_OFF_ATTEMPTS
        for _ in range(attempts):
            left_drops = []
            
            # Çift → Tek: Bir çifti kırmak için pozisyon seç
            if break_pair_for_middle:
                if len(search_indices) == 0:
                    continue
                # Ortaya yakın bir pozisyon seç (sol yarının sonlarından)
                # Bu pozisyondaki ply ortaya geçecek, mirror'ı drop edilecek
                break_pair_idx = random.choice(search_indices)
            
            # Normal çift drop pozisyonları seç
            if pairs_to_remove > 0 and len(search_indices) > 0:
                # break_pair_idx zaten kullanıldıysa onu hariç tut
                available_indices = [i for i in search_indices if i != break_pair_idx]
                sample_size = min(pairs_to_remove, len(available_indices))
                if sample_size > 0:
                    left_drops = random.sample(available_indices, sample_size)
                    left_drops.sort()

            # ✅ 1. NO GROUPING CHECK - Ardışık drop pozisyonları yasak
            # Drop pozisyonları birbirine çok yakın olmamalı (gruplama önleme)
            if len(left_drops) > 1:
                has_consecutive = any(left_drops[i+1] - left_drops[i] == 1 for i in range(len(left_drops)-1))
                if has_consecutive:
                    continue  # Grouped drops = reddet

            # ✅ 2. UNIFORM DISTRIBUTION CHECK - Drop'lar düzgün dağıtılmış olmalı
            spacing_std = 0.0  # Default değer
            if len(left_drops) > 2:
                spacings = [left_drops[i+1] - left_drops[i] for i in range(len(left_drops)-1)]
                spacing_mean = np.mean(spacings)
                spacing_std = np.std(spacings)
                # Çok yüksek standart sapma = kötü dağılım (AVOID örneği gibi)
                if spacing_std > spacing_mean * 0.7:  # Çok düzensiz dağılım
                    continue

            all_drops = []
            for idx in left_drops:
                all_drops.append(idx)
                all_drops.append(self.total_plies - 1 - idx)
            
            # Ortadaki ply'ı drop et (eğer gerekiyorsa - Tek → Çift)
            if drop_middle and middle_idx is not None:
                all_drops.append(middle_idx)
            
            # Çift → Tek: Bir çifti kır - sadece sağ yarıdaki mirror'ı drop et
            # Sol yarıdaki ply otomatik olarak yeni ortada kalır
            if break_pair_for_middle and break_pair_idx is not None:
                mirror_idx = self.total_plies - 1 - break_pair_idx
                all_drops.append(mirror_idx)
            
            all_drops.sort()

            temp_seq = [ang for i, ang in enumerate(self.master_sequence) if i not in all_drops]

            # ✅ 3. MULTI-ANGLE CHECK - Sadece bir açıdan drop olmasın (0° dahil tüm açılar)
            dropped_angles_left = [self.master_sequence[idx] for idx in left_drops]
            if drop_middle and middle_idx is not None:
                dropped_angles_left.append(self.master_sequence[middle_idx])
            if break_pair_for_middle and break_pair_idx is not None:
                # Kırılan çiftin mirror'ını da ekle (sağ yarıdaki drop edilen)
                mirror_idx = self.total_plies - 1 - break_pair_idx
                dropped_angles_left.append(self.master_sequence[mirror_idx])
            unique_angles_dropped = set(dropped_angles_left)
            
            # Eğer sadece bir açıdan drop varsa ve toplam drop sayısı 2'den fazlaysa, reddet
            # Bu, 0°, 90°, 45°, -45° tüm açılar için geçerli
            # Özellikle 0°'dan da drop yapılabilmeli, ama tek başına olmamalı
            if len(unique_angles_dropped) == 1 and len(dropped_angles_left) > 2:
                continue  # Sadece bir açıdan drop yapılmış = reddet
            
            # 0° açısından drop yapılmış mı kontrol et (isteğe bağlı, bilgi amaçlı)
            # Kod zaten 0°'dan drop yapılmasına izin veriyor, sadece tek başına olmamalı

            # ✅ 4. BALANCE CHECK (45°/-45° alternasyon + tüm açılar için denge)
            # Drop edilen açıların dağılımı dengeli olmalı
            count_45 = dropped_angles_left.count(45)
            count_minus45 = dropped_angles_left.count(-45)
            count_0 = dropped_angles_left.count(0)
            count_90 = dropped_angles_left.count(90)

            # 90°'dan aşırı drop yapılmasını engelle (en fazla 3 çift = 6 ply)
            if count_90 > 3:
                continue

            # 45°/-45° düşüşünü teşvik et: 4+ drop varsa en az bir 45° veya -45° olmalı
            total_drops = len(dropped_angles_left)
            if total_drops >= 4 and count_45 == 0 and count_minus45 == 0:
                continue
            
            # 45°/-45° dengesi kontrolü
            if count_45 > 0 or count_minus45 > 0:
                # Eğer her ikisi de varsa, sayıları yakın olmalı
                if count_45 > 0 and count_minus45 > 0:
                    if abs(count_45 - count_minus45) > 2:  # Çok dengesiz
                        continue
                # Eğer sadece biri varsa ve sayı 2'den fazlaysa, bu da dengesizlik
                elif (count_45 > 2 and count_minus45 == 0) or (count_minus45 > 2 and count_45 == 0):
                    continue

            total_score, details = self.base_opt.calculate_fitness(temp_seq)

            # 🚫 HARD FAIL (Hard constraints ihlali)
            if total_score <= 0:
                continue

            rules = details["rules"]

            # ✅ 5. RULE 6 (GROUPING) ÖZEL KONTROL - Drop sonrası grouping kontrolü
            # 3'lü veya daha fazla grouping varsa reddet
            groups_of_3 = self.base_opt._find_groups_of_size(temp_seq, 3)
            groups_of_4 = self.base_opt._find_groups_of_size(temp_seq, 4)
            groups_of_5 = self.base_opt._find_groups_of_size(temp_seq, 5)
            groups_of_4_or_more = groups_of_4 + groups_of_5  # 4 veya daha fazla
            
            # 4 veya daha fazla grouping varsa kesinlikle reddet
            if groups_of_4_or_more > 0:
                continue
            
            # 3'lü grouping sayısı fazla ise (3'ten fazla) reddet
            if groups_of_3 > 3:
                continue

            # ✅ 6. TÜM KURALLAR (R1-R8) MİNİMUM SKOR KONTROLÜ
            # Drop-off yapınca kuralların dışına çıkmamalı - minimum skorları koru
            min_scores = {
                "R1": 0.85,  # Symmetry - %85 minimum
                "R2": 0.80,  # Balance - %80 minimum
                "R3": 0.80,  # Percentage - %80 minimum
                "R4": 0.75,  # External plies - %75 minimum
                "R5": 0.70,  # Distribution - %70 minimum
                "R6": 0.75,  # Grouping - %75 minimum (önemli!)
                "R7": 0.75,  # Buckling - %75 minimum
                "R8": 0.85,  # Lateral bending - %85 minimum
            }
            
            # Her kural için minimum skor kontrolü
            rule_violations = 0
            for rule_name, min_ratio in min_scores.items():
                if rule_name in rules:
                    rule_weight = rules[rule_name]["weight"]
                    rule_score = rules[rule_name]["score"]
                    rule_ratio = rule_score / rule_weight if rule_weight > 0 else 0
                    
                    if rule_ratio < min_ratio:
                        rule_violations += 1
            
            # Çok fazla kural ihlali varsa reddet (2'den fazla kural %75'in altındaysa)
            if rule_violations > 2:
                continue

            # ✅ 7. IMPROVED SELECTION KEY (lexicographic) - Tüm kuralları dikkate al
            # Uniform distribution score (düşük std = iyi)
            dist_score = spacing_std  # Zaten yukarıda hesaplandı
            
            # Angle diversity score (daha fazla farklı açı = iyi)
            angle_diversity = len(unique_angles_dropped)
            
            # Balance score (45°/-45° dengesi)
            balance_score = abs(count_45 - count_minus45) if (count_45 > 0 or count_minus45 > 0) else 0
            
            # Rule 6 (Grouping) penalty - düşük olmalı
            r6_penalty = rules.get("R6", {}).get("penalty", 0)
            
            # Tüm kuralların toplam penalty'si (düşük = iyi)
            total_penalty = sum(r.get("penalty", 0) for r in rules.values())
            
            # 0° drop bonusu - 0°'dan da drop yapıldıysa bonus ver (daha çeşitli drop için)
            # Ancak tek başına 0° olmamalı (zaten yukarıda kontrol edildi)
            has_0_drop = 1 if count_0 > 0 else 0

            # 90° drop penalty: çok sayıda 90° drop'u ittir
            ninety_drop_penalty = count_90 * 0.5

            # 45°/-45° drop bonusu: bu açılardan drop varsa ödüllendir
            has_45_drop_bonus = -1 if (count_45 > 0 or count_minus45 > 0) else 0
            
            key = (
                rule_violations,  # Primary: Kural ihlali sayısı (düşük = iyi, 0 = hiç ihlal yok)
                groups_of_3,  # Secondary: 3'lü grup sayısı (düşük = iyi)
                groups_of_4_or_more,  # Tertiary: 4+ grup sayısı (düşük = iyi, 0 olmalı)
                r6_penalty,  # Quaternary: Rule 6 grouping penalty (düşük = iyi)
                ninety_drop_penalty,  # 90° drop penalty (düşük = iyi)
                rules["R1"]["penalty"] + rules["R8"]["penalty"],  # Quinary: R1 + R8 penalty
                dist_score,  # Senary: Uniform distribution (düşük = iyi)
                balance_score,  # Senaryedi: Balance (düşük = iyi)
                -angle_diversity,  # Sekizinci: Angle diversity (yüksek = iyi, negatif çünkü min istiyoruz)
                has_45_drop_bonus,  # 45°/-45° drop bonusu (negatif = ödül)
                -has_0_drop,  # Dokuzuncu: 0° drop bonusu (0° varsa -1, yoksa 0, negatif çünkü min istiyoruz, daha yüksek priority için)
                total_penalty,  # Onuncu: Toplam penalty
                -total_score  # On birinci: Total fitness score (yüksek = iyi)
            )

            if best_key is None or key < best_key:
                best_key = key
                best_candidate = temp_seq
                best_dropped = all_drops

        if best_candidate is None:
            return self.master_sequence, 0.0, []

        return best_candidate, best_key[10] * -1, best_dropped  # Total score'u döndür (11. eleman)

    def optimize_drop_with_angle_targets(
        self, target_ply_counts: Dict[int, int]
    ) -> Tuple[List[int], float, Dict[int, List[int]]]:
        """
        Master sequence'den spesifik açı sayılarına göre drop yapar.
        
        Args:
            target_ply_counts: Hedef açı sayıları (örn: {0: 12, 90: 14, 45: 16, -45: 14})
        
        Returns:
            (new_sequence, fitness_score, dropped_indices_by_angle)
        """
        from collections import Counter
        
        # 1. Validation: Target counts kontrolü
        current_counts = dict(Counter(self.master_sequence))
        
        for angle, target_count in target_ply_counts.items():
            current = current_counts.get(angle, 0)
            if target_count > current:
                raise ValueError(
                    f"Angle {angle}°: hedef {target_count} ama mevcut sadece {current} katman var"
                )
            if target_count < 0:
                raise ValueError(f"Angle {angle}°: hedef sayı negatif olamaz")
        
        # 2. Her açıdan kaç ply düşeceğini hesapla
        drops_needed = {}
        for angle, target_count in target_ply_counts.items():
            current = current_counts.get(angle, 0)
            if current > target_count:
                drops_needed[angle] = current - target_count
        
        # Toplam düşürülecek ply sayısı
        total_drops = sum(drops_needed.values())
        
        if total_drops == 0:
            # Hiç drop gerekmiyorsa master sequence'i döndür
            score, _ = self.base_opt.calculate_fitness(self.master_sequence)
            return self.master_sequence[:], score, {}
        
        # Original drops'u sakla (adjusted target hesabı için)
        original_drops_needed = drops_needed.copy()
        
        # 3. Her açı için drop edilebilir pozisyonları bul (sol yarıdan)
        n = len(self.master_sequence)
        half = n // 2
        master_is_odd = n % 2 == 1
        middle_idx = half if master_is_odd else None
        middle_angle = self.master_sequence[middle_idx] if middle_idx is not None else None
        
        # Tek/çift durumu kontrolü
        target_total = sum(target_ply_counts.values())
        target_is_odd = target_total % 2 == 1
        
        # Ortadaki ply drop edilecek mi? / Bir çift kırılacak mı?
        drop_middle = False
        break_pair_for_middle = False
        break_pair_angle = None  # Çift kırılacak açı
        
        if master_is_odd and not target_is_odd:
            # Tek → Çift: Ortadaki ply'ı drop et
            drop_middle = True
            if middle_angle in drops_needed:
                drops_needed[middle_angle] -= 1
                if drops_needed[middle_angle] == 0:
                    del drops_needed[middle_angle]
        elif not master_is_odd and target_is_odd:
            # Çift → Tek: Bir çifti kır - bir ply ortaya geçecek
            break_pair_for_middle = True
            # Tek sayıda drop gereken açıyı bul
            for angle, count in drops_needed.items():
                if count % 2 == 1:
                    break_pair_angle = angle
                    drops_needed[angle] -= 1  # Çift yap (bir tanesi ortaya geçecek)
                    if drops_needed[angle] == 0:
                        del drops_needed[angle]
                    break
            # Eğer hiçbiri tek değilse, herhangi birinden kır
            if break_pair_angle is None and drops_needed:
                break_pair_angle = list(drops_needed.keys())[0]
                # Bu açıdan 1 eksik drop yap çünkü 1 tanesi ortaya geçecek
                # Ama eğer drops_needed[angle] = 0 ise sadece ortaya 1 ply geçecek
        
        # Her açının drop sayısı çift olmalı (simetrik drop için)
        for angle in list(drops_needed.keys()):
            if drops_needed[angle] % 2 != 0:
                # Tek sayıda drop varsa, ortadaki ply bu açıdansa onu kullan
                if master_is_odd and middle_angle == angle and not drop_middle:
                    drop_middle = True
                    drops_needed[angle] -= 1
                    if drops_needed[angle] == 0:
                        del drops_needed[angle]
                else:
                    # Çift yap (bir fazla drop)
                    drops_needed[angle] += 1
        
        angle_positions_left = {}  # Her açının sol yarıdaki pozisyonları
        all_angles_to_check = set(drops_needed.keys())
        if break_pair_angle:
            all_angles_to_check.add(break_pair_angle)
        
        for angle in all_angles_to_check:
            positions = [i for i in range(half) if self.master_sequence[i] == angle]
            # External plies koruması: ilk 2 katmanı koru (pozisyon 0 ve 1)
            # Rule 4'e göre ilk 2 ve son 2 katman korunmalı
            positions = [p for p in positions if p > 1]  # Pozisyon 0 ve 1 hariç
            angle_positions_left[angle] = positions
        
        # 4. En iyi drop kombinasyonunu bul
        best_candidate = None
        best_score = -1
        best_dropped_by_angle = {}
        
        attempts = self.base_opt.ANGLE_TARGET_DROP_ATTEMPTS
        for _ in range(attempts):
            # Her açı için random drop pozisyonları seç (sol yarıdan)
            left_drops_by_angle = {}
            valid = True
            
            for angle, drop_count in drops_needed.items():
                pairs_needed = drop_count // 2  # Simetrik droplar
                available = angle_positions_left.get(angle, [])
                
                if len(available) < pairs_needed:
                    valid = False
                    break
                
                selected = random.sample(available, pairs_needed)
                left_drops_by_angle[angle] = sorted(selected)
            
            if not valid:
                continue
            
            # Tüm drop pozisyonlarını birleştir
            all_left_drops = []
            for positions in left_drops_by_angle.values():
                all_left_drops.extend(positions)
            all_left_drops.sort()
            
            # Ardışık drop kontrolü
            if len(all_left_drops) > 1:
                has_consecutive = any(
                    all_left_drops[i+1] - all_left_drops[i] == 1 
                    for i in range(len(all_left_drops)-1)
                )
                if has_consecutive:
                    continue
            
            # Simetrik pozisyonları ekle (sağ yarıdan)
            all_drops = []
            dropped_by_angle = {angle: [] for angle in drops_needed.keys()}
            
            for angle, left_positions in left_drops_by_angle.items():
                for idx in left_positions:
                    all_drops.append(idx)
                    mirror_idx = n - 1 - idx
                    all_drops.append(mirror_idx)
                    dropped_by_angle[angle].extend([idx, mirror_idx])
            
            # Ortadaki ply'ı drop et (eğer gerekiyorsa - Tek → Çift)
            if drop_middle and middle_idx is not None:
                all_drops.append(middle_idx)
                if middle_angle not in dropped_by_angle:
                    dropped_by_angle[middle_angle] = []
                dropped_by_angle[middle_angle].append(middle_idx)
            
            # Çift → Tek: Bir çifti kır - sadece mirror'ı drop et
            break_pair_idx = None
            if break_pair_for_middle and break_pair_angle is not None:
                available_for_break = angle_positions_left.get(break_pair_angle, [])
                # left_drops_by_angle'da kullanılmamış bir pozisyon seç
                used_positions = left_drops_by_angle.get(break_pair_angle, [])
                available_for_break = [p for p in available_for_break if p not in used_positions]
                
                if available_for_break:
                    break_pair_idx = random.choice(available_for_break)
                    mirror_idx = n - 1 - break_pair_idx
                    all_drops.append(mirror_idx)
                    if break_pair_angle not in dropped_by_angle:
                        dropped_by_angle[break_pair_angle] = []
                    dropped_by_angle[break_pair_angle].append(mirror_idx)
            
            all_drops.sort()
            
            # Yeni sequence oluştur
            temp_seq = [
                ang for i, ang in enumerate(self.master_sequence) 
                if i not in all_drops
            ]
            
            # Fitness hesapla
            score, details = self.base_opt.calculate_fitness(temp_seq)
            
            # Hard constraint ihlali varsa atla
            if score <= 0:
                continue
            
            # Hedef açı sayılarına ulaşıldı mı kontrol et
            # Not: Simetri için drop sayıları ayarlanmış olabilir, bu yüzden 
            # orijinal hedeften biraz sapma kabul edilebilir
            temp_counts = Counter(temp_seq)
            
            # Adjusted (düzeltilmiş) hedefleri hesapla
            adjusted_targets = {}
            for angle, orig_target in target_ply_counts.items():
                current = current_counts.get(angle, 0)
                orig_drop = original_drops_needed.get(angle, 0)
                actual_drop = 0
                
                # Bu açıdan gerçekte kaç drop yapıldı?
                if angle in drops_needed:
                    actual_drop = drops_needed[angle]
                
                # Ortadaki ply drop edildi mi?
                if drop_middle and middle_angle == angle:
                    actual_drop += 1
                
                # Çift kırma durumunda
                if break_pair_for_middle and break_pair_angle == angle:
                    actual_drop += 1
                
                adjusted_targets[angle] = current - actual_drop
            
            # Tolerans ile kontrol - her açı için ±1 sapma kabul et (simetri ayarlaması için)
            matches_target = True
            for angle, orig_target in target_ply_counts.items():
                actual_count = temp_counts.get(angle, 0)
                # Orijinal hedefe yakın mı? (±1 tolerans)
                if abs(actual_count - orig_target) > 1:
                    matches_target = False
                    break
            
            if not matches_target:
                continue
            
            # En iyi skoru güncelle
            if score > best_score:
                best_score = score
                best_candidate = temp_seq
                best_dropped_by_angle = {
                    angle: sorted(positions) 
                    for angle, positions in dropped_by_angle.items()
                }
        
        if best_candidate is None:
            # Hiç geçerli çözüm bulunamadıysa master sequence'i döndür
            print("UYARI: Hedef açı sayılarına uygun drop kombinasyonu bulunamadı")
            score, _ = self.base_opt.calculate_fitness(self.master_sequence)
            return self.master_sequence[:], score, {}
        
        return best_candidate, best_score, best_dropped_by_angle


# -----------------------------------------------------------------------------
# Zone Management System
# -----------------------------------------------------------------------------

class Zone:
    """Zone (katman) temsil eden sınıf"""
    def __init__(self, zone_id: int, name: str, sequence: List[int], ply_count: int):
        self.zone_id = zone_id
        self.name = name
        self.sequence = sequence
        self.ply_count = ply_count
        self.fitness_score = 0.0
        self.source_zones = []  # Bu zone'u oluşturan kaynak zone'lar
        self.transition_type = "drop_off"  # "drop_off" veya "merge"
    
    def to_dict(self):
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "sequence": self.sequence,
            "ply_count": self.ply_count,
            "fitness_score": self.fitness_score,
            "source_zones": self.source_zones,
            "transition_type": self.transition_type
        }

class ZoneManager:
    """Zone'ları ve geçişleri yöneten sınıf"""
    def __init__(self):
        self.zones: Dict[int, Zone] = {}
        self.transitions: List[Dict] = []  # Geçiş bilgileri
        self.next_zone_id = 1
    
    def create_zone_from_dropoff(self, source_zone_id: int, target_ply: int, 
                                  optimizer: 'LaminateOptimizer', 
                                  drop_optimizer: 'DropOffOptimizer') -> Zone:
        """Kaynak zone'dan drop-off yaparak yeni zone oluştur"""
        source_zone = self.zones[source_zone_id]
        drop_optimizer.master_sequence = source_zone.sequence
        drop_optimizer.total_plies = source_zone.ply_count
        
        new_sequence, score, dropped = drop_optimizer.optimize_drop(target_ply)
        
        zone_id = self.next_zone_id
        self.next_zone_id += 1
        
        new_zone = Zone(
            zone_id=zone_id,
            name=f"Zone {zone_id}",
            sequence=new_sequence,
            ply_count=target_ply
        )
        new_zone.fitness_score = score
        new_zone.source_zones = [source_zone_id]
        new_zone.transition_type = "drop_off"
        
        self.zones[zone_id] = new_zone
        
        # Geçiş bilgisini kaydet
        self.transitions.append({
            "from": source_zone_id,
            "to": zone_id,
            "type": "drop_off",
            "dropped_indices": dropped,
            "target_ply": target_ply
        })
        
        return new_zone
    
    def create_zone_from_merge(self, source_zone_ids: List[int], target_ply: int,
                               optimizer: 'LaminateOptimizer') -> Zone:
        """Birden fazla zone'u birleştirerek yeni zone oluştur"""
        # Kaynak zone'ların sequence'lerini birleştir
        merged_sequences = []
        for zone_id in source_zone_ids:
            if zone_id in self.zones:
                merged_sequences.append(self.zones[zone_id].sequence)
        
        if not merged_sequences:
            raise ValueError("Geçerli kaynak zone bulunamadı")
        
        # Birleştirme mantığı: En uzun sequence'i al, diğerlerini ona göre optimize et
        # Veya tüm sequence'leri birleştirip optimize et
        # Şimdilik en uzun olanı alıyoruz
        longest_seq = max(merged_sequences, key=len)
        
        # Eğer target_ply belirtilmişse ve uzunluktan küçükse, drop-off yap
        if target_ply is not None and target_ply < len(longest_seq):
            drop_optimizer = DropOffOptimizer(longest_seq, optimizer)
            new_sequence, score, dropped = drop_optimizer.optimize_drop(target_ply)
        else:
            # Target ply belirtilmemiş veya daha büyükse, en uzun sequence'i kullan
            new_sequence = longest_seq
            score, _ = optimizer.calculate_fitness(new_sequence)
        
        zone_id = self.next_zone_id
        self.next_zone_id += 1
        
        new_zone = Zone(
            zone_id=zone_id,
            name=f"Merge Zone {zone_id}",
            sequence=new_sequence,
            ply_count=len(new_sequence)
        )
        new_zone.fitness_score = score
        new_zone.source_zones = source_zone_ids
        new_zone.transition_type = "merge"
        
        self.zones[zone_id] = new_zone
        
        # Geçiş bilgisini kaydet
        self.transitions.append({
            "from": source_zone_ids,
            "to": zone_id,
            "type": "merge",
            "target_ply": target_ply
        })
        
        return new_zone
    
    def create_zone_from_angle_dropoff(self, source_zone_id: int, target_ply_counts: Dict[int, int],
                                       optimizer: 'LaminateOptimizer',
                                       drop_optimizer: 'DropOffOptimizer') -> Zone:
        """Kaynak zone'dan açıya özel drop-off yaparak yeni zone oluştur"""
        source_zone = self.zones[source_zone_id]
        drop_optimizer.master_sequence = source_zone.sequence
        drop_optimizer.total_plies = source_zone.ply_count
        
        new_sequence, score, dropped_by_angle = drop_optimizer.optimize_drop_with_angle_targets(target_ply_counts)
        
        zone_id = self.next_zone_id
        self.next_zone_id += 1
        
        new_zone = Zone(
            zone_id=zone_id,
            name=f"Zone {zone_id}",
            sequence=new_sequence,
            ply_count=len(new_sequence)
        )
        new_zone.fitness_score = score
        new_zone.source_zones = [source_zone_id]
        new_zone.transition_type = "angle_drop_off"
        
        self.zones[zone_id] = new_zone
        
        # Geçiş bilgisini kaydet
        self.transitions.append({
            "from": source_zone_id,
            "to": zone_id,
            "type": "angle_drop_off",
            "target_ply_counts": target_ply_counts,
            "dropped_by_angle": dropped_by_angle
        })
        
        return new_zone
    
    def get_zone(self, zone_id: int) -> Optional[Zone]:
        return self.zones.get(zone_id)
    
    def get_all_zones(self) -> List[Dict]:
        return [zone.to_dict() for zone in self.zones.values()]
    
    def get_transitions(self) -> List[Dict]:
        return self.transitions
    
    def add_root_zone(self, sequence: List[int], optimizer: 'LaminateOptimizer'):
        """Root zone'u ekle (master sequence)"""
        score, _ = optimizer.calculate_fitness(sequence)
        root_zone = Zone(
            zone_id=0,
            name="Root",
            sequence=sequence,
            ply_count=len(sequence)
        )
        root_zone.fitness_score = score
        self.zones[0] = root_zone

# Global zone manager (session bazlı olabilir, şimdilik global)
zone_managers: Dict[str, ZoneManager] = {}


# -----------------------------------------------------------------------------
# Flask App
# -----------------------------------------------------------------------------


def create_app() -> Flask:
    app = Flask(__name__, static_folder=".", static_url_path="")

    def check_symmetry_compatibility(ply_counts: Dict[int, int]) -> Dict:
        """
        Simetri uyumluluğunu kontrol et.
        
        Returns:
            {
                'requires_user_choice': bool,
                'issues': list,
                'suggestions': list
            }
        """
        total = sum(ply_counts.values())
        is_odd_total = total % 2 == 1
        
        # Tek sayıda olan açıları bul
        odd_angles = [angle for angle, count in ply_counts.items() if count % 2 == 1]
        
        issues = []
        suggestions = []
        
        # Durum 1: Tek sayılı toplam + 2+ tek sayılı açı
        if is_odd_total and len(odd_angles) > 1:
            issues.append({
                'type': 'multiple_odd_angles',
                'angles': odd_angles,
                'total': total,
                'message': f'{len(odd_angles)} açı tek sayıda ({", ".join(map(str, odd_angles))}°). Sadece biri ortaya gidebilir.'
            })
            
            # Öneri: Her birini ortaya koyarak denemek
            for angle in odd_angles:
                adjusted = ply_counts.copy()
                adjusted[angle] = adjusted[angle] - 1  # Ortaya gider
                suggestions.append({
                    'type': 'set_middle',
                    'middle_angle': angle,
                    'adjusted_counts': adjusted,
                    'description': f'{angle}° ortaya koy, diğer tek sayılı açıları çift yap'
                })
        
        # Durum 2: Çift sayılı toplam + tek sayılı açılar
        if not is_odd_total and len(odd_angles) > 0:
            # ±45° dengesi kontrolü
            angle_45 = ply_counts.get(45, 0)
            angle_minus45 = ply_counts.get(-45, 0)
            
            if angle_45 % 2 == 1 and angle_minus45 % 2 == 1 and angle_45 == angle_minus45:
                # İkisi de tek ve eşit - özel durum
                issues.append({
                    'type': 'odd_45_balance',
                    'angles': [45, -45],
                    'total': total,
                    'message': f'45° ve -45° her ikisi de tek sayıda ({angle_45}). Eşit sayıda kalmalı (Rule 2: Balance).'
                })
                
                # Öneri 1: İkisini de +1 yap
                adjusted1 = ply_counts.copy()
                adjusted1[45] = angle_45 + 1
                adjusted1[-45] = angle_minus45 + 1
                # Toplam artıyor, diğer açılardan çıkar
                total_adjustment = 2
                for adj_angle in [0, 90]:
                    if adj_angle in adjusted1 and adjusted1[adj_angle] >= total_adjustment:
                        adjusted1[adj_angle] -= total_adjustment
                        suggestions.append({
                            'type': 'increase_45_pair',
                            'compensation_angle': adj_angle,
                            'compensation_amount': total_adjustment,
                            'adjusted_counts': adjusted1,
                            'description': f'45° ve -45° → {angle_45 + 1} (her ikisi +1), {adj_angle}° → {adjusted1[adj_angle]} ({adjusted1[adj_angle] + total_adjustment} - {total_adjustment})'
                        })
                        break
                
                # Öneri 2: İkisini de -1 yap
                adjusted2 = ply_counts.copy()
                adjusted2[45] = angle_45 - 1
                adjusted2[-45] = angle_minus45 - 1
                # Toplam azalıyor, diğer açılara ekle
                total_adjustment = 2
                for adj_angle in [0, 90]:
                    if adj_angle in adjusted2:
                        adjusted2[adj_angle] += total_adjustment
                        suggestions.append({
                            'type': 'decrease_45_pair',
                            'compensation_angle': adj_angle,
                            'compensation_amount': total_adjustment,
                            'adjusted_counts': adjusted2,
                            'description': f'45° ve -45° → {angle_45 - 1} (her ikisi -1), {adj_angle}° → {adjusted2[adj_angle]} ({adjusted2[adj_angle] - total_adjustment} + {total_adjustment})'
                        })
                        break
            else:
                # Genel durum: Tek sayılı açılar var
                issues.append({
                    'type': 'odd_angles_even_total',
                    'angles': odd_angles,
                    'total': total,
                    'message': f'Çift sayılı toplamda {len(odd_angles)} açı tek sayıda. Simetri için her açı çift sayıda olmalı.'
                })
                
                # Öneri: Her birini çift yap
                for angle in odd_angles:
                    adjusted = ply_counts.copy()
                    adjusted[angle] = adjusted[angle] + 1
                    suggestions.append({
                        'type': 'make_even',
                        'angle': angle,
                        'adjusted_counts': adjusted,
                        'description': f'{angle}°: {ply_counts[angle]} → {adjusted[angle]} (çift sayı yap)'
                    })
        
        return {
            'requires_user_choice': len(issues) > 0,
            'issues': issues,
            'suggestions': suggestions
        }

    @app.route("/")
    def index():
        return send_from_directory(".", "index.html")

    @app.post("/optimize")
    def optimize():
        payload = request.get_json(force=True, silent=True) or {}
        ply_counts = payload.get("ply_counts", {})
        # Default example if nothing provided
        ply_counts = {
            int(k): int(v)
            for k, v in ply_counts.items()
            if str(v).isdigit()
        } or {0: 18, 90: 18, 45: 18, -45: 18}

        population_size = int(payload.get("population_size", 120))
        generations = int(payload.get("generations", 600))
        min_drop = int(payload.get("min_drop", 48))
        drop_step = int(payload.get("drop_step", 8))

        # Simetri uyumluluğu kontrolü
        symmetry_check = check_symmetry_compatibility(ply_counts)
        
        # Eğer kullanıcı seçimi gerekiyorsa ve user_choice verilmemişse uyarı döndür
        user_choice = payload.get("symmetry_user_choice")
        if symmetry_check['requires_user_choice'] and not user_choice:
            return jsonify({
                'requires_symmetry_choice': True,
                'symmetry_info': symmetry_check,
                'message': 'Simetri uyarısı: Kullanıcı seçimi gerekiyor'
            }), 200
        
        # Eğer kullanıcı "mevcut sayılarla devam" seçtiyse, ply_counts değişmez (sadece penalty olur)
        # Eğer bir öneri seçtiyse, adjusted_counts kullan
        if user_choice:
            if user_choice.get('continue_with_current'):
                # Mevcut sayılarla devam - ply_counts değişmez
                pass
            elif 'adjusted_counts' in user_choice:
                ply_counts = user_choice['adjusted_counts']
        
        # Ply sayısına göre parametreleri otomatik ayarla (varsayılan değerler kullanılıyorsa)
        total_plies = sum(ply_counts.values())
        if population_size <= 120:
            # Otomatik popülasyon boyutu hesaplama
            population_size = max(120, min(400, int(total_plies * 2.0)))
        if generations <= 600:
            # Otomatik jenerasyon sayısı hesaplama
            generations = max(600, min(1500, int(total_plies * 10.0)))

        optimizer = LaminateOptimizer(ply_counts)
        start_time = time.time()
        # Hybrid optimization kullan
        master_seq, master_score, details, history = optimizer.run_hybrid_optimization()
        ga_elapsed = time.time() - start_time

        drop_targets = []
        temp = len(master_seq)
        while temp > min_drop:
            temp -= drop_step
            if temp > 0:
                drop_targets.append(temp)

        drop_opt = DropOffOptimizer(master_seq, optimizer)
        drop_results_list = []
        current_seq = master_seq
        for target in drop_targets:
            drop_opt.master_sequence = current_seq
            drop_opt.total_plies = len(current_seq)
            new_seq, sc, dropped_indices = drop_opt.optimize_drop(target)
            drop_results_list.append(
                {"target": target, "seq": new_seq, "score": sc, "dropped": dropped_indices}
            )
            current_seq = new_seq

        # details is now a dict with "total_score", "max_score", "rules"
        response = {
            "master_sequence": master_seq,
            "fitness_score": details.get("total_score", master_score),
            "max_score": details.get("max_score", 100),
            "penalties": details.get("rules", {}),
            "history": history,
            "drop_off_results": drop_results_list,
            "stats": {
                "plies": len(master_seq),
                "duration_seconds": round(ga_elapsed, 2),
                "population_size": population_size,
                "generations": generations,
            },
        }
        return jsonify(response)

    @app.post("/evaluate")
    def evaluate():
        """
        Verilen bir sequence'ın fitness skorunu hesaplar.
        Manuel düzenlenmiş sequence'ları test etmek için kullanılır.
        
        Request body (JSON):
            - sequence: List of ply angles (e.g., [45, 0, -45, 90, ...])
            - ply_counts: Dict of ply counts (optional, for validation)
        
        Returns:
            JSON response with fitness score and penalty details
        """
        payload = request.get_json(force=True, silent=True) or {}
        sequence = payload.get("sequence", [])
        ply_counts = payload.get("ply_counts", {})
        
        if not sequence:
            return jsonify({"error": "Sequence required"}), 400
        
        # Eğer ply_counts verilmemişse, sequence'dan çıkar
        if not ply_counts:
            from collections import Counter
            ply_counts = dict(Counter(sequence))
        
        # Sequence'ı integer list'e çevir
        try:
            sequence = [int(x) for x in sequence]
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid sequence format"}), 400
        
        optimizer = LaminateOptimizer(ply_counts)
        fitness_score, details = optimizer.calculate_fitness(sequence)
        
        return jsonify({
            "sequence": sequence,
            "fitness_score": fitness_score,
            "max_score": details.get("max_score", 100),
            "penalties": details.get("rules", {}),
            "valid": fitness_score > 0  # Hard rule ihlali varsa False
        })

    @app.post("/dropoff")
    def dropoff():
        """
        Drop-off optimization endpoint.
        Verilen master sequence'den belirtilen hedef ply sayısına düşürmek için
        kurallara uygun katmanları siler.
        
        Request body (JSON):
            - master_sequence: List of ply angles (e.g., [45, 0, -45, 90, ...])
            - target_ply: Hedef ply sayısı (e.g., 26)
            - ply_counts: Dict of ply counts (optional, sequence'dan çıkarılabilir)
        
        Returns:
            JSON response with:
                - sequence: Drop-off sonrası sequence
                - fitness_score: Fitness score
                - max_score: Max score (100)
                - penalties: Penalty details
                - dropped_indices: Silinen katmanların index'leri
                - target_ply: Hedef ply sayısı
        """
        payload = request.get_json(force=True, silent=True) or {}
        master_sequence = payload.get("master_sequence", [])
        target_ply = payload.get("target_ply")
        ply_counts = payload.get("ply_counts", {})
        
        if not master_sequence:
            return jsonify({"error": "master_sequence required"}), 400
        
        if target_ply is None:
            return jsonify({"error": "target_ply required"}), 400
        
        try:
            master_sequence = [int(x) for x in master_sequence]
            target_ply = int(target_ply)
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid format"}), 400
        
        # Ply counts'u hesapla (verilmemişse)
        if not ply_counts:
            from collections import Counter
            ply_counts = dict(Counter(master_sequence))
        else:
            ply_counts = {
                int(k): int(v)
                for k, v in ply_counts.items()
                if str(v).isdigit()
            }
        
        # Hedef ply sayısı kontrolü
        current_ply = len(master_sequence)
        if target_ply >= current_ply:
            return jsonify({"error": f"target_ply ({target_ply}) must be less than current ply count ({current_ply})"}), 400
        
        if target_ply <= 0:
            return jsonify({"error": "target_ply must be greater than 0"}), 400
        
        # Optimizer oluştur
        optimizer = LaminateOptimizer(ply_counts)
        
        # Drop-off optimizer oluştur ve çalıştır
        drop_opt = DropOffOptimizer(master_sequence, optimizer)
        new_seq, score, dropped_indices = drop_opt.optimize_drop(target_ply)
        
        # Fitness hesapla
        fitness_score, details = optimizer.calculate_fitness(new_seq)
        
        return jsonify({
            "sequence": new_seq,
            "fitness_score": fitness_score,
            "max_score": details.get("max_score", 100),
            "penalties": details.get("rules", {}),
            "dropped_indices": dropped_indices,
            "target_ply": target_ply,
            "original_ply": current_ply,
            "removed_count": len(dropped_indices)
        })
    
    @app.post("/dropoff_angle_targets")
    def dropoff_angle_targets():
        """
        Açıya özel drop-off optimization endpoint.
        
        Master sequence'den belirtilen açı sayılarına göre drop yapar.
        Örneğin: Master'da her açıdan 18 varken, hedefte 0°:12, 90°:14, 45°:16, -45°:14 olabilir.
        
        Request body (JSON):
            - master_sequence: List of ply angles (e.g., [45, 0, -45, 90, ...])
            - target_ply_counts: Dict of target angle counts (e.g., {0: 12, 90: 14, 45: 16, -45: 14})
            - ply_counts: Dict of current ply counts (optional, sequence'dan çıkarılabilir)
        
        Returns:
            JSON response with:
                - sequence: Drop-off sonrası sequence
                - fitness_score: Fitness score
                - max_score: Max score (100)
                - penalties: Penalty details
                - dropped_by_angle: Her açıdan düşen katmanların index'leri
                - target_ply_counts: Hedef açı sayıları
                - current_ply_counts: Master sequence'deki açı sayıları
                - total_removed: Toplam silinen katman sayısı
        """
        payload = request.get_json(force=True, silent=True) or {}
        master_sequence = payload.get("master_sequence", [])
        target_ply_counts = payload.get("target_ply_counts", {})
        ply_counts = payload.get("ply_counts", {})
        
        if not master_sequence:
            return jsonify({"error": "master_sequence required"}), 400
        
        if not target_ply_counts:
            return jsonify({"error": "target_ply_counts required"}), 400
        
        try:
            master_sequence = [int(x) for x in master_sequence]
            target_ply_counts = {
                int(k): int(v)
                for k, v in target_ply_counts.items()
            }
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid format"}), 400
        
        # Ply counts'u hesapla (verilmemişse)
        if not ply_counts:
            from collections import Counter
            ply_counts = dict(Counter(master_sequence))
        else:
            ply_counts = {
                int(k): int(v)
                for k, v in ply_counts.items()
                if str(v).isdigit()
            }
        
        # Master sequence'deki açı sayıları
        from collections import Counter
        current_ply_counts = dict(Counter(master_sequence))
        
        # Validation: Hedef sayılar mevcut sayıları aşmamalı
        for angle, target_count in target_ply_counts.items():
            current = current_ply_counts.get(angle, 0)
            if target_count > current:
                return jsonify({
                    "error": f"Angle {angle}°: hedef {target_count} ama mevcut sadece {current} katman var"
                }), 400
        
        # Optimizer oluştur
        optimizer = LaminateOptimizer(ply_counts)
        
        # Drop-off optimizer oluştur ve açıya özel drop yap
        drop_opt = DropOffOptimizer(master_sequence, optimizer)
        
        try:
            new_seq, score, dropped_by_angle = drop_opt.optimize_drop_with_angle_targets(
                target_ply_counts
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        
        # Fitness hesapla
        fitness_score, details = optimizer.calculate_fitness(new_seq)
        
        # Yeni sequence'deki açı sayıları
        new_ply_counts = dict(Counter(new_seq))
        
        # Toplam silinen katman sayısı
        total_removed = len(master_sequence) - len(new_seq)
        
        return jsonify({
            "sequence": new_seq,
            "fitness_score": fitness_score,
            "max_score": details.get("max_score", 100),
            "penalties": details.get("rules", {}),
            "dropped_by_angle": dropped_by_angle,
            "target_ply_counts": target_ply_counts,
            "current_ply_counts": current_ply_counts,
            "new_ply_counts": new_ply_counts,
            "total_removed": total_removed,
            "original_total": len(master_sequence),
            "new_total": len(new_seq)
        })
    
    @app.post("/auto_optimize")
    def auto_optimize():
        """
        Automatic multi-run optimization endpoint.
        
        Runs the genetic algorithm multiple times with convergence detection
        and returns the best solution found across all runs.
        
        Request body (JSON):
            - ply_counts: Dict of ply angles and counts (e.g., {0: 18, 90: 18, 45: 18, -45: 18})
            - runs: Number of independent GA runs (default: 10)
            - population_size: Population size per run (default: 180)
            - generations: Maximum generations per run (default: 800)
            - stagnation_window: Generations to check for stagnation (default: 150)
        
        Returns:
            JSON response with:
                - best_sequence: Best stacking sequence found
                - best_fitness: Fitness score of best sequence
                - penalties: Penalty details for best sequence
                - history: Combined fitness history across all runs
        """
        payload = request.get_json(force=True, silent=True) or {}
        ply_counts = payload.get("ply_counts", {})
        # Default example if nothing provided
        ply_counts = {
            int(k): int(v)
            for k, v in ply_counts.items()
            if str(v).isdigit()
        } or {0: 18, 90: 18, 45: 18, -45: 18}

        # Auto-optimization parameters with defaults
        runs = int(payload.get("runs", 10))
        population_size = int(payload.get("population_size", 180))
        generations = int(payload.get("generations", 800))
        stagnation_window = int(payload.get("stagnation_window", 150))

        # Ply sayısına göre parametreleri otomatik ayarla (varsayılan değerler kullanılıyorsa)
        total_plies = sum(ply_counts.values())
        if population_size <= 180:
            # Otomatik popülasyon boyutu hesaplama
            population_size = max(180, min(400, int(total_plies * 2.0)))
        if generations <= 800:
            # Otomatik jenerasyon sayısı hesaplama
            generations = max(800, min(1500, int(total_plies * 10.0)))

        # Create optimizer and run auto-optimization
        optimizer = LaminateOptimizer(ply_counts)
        start_time = time.time()
        
        result = optimizer.auto_optimize(
            runs=runs,
            population_size=population_size,
            generations=generations,
            stagnation_window=stagnation_window,
        )
        
        elapsed = time.time() - start_time

        # Get the detailed fitness information
        optimizer = LaminateOptimizer(ply_counts)
        _, fitness_details = optimizer.calculate_fitness(result["best_sequence"])
        
        # Build response with additional stats
        response = {
            "master_sequence": result["best_sequence"],
            "fitness_score": fitness_details.get("total_score", result["best_fitness"]),
            "max_score": fitness_details.get("max_score", 100),
            "penalties": fitness_details.get("rules", {}),
            "history": result["history"],
            "stats": {
                "plies": len(result["best_sequence"]),
                "duration_seconds": round(elapsed, 2),
                "runs": runs,
                "population_size": population_size,
                "generations": generations,
                "stagnation_window": stagnation_window,
            },
        }
        return jsonify(response)

    # -----------------------------------------------------------------------------
    # Zone Management Endpoints
    # -----------------------------------------------------------------------------
    
    @app.post("/zones/create_from_dropoff")
    def create_zone_from_dropoff():
        """
        Kaynak zone'dan drop-off yaparak yeni zone oluştur.
        
        Request body (JSON):
            - session_id: Session ID (opsiyonel, default: "default")
            - source_zone_id: Kaynak zone ID
            - target_ply: Hedef ply sayısı
            - ply_counts: Dict of ply counts (opsiyonel)
        
        Returns:
            JSON response with new zone information
        """
        payload = request.get_json(force=True, silent=True) or {}
        session_id = payload.get("session_id", "default")
        source_zone_id = payload.get("source_zone_id")
        target_ply = payload.get("target_ply")
        ply_counts = payload.get("ply_counts", {})
        
        if source_zone_id is None:
            return jsonify({"error": "source_zone_id required"}), 400
        if target_ply is None:
            return jsonify({"error": "target_ply required"}), 400
        
        # Zone manager'ı al veya oluştur
        if session_id not in zone_managers:
            return jsonify({"error": "Session not found. Create root zone first."}), 400
        
        zone_manager = zone_managers[session_id]
        source_zone = zone_manager.get_zone(source_zone_id)
        
        if not source_zone:
            return jsonify({"error": f"Source zone {source_zone_id} not found"}), 404
        
        # Ply counts'u hesapla
        if not ply_counts:
            from collections import Counter
            ply_counts = dict(Counter(source_zone.sequence))
        
        # Optimizer oluştur
        optimizer = LaminateOptimizer(ply_counts)
        drop_optimizer = DropOffOptimizer(source_zone.sequence, optimizer)
        
        # Yeni zone oluştur
        try:
            new_zone = zone_manager.create_zone_from_dropoff(
                source_zone_id, target_ply, optimizer, drop_optimizer
            )
            return jsonify({
                "success": True,
                "zone": new_zone.to_dict(),
                "transitions": zone_manager.get_transitions()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.post("/zones/create_from_angle_dropoff")
    def create_zone_from_angle_dropoff():
        """
        Kaynak zone'dan açıya özel drop-off yaparak yeni zone oluştur.
        
        Request body (JSON):
            - session_id: Session ID (opsiyonel, default: "default")
            - source_zone_id: Kaynak zone ID
            - target_ply_counts: Hedef açı sayıları (örn: {0: 12, 90: 14, 45: 16, -45: 14})
            - ply_counts: Dict of ply counts (opsiyonel)
        
        Returns:
            JSON response with new zone information
        """
        payload = request.get_json(force=True, silent=True) or {}
        session_id = payload.get("session_id", "default")
        source_zone_id = payload.get("source_zone_id")
        target_ply_counts = payload.get("target_ply_counts", {})
        ply_counts = payload.get("ply_counts", {})
        
        if source_zone_id is None:
            return jsonify({"error": "source_zone_id required"}), 400
        if not target_ply_counts:
            return jsonify({"error": "target_ply_counts required"}), 400
        
        # Zone manager'ı al veya oluştur
        if session_id not in zone_managers:
            return jsonify({"error": "Session not found. Create root zone first."}), 400
        
        zone_manager = zone_managers[session_id]
        source_zone = zone_manager.get_zone(source_zone_id)
        
        if not source_zone:
            return jsonify({"error": f"Source zone {source_zone_id} not found"}), 404
        
        # Ply counts'u hesapla
        if not ply_counts:
            from collections import Counter
            ply_counts = dict(Counter(source_zone.sequence))
        
        # Target ply counts'u integer'a çevir
        try:
            target_ply_counts = {
                int(k): int(v)
                for k, v in target_ply_counts.items()
            }
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid target_ply_counts format"}), 400
        
        # Optimizer oluştur
        optimizer = LaminateOptimizer(ply_counts)
        drop_optimizer = DropOffOptimizer(source_zone.sequence, optimizer)
        
        # Yeni zone oluştur
        try:
            new_zone = zone_manager.create_zone_from_angle_dropoff(
                source_zone_id, target_ply_counts, optimizer, drop_optimizer
            )
            return jsonify({
                "success": True,
                "zone": new_zone.to_dict(),
                "zones": zone_manager.get_all_zones(),
                "transitions": zone_manager.get_transitions()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.post("/zones/create_from_merge")
    def create_zone_from_merge():
        """
        Birden fazla zone'u birleştirerek yeni zone oluştur.
        
        Request body (JSON):
            - session_id: Session ID (opsiyonel, default: "default")
            - source_zone_ids: Kaynak zone ID'leri listesi
            - target_ply: Hedef ply sayısı (opsiyonel)
            - ply_counts: Dict of ply counts (opsiyonel)
        
        Returns:
            JSON response with new merged zone information
        """
        payload = request.get_json(force=True, silent=True) or {}
        session_id = payload.get("session_id", "default")
        source_zone_ids = payload.get("source_zone_ids", [])
        target_ply = payload.get("target_ply")
        ply_counts = payload.get("ply_counts", {})
        
        if not source_zone_ids:
            return jsonify({"error": "source_zone_ids required"}), 400
        
        # Zone manager'ı al veya oluştur
        if session_id not in zone_managers:
            return jsonify({"error": "Session not found. Create root zone first."}), 400
        
        zone_manager = zone_managers[session_id]
        
        # Kaynak zone'ları kontrol et
        for zone_id in source_zone_ids:
            if not zone_manager.get_zone(zone_id):
                return jsonify({"error": f"Source zone {zone_id} not found"}), 404
        
        # Ply counts'u hesapla (ilk kaynak zone'dan)
        if not ply_counts and source_zone_ids:
            from collections import Counter
            first_zone = zone_manager.get_zone(source_zone_ids[0])
            if first_zone:
                ply_counts = dict(Counter(first_zone.sequence))
        
        # Optimizer oluştur
        optimizer = LaminateOptimizer(ply_counts)
        
        # Yeni zone oluştur
        try:
            if target_ply is None:
                # Target ply belirtilmemişse, en uzun sequence'in uzunluğunu kullan
                max_ply = max([zone_manager.get_zone(zid).ply_count for zid in source_zone_ids])
                target_ply = max_ply
            
            new_zone = zone_manager.create_zone_from_merge(
                source_zone_ids, target_ply, optimizer
            )
            return jsonify({
                "success": True,
                "zone": new_zone.to_dict(),
                "zones": zone_manager.get_all_zones(),
                "transitions": zone_manager.get_transitions()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.post("/zones/add_from_dropoff")
    def add_zone_from_dropoff():
        """
        Eski drop-off sonucunu zone sistemine ekle.
        
        Request body (JSON):
            - session_id: Session ID (opsiyonel, default: "default")
            - source_zone_id: Kaynak zone ID
            - sequence: Yeni sequence
            - ply_count: Ply sayısı
            - fitness_score: Fitness score
            - dropped_indices: Çıkarılan index'ler
        
        Returns:
            JSON response with new zone information
        """
        payload = request.get_json(force=True, silent=True) or {}
        session_id = payload.get("session_id", "default")
        source_zone_id = payload.get("source_zone_id")
        sequence = payload.get("sequence", [])
        ply_count = payload.get("ply_count")
        fitness_score = payload.get("fitness_score", 0.0)
        dropped_indices = payload.get("dropped_indices", [])
        
        if not sequence:
            return jsonify({"error": "sequence required"}), 400
        if ply_count is None:
            ply_count = len(sequence)
        if source_zone_id is None:
            return jsonify({"error": "source_zone_id required"}), 400
        
        # Zone manager'ı al veya oluştur
        if session_id not in zone_managers:
            return jsonify({"error": "Session not found. Create root zone first."}), 400
        
        zone_manager = zone_managers[session_id]
        source_zone = zone_manager.get_zone(source_zone_id)
        
        if not source_zone:
            return jsonify({"error": f"Source zone {source_zone_id} not found"}), 404
        
        # Sequence'i integer list'e çevir
        try:
            sequence = [int(x) for x in sequence]
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid sequence format"}), 400
        
        # Yeni zone oluştur
        zone_id = zone_manager.next_zone_id
        zone_manager.next_zone_id += 1
        
        new_zone = Zone(
            zone_id=zone_id,
            name=f"Zone {zone_id}",
            sequence=sequence,
            ply_count=ply_count
        )
        new_zone.fitness_score = fitness_score
        new_zone.source_zones = [source_zone_id]
        new_zone.transition_type = "drop_off"
        
        zone_manager.zones[zone_id] = new_zone
        
        # Geçiş bilgisini kaydet
        zone_manager.transitions.append({
            "from": source_zone_id,
            "to": zone_id,
            "type": "drop_off",
            "target_ply": ply_count,
            "dropped_indices": dropped_indices
        })
        
        return jsonify({
            "success": True,
            "zone": new_zone.to_dict(),
            "zones": zone_manager.get_all_zones(),
            "transitions": zone_manager.get_transitions()
        })
    
    @app.post("/zones/init_root")
    def init_root_zone():
        """
        Root zone'u başlat (master sequence ile).
        
        Request body (JSON):
            - session_id: Session ID (opsiyonel, default: "default")
            - master_sequence: Master sequence listesi
            - ply_counts: Dict of ply counts (opsiyonel)
        
        Returns:
            JSON response with root zone information
        """
        payload = request.get_json(force=True, silent=True) or {}
        session_id = payload.get("session_id", "default")
        master_sequence = payload.get("master_sequence", [])
        ply_counts = payload.get("ply_counts", {})
        
        if not master_sequence:
            return jsonify({"error": "master_sequence required"}), 400
        
        try:
            master_sequence = [int(x) for x in master_sequence]
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid sequence format"}), 400
        
        # Ply counts'u hesapla
        if not ply_counts:
            from collections import Counter
            ply_counts = dict(Counter(master_sequence))
        
        # Zone manager oluştur
        zone_manager = ZoneManager()
        optimizer = LaminateOptimizer(ply_counts)
        zone_manager.add_root_zone(master_sequence, optimizer)
        zone_managers[session_id] = zone_manager
        
        root_zone = zone_manager.get_zone(0)
        return jsonify({
            "success": True,
            "zone": root_zone.to_dict(),
            "all_zones": zone_manager.get_all_zones(),
            "transitions": zone_manager.get_transitions()
        })
    
    @app.get("/zones/list")
    def list_zones():
        """
        Tüm zone'ları listele.
        
        Query params:
            - session_id: Session ID (opsiyonel, default: "default")
        
        Returns:
            JSON response with all zones and transitions
        """
        session_id = request.args.get("session_id", "default")
        
        if session_id not in zone_managers:
            return jsonify({
                "zones": [],
                "transitions": [],
                "message": "No zones found. Initialize root zone first."
            })
        
        zone_manager = zone_managers[session_id]
        return jsonify({
            "zones": zone_manager.get_all_zones(),
            "transitions": zone_manager.get_transitions()
        })
    
    @app.get("/zones/<int:zone_id>")
    def get_zone(zone_id):
        """
        Belirli bir zone'u getir.
        
        Query params:
            - session_id: Session ID (opsiyonel, default: "default")
        
        Returns:
            JSON response with zone information
        """
        session_id = request.args.get("session_id", "default")
        
        if session_id not in zone_managers:
            return jsonify({"error": "Session not found"}), 404
        
        zone_manager = zone_managers[session_id]
        zone = zone_manager.get_zone(zone_id)
        
        if not zone:
            return jsonify({"error": f"Zone {zone_id} not found"}), 404
        
        return jsonify({
            "zone": zone.to_dict(),
            "transitions": zone_manager.get_transitions()
        })

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

