import random
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np


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

    # Maximum cache size to prevent memory issues
    MAX_CACHE_SIZE = 50000

    def __init__(self, ply_counts: Dict[int, int]):
        self.ply_counts = ply_counts
        self.initial_pool = []  # type: List[int]
        for angle, count in ply_counts.items():
            self.initial_pool.extend([angle] * int(count))

        self.total_plies = len(self.initial_pool)
        
        # Fitness cache for memoization (sequence tuple -> (score, details))
        self._fitness_cache = {}  # type: Dict[tuple, Tuple[float, Dict]]
        self._cache_hits = 0
        self._cache_misses = 0

    def clear_cache(self) -> None:
        """Clear the fitness cache and reset statistics."""
        self._fitness_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            "cache_size": len(self._fitness_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
        }

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
        assert len(sequence) == total, "Sequence length mismatch: {} != {}".format(len(sequence), total)
        for angle in set(self.initial_pool):
            expected = self.initial_pool.count(angle)
            actual = sequence.count(angle)
            assert expected == actual, "Angle {} count mismatch: expected {}, got {}".format(angle, expected, actual)

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

    def _grouping_stats(self, sequence: List[int]) -> Dict[str, int]:
        """
        Grouping istatistikleri:
        - adjacent_pairs: yan yana aynı açı sayısı toplamı (her run için run_len-1)
        - group_runs: uzunluğu >=2 olan run sayısı
        - groups_len_2: uzunluğu ==2 olan run sayısı
        - groups_len_3: uzunluğu ==3 olan run sayısı
        - groups_len_ge4: uzunluğu >=4 olan run sayısı
        - max_run: en uzun run uzunluğu
        """
        if not sequence:
            return {
                "adjacent_pairs": 0,
                "group_runs": 0,
                "groups_len_2": 0,
                "groups_len_3": 0,
                "groups_len_ge4": 0,
                "max_run": 0,
            }

        adjacent_pairs = 0
        group_runs = 0
        groups_len_2 = 0
        groups_len_3 = 0
        groups_len_ge4 = 0
        max_run = 1

        curr = 1
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i - 1]:
                curr += 1
            else:
                if curr >= 2:
                    group_runs += 1
                    adjacent_pairs += (curr - 1)
                    if curr == 2:
                        groups_len_2 += 1
                    elif curr == 3:
                        groups_len_3 += 1
                    elif curr >= 4:
                        groups_len_ge4 += 1
                max_run = max(max_run, curr)
                curr = 1

        # finalize last run
        if curr >= 2:
            group_runs += 1
            adjacent_pairs += (curr - 1)
            if curr == 2:
                groups_len_2 += 1
            elif curr == 3:
                groups_len_3 += 1
            elif curr >= 4:
                groups_len_ge4 += 1
        max_run = max(max_run, curr)

        return {
            "adjacent_pairs": int(adjacent_pairs),
            "group_runs": int(group_runs),
            "groups_len_2": int(groups_len_2),
            "groups_len_3": int(groups_len_3),
            "groups_len_ge4": int(groups_len_ge4),
            "max_run": int(max_run),
        }

    def _check_grouping(self, sequence: List[int], max_group: int = 3, gstats: Optional[Dict[str, int]] = None) -> float:
        """Rule 6: Grouping kontrolü - max 3 ply üst üste + toplam grouping minimize + 3'lü grup penalty.
        
        Args:
            sequence: The laminate sequence
            max_group: Maximum allowed consecutive plies
            gstats: Pre-computed grouping stats (optional, to avoid redundant calculation)
        """
        max_penalty = self.WEIGHTS["R6"]
        
        # Use pre-computed stats if provided, otherwise compute
        if gstats is None:
            gstats = self._grouping_stats(sequence)
        
        penalty = 0.0
        max_group_found = gstats["max_run"]
        total_adjacent_pairs = gstats["adjacent_pairs"]
        groups_of_3 = gstats["groups_len_3"]

        # Penalty 1: Max group > 3 ise büyük penalty (max_penalty'nin %25'i her fazla için)
        if max_group_found > max_group:
            excess = max_group_found - max_group
            penalty += excess * (max_penalty * 0.25)

        # Penalty 2: 3'lü gruplar için küçük penalty (ideal: 2'li veya daha az)
        penalty += groups_of_3 * 1.0  # Her 3'lü grup için 1 puan penalty

        # Penalty 3: Toplam adjacent pairs'e göre küçük penalty
        n = len(sequence)
        if n > 1:
            adjacent_ratio = total_adjacent_pairs / float(n - 1)
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

    def _build_smart_skeleton(self) -> List[int]:
        """Kuralları sırayla tatmin eden başlangıç sequence oluştur (simetrik)."""
        # Simetrik skeleton oluşturmak için _create_symmetric_individual metodunu kullan
        # Bu metod her açının sayısını koruyarak simetrik bir sequence oluşturur
        return self._create_symmetric_individual()

    def _multi_start_ga(self, skeleton: List[int], n_runs: int = 7) -> Tuple[List[int], float]:
        """Multi-start GA: Skeleton'dan başlayarak farklı local optima'lara bakar."""
        print("Phase 2: Multi-Start GA")

        skeleton_score, _ = self.calculate_fitness(skeleton)
        print("  Skeleton score: {:.2f}/100".format(skeleton_score))

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
            print("  Run {}/{}...".format(run + 1, n_runs), end=" ")

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

            for _gen in range(generations):
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

            print("Score: {:.2f}".format(best_fit))

            if best_fit > best_score:
                best_score = best_fit
                best_global = best_seq[:]

        print("  Best across runs: {:.2f}/100".format(best_score))
        return best_global, best_score

    def _local_search(self, sequence: List[int], max_iter: int = 100) -> Tuple[List[int], float]:
        """Hill climbing with first-improvement strategy.
        
        Uses first-improvement instead of best-improvement for faster convergence.
        Still prioritizes: 3'lü grup azaltma > grouping azaltma > score artışı.
        """
        print("Phase 3: Local Search (First-Improvement)")

        current = sequence[:]
        current_score, _ = self.calculate_fitness(current)
        current_groupings = self._count_groupings(current)
        current_groups_of_3 = self._find_groups_of_size(current, 3)
        print(
            "  Initial score: {:.2f}, Groupings: {}, Groups of 3: {}".format(
                current_score, current_groupings, current_groups_of_3
            )
        )

        iteration = 0
        improvements = 0

        while iteration < max_iter:
            improved = False
            n = len(current)
            half = n // 2
            
            # Randomize search order for diversity
            indices = list(range(half))
            random.shuffle(indices)

            # FIRST-IMPROVEMENT: İlk iyileştirmeyi bulunca uygula
            for i in indices:
                if improved:
                    break
                for j in range(i + 1, half):
                    candidate = current[:]
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    mirror_i = n - 1 - i
                    mirror_j = n - 1 - j
                    candidate[mirror_i], candidate[mirror_j] = candidate[mirror_j], candidate[mirror_i]

                    # Rule 4 kontrolü: Başlangıç ve bitiş ±45 olmalı
                    rule4_preserved = (candidate[0] in [45, -45] and candidate[-1] in [45, -45])
                    
                    # Rule 4 bozuluyorsa hızlı atla (sadece 3'lü grup azaltıyorsa kabul)
                    if not rule4_preserved:
                        candidate_groups_of_3 = self._find_groups_of_size(candidate, 3)
                        if candidate_groups_of_3 >= current_groups_of_3:
                            continue

                    candidate_score, _ = self.calculate_fitness(candidate)
                    
                    # İlk iyileştirmeyi kabul et
                    if candidate_score > current_score:
                        candidate_groupings = self._count_groupings(candidate)
                        candidate_groups_of_3 = self._find_groups_of_size(candidate, 3)
                        grouping_change = current_groupings - candidate_groupings
                        groups_of_3_change = current_groups_of_3 - candidate_groups_of_3
                        
                        current = candidate
                        current_score = candidate_score
                        current_groupings = candidate_groupings
                        current_groups_of_3 = candidate_groups_of_3
                        improved = True
                        improvements += 1
                        
                        if rule4_preserved:
                            print(
                                "  Iteration {}: Improved to {:.2f} (Rule4 preserved), Groupings: {} ({:+d}), Groups of 3: {} ({:+d})".format(
                                    iteration,
                                    current_score,
                                    current_groupings,
                                    grouping_change,
                                    current_groups_of_3,
                                    groups_of_3_change,
                                )
                            )
                        else:
                            print(
                                "  Iteration {}: Improved to {:.2f} (Rule4 violated for grouping reduction), Groupings: {} ({:+d}), Groups of 3: {} ({:+d})".format(
                                    iteration,
                                    current_score,
                                    current_groupings,
                                    grouping_change,
                                    current_groups_of_3,
                                    groups_of_3_change,
                                )
                            )
                        break

            if not improved:
                print("  Converged after {} iterations ({} improvements)".format(iteration, improvements))
                break

            iteration += 1

        final_groups_of_3 = self._find_groups_of_size(current, 3)
        print(
            "  Final score: {:.2f}/100, Final groupings: {}, Final groups of 3: {}".format(
                current_score, self._count_groupings(current), final_groups_of_3
            )
        )
        return current, current_score

    def run_hybrid_optimization(self) -> Tuple[List[int], float, Dict[str, Any], List[float]]:
        """3-phase hybrid optimization pipeline."""
        print("=" * 60)
        print("3-PHASE HYBRID OPTIMIZATION")
        print("=" * 60)

        start_time = time.time()

        # Phase 1
        print("\nPhase 1: Smart Skeleton Construction")
        skeleton = self._build_smart_skeleton()
        phase1_score, phase1_details = self.calculate_fitness(skeleton)
        print("  Score: {:.2f}/100".format(phase1_score))
        print("  Time: {:.2f}s".format(time.time() - start_time))

        # Phase 2
        phase2_start = time.time()
        # Ply sayısına göre run sayısını ayarla
        n_runs = 7 if self.total_plies <= 40 else 10
        best_seq, phase2_score = self._multi_start_ga(skeleton, n_runs=n_runs)
        print("  Time: {:.2f}s".format(time.time() - phase2_start))

        # Phase 3
        phase3_start = time.time()
        final_seq, final_score = self._local_search(best_seq, max_iter=100)
        print("  Time: {:.2f}s".format(time.time() - phase3_start))

        total_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("FINAL RESULT: {:.2f}/100 (in {:.2f}s)".format(final_score, total_time))
        
        # Print cache statistics
        cache_stats = self.get_cache_stats()
        print("Cache Stats: {} cached, {} hits, {} misses, {:.1f}% hit rate".format(
            cache_stats["cache_size"],
            cache_stats["cache_hits"],
            cache_stats["cache_misses"],
            cache_stats["hit_rate_percent"]
        ))
        print("=" * 60)

        # Final details
        _, final_details = self.calculate_fitness(final_seq)

        # History (dummy - hybrid optimization'da history yok)
        history = [final_score]

        return final_seq, final_score, final_details, history

    def calculate_fitness(self, sequence: List[int]):
        """
        PDF kurallarına göre fitness hesapla (with caching).
        Max score = 100 (tüm rule weights toplamı)
        """
        # Convert to tuple for hashing
        seq_key = tuple(sequence)
        
        # Check cache first
        if seq_key in self._fitness_cache:
            self._cache_hits += 1
            return self._fitness_cache[seq_key]
        
        self._cache_misses += 1
        
        # Calculate fitness
        result = self._calculate_fitness_impl(sequence)
        
        # Store in cache (with size limit)
        if len(self._fitness_cache) < self.MAX_CACHE_SIZE:
            self._fitness_cache[seq_key] = result
        
        return result
    
    def _calculate_fitness_impl(self, sequence: List[int]):
        """
        Actual fitness calculation implementation.
        """
        WEIGHTS = self.WEIGHTS

        rules_result = {}

        # ========== HARD CONSTRAINTS ==========

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
                        "reason": "0° başlangıç veya bitiş katmanı (YASAK)",
                    }
                },
            }

        # ========== SOFT CONSTRAINTS ==========

        # Rule 1: Symmetry (distance-weighted)
        penalty_r1 = self._check_symmetry_distance_weighted(sequence)
        score_r1 = max(0, WEIGHTS["R1"] - penalty_r1)
        rules_result["R1"] = {
            "weight": WEIGHTS["R1"],
            "score": round(score_r1, 2),
            "penalty": round(penalty_r1, 2),
            "reason": "Asimetri var" if penalty_r1 > 0 else "",
        }

        # Rule 2: Balance (sadece ±45 için)
        penalty_r2 = self._check_balance_45(sequence)
        score_r2 = max(0, WEIGHTS["R2"] - penalty_r2)
        rules_result["R2"] = {
            "weight": WEIGHTS["R2"],
            "score": round(score_r2, 2),
            "penalty": round(penalty_r2, 2),
            "reason": "+45/-45 sayıları eşit değil" if penalty_r2 > 0 else "",
        }

        # Rule 3: Percentage (8-67%)
        penalty_r3 = self._check_percentage_rule(sequence)
        score_r3 = max(0, WEIGHTS["R3"] - penalty_r3)
        rules_result["R3"] = {
            "weight": WEIGHTS["R3"],
            "score": round(score_r3, 2),
            "penalty": round(penalty_r3, 2),
            "reason": "Bazı açılar %8-67 dışında" if penalty_r3 > 0 else "",
        }

        # Rule 4: External plies (ilk/son 2 katman)
        score_r4 = self._check_external_plies(sequence)
        penalty_r4 = WEIGHTS["R4"] - score_r4
        rules_result["R4"] = {
            "weight": WEIGHTS["R4"],
            "score": round(score_r4, 2),
            "penalty": round(penalty_r4, 2),
            "reason": "Dış katmanlar ideal değil" if penalty_r4 > 0 else "",
        }

        # Rule 5: Distribution (variance-based)
        penalty_r5 = self._check_distribution_variance(sequence)
        score_r5 = max(0, WEIGHTS["R5"] - penalty_r5)
        rules_result["R5"] = {
            "weight": WEIGHTS["R5"],
            "score": round(score_r5, 2),
            "penalty": round(penalty_r5, 2),
            "reason": "Dağılım uniform değil" if penalty_r5 > 0 else "",
        }

        # Rule 6: Grouping (max 3) - compute stats once and reuse
        gstats = self._grouping_stats(sequence)
        penalty_r6 = self._check_grouping(sequence, max_group=3, gstats=gstats)
        score_r6 = max(0, WEIGHTS["R6"] - penalty_r6)
        if penalty_r6 > 0:
            # Sadece istenen sayılar: 2'li / 3'lü / 4+ grup adedi
            reason_r6 = "2'li grup: {}, 3'lü grup: {}, 4+ grup: {}".format(
                gstats["groups_len_2"], gstats["groups_len_3"], gstats["groups_len_ge4"]
            )
        else:
            reason_r6 = ""
        rules_result["R6"] = {
            "weight": WEIGHTS["R6"],
            "score": round(score_r6, 2),
            "penalty": round(penalty_r6, 2),
            "reason": reason_r6,
        }

        # Rule 7: Buckling (±45 uzakta)
        penalty_r7 = self._check_buckling(sequence)
        score_r7 = max(0, WEIGHTS["R7"] - penalty_r7)
        rules_result["R7"] = {
            "weight": WEIGHTS["R7"],
            "score": round(score_r7, 2),
            "penalty": round(penalty_r7, 2),
            "reason": "±45 middle plane'e yakın" if penalty_r7 > 0 else "",
        }

        # Rule 8: Lateral bending (90° uzakta)
        penalty_r8 = self._check_lateral_bending(sequence)
        score_r8 = max(0, WEIGHTS["R8"] - penalty_r8)
        rules_result["R8"] = {
            "weight": WEIGHTS["R8"],
            "score": round(score_r8, 2),
            "penalty": round(penalty_r8, 2),
            "reason": "90° middle plane'e yakın" if penalty_r8 > 0 else "",
        }

        # FINAL SCORE
        # Ensure plain Python float (avoid numpy scalar propagation)
        total_score = float(sum(r["score"] for r in rules_result.values()))

        return total_score, {"total_score": round(total_score, 2), "max_score": 100.0, "rules": rules_result}

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
        population = []  # type: List[List[int]]
        for _ in range(population_size):
            ind = self._create_symmetric_individual()
            population.append(ind)

        best_sol = None  # type: Optional[List[int]]
        best_fit = -1.0
        best_det = {}  # type: Dict[str, float]
        history = []  # type: List[float]

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
        """
        global_best_sequence = None  # type: Optional[List[int]]
        global_best_fitness = -1.0
        global_best_penalties = {}  # type: Dict[str, float]
        all_histories = []  # type: List[List[float]]

        print(
            "Starting auto-optimization: {} runs, pop={}, gen={}".format(
                runs, population_size, generations
            )
        )

        for run_num in range(1, runs + 1):
            print("Run {}/{}...".format(run_num, runs))

            sequence, fitness, penalties, history = self.run_genetic_algorithm(
                population_size=population_size, generations=generations
            )

            all_histories.append(history)

            if len(history) >= stagnation_window:
                recent_fitness = history[-stagnation_window:]
                max_recent = max(recent_fitness)
                min_recent = min(recent_fitness)
                fitness_range = max_recent - min_recent

                if fitness_range < 0.01:
                    print(
                        "  Run {}: Converged early (fitness range: {:.6f})".format(
                            run_num, fitness_range
                        )
                    )

            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_sequence = sequence[:]
                global_best_penalties = penalties.copy()
                print("  Run {}: New best fitness = {:.2f}".format(run_num, fitness))

        combined_history = []  # type: List[float]
        max_gen_length = max(len(h) for h in all_histories) if all_histories else 0

        for gen_idx in range(max_gen_length):
            gen_best = -1.0
            for history in all_histories:
                if gen_idx < len(history):
                    gen_best = max(gen_best, history[gen_idx])
            combined_history.append(gen_best)

        print("Auto-optimization complete. Best fitness: {:.2f}".format(global_best_fitness))

        return {
            "best_sequence": global_best_sequence or [],
            "best_fitness": round(global_best_fitness, 2),
            "penalties": global_best_penalties,
            "history": combined_history,
        }

