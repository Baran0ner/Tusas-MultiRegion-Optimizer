import random
from typing import Dict, List, Tuple

import numpy as np

from .laminate_optimizer import LaminateOptimizer


class DropOffOptimizer:
    """
    Tapering optimizer for ply drop-off.
    """

    def __init__(self, master_sequence: List[int], base_optimizer: LaminateOptimizer):
        self.master_sequence = master_sequence
        self.base_opt = base_optimizer
        self.total_plies = len(master_sequence)

    def optimize_drop(self, target_ply: int, fixed_drop_indices: List[int] = None) -> Tuple[List[int], float, List[int]]:
        """
        Drop-off optimization with odd/even ply support.

        Supports:
        - Even → Even: Normal symmetric drop (pairs)
        - Odd → Odd: Symmetric drop (pairs), middle ply preserved
        - Odd → Even: Drop middle ply + symmetric pairs
        - Even → Odd: Break one pair - keep one as middle, drop its mirror
        
        Args:
            target_ply: Target ply count
            fixed_drop_indices: List of indices that MUST be dropped (from thicker neighbors)
        """
        if fixed_drop_indices is None:
            fixed_drop_indices = []
            
        remove_cnt = self.total_plies - target_ply
        if remove_cnt <= 0:
            return self.master_sequence, 0.0, []
            
        # Validate fixed drops
        if len(fixed_drop_indices) > remove_cnt:
            print(f"Warning: Fixed drops ({len(fixed_drop_indices)}) > removes needed ({remove_cnt})")
            # We must drop at least fixed ones, so target ply might be lower than requested
            # But for now let's just proceed and maybe fail or clip
            
        # Current valid drops so far (from constraints)
        current_drops = set(fixed_drop_indices)

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
            
            # Eğer middle zaten fixed drops içindeyse, stratejiye uyuyor
            if middle_idx in current_drops:
                pass # Already handled
                
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
        
        # Fixed drop indices'i analize et
        # Kaç çift fixed? Orta ply fixed mi?
        fixed_count = len(current_drops)
        
        # Fixed drops simetrik olmalı
        # Sol yarıdaki fixed pozisyonları bul
        fixed_left = []
        for idx in current_drops:
            if idx < half_len:
                fixed_left.append(idx)
            elif idx == middle_idx:
                # Middleware drop is fixed
                if not drop_middle:
                    # Strateji uyuşmazlığı, ama yapacak bir şey yok
                    pass
                    
        # Calculate remaining pairs needed
        # We need total `pairs_to_remove` pairs
        # We already have `len(fixed_left)` pairs dropped (assuming symmetry)
        pairs_needed = pairs_to_remove - len(fixed_left)
        
        best_candidate = None
        best_key = None
        best_dropped = []

        attempts = self.base_opt.DROP_OFF_ATTEMPTS
        
        for _ in range(attempts):
            left_drops = list(fixed_left) # Start with fixed drops

            # Çift → Tek: Bir çifti kırmak için pozisyon seç
            if break_pair_for_middle:
                if len(search_indices) == 0:
                    continue
                # Ortaya yakın bir pozisyon seç (sol yarının sonlarından)
                # Bu pozisyondaki ply ortaya geçecek, mirror'ı drop edilecek
                break_pair_idx = random.choice(search_indices)
                
            # Normal çift drop pozisyonları seç
            # Sadece eğer daha fazla çift lazımsa
            if pairs_needed > 0 and len(search_indices) > 0:
                # break_pair_idx zaten kullanıldıysa onu hariç tut
                # fixed_left zaten kullanıldı, onları da hariç tut
                available_indices = [i for i in search_indices if i != break_pair_idx and i not in current_drops]
                
                sample_size = min(pairs_needed, len(available_indices))
                if sample_size > 0:
                    random_drops = random.sample(available_indices, sample_size)
                    left_drops.extend(random_drops)
                    left_drops.sort()

            # ✅ 1. NO GROUPING CHECK - Ardışık drop pozisyonları yasak
            # Drop pozisyonları birbirine çok yakın olmamalı (gruplama önleme)
            # Relax this check if drop density is high (> 45%) to avoid infinite loops
            density = len(left_drops) / (len(search_indices) if search_indices else 1)
            
            if density < 0.45 and len(left_drops) > 1:
                has_consecutive = any(left_drops[i + 1] - left_drops[i] == 1 for i in range(len(left_drops) - 1))
                if has_consecutive:
                    continue  # Grouped drops = reddet

            # ✅ 2. UNIFORM DISTRIBUTION CHECK - Drop'lar düzgün dağıtılmış olmalı
            spacing_std = 0.0  # Default değer
            if len(left_drops) > 2:
                spacings = [left_drops[i + 1] - left_drops[i] for i in range(len(left_drops) - 1)]
                spacing_mean = np.mean(spacings)
                spacing_std = np.std(spacings)
                # Çok yüksek standart sapma = kötü dağılım (AVOID örneği gibi)
                if spacing_mean > 0 and spacing_std > spacing_mean * 0.7:  # Çok düzensiz dağılım
                    continue

            all_drops = []
            # Add fixed drops explicitly to all_drops to be safe, although left_drops should cover left half
            # But we need to handle full sequence construction carefully
            
            # Use left_drops to construct all_drops
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
                -has_0_drop,  # Dokuzuncu: 0° drop bonusu (0° varsa -1, yoksa 0, negatif çünkü min istiyoruz)
                total_penalty,  # Onuncu: Toplam penalty
                -total_score,  # On birinci: Total fitness score (yüksek = iyi)
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
        """
        from collections import Counter

        # 1. Validation: Target counts kontrolü
        current_counts = dict(Counter(self.master_sequence))

        for angle, target_count in target_ply_counts.items():
            current = current_counts.get(angle, 0)
            if target_count > current:
                raise ValueError(
                    "Angle {}°: hedef {} ama mevcut sadece {} katman var".format(angle, target_count, current)
                )
            if target_count < 0:
                raise ValueError("Angle {}°: hedef sayı negatif olamaz".format(angle))

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
            positions = [p for p in positions if p > 1]
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
                    all_left_drops[i + 1] - all_left_drops[i] == 1 for i in range(len(all_left_drops) - 1)
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
            temp_seq = [ang for i, ang in enumerate(self.master_sequence) if i not in all_drops]

            # Fitness hesapla
            score, details = self.base_opt.calculate_fitness(temp_seq)

            # Hard constraint ihlali varsa atla
            if score <= 0:
                continue

            # Hedef açı sayılarına ulaşıldı mı kontrol et (±1 tolerans)
            temp_counts = Counter(temp_seq)

            matches_target = True
            for angle, orig_target in target_ply_counts.items():
                actual_count = temp_counts.get(angle, 0)
                if abs(actual_count - orig_target) > 1:
                    matches_target = False
                    break

            if not matches_target:
                continue

            # En iyi skoru güncelle
            if score > best_score:
                best_score = score
                best_candidate = temp_seq
                best_dropped_by_angle = {angle: sorted(positions) for angle, positions in dropped_by_angle.items()}

        if best_candidate is None:
            print("UYARI: Hedef açı sayılarına uygun drop kombinasyonu bulunamadı")
            score, _ = self.base_opt.calculate_fitness(self.master_sequence)
            return self.master_sequence[:], score, {}

        return best_candidate, best_score, best_dropped_by_angle

