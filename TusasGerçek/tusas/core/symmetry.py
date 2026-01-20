from typing import Dict, Any


def check_symmetry_compatibility(ply_counts: Dict[int, int]) -> Dict[str, Any]:
    """
    Simetri uyumluluğunu kontrol eder.

    Returns:
        {
            'requires_user_choice': bool,
            'issues': list,
            'suggestions': list
        }
    """
    total = sum(ply_counts.values())
    is_odd_total = total % 2 == 1

    odd_angles = [angle for angle, count in ply_counts.items() if count % 2 == 1]

    issues = []
    suggestions = []

    # Durum 1: Tek sayılı toplam + 2+ tek sayılı açı
    if is_odd_total and len(odd_angles) > 1:
        issues.append(
            {
                "type": "multiple_odd_angles",
                "angles": odd_angles,
                "total": total,
                "message": "{} açı tek sayıda ({}°). Sadece biri ortaya gidebilir.".format(
                    len(odd_angles), ", ".join(map(str, odd_angles))
                ),
            }
        )

        for angle in odd_angles:
            adjusted = ply_counts.copy()
            adjusted[angle] = adjusted[angle] - 1
            suggestions.append(
                {
                    "type": "set_middle",
                    "middle_angle": angle,
                    "adjusted_counts": adjusted,
                    "description": "{}° ortaya koy, diğer tek sayılı açıları çift yap".format(angle),
                }
            )

    # Durum 2: Çift sayılı toplam + tek sayılı açılar
    if (not is_odd_total) and len(odd_angles) > 0:
        angle_45 = ply_counts.get(45, 0)
        angle_minus45 = ply_counts.get(-45, 0)

        if angle_45 % 2 == 1 and angle_minus45 % 2 == 1 and angle_45 == angle_minus45:
            issues.append(
                {
                    "type": "odd_45_balance",
                    "angles": [45, -45],
                    "total": total,
                    "message": "45° ve -45° her ikisi de tek sayıda ({}). Eşit sayıda kalmalı (Rule 2: Balance).".format(
                        angle_45
                    ),
                }
            )

            # Öneri 1: İkisini de +1 yap (toplam sabit kalsın diye 0 veya 90'dan düş)
            adjusted1 = ply_counts.copy()
            adjusted1[45] = angle_45 + 1
            adjusted1[-45] = angle_minus45 + 1
            total_adjustment = 2
            for adj_angle in [0, 90]:
                if adj_angle in adjusted1 and adjusted1[adj_angle] >= total_adjustment:
                    adjusted1[adj_angle] -= total_adjustment
                    suggestions.append(
                        {
                            "type": "increase_45_pair",
                            "compensation_angle": adj_angle,
                            "compensation_amount": total_adjustment,
                            "adjusted_counts": adjusted1,
                            "description": "45° ve -45° → {} (her ikisi +1), {}° → {} (... - {})".format(
                                angle_45 + 1, adj_angle, adjusted1[adj_angle], total_adjustment
                            ),
                        }
                    )
                    break

            # Öneri 2: İkisini de -1 yap (toplam sabit kalsın diye 0 veya 90'a ekle)
            adjusted2 = ply_counts.copy()
            adjusted2[45] = angle_45 - 1
            adjusted2[-45] = angle_minus45 - 1
            total_adjustment = 2
            for adj_angle in [0, 90]:
                if adj_angle in adjusted2:
                    adjusted2[adj_angle] += total_adjustment
                    suggestions.append(
                        {
                            "type": "decrease_45_pair",
                            "compensation_angle": adj_angle,
                            "compensation_amount": total_adjustment,
                            "adjusted_counts": adjusted2,
                            "description": "45° ve -45° → {} (her ikisi -1), {}° → {} (... + {})".format(
                                angle_45 - 1, adj_angle, adjusted2[adj_angle], total_adjustment
                            ),
                        }
                    )
                    break
        else:
            issues.append(
                {
                    "type": "odd_angles_even_total",
                    "angles": odd_angles,
                    "total": total,
                    "message": "Çift sayılı toplamda {} açı tek sayıda. Simetri için her açı çift sayıda olmalı.".format(
                        len(odd_angles)
                    ),
                }
            )

            for angle in odd_angles:
                adjusted = ply_counts.copy()
                adjusted[angle] = adjusted[angle] + 1
                suggestions.append(
                    {
                        "type": "make_even",
                        "angle": angle,
                        "adjusted_counts": adjusted,
                        "description": "{}°: {} → {} (çift sayı yap)".format(angle, ply_counts[angle], adjusted[angle]),
                    }
                )

    return {"requires_user_choice": len(issues) > 0, "issues": issues, "suggestions": suggestions}

