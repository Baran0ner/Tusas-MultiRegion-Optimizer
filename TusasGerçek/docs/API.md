# API

Tüm endpoint’ler JSON alır/verir.

## `POST /optimize`

Laminate master sequence üretir ve otomatik drop-off zinciri döner.

**Body**

```json
{
  "ply_counts": { "0": 18, "90": 18, "45": 18, "-45": 18 },
  "population_size": 120,
  "generations": 600,
  "min_drop": 48,
  "drop_step": 8,
  "symmetry_user_choice": { "continue_with_current": true }
}
```

**Response (özet)**

```json
{
  "master_sequence": [45, 0, -45, 90],
  "fitness_score": 96.2,
  "max_score": 100,
  "penalties": { "R1": { "weight": 20, "score": 20, "penalty": 0, "reason": "" } },
  "drop_off_results": [{ "target": 34, "seq": [], "score": 90.1, "dropped": [1, 32] }],
  "stats": { "plies": 36, "duration_seconds": 5.2 }
}
```

## `POST /evaluate`

Verilen sequence’ı skorlar.

## `POST /dropoff`

Tek hedef ply sayısına drop-off.

## `POST /dropoff_angle_targets`

Açı bazlı hedef sayılara drop-off.

## `POST /auto_optimize`

Çoklu-run GA ile en iyi master sequence’i döner.

## Zone Endpoint’leri

Zone state RAM’de ve `session_id` ile ayrılır.

- `POST /zones/init_root`
- `GET /zones/list?session_id=...`
- `GET /zones/<zone_id>?session_id=...`
- `POST /zones/create_from_dropoff`
- `POST /zones/create_from_angle_dropoff`
- `POST /zones/create_from_merge`
- `POST /zones/add_from_dropoff`

