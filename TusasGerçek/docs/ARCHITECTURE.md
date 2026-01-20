# Architecture

Bu proje, kompozit laminate stacking optimizasyonu + drop-off + zone (bölge) yönetimi yapan bir Flask uygulamasıdır.

## Klasör Yapısı

```
TusasGerçek/
  app.py                 # Uygulama entrypoint (Flask)
  main.py                # Alternatif entrypoint (aynı app)
  tusas/                 # Backend package
    app_factory.py        # create_app() (Flask factory)
    api/
      routes.py           # HTTP endpoint’ler (Blueprint)
    core/
      laminate_optimizer.py # GA + fitness (R1–R8)
      dropoff_optimizer.py  # Drop-off algoritmaları
      symmetry.py           # Simetri uyumluluğu kontrolü
    zones/
      models.py           # Zone modeli
      manager.py          # ZoneManager (in-memory)
    state.py              # session_id -> ZoneManager store
  static/
    css/app.css           # UI custom CSS
  index.html              # Frontend (şimdilik tek HTML)
  requirements.txt
```

## Veri Akışı

- Frontend `fetch()` ile backend endpoint’lerine istek atar.
- Backend:
  - `LaminateOptimizer` ile master sequence üretir (`/optimize`)
  - `DropOffOptimizer` ile drop-off yapar (`/dropoff`, `/dropoff_angle_targets`)
  - `ZoneManager` ile zone/transition yönetir (`/zones/*`)
- Zone state’i şu an **RAM’de** tutulur (`tusas/state.py`). Sunucu restart olursa resetlenir.

## Neden Flask Factory?

`create_app()` yaklaşımı ile:
- test/konfig ayrımı kolaylaşır
- blueprint ile route’lar modüler olur
- tek dosyada büyüme engellenir

