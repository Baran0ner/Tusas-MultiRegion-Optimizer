# Frontend

Frontend şu an `index.html` içinde (Tailwind CDN + Chart.js CDN).

## Custom CSS

Inline `<style>` bloğu kaldırıldı ve `static/css/app.css` altına taşındı.

`index.html` artık bunu yükler:

- `/static/css/app.css`

## İstekler

Frontend `API_BASE` üzerinden backend’e istek atar. Localhost’ta `API_BASE = ''` olacak şekilde tasarlanmıştır.

## Notlar

- Zone şeması ve tooltip gibi etkileşimler tamamen frontend JS tarafında yönetilir.
- İleride JS’i de `static/js/*` altına bölmek kolay olacak şekilde backend artık modüler.

