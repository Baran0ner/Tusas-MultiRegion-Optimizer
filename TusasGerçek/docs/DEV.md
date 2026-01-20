# Development

## Kurulum

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma

```powershell
python app.py
```

Tarayıcı:
- `http://127.0.0.1:5000`

## Windows Türkçe karakter (ç/ğ) notu

Bazı ortamlarda çalışma dizini `TusasGerçek` yolu `TusasGer��ek` gibi yanlış görünebiliyor. Uygulama yine de çalışsa da, mümkünse proje yolunda Türkçe karakter kullanmamak daha stabil olur.

