# llm-quality-testing

## Tujuan Proyek
Alat CLI untuk membandingkan kualitas beberapa model LLM (Hugging Face) pada dataset yang sama menggunakan metrik perplexity, accuracy, dan BLEU. Hasil dapat dilihat langsung di terminal (tabel) dan disimpan ke file (CSV/JSON) untuk dokumentasi dan analisis lebih lanjut.

## Instalasi
1. Pastikan Python 3.8+ sudah terpasang.
2. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

## Struktur Konfigurasi YAML
Contoh file: `configs/default_config.yaml`
```yaml
models:
  - gpt2
  - facebook/bart-large-cnn
dataset: data/sample_data.json
output_csv: results/leaderboard.csv
output_json: results/leaderboard.json
```
- `models`: Daftar nama model Hugging Face yang akan dibandingkan.
- `dataset`: Path ke file data uji (format JSON, list of {"text", "label"}).
- `output_csv`: Path file output hasil leaderboard (CSV).
- `output_json`: Path file output hasil leaderboard (JSON).

## Menjalankan Perbandingan Model
Jalankan perintah berikut dari root project:
```bash
python scripts/compare_models.py
```

## Contoh Output di Terminal
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                Comparative Model Leaderboard     ┃
┡━━━━━━━━━━━━━━━┯━━━━━━━━━━━━┯━━━━━━━━━━┯━━━━━━━━━━┩
│ model_name    │ perplexity │ accuracy │ bleu     │
├───────────────┼────────────┼──────────┼──────────┤
│ gpt2          │ 34.1234    │ 0.8125   │ 0.5123   │
│ bart-large-cnn│ 28.5678    │ 0.8450   │ 0.6012   │
└───────────────┴────────────┴──────────┴──────────┘

Leaderboard saved to: results/leaderboard.csv and results/leaderboard.json
```

## Hasil File Output
- `results/leaderboard.csv`: Tabel hasil perbandingan dalam format CSV.
- `results/leaderboard.json`: Hasil yang sama dalam format JSON.

## Testing
Jalankan seluruh tes dengan:
```bash
pytest
```

## Struktur Proyek
```
llm-quality-testing/
├── llm_eval/          # Kode utama evaluasi
├── scripts/           # Script CLI
├── tests/             # Unit & integration tests
├── configs/           # File konfigurasi YAML
├── results/           # Output leaderboard
├── requirements.txt
└── README.md
```

## Catatan
- Untuk model besar, pastikan resource (RAM/GPU) cukup.
- Untuk menambah model, cukup edit file YAML konfigurasi.
- Untuk memperluas ke custom loader, paralelisasi, atau API, struktur sudah siap untuk dikembangkan lebih lanjut.
