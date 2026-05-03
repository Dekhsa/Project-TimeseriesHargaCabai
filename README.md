# Project Data Mining dengan MLflow

Proyek ini dibuat berdasarkan `src/notebook/experiment.ipynb` sebagai sumber utama seluruh pipeline. Kode Python telah disusun ulang untuk merekam alur pengambilan data, eksplorasi, preprocessing, dan pelacakan hasil dengan MLflow.

## Struktur proyek

- `data/` - folder input/output data mentah dan terproses
- `src/` - kode utama aplikasi
- `src/notebook/experiment.ipynb` - sumber eksperimen utama
- `artifacts/` - artefak MLflow dan figure
- `mlruns/` - direktori tracking MLflow
- `tests/` - unit test sederhana

## Instalasi

Gunakan conda atau pip.

Dengan Conda:

```bash
conda env create -f environment.yml
conda activate datamining-mlflow
```

Dengan pip:

```bash
pip install -r requirements.txt
```

## Menjalankan pipeline

Jika ingin mengambil data langsung dari API SP2KP:

```bash
python src/data_mining/train.py --use_api True --start_date 2023-01-23 --end_date 2026-04-22
```

Jika ingin menggunakan CSV lokal:

```bash
python src/data_mining/train.py --use_api False --data_path data/raw/hnt_cabai_merah_keriting.csv
```

## Menjalankan dengan MLflow

```bash
mlflow run .
```

atau

```bash
mlflow run . -P use_api=True
```

## Tracking MLflow

Setelah pipeline selesai, buka UI MLflow:

```bash
mlflow ui
```

Lalu akses `http://127.0.0.1:5000`.

## Catatan

`experiment.ipynb` adalah referensi utama. Semua operasi di `src/data_mining/train.py` mengikuti alur notebook:

- ambil data dari SP2KP API atau CSV
- set index tanggal dan sorting
- interpolasi missing value
- plot time series, rolling mean/std, dekomposisi, ACF/PACF, boxplot musiman
- log artefak dan metrik ke MLflow
