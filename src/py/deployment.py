import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from pathlib import Path
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder
import dagshub

# Inisialisasi Dagshub & MLflow sesuai experiment_latest.ipynb
dagshub.init(repo_owner='Dekhsa', repo_name='Project-TimeseriesHargaCabai', mlflow=True)
mlflow.set_experiment("Analisis_Harga_Cabai_Rolling")

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "data_with_holidays.csv"
GB_MODEL_PATH = ROOT / "src" / "notebook" / "mlruns" / "0" / "models" / "m-5cd1077a94834bf881fa2860b0ab4105" / "artifacts"
SARIMAX_MODEL_PATH = ROOT / "src" / "notebook" / "mlruns" / "0" / "models" / "m-6112446c237e439c9c46d6156934d187" / "artifacts"


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, LabelEncoder]:
    df = pd.read_csv(data_path)
    df["tanggal_data"] = pd.to_datetime(df["tanggal_data"])
    df = df.sort_values("tanggal_data").reset_index(drop=True)

    df["jenis_libur"] = df["jenis_libur"].fillna("Bukan Libur")
    le = LabelEncoder()
    df["jenis_libur_encoded"] = le.fit_transform(df["jenis_libur"])

    df["is_libur"] = df["is_libur"].astype(int)

    full_range = pd.date_range(df["tanggal_data"].min(), df["tanggal_data"].max(), freq="D")
    df = df.set_index("tanggal_data").reindex(full_range)
    df.index.name = "tanggal_data"
    df["harga"] = df["harga"].interpolate(method="linear")
    df["is_libur"] = df["is_libur"].fillna(0).astype(int)
    df["jenis_libur_encoded"] = df["jenis_libur_encoded"].fillna(le.transform(["Bukan Libur"])[0]).astype(int)

    df = df.reset_index()
    return df, le


@st.cache_data
def load_data() -> tuple[pd.DataFrame, LabelEncoder]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")
    return load_dataset(DATA_PATH)


@st.cache_resource
def load_gb_model():
    if not GB_MODEL_PATH.exists():
        raise FileNotFoundError(f"Gradient Boosting model tidak ditemukan: {GB_MODEL_PATH}")
    return mlflow.sklearn.load_model(str(GB_MODEL_PATH))


@st.cache_resource
def load_sarimax_model():
    if not SARIMAX_MODEL_PATH.exists():
        raise FileNotFoundError(f"SARIMAX model tidak ditemukan: {SARIMAX_MODEL_PATH}")
    return mlflow.pyfunc.load_model(str(SARIMAX_MODEL_PATH))


def build_input_features(last_prices: list[float], is_libur: int, jenis_libur_encoded: int) -> np.ndarray:
    return np.array(last_prices + [int(is_libur), int(jenis_libur_encoded)], dtype=float).reshape(1, -1)


def predict_gb(model, last_prices: list[float], is_libur: int, jenis_libur_encoded: int) -> float:
    features = build_input_features(last_prices, is_libur, jenis_libur_encoded)
    return float(model.predict(features)[0])


def predict_sarimax(model, is_libur: int, jenis_libur_encoded: int) -> float:
    exog = pd.DataFrame({"is_libur": [int(is_libur)], "jenis_libur_encoded": [int(jenis_libur_encoded)]})
    if hasattr(model, "forecast"):
        try:
            forecast = model.forecast(steps=1, exog=exog)
            return float(np.asarray(forecast)[0])
        except Exception:
            pass

    prediction = model.predict(exog)
    if isinstance(prediction, (pd.Series, np.ndarray, list)):
        return float(prediction[0])
    return float(prediction)


def get_holiday_choices(df: pd.DataFrame) -> list[str]:
    unique_holidays = sorted(df["jenis_libur"].unique())
    if "Bukan Libur" in unique_holidays:
        unique_holidays.remove("Bukan Libur")
    return ["Bukan Libur"] + unique_holidays


def main() -> None:
    st.set_page_config(page_title="Deploy Harga Cabai", layout="wide")
    st.title("Forecast Harga Cabai dengan Streamlit")
    st.markdown(
        "Aplikasi ini memuat model Gradient Boosting dan SARIMAX yang dibuat pada `experiment_latest.ipynb`. "
        "Gunakan data harga terakhir dan fitur libur untuk memprediksi harga besok."
    )

    try:
        df, encoder = load_data()
    except Exception as exc:
        st.error(str(exc))
        return

    last_date = df["tanggal_data"].max()
    next_date = last_date + timedelta(days=1)
    last_prices = df["harga"].tail(7).tolist()

    st.sidebar.header("Pengaturan Prediksi")
    use_gb = st.sidebar.checkbox("Gradient Boosting", value=True)
    use_sarima = st.sidebar.checkbox("SARIMAX", value=True)
    st.sidebar.markdown("---")
    st.sidebar.write(f"Data terakhir: **{last_date.date()}**")
    st.sidebar.write(f"Tanggal prediksi: **{next_date.date()}**")

    holiday_option = st.sidebar.radio("Apakah tanggal prediksi libur?", ["Tidak", "Ya"], index=0)
    holiday_choices = get_holiday_choices(df)
    if holiday_option == "Ya":
        jenis_libur = st.sidebar.selectbox("Pilih jenis libur", holiday_choices[1:], index=0)
        is_libur = 1
    else:
        jenis_libur = "Bukan Libur"
        is_libur = 0

    manual_prices = st.sidebar.checkbox("Masukkan harga secara manual", value=False)
    if manual_prices:
        st.sidebar.markdown("**Harga input (7 hari terakhir / nilai manual)**")
        user_prices = []
        price_cols = st.sidebar.columns(7)
        for idx, col in enumerate(price_cols, start=1):
            user_prices.append(
                col.number_input(
                    f"Hari -{8-idx}",
                    min_value=0.0,
                    value=float(last_prices[idx - 1]),
                    format="%.2f",
                )
            )
        last_prices = user_prices
    else:
        st.sidebar.markdown("**Menggunakan 7 harga terakhir dari dataset**")
        for idx, price in enumerate(last_prices, start=1):
            st.sidebar.write(f"Hari -{8-idx}: {price:.2f}")

    if jenis_libur not in encoder.classes_:
        jenis_libur_encoded = encoder.transform(["Bukan Libur"])[0]
    else:
        jenis_libur_encoded = int(encoder.transform([jenis_libur])[0])

    st.subheader("Ringkasan Input")
    st.metric("Tanggal prediksi", str(next_date.date()))
    st.write(
        f"Harga window (7 hari terakhir): {', '.join(f'{p:.2f}' for p in last_prices)}"
    )
    st.write(f"Libur: {holiday_option} - {jenis_libur}")

    if st.button("Jalankan Prediksi"):
        if not use_gb and not use_sarima:
            st.warning("Pilih setidaknya satu model untuk melakukan prediksi.")
        else:
            cols = st.columns(2)
            if use_gb:
                try:
                    gb_model = load_gb_model()
                    gb_prediction = predict_gb(gb_model, last_prices, is_libur, jenis_libur_encoded)
                    cols[0].metric("Prediksi Gradient Boosting", f"Rp {gb_prediction:,.0f}")
                except Exception as exc:
                    cols[0].error(f"Gradient Boosting error: {exc}")

            if use_sarima:
                try:
                    sarimax_model = load_sarimax_model()
                    sarimax_prediction = predict_sarimax(sarimax_model, is_libur, jenis_libur_encoded)
                    cols[1 if use_gb else 0].metric("Prediksi SARIMAX", f"Rp {sarimax_prediction:,.0f}")
                except Exception as exc:
                    cols[1 if use_gb else 0].error(f"SARIMAX error: {exc}")

    with st.expander("Tampilan dataset asli"):
        st.write(df.tail(15))

    st.sidebar.markdown("---")
    st.sidebar.write("Jika model tidak ditemukan, pastikan `src/notebook/mlruns/0/models` tersedia dan file `data/processed/data_with_holidays.csv` ada.")


if __name__ == "__main__":
    main()
