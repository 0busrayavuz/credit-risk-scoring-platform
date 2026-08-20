"""Skorlanmis portfoyu veritabanina yazar - BI panelinin veri kaynagi.

NEDEN AYRI BIR TABLO:
    Power BI'i dogrudan features.model_input'a baglamak kotu bir tasarim olurdu:
    230 kolonluk ham bir tablo, is birimi icin anlamsizdir ve panel her
    yenilendiginde modelin yeniden calistirilmasi gerekirdi.

    Bunun yerine BI katmani icin ozel, dar ve anlamli bir tablo uretiyoruz:
    her basvuru icin skor, karar, risk bandi ve segment alanlari. Veri
    ambarciligi terimiyle bu bir "semantic layer" (anlam katmani) - ham veriyi
    is diliyle konusan bir yapiya cevirir.

NE ICERIR:
    - model ciktisi   : temerrut olasiligi, skorkart puani, karar, risk bandi
    - gerceklesme     : gercek temerrut (model degerlendirmesi icin)
    - segment alanlari: yas bandi, egitim, gelir bandi, bolge, meslek
    - tutar           : kredi tutari ve beklenen kar/zarar katkisi

    Bu haliyle panel; onay oranini bolgeye gore, temerrudu yas bandina gore,
    kari risk dilimine gore kesip bicebilir - Python calistirmadan.

Calistirmak icin (proje kokunden):
    .\\.venv\\Scripts\\python.exe -m src.izleme_tablosu
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import text

from src.config import MODELS_DIR, REPORTS_DIR, TARGET
from src.data import kolon_tipleri, model_input_yukle, motor, veri_bol
from src.kar_optimizasyonu import LGD, MARJ

TABLO = "monitoring.skor_portfoy"


def _esik() -> float:
    yol = REPORTS_DIR / "adalet_analizi.json"
    if yol.exists():
        return float(json.loads(yol.read_text(encoding="utf-8"))["esikler"]["cinsiyetsiz"])
    return 0.162


def main() -> None:
    print("=" * 78)
    print("SKORLANMIS PORTFOY TABLOSU")
    print("=" * 78)

    df = model_input_yukle()
    train, valid, test = veri_bol(df)

    model = xgb.XGBClassifier()
    model.load_model(str(MODELS_DIR / "xgboost_adil.json"))
    ozellikler = list(model.get_booster().feature_names)
    esik = _esik()

    # Hangi basvurunun hangi kumede oldugunu isaretliyoruz.
    # Panelde bu KRITIK: egitim kumesindeki performansa bakip "modelim
    # harika" demek yaniltici olur; is birimi test kumesine bakmali.
    kume = pd.Series("egitim", index=df.index, dtype=object)
    kume.loc[valid.index] = "dogrulama"
    kume.loc[test.index] = "test"

    print(f"skorlaniyor: {len(df):,} basvuru...")
    olasilik = model.predict_proba(df[ozellikler])[:, 1]

    karar = np.where(olasilik < esik, "ONAY", "RED")
    oder = (df[TARGET].values == 0)

    # Beklenen kar katkisi: yalnizca ONAY alanlar icin.
    # Reddedilenler portfoye girmedigi icin katkilari sifirdir.
    kar = np.where(
        karar == "ONAY",
        np.where(oder,
                 df["amt_credit"].values * MARJ,
                 -df["amt_credit"].values * LGD),
        0.0,
    )

    cikti = pd.DataFrame({
        "sk_id_curr": df["sk_id_curr"].values,
        "kume": kume.values,
        "temerrut_olasiligi": olasilik.round(6),
        "risk_dilimi": pd.qcut(olasilik, 10, labels=False, duplicates="drop") + 1,
        "risk_bandi": pd.cut(
            olasilik, [-0.001, 0.02, 0.05, 0.10, 0.20, 1.0],
            labels=["çok düşük", "düşük", "orta", "yüksek", "çok yüksek"],
        ).astype(str),
        "karar": karar,
        "gercek_temerrut": df[TARGET].values,
        "amt_credit": df["amt_credit"].values,
        "amt_income_total": df["amt_income_total"].values,
        "beklenen_kar": kar.round(2),
        # --- segment alanlari ---
        "yas_bandi": pd.cut(
            df["age_years"].values, [0, 25, 35, 45, 55, 120],
            labels=["18-25", "26-35", "36-45", "46-55", "56+"],
        ).astype(str),
        "gelir_bandi": pd.qcut(
            df["amt_income_total"], 5,
            labels=["çok düşük", "düşük", "orta", "yüksek", "çok yüksek"],
            duplicates="drop",
        ).astype(str),
        "cinsiyet": df["code_gender"].astype(str).values,
        "egitim": df["name_education_type"].astype(str).values,
        "aile_durumu": df["name_family_status"].astype(str).values,
        "meslek": df["occupation_type"].astype(str).values,
        "sozlesme_turu": df["name_contract_type"].astype(str).values,
        "bolge_puani": df["region_rating_client_w_city"].values,
        "calismiyor": df["flag_not_employed"].values,
    })

    print(f"  onay orani     : %{100*(cikti['karar']=='ONAY').mean():.1f}")
    print(f"  ortalama tahmin: %{100*cikti['temerrut_olasiligi'].mean():.2f}")
    print(f"  esik           : {esik:.4f}")

    with motor().begin() as baglanti:
        baglanti.execute(text("CREATE SCHEMA IF NOT EXISTS monitoring"))
        baglanti.execute(text(f"DROP TABLE IF EXISTS {TABLO}"))

    # chunksize: 307 bin satiri tek seferde gondermek bellek ve ag acisindan
    # verimsizdir; parcalar halinde yaziyoruz.
    cikti.to_sql(
        "skor_portfoy", motor(), schema="monitoring",
        if_exists="append", index=False, chunksize=20000, method="multi",
    )

    with motor().begin() as baglanti:
        baglanti.execute(text(
            f"ALTER TABLE {TABLO} ADD PRIMARY KEY (sk_id_curr)"))
        for k in ("kume", "karar", "risk_bandi", "yas_bandi"):
            baglanti.execute(text(
                f"CREATE INDEX idx_skor_portfoy_{k} ON {TABLO} ({k})"))
        baglanti.execute(text(f"ANALYZE {TABLO}"))

    print(f"\n{TABLO} yazildi: {len(cikti):,} satir x {cikti.shape[1]} kolon")


if __name__ == "__main__":
    main()
