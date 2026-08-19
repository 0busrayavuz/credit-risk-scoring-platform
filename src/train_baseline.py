"""Temel cizgi (baseline) modelleri.

AMAC YUKSEK SKOR DEGIL, REFERANS KURMAK.
Sonraki asamalarda scorecard ve XGBoost'u bir seyle kiyaslamamiz gerekiyor.
"Modelim 0.75 AUC aldi" tek basina anlamsizdir; "tek degiskenli basit model
0.70 aliyordu, ben 0.78'e cikardim" anlamlidir.

Uc seviye kuruyoruz:
    1. Tek degisken  : ext_source_mean (veri setinin en guclu tekil sinyali)
    2. El ile secilmis 10 degisken + lojistik regresyon
    3. (sonraki asamalar bunlarin uzerine kurulacak)

Calistirmak icin (proje kokunden):
    .\\.venv\\Scripts\\python.exe -m src.train_baseline
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, REPORTS_DIR, TARGET
from src.data import model_input_yukle, ozet, veri_bol
from src.metrics import degerlendir, dilim_analizi, rapor_yazdir

# El ile secilmis degiskenler.
# Secim kriteri: onceki asamada SQL ile OLCTUGUMUZ sinyal gucu.
# Tahminle degil, veriye bakarak secildiler.
BASELINE_OZELLIKLER = [
    "ext_source_mean",        # dis kredi skorlari ortalamasi - en guclu tekil sinyal
    "ext_source_min",         # en dusuk dis skor (kotu senaryo gostergesi)
    "cc_utilization_avg",     # kredi karti limit kullanimi - olcumde 3,1x ayrim
    "bb_dpd_ratio",           # dis kredilerde gecikmeli ay orani - 2,3x
    "prev_refused_ratio",     # gecmis basvurularda red orani - 2,25x
    "inst_late_ratio",        # taksit gecikme orani - 2,0x
    "credit_income_ratio",    # kredi / gelir
    "annuity_income_ratio",   # taksit / gelir (DTI)
    "age_years",              # yas
    "flag_not_employed",      # calismiyor bayragi (buyuk cogunlugu emekli)
]


def pipeline_kur() -> Pipeline:
    """Lojistik regresyon icin on isleme + model zinciri.

    add_indicator=True KRITIK:
        Onceki asamada olctuk ki eksik degerin KENDISI bilgi tasiyor
        (dis kredi gecmisi yok -> %10,12 temerrut, ortalamanin ustu).
        Sadece medyanla doldurursak bu bilgiyi silmis oluruz.
        add_indicator, her eksikli kolon icin "bu deger eksikti mi?"
        seklinde ikili bir kolon daha ekler; model bosluk bilgisini kaybetmez.

    StandardScaler:
        Lojistik regresyon olcek duyarlidir. age_years (0-70) ile
        cc_utilization_avg (0-1) ayni olceklerde degil; olceklemeden
        katsayilar karsilastirilamaz ve optimizasyon yavas yakinsar.

    class_weight KULLANMIYORUZ:
        Dengesizligi duzeltmek AUC'yi pek degistirmez ama OLASILIKLARI bozar -
        model gercekte %8 olan riski %50 gibi tahmin etmeye baslar.
        Ilerideki kar hesabi dogru kalibre edilmis olasiliklara dayanacagi
        icin siniflari oldugu gibi birakiyoruz.
    """
    return Pipeline(
        steps=[
            ("doldur", SimpleImputer(strategy="median", add_indicator=True)),
            ("olcekle", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    # lbfgs coklu degiskende kararli ve hizlidir
                    solver="lbfgs",
                ),
            ),
        ]
    )


def main() -> None:
    baslangic = time.time()

    print("=" * 72)
    print("VERI YUKLENIYOR")
    print("=" * 72)
    df = model_input_yukle()
    print(f"features.model_input -> {df.shape[0]:,} satir x {df.shape[1]} kolon")
    print(f"bellek: {df.memory_usage(deep=True).sum() / 1024**2:.0f} MB")

    train, valid, test = veri_bol(df)
    print()
    ozet(train, "egitim")
    ozet(valid, "dogrulama")
    ozet(test, "test")
    print("\nNot: test kumesine bu asamada BAKILMIYOR. Sadece en sonda, bir kez.")

    sonuclar = []

    # ------------------------------------------------------------------
    # 1. Tek degiskenli referans
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("1) TEK DEGISKENLI REFERANS: ext_source_mean")
    print("=" * 72)
    print("Model yok - degiskenin kendisini skor olarak kullaniyoruz.")
    print("Dis skor DUSUKSE risk YUKSEK oldugu icin isaretini ters ceviriyoruz.")

    # Eksik olanlari medyanla dolduruyoruz ki tum satirlar degerlendirilebilsin.
    tek = valid["ext_source_mean"].fillna(train["ext_source_mean"].median())
    sonuclar.append(degerlendir(valid[TARGET], -tek, "tek degisken (ext_source_mean)"))

    # ------------------------------------------------------------------
    # 2. Lojistik regresyon
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"2) LOJISTIK REGRESYON: el ile secilmis {len(BASELINE_OZELLIKLER)} degisken")
    print("=" * 72)

    X_train = train[BASELINE_OZELLIKLER].astype("float64")
    X_valid = valid[BASELINE_OZELLIKLER].astype("float64")
    y_train, y_valid = train[TARGET], valid[TARGET]

    eksik = X_train.isna().mean().sort_values(ascending=False)
    print("Eksik oranlari:")
    for k, v in eksik.items():
        if v > 0:
            print(f"  {k:24s} {100 * v:5.1f}%")

    pipe = pipeline_kur()
    pipe.fit(X_train, y_train)
    skor_valid = pipe.predict_proba(X_valid)[:, 1]
    sonuclar.append(degerlendir(y_valid, skor_valid, "lojistik regresyon (10 degisken)"))

    # Katsayilar: yonleri is mantigina uyuyor mu?
    # Bu kontrol atlanmamali - isaret ters cikan bir degisken ya veri
    # sorununa ya da coklu dogrusal baglantiya (multicollinearity) isarettir.
    model = pipe.named_steps["model"]
    ad_listesi = pipe.named_steps["doldur"].get_feature_names_out(BASELINE_OZELLIKLER)
    katsayi = (
        pd.DataFrame({"degisken": ad_listesi, "katsayi": model.coef_[0]})
        .assign(mutlak=lambda d: d["katsayi"].abs())
        .sort_values("mutlak", ascending=False)
    )
    print("\nKatsayilar (buyukten kucuge, + = riski ARTIRIR):")
    for _, r in katsayi.head(12).iterrows():
        yon = "+" if r["katsayi"] > 0 else "-"
        print(f"  {yon} {r['degisken']:34s} {r['katsayi']:+.4f}")

    # ------------------------------------------------------------------
    # Sonuclar
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("DOGRULAMA KUMESI SONUCLARI")
    print("=" * 72)
    ozet_df = rapor_yazdir(sonuclar)

    print("\n" + "=" * 72)
    print("SKOR DILIMLERI - lojistik regresyon (en riskli dilim ustte)")
    print("=" * 72)
    dilimler = dilim_analizi(y_valid, skor_valid, 10)
    print(dilimler.to_string(index=False))

    # Sonuclari diske yaz: sonraki modellerle karsilastirabilmek icin.
    cikti = REPORTS_DIR / "baseline_sonuclar.json"
    cikti.write_text(
        json.dumps(ozet_df.to_dict("records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSonuclar kaydedildi: {cikti}")
    print(f"Toplam sure: {time.time() - baslangic:.1f} sn")


if __name__ == "__main__":
    main()
