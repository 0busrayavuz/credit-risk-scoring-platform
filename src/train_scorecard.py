"""WOE tabanli skorkart (scorecard) modeli.

BU NEDEN BANKALARIN STANDART YONTEMI:
    Cikti insan tarafindan okunabilir bir PUAN TABLOSUDUR:
        "dis skor 0.68 uzeri        -> +14 puan"
        "kart limit kullanimi %57+  ->  -6 puan"
    Bir kredi uzmani bu tabloyu okuyup "bu musteri neden 512 puan aldi?"
    sorusunu satir satir cevaplayabilir. Regulator de ayni tabloyu denetler.
    XGBoost'un yuzlerce agacini kimse bu sekilde denetleyemez; bankacilikta
    aciklanabilirlik cogu zaman tercih degil, mevzuat geregidir.

WOE (Weight of Evidence):
    Her degisken dilimlere ayrilir, her dilim icin
        WOE = ln(dilimdeki iyi musteri orani / kotu musteri orani)
    hesaplanir ve ham deger yerine bu kullanilir.
    Kazanimi: baseline'da gordugumuz TERS U sorununu cozer (credit_income_ratio
    riski once artirip sonra azaltiyordu; dogrusal model bunu temsil edemiyordu).
    Ayrica aykiri degerler dilim icinde eridigi icin model saglamlasir.

IV (Information Value) - degisken secme olcusu:
    < 0.02 kullanissiz | 0.02-0.10 zayif | 0.10-0.30 orta
    0.30-0.50 guclu    | > 0.50 SUPHELI (genelde veri sizintisi isareti)

UC ASAMALI SECIM (bu projenin ayirt edici kismi):
    1. IV filtresi          -> zayif degiskenleri ele
    2. Korelasyon budama    -> birbirinin ayni olan degiskenlerden birini ele
    3. Isaret duzeltme      -> katsayisi is mantigina aykiri olanlari ele
    Ucuncu adim olmadan uretilen skorkartta 108 degiskenin 35'inin isareti
    ters cikmisti (riskli musteriye daha cok puan). Ayrintili gerekce:
    src/scorecard_secim.py

Calistirmak icin (proje kokunden):
    .\\.venv\\Scripts\\python.exe -m src.train_scorecard
"""

from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd

# optbinning 0.20, sklearn'un 'force_all_finite' parametresini kullaniyor;
# sklearn 1.6+ bunun adini degistirdigi icin her degiskende bir uyari basiyor.
# Islevsel bir sorun degil (bu yuzden sklearn'u <1.8'e sabitledik), sadece
# ciktiyi okunmaz hale getiriyor. Yalnizca BU uyariyi susturuyoruz -
# tum uyarilari kapatmak kotu fikirdir, gercek sorunlari da gizler.
warnings.filterwarnings(
    "ignore", message=".*force_all_finite.*", category=FutureWarning
)

from optbinning import BinningProcess, Scorecard  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TARGET  # noqa: E402
from src.data import kolon_tipleri, model_input_yukle, ozet, veri_bol  # noqa: E402
from src.metrics import degerlendir, dilim_analizi, rapor_yazdir  # noqa: E402
from src.scorecard_secim import (  # noqa: E402
    isaret_duzelt,
    kalite_raporu,
    korelasyon_buda,
)

IV_MIN = 0.02          # bu esigin altindaki degiskenler gurultu ekler
IV_MAX = 0.60          # ustu sizinti supheli - kontrol edilmeden alinmaz
KORELASYON_ESIK = 0.75


def veri_hazirla(df: pd.DataFrame, ozellikler: list[str], kategorik: list[str]) -> pd.DataFrame:
    """optbinning'in bekledigi tiplere donusturur.

    - bool -> float (optbinning sayisal bekler)
    - category -> object (NaN korunur; optbinning eksikleri ayri dilim yapar)
    """
    X = df[ozellikler].copy()
    for k in ozellikler:
        if pd.api.types.is_bool_dtype(X[k]):
            X[k] = X[k].astype("float64")
        elif k in kategorik:
            X[k] = X[k].astype(object)
    return X


def puan_yonu_denetle(puan_tablosu: pd.DataFrame) -> pd.DataFrame:
    """Her degisken icin puanlarin is mantigina uygun yonde olup olmadigini olcer.

    Beklenen: dilimdeki temerrut orani ARTTIKCA puan AZALMALI.
    Yani (temerrut orani, puan) korelasyonu NEGATIF olmali.
    Bu denetim, ilk surumde 35 degiskende yakalanan yon hatasinin
    bir daha sessizce gecmemesi icin eklendi.
    """
    satirlar = []
    for ad, alt in puan_tablosu.groupby("Variable"):
        # Ozel/eksik dilimleri disarida birakiyoruz: bunlarin "temerrut orani"
        # bazen 0 kayittan hesaplandigi icin korelasyonu bozar.
        g = alt[~alt["Bin"].astype(str).isin(["Special", "Missing"])]
        g = g[g["Count"] > 0]
        if len(g) < 2:
            continue
        r = np.corrcoef(g["Event rate"], g["Points"])[0, 1]
        satirlar.append({"degisken": ad, "yon_korelasyon": r,
                         "durum": "DOGRU" if r < 0 else "TERS"})
    return pd.DataFrame(satirlar)


def main() -> None:
    baslangic = time.time()

    print("=" * 78)
    print("VERI")
    print("=" * 78)
    df = model_input_yukle()
    train, valid, test = veri_bol(df)
    ozet(train, "egitim")
    ozet(valid, "dogrulama")

    tipler = kolon_tipleri(df)
    ozellikler, kategorik = tipler["ozellikler"], tipler["kategorik"]
    print(f"\naday degisken: {len(ozellikler)}")

    X_train = veri_hazirla(train, ozellikler, kategorik)
    X_valid = veri_hazirla(valid, ozellikler, kategorik)
    y_train, y_valid = train[TARGET].values, valid[TARGET].values

    # =================================================================
    # ASAMA 1 - Binning ve IV filtresi
    # =================================================================
    print("\n" + "=" * 78)
    print("ASAMA 1: WOE BINNING + IV FILTRESI")
    print("=" * 78)
    print(f"Her degisken icin optimal dilimler araniyor (IV {IV_MIN} - {IV_MAX})...")

    binning = BinningProcess(
        variable_names=ozellikler,
        categorical_variables=kategorik,
        selection_criteria={"iv": {"min": IV_MIN, "max": IV_MAX, "strategy": "highest"}},
    )
    binning.fit(X_train, y_train)

    ozet_bin = binning.summary()
    secilen = ozet_bin[ozet_bin["selected"]].sort_values("iv", ascending=False)
    iv_serisi = secilen.set_index("name")["iv"]
    print(f"  IV filtresi: {len(ozellikler)} -> {len(secilen)} degisken")

    print("\n  EN GUCLU 12 DEGISKEN:")
    for _, r in secilen.head(12).iterrows():
        guc = "zayif" if r["iv"] < 0.10 else "orta" if r["iv"] < 0.30 else "guclu"
        print(f"    {r['name']:38s} IV={r['iv']:.4f}  {guc}")

    secilen[["name", "dtype", "iv", "js", "n_bins"]].to_csv(
        REPORTS_DIR / "scorecard_iv_tablosu.csv", index=False, encoding="utf-8"
    )

    # =================================================================
    # ASAMA 2 - Korelasyon budama
    # =================================================================
    print("\n" + "=" * 78)
    print("ASAMA 2: KORELASYON BUDAMA")
    print("=" * 78)
    X_woe = binning.transform(X_train, metric="woe")
    kalan = korelasyon_buda(X_woe, iv_serisi, esik=KORELASYON_ESIK)

    # =================================================================
    # ASAMA 3 - Isaret duzeltme
    # =================================================================
    print("\n" + "=" * 78)
    print("ASAMA 3: KATSAYI ISARETI DUZELTME")
    print("=" * 78)
    nihai, lr = isaret_duzelt(X_woe, y_train, kalan, random_state=RANDOM_STATE)

    print(f"\n  SECIM OZETI: {len(ozellikler)} -> {len(secilen)} (IV) "
          f"-> {len(kalan)} (korelasyon) -> {len(nihai)} (isaret)")

    kalite = kalite_raporu(X_woe, y_train, nihai, lr)
    print("\n  EN ONEMLI 12 DEGISKEN (katsayi buyuklugune gore):")
    for _, r in kalite.head(12).iterrows():
        print(f"    {r['degisken']:38s} katsayi={r['katsayi']:+.4f}")

    # =================================================================
    # Nihai skorkart
    # =================================================================
    print("\n" + "=" * 78)
    print("NIHAI SKORKART EGITIMI")
    print("=" * 78)

    nihai_binning = BinningProcess(
        variable_names=nihai,
        categorical_variables=[k for k in kategorik if k in nihai],
    )
    scorecard = Scorecard(
        binning_process=nihai_binning,
        estimator=LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs"
        ),
        scaling_method="pdo_odds",
        # pdo=20, odds=20, 600 puan: "600 puan iyi/kotu oraninin 20'ye 1
        # oldugu nokta; her 20 puan bu orani ikiye katlar." Bankacilikta standart.
        scaling_method_params={"pdo": 20, "odds": 20, "scorecard_points": 600},
        # DIKKAT: optbinning'de bu parametrenin yonu sezgiye aykiri.
        # True denendi, puan-temerrut iliskisi POZITIF cikti (yanlis).
        # Dogrusu False. Asagidaki yon kontrolu bunu her calistirmada dogrular.
        reverse_scorecard=False,
    )
    scorecard.fit(X_train[nihai], y_train)

    # =================================================================
    # Degerlendirme
    # =================================================================
    print("\n" + "=" * 78)
    print("DOGRULAMA KUMESI SONUCLARI")
    print("=" * 78)

    pd_valid = scorecard.predict_proba(X_valid[nihai])[:, 1]
    sonuclar = [degerlendir(y_valid, pd_valid, f"WOE scorecard ({len(nihai)} degisken)")]

    baseline_dosya = REPORTS_DIR / "baseline_sonuclar.json"
    if baseline_dosya.exists():
        sonuclar = json.loads(baseline_dosya.read_text(encoding="utf-8")) + sonuclar
    ozet_df = rapor_yazdir(sonuclar)

    print("\n" + "=" * 78)
    print("SKOR DILIMLERI (en riskli ustte)")
    print("=" * 78)
    print(dilim_analizi(y_valid, pd_valid, 10).to_string(index=False))

    # =================================================================
    # Puan tablosu ve yon denetimleri
    # =================================================================
    puan_tablosu = scorecard.table(style="detailed")
    puan_tablosu.to_csv(
        REPORTS_DIR / "scorecard_puan_tablosu.csv", index=False, encoding="utf-8"
    )

    puanlar = scorecard.score(X_valid[nihai])

    # Denetim 1: toplam puanin yonu
    yon = np.corrcoef(puanlar, y_valid)[0, 1]
    if yon > 0:
        raise RuntimeError(
            f"PUAN YONU TERS! puan-temerrut korelasyonu = {yon:+.4f} (negatif olmali)."
        )

    # Denetim 2: her degiskenin kendi icindeki yonu
    yon_df = puan_yonu_denetle(puan_tablosu)
    ters_sayi = int((yon_df["durum"] == "TERS").sum())

    print("\n" + "=" * 78)
    print("PUAN TABLOSU DENETIMI")
    print("=" * 78)
    print(f"  toplam puan yonu     : {yon:+.4f}  (negatif = DOGRU)")
    print(f"  degisken yon denetimi: {len(yon_df) - ters_sayi}/{len(yon_df)} DOGRU")
    if ters_sayi:
        print("  TERS kalanlar:")
        for _, r in yon_df[yon_df["durum"] == "TERS"].iterrows():
            print(f"    {r['degisken']:38s} r={r['yon_korelasyon']:+.3f}")
    else:
        print("  tum degiskenlerde puan, temerrut arttikca AZALIYOR - is mantigina uygun")

    print(f"\n  puan araligi: {puanlar.min():.0f} - {puanlar.max():.0f} "
          f"(ortalama {puanlar.mean():.0f}) | yuksek puan = dusuk risk")

    p = pd.DataFrame({"puan": puanlar, "y": y_valid})
    p["bant"] = pd.cut(p["puan"], bins=10)
    bant = (
        p.groupby("bant", observed=True)
        .agg(musteri=("y", "size"),
             temerrut_yuzde=("y", lambda s: round(100 * s.mean(), 2)))
        .reset_index()
    )
    print("\nPUAN BANDINA GORE TEMERRUT:")
    print(bant.to_string(index=False))

    scorecard.save(str(MODELS_DIR / "scorecard.pkl"))
    (REPORTS_DIR / "scorecard_secilen_degiskenler.json").write_text(
        json.dumps(nihai, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ozet_df.to_json(
        REPORTS_DIR / "model_karsilastirma.json",
        orient="records", indent=2, force_ascii=False,
    )

    print("\nKaydedilenler:")
    for f in ["models/scorecard.pkl", "reports/scorecard_iv_tablosu.csv",
              "reports/scorecard_puan_tablosu.csv",
              "reports/scorecard_secilen_degiskenler.json",
              "reports/model_karsilastirma.json"]:
        print(f"  {f}")
    print(f"\nToplam sure: {time.time() - baslangic:.0f} sn")


if __name__ == "__main__":
    main()
