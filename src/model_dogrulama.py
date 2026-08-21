"""Model dogrulama: adil karsilastirma, guven araliklari ve kalibrasyon.

BU DOSYA UC SORUYU CEVAPLAR:

  1. XGBoost gercekten scorecard'dan IYI MI?
     Simdiye kadar "0.7807 > 0.7621, demek ki iyi" dedik. Ama bu fark
     ORNEKLEME GURULTUSU olabilir. Guven araligi hesaplanmadan iki model
     arasinda "daha iyi" demek istatistiksel olarak dayanaksizdir.
     Yontem: test kumesi uzerinde EŞLI (paired) bootstrap. Ayni yeniden
     ornekleme iki modele de uygulanir, farkin dagilimi cikarilir. Esli olmasi
     onemli: iki model AYNI musterileri skorluyor, bagimsiz varsaymak
     belirsizligi oldugundan buyuk gosterirdi.

  2. Farkin ne kadari MODEL SINIFINDAN, ne kadari DEGISKEN SAYISINDAN?
     Scorecard 50, XGBoost 224 degisken kullaniyor. Bu haliyle karsilastirma
     esit kosullu degil. Cozum: XGBoost'u scorecard'in TAM OLARAK ayni 50
     degiskeniyle egitip tekrar olcmek.

  3. Olasiliklar KALIBRE MI?
     Kar bazli esik, modelin "%10 risk" dedigi musterilerin gercekten %10
     oraninda batmasina dayanir. Siralama dogru olsa bile olasiliklar kayik
     olabilir. Brier skoru bunu tek sayiya indirger; guvenilirlik egrisi
     NEREDE kayik oldugunu gosterir.

Calistirmak icin (proje kokunden):
    .\\.venv\\Scripts\\python.exe -m src.model_dogrulama
"""

from __future__ import annotations

import json
import warnings

import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from optbinning import Scorecard  # noqa: E402

from src.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TARGET  # noqa: E402
from src.data import model_input_yukle, veri_bol  # noqa: E402
from src.train_xgboost import PARAMETRELER  # noqa: E402

BOOTSTRAP = 2000

RENK_1 = "#2a78d6"
RENK_2 = "#eb6834"
YUZEY = "#fcfcfb"
MUREKKEP = "#0b0b0b"
IKINCIL = "#52514e"
SOLUK = "#898781"
IZGARA = "#e1e0d9"
EKSEN = "#c3c2b7"


def bootstrap_auc(y, skor, n=BOOTSTRAP, tohum=RANDOM_STATE) -> tuple[float, float, float]:
    """Test AUC'si icin %95 bootstrap guven araligi."""
    rng = np.random.default_rng(tohum)
    y, skor = np.asarray(y), np.asarray(skor)
    degerler = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        # Cok nadir de olsa tum ornek tek sinif olabilir; o turu atlıyoruz.
        if len(np.unique(y[idx])) < 2:
            degerler[i] = np.nan
            continue
        degerler[i] = roc_auc_score(y[idx], skor[idx])
    degerler = degerler[~np.isnan(degerler)]
    return (float(roc_auc_score(y, skor)),
            float(np.percentile(degerler, 2.5)),
            float(np.percentile(degerler, 97.5)))


def esli_bootstrap_fark(y, skor_a, skor_b, n=BOOTSTRAP, tohum=RANDOM_STATE) -> dict:
    """Iki modelin AUC farki icin esli bootstrap.

    ESLI olmasi kritik: ayni yeniden ornekleme her iki modele uygulanir.
    Bagimsiz bootstrap yapmak, iki modelin AYNI musterileri skorladigi
    gercegini yok sayar ve belirsizligi oldugundan buyuk gosterir.
    """
    rng = np.random.default_rng(tohum)
    y, a, b = np.asarray(y), np.asarray(skor_a), np.asarray(skor_b)
    farklar = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            farklar[i] = np.nan
            continue
        farklar[i] = roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx])
    farklar = farklar[~np.isnan(farklar)]
    gozlenen = roc_auc_score(y, a) - roc_auc_score(y, b)
    # Iki yonlu p-degeri yaklasimi: farkin sifiri kapsayan kuyrugu.
    p = 2 * min((farklar <= 0).mean(), (farklar >= 0).mean())
    return {
        "fark": float(gozlenen),
        "alt": float(np.percentile(farklar, 2.5)),
        "ust": float(np.percentile(farklar, 97.5)),
        "p": float(min(p, 1.0)),
    }


def scorecard_hazirla(d: pd.DataFrame, degiskenler: list[str]) -> pd.DataFrame:
    X = d[degiskenler].copy()
    for k in X.columns:
        if pd.api.types.is_bool_dtype(X[k]):
            X[k] = X[k].astype("float64")
        elif isinstance(X[k].dtype, pd.CategoricalDtype):
            X[k] = X[k].astype(object)
    return X


def _eksen(eks):
    eks.set_facecolor(YUZEY)
    eks.grid(True, color=IZGARA, linewidth=0.8, zorder=0)
    eks.set_axisbelow(True)
    for kenar in ("top", "right"):
        eks.spines[kenar].set_visible(False)
    for kenar in ("left", "bottom"):
        eks.spines[kenar].set_color(EKSEN)
        eks.spines[kenar].set_linewidth(1)
    eks.tick_params(colors=SOLUK, labelsize=9)


def grafik(kalibrasyon: dict, auc_ci: pd.DataFrame, yol) -> None:
    fig, (sol, sag) = plt.subplots(1, 2, figsize=(13, 5.6),
                                   gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.26})
    fig.patch.set_facecolor(YUZEY)

    # --- Sol: guvenilirlik egrisi ---
    _eksen(sol)
    sol.plot([0, 0.5], [0, 0.5], color=EKSEN, linewidth=1.4,
             linestyle=(0, (4, 4)), zorder=2)
    sol.text(0.30, 0.325, "kusursuz kalibrasyon", color=SOLUK, fontsize=9,
             rotation=39, ha="center", va="center")
    for (ad, renk) in (("WOE scorecard", RENK_1), ("XGBoost", RENK_2)):
        t, g = kalibrasyon[ad]
        sol.plot(t, g, color=renk, linewidth=2, marker="o", markersize=6,
                 markeredgecolor=YUZEY, markeredgewidth=1.4, label=ad, zorder=3)
    sol.set_xlabel("Modelin tahmin ettiği temerrüt olasılığı", color=IKINCIL, fontsize=10)
    sol.set_ylabel("Gerçekleşen temerrüt oranı", color=IKINCIL, fontsize=10)
    sol.set_title("Kalibrasyon", color=MUREKKEP, fontsize=13.5,
                  fontweight="bold", loc="left", pad=12)
    sol.legend(frameon=False, labelcolor=IKINCIL, fontsize=10, loc="upper left")
    sol.set_xlim(0, 0.52)
    sol.set_ylim(0, 0.52)

    # --- Sag: AUC ve %95 guven araliklari ---
    _eksen(sag)
    d = auc_ci.iloc[::-1]
    y = np.arange(len(d))
    for i, (_, r) in enumerate(d.iterrows()):
        renk = RENK_2 if "XGBoost" in r["model"] else RENK_1
        sag.plot([r["alt"], r["ust"]], [i, i], color=renk, linewidth=2.4,
                 solid_capstyle="butt", zorder=3)
        sag.plot(r["auc"], i, "o", color=renk, markersize=9,
                 markeredgecolor=YUZEY, markeredgewidth=1.6, zorder=4)
        sag.text(r["ust"] + 0.0012, i, f"{r['auc']:.4f}", va="center",
                 color=IKINCIL, fontsize=9.5)
    sag.set_yticks(y)
    sag.set_yticklabels(d["model"], color=IKINCIL, fontsize=10)
    sag.set_xlabel("Test AUC  ·  çizgi: %95 bootstrap güven aralığı",
                   color=IKINCIL, fontsize=10)
    sag.set_title("Belirsizlik", color=MUREKKEP, fontsize=13.5,
                  fontweight="bold", loc="left", pad=12)
    sag.margins(x=0.12, y=0.18)

    fig.suptitle("Model doğrulama: kalibrasyon ve belirsizlik",
                 color=MUREKKEP, fontsize=15.5, fontweight="bold",
                 x=0.062, ha="left", y=1.04)
    fig.text(0.062, 0.985,
             f"Test kümesi · {BOOTSTRAP:,} bootstrap yinelemesi",
             color=SOLUK, fontsize=10, ha="left")
    fig.savefig(yol, dpi=160, bbox_inches="tight", facecolor=YUZEY)
    plt.close(fig)


def main() -> None:
    print("=" * 78)
    print("VERI VE MODELLER")
    print("=" * 78)
    df = model_input_yukle()
    train, valid, test = veri_bol(df)
    y_train, y_test = train[TARGET].values, test[TARGET].values

    sc = Scorecard.load(str(MODELS_DIR / "scorecard_adil.pkl"))
    sc_degisken = json.loads(
        (REPORTS_DIR / "scorecard_secilen_degiskenler_adil.json").read_text(encoding="utf-8")
    )
    xgbm = xgb.XGBClassifier()
    xgbm.load_model(str(MODELS_DIR / "xgboost_adil.json"))
    xgb_degisken = list(xgbm.get_booster().feature_names)

    p_sc = sc.predict_proba(scorecard_hazirla(test, sc_degisken))[:, 1]
    p_xgb = xgbm.predict_proba(test[xgb_degisken])[:, 1]

    print(f"scorecard: {len(sc_degisken)} degisken | XGBoost: {len(xgb_degisken)} degisken")

    # ------------------------------------------------------------------
    # 1) ADIL KARSILASTIRMA - XGBoost, scorecard'in AYNI degiskenleriyle
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("1) ADIL KARSILASTIRMA: XGBoost, scorecard'in ayni 50 degiskeniyle")
    print("=" * 78)
    print("  Soru: fark model sinifindan mi, degisken sayisindan mi geliyor?\n")

    par = dict(PARAMETRELER)
    esit = xgb.XGBClassifier(**par)
    esit.fit(train[sc_degisken], y_train,
             eval_set=[(valid[sc_degisken], valid[TARGET].values)], verbose=False)
    p_esit = esit.predict_proba(test[sc_degisken])[:, 1]
    print(f"  egitim tamam (en iyi tur: {esit.best_iteration})")

    # ------------------------------------------------------------------
    # 2) GUVEN ARALIKLARI
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"2) TEST AUC VE %95 GUVEN ARALIKLARI ({BOOTSTRAP:,} bootstrap)")
    print("=" * 78)

    modeller = [
        (f"WOE scorecard ({len(sc_degisken)} değişken)", p_sc),
        (f"XGBoost ({len(sc_degisken)} değişken)", p_esit),
        (f"XGBoost ({len(xgb_degisken)} değişken)", p_xgb),
    ]
    satirlar = []
    for ad, p in modeller:
        auc, alt, ust = bootstrap_auc(y_test, p)
        satirlar.append({"model": ad, "auc": auc, "alt": alt, "ust": ust,
                         "genislik": ust - alt})
        print(f"  {ad:36s} {auc:.4f}  [{alt:.4f} - {ust:.4f}]  ±{(ust-alt)/2:.4f}")
    auc_ci = pd.DataFrame(satirlar)

    print("\n  ESLI KARSILASTIRMALAR (fark ve %95 aralik):")
    kiyaslar = [
        ("XGBoost(224) - scorecard(50)", p_xgb, p_sc),
        ("XGBoost(50)  - scorecard(50)", p_esit, p_sc),
        ("XGBoost(224) - XGBoost(50)", p_xgb, p_esit),
    ]
    farklar = {}
    for ad, a, b in kiyaslar:
        r = esli_bootstrap_fark(y_test, a, b)
        farklar[ad] = r
        anlamli = "ANLAMLI" if (r["alt"] > 0 or r["ust"] < 0) else "anlamli degil"
        print(f"    {ad:32s} {r['fark']:+.4f}  "
              f"[{r['alt']:+.4f}, {r['ust']:+.4f}]  p={r['p']:.4f}  {anlamli}")

    toplam = farklar["XGBoost(224) - scorecard(50)"]["fark"]
    sinif = farklar["XGBoost(50)  - scorecard(50)"]["fark"]
    print(f"\n  AYRISTIRMA - toplam fark {toplam:+.4f}:")
    print(f"    model sinifi (agac vs dogrusal, ayni degiskenler) : {sinif:+.4f}"
          f"  (%{100*sinif/toplam:.0f})")
    print(f"    ek degiskenler (50 -> {len(xgb_degisken)})              : "
          f"{toplam-sinif:+.4f}  (%{100*(toplam-sinif)/toplam:.0f})")

    # ------------------------------------------------------------------
    # 3) KALIBRASYON
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("3) KALIBRASYON")
    print("=" * 78)
    print("  Modelin '%X risk' dedigi musterilerin gercekten %X'i batiyor mu?\n")

    kalibrasyon = {}
    for ad, p in (("WOE scorecard", p_sc), ("XGBoost", p_xgb)):
        # strategy='quantile': esit BUYUKLUKTE dilimler. 'uniform' olsaydi
        # yuksek risk bolgesinde neredeyse hic gozlem olmayan dilimler cikardi.
        g, t = calibration_curve(y_test, p, n_bins=12, strategy="quantile")
        kalibrasyon[ad] = (t, g)
        sapma = float(np.mean(np.abs(t - g)))
        maks = float(np.max(np.abs(t - g)))
        print(f"  {ad:16s} ortalama sapma {100*sapma:.2f} puan | "
              f"en buyuk sapma {100*maks:.2f} puan")
        for tt, gg in zip(t, g):
            isaret = "üstü" if tt > gg else "altı"
            print(f"      tahmin %{100*tt:5.2f}  ->  gerçek %{100*gg:5.2f}  "
                  f"({100*abs(tt-gg):4.2f} puan {isaret} tahmin)")
        print()

    # ------------------------------------------------------------------
    grafik(kalibrasyon, auc_ci, REPORTS_DIR / "model_dogrulama.png")

    (REPORTS_DIR / "model_dogrulama.json").write_text(
        json.dumps({
            "bootstrap_yineleme": BOOTSTRAP,
            "auc_guven_araliklari": auc_ci.to_dict("records"),
            "esli_farklar": farklar,
            "ayristirma": {"toplam": toplam, "model_sinifi": sinif,
                           "ek_degiskenler": toplam - sinif},
            "kalibrasyon_sapmasi": {
                ad: float(np.mean(np.abs(np.array(t) - np.array(g))))
                for ad, (t, g) in kalibrasyon.items()
            },
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # Bu model yalnizca "adil karsilastirma" sorusunu cevaplamak icin egitildi;
    # servise konmuyor ve baska hicbir yerde kullanilmiyor. Diske yaziyoruz ki
    # analiz tekrar calistirilmadan incelenebilsin, ancak .gitignore ile repo
    # disinda tutuluyor - 2,7 MB'lik tek kullanimlik bir dosyayi versiyon
    # kontrolunde tasimanin anlami yok, script'i calistirmak yeterli.
    esit.save_model(str(MODELS_DIR / "xgboost_esit_karsilastirma.json"))
    print(f"Grafik  : {REPORTS_DIR / 'model_dogrulama.png'}")
    print(f"Sonuclar: {REPORTS_DIR / 'model_dogrulama.json'}")


if __name__ == "__main__":
    main()
