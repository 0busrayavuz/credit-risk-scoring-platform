"""Kar bazli kesim noktasi optimizasyonu.

BU ADIM PROJENIN IS TARAFINDAKI ZIRVESI.
"Modelim 0.78 AUC aldi" cumlesi bir is biriminde hicbir sey ifade etmez.
"Bu modelle su esikten kesersek portfoy karimiz %X artiyor" cumlesi ise
dogrudan karar uretir. Bu dosya birinciyi ikinciye cevirir.

TEMEL FIKIR - HATA MALIYETLERI SIMETRIK DEGILDIR:
    Bir bankanin yapabilecegi iki hata var:
      1. Odeyecek musteriyi reddetmek  -> kacirilan kar
      2. Odemeyecek musteriye kredi vermek -> anaparanin buyuk kismi zarar
    Ikincisi cok daha pahalidir. 100.000 TL'lik bir krediden yillar icinde
    kazanilan faiz marji, batan bir 100.000 TL'nin yanina yaklasmaz.

    Bu asimetri yuzunden "olasilik 0.5'ten buyukse reddet" kurali YANLISTIR.
    0.5 esigi, iki hatanin esit maliyetli oldugunu varsayar. Kredi riskinde
    dogru esik cok daha DUSUKTUR - cunku batigin maliyeti agir basar.

MODEL (basit ve acikca varsayimsal):
    Onayla + oderse : kar    = kredi_tutari x MARJ
    Onayla + batarsa: zarar  = kredi_tutari x LGD
    Reddet          : 0

    MARJ: kredi omru boyunca elde kalan net marj (fonlama ve operasyon
          maliyetleri dusulmus).
    LGD (Loss Given Default): battiginda kaybedilen kismin orani. Teminatsiz
          tuketici kredilerinde sektorde tipik olarak %50-80 araliginda anilir.

    Bu iki sayi VARSAYIMDIR, veriden gelmez. O yuzden tek bir sonuca
    guvenmiyoruz: asagida duyarlilik analizi ile esigin bu varsayimlara
    ne kadar bagli oldugunu da olcuyoruz. Gercek bir projede bu degerler
    bankanin kendi fiyatlama ve tahsilat verisinden alinir.

YONTEM - DURUSTLUK KURALI:
    Esik DOGRULAMA kumesinde secilir, sonuc TEST kumesinde raporlanir.
    Esigi test uzerinde secip yine test uzerinde raporlamak, kendi sinavinin
    sorularini kendin yazip yuksek not almaya benzer.

Calistirmak icin (proje kokunden):
    .\\.venv\\Scripts\\python.exe -m src.kar_optimizasyonu
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # ekransiz ortamda dosyaya cizim
import matplotlib.pyplot as plt  # noqa: E402
import xgboost as xgb  # noqa: E402
from optbinning import Scorecard  # noqa: E402

from src.config import MODELS_DIR, REPORTS_DIR, TARGET  # noqa: E402
from src.data import kolon_tipleri, model_input_yukle, veri_bol  # noqa: E402

# --- Ekonomik varsayimlar (temel senaryo) -----------------------------------
MARJ = 0.12   # oderse: kredi tutarinin %12'si net kar
LGD = 0.65    # batarsa: kredi tutarinin %65'i zarar

# --- Gorsel kimlik (dogrulanmis palet) --------------------------------------
# Renkler dataviz paletinin 1. ve 2. kategorik yuvasi; validate_palette.js ile
# hem acik hem koyu modda dogrulandi (renk korlugu ayrimi dE 24.7, esik 8).
RENK_1 = "#2a78d6"   # mavi   - scorecard
RENK_2 = "#eb6834"   # turuncu - XGBoost
YUZEY = "#fcfcfb"
MUREKKEP = "#0b0b0b"
IKINCIL = "#52514e"
SOLUK = "#898781"
IZGARA = "#e1e0d9"
EKSEN = "#c3c2b7"


def kar_hesapla(y, olasilik, tutar, esik, marj=MARJ, lgd=LGD) -> dict:
    """Verilen esikte portfoy karini ve is metriklerini hesaplar.

    esik: bu olasiligin ALTINDAKI basvurular onaylanir.
    """
    onay = olasilik < esik
    n = len(y)

    if onay.sum() == 0:
        return {"esik": esik, "onay_orani": 0.0, "kar": 0.0, "kar_basvuri": 0.0,
                "onaylanan": 0, "onaylananda_temerrut": 0.0,
                "kacirilan_iyi": int((y == 0).sum()), "onlenen_batik": int((y == 1).sum())}

    oder = onay & (y == 0)
    batar = onay & (y == 1)

    kar = float((tutar[oder] * marj).sum() - (tutar[batar] * lgd).sum())

    return {
        "esik": float(esik),
        "onay_orani": float(onay.mean()),
        "kar": kar,
        "kar_basvuri": kar / n,
        "onaylanan": int(onay.sum()),
        "onaylananda_temerrut": float(y[onay].mean()),
        "kacirilan_iyi": int(((~onay) & (y == 0)).sum()),
        "onlenen_batik": int(((~onay) & (y == 1)).sum()),
    }


def kar_egrisi(y, olasilik, tutar, esikler, marj=MARJ, lgd=LGD) -> pd.DataFrame:
    return pd.DataFrame([kar_hesapla(y, olasilik, tutar, e, marj, lgd) for e in esikler])


def para(x: float) -> str:
    """Buyuk tutarlari okunakli yazar."""
    if abs(x) >= 1e9:
        return f"{x/1e9:.2f} milyar"
    if abs(x) >= 1e6:
        return f"{x/1e6:.1f} milyon"
    return f"{x:,.0f}"


def grafik_ciz(egri_sc, egri_xgb, opt_sc, opt_xgb, yol) -> None:
    """Kar egrisi ve onay/temerrut egrileri - iki panel, tek eksen kurali."""
    fig, (ust, alt) = plt.subplots(
        2, 1, figsize=(10, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.16},
    )
    fig.patch.set_facecolor(YUZEY)

    for eks in (ust, alt):
        eks.set_facecolor(YUZEY)
        eks.grid(True, color=IZGARA, linewidth=0.8, zorder=0)
        eks.set_axisbelow(True)
        for kenar in ("top", "right"):
            eks.spines[kenar].set_visible(False)
        for kenar in ("left", "bottom"):
            eks.spines[kenar].set_color(EKSEN)
            eks.spines[kenar].set_linewidth(1)
        eks.tick_params(colors=SOLUK, labelsize=9)

    # --- Ust panel: kar egrisi ---
    ust.plot(egri_sc["esik"], egri_sc["kar"] / 1e6, color=RENK_1, linewidth=2,
             label="WOE scorecard", zorder=3)
    ust.plot(egri_xgb["esik"], egri_xgb["kar"] / 1e6, color=RENK_2, linewidth=2,
             label="XGBoost", zorder=3)

    # Optimum noktalar. Etiketler el ile konumlandirildi: otomatik yerlesim
    # ikisini ust uste bindiriyordu (ilk cizimde okunmuyordu).
    ust.plot(opt_xgb["esik"], opt_xgb["kar"] / 1e6, "o", color=RENK_2, markersize=10,
             markeredgecolor=YUZEY, markeredgewidth=2, zorder=5)
    ust.annotate(
        f"XGBoost optimum\neşik {opt_xgb['esik']:.3f} · onay %{100*opt_xgb['onay_orani']:.0f}\n"
        f"kâr {opt_xgb['kar']/1e9:.2f} milyar",
        xy=(opt_xgb["esik"], opt_xgb["kar"] / 1e6),
        xytext=(0.30, 2780), color=IKINCIL, fontsize=9.5,
        arrowprops=dict(arrowstyle="-", color=EKSEN, linewidth=1,
                        connectionstyle="angle,angleA=0,angleB=90,rad=4"),
    )
    ust.plot(opt_sc["esik"], opt_sc["kar"] / 1e6, "o", color=RENK_1, markersize=10,
             markeredgecolor=YUZEY, markeredgewidth=2, zorder=5)
    ust.annotate(
        f"scorecard optimum\neşik {opt_sc['esik']:.3f} · onay %{100*opt_sc['onay_orani']:.0f}\n"
        f"kâr {opt_sc['kar']/1e9:.2f} milyar",
        xy=(opt_sc["esik"], opt_sc["kar"] / 1e6),
        xytext=(0.47, 1780), color=IKINCIL, fontsize=9.5,
        arrowprops=dict(arrowstyle="-", color=EKSEN, linewidth=1,
                        connectionstyle="angle,angleA=0,angleB=90,rad=4"),
    )

    # Referans: herkese kredi ver
    hepsi = egri_xgb["kar"].iloc[-1] / 1e6
    ust.axhline(hepsi, color=SOLUK, linewidth=1, linestyle=(0, (4, 4)), zorder=2)
    ust.annotate(f"model yok: herkese onay · {hepsi/1e3:.2f} milyar",
                 xy=(0.60, hepsi), color=SOLUK, fontsize=9.5, va="bottom")
    ust.axhline(0, color=EKSEN, linewidth=1, zorder=2)

    ust.set_ylim(-150, 3150)
    ust.set_ylabel("Portföy kârı (milyon)", color=IKINCIL, fontsize=10)
    ust.set_title(
        "Kesim noktası ile portföy kârı",
        color=MUREKKEP, fontsize=15, fontweight="bold", loc="left", pad=34,
    )
    ust.text(0, 1.035,
             f"test kümesi · 61.503 başvuru · marj %{100*MARJ:.0f} · LGD %{100*LGD:.0f}",
             transform=ust.transAxes, color=SOLUK, fontsize=10)
    ust.legend(frameon=False, labelcolor=IKINCIL, fontsize=10, loc="center right")

    # --- Alt panel: onay orani ve onaylananlarda temerrut (ikisi de yuzde) ---
    alt.plot(egri_xgb["esik"], 100 * egri_xgb["onay_orani"], color=RENK_2,
             linewidth=2, zorder=3)
    alt.plot(egri_xgb["esik"], 100 * egri_xgb["onaylananda_temerrut"], color=RENK_2,
             linewidth=2, linestyle=(0, (5, 3)), zorder=3)
    # Iki egri de yuzde olcusunde oldugu icin tek eksen dogru; ikinci bir
    # y ekseni eklemek (dual axis) grafik tasariminda en yaygin hatadir.
    alt.text(0.46, 92, "onay oranı", color=IKINCIL, fontsize=10)
    alt.text(0.46, 13, "onaylananlarda temerrüt", color=IKINCIL, fontsize=10)
    alt.axvline(opt_xgb["esik"], color=EKSEN, linewidth=1, linestyle=(0, (4, 4)), zorder=2)
    alt.text(opt_xgb["esik"] + 0.012, 46,
             f"seçilen eşik {opt_xgb['esik']:.3f}\nonay %{100*opt_xgb['onay_orani']:.0f} · "
             f"temerrüt %{100*opt_xgb['onaylananda_temerrut']:.2f}",
             color=IKINCIL, fontsize=9.5)

    alt.set_xlabel("Kesim noktası (bu olasılığın altındaki başvurular onaylanır)",
                   color=IKINCIL, fontsize=10)
    alt.set_ylabel("Yüzde", color=IKINCIL, fontsize=10)
    alt.set_ylim(0, 100)

    fig.savefig(yol, dpi=160, bbox_inches="tight", facecolor=YUZEY)
    plt.close(fig)


def main() -> None:
    print("=" * 78)
    print("VERI VE MODELLER")
    print("=" * 78)
    df = model_input_yukle()
    train, valid, test = veri_bol(df)
    ozellikler = kolon_tipleri(df)["ozellikler"]

    scorecard = Scorecard.load(str(MODELS_DIR / "scorecard.pkl"))
    sc_degisken = json.loads(
        (REPORTS_DIR / "scorecard_secilen_degiskenler.json").read_text(encoding="utf-8")
    )
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(MODELS_DIR / "xgboost.json"))

    def hazirla(d):
        X = d[ozellikler].copy()
        for k in X.columns:
            if pd.api.types.is_bool_dtype(X[k]):
                X[k] = X[k].astype("float64")
            elif isinstance(X[k].dtype, pd.CategoricalDtype) and k in sc_degisken:
                pass
        return X

    y_valid, y_test = valid[TARGET].values, test[TARGET].values
    tutar_valid = valid["amt_credit"].values.astype(float)
    tutar_test = test["amt_credit"].values.astype(float)

    X_valid_sc = hazirla(valid)[sc_degisken]
    X_test_sc = hazirla(test)[sc_degisken]
    for k in X_valid_sc.columns:
        if isinstance(X_valid_sc[k].dtype, pd.CategoricalDtype):
            X_valid_sc[k] = X_valid_sc[k].astype(object)
            X_test_sc[k] = X_test_sc[k].astype(object)

    p_sc_valid = scorecard.predict_proba(X_valid_sc)[:, 1]
    p_sc_test = scorecard.predict_proba(X_test_sc)[:, 1]
    p_xgb_valid = xgb_model.predict_proba(valid[ozellikler])[:, 1]
    p_xgb_test = xgb_model.predict_proba(test[ozellikler])[:, 1]

    print(f"dogrulama: {len(y_valid):,} basvuru | test: {len(y_test):,} basvuru")
    print(f"ortalama kredi tutari (test): {tutar_test.mean():,.0f}")
    print(f"\nVarsayimlar: marj %{100*MARJ:.0f} (oderse) | LGD %{100*LGD:.0f} (batarsa)")

    # ------------------------------------------------------------------
    # 1) Esigi DOGRULAMA kumesinde sec
    # ------------------------------------------------------------------
    esikler = np.linspace(0.005, 1.0, 400)
    egri_sc_v = kar_egrisi(y_valid, p_sc_valid, tutar_valid, esikler)
    egri_xgb_v = kar_egrisi(y_valid, p_xgb_valid, tutar_valid, esikler)

    opt_sc_esik = float(egri_sc_v.loc[egri_sc_v["kar"].idxmax(), "esik"])
    opt_xgb_esik = float(egri_xgb_v.loc[egri_xgb_v["kar"].idxmax(), "esik"])

    print("\n" + "=" * 78)
    print("1) OPTIMUM ESIK (dogrulama kumesinde secildi)")
    print("=" * 78)
    print(f"  WOE scorecard : {opt_sc_esik:.3f}")
    print(f"  XGBoost       : {opt_xgb_esik:.3f}")
    print("\n  Not: her iki esik de 0.50'nin cok altinda. Sebep, hatalarin")
    print("  asimetrik maliyeti: bir batik krediyi onlemek, birkac iyi")
    print("  musteriyi reddetmekten daha degerli.")

    # ------------------------------------------------------------------
    # 2) TEST kumesinde raporla (ilk ve son kez bakiliyor)
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("2) TEST KUMESI SONUCLARI (esik disaridan geldi, burada secilmedi)")
    print("=" * 78)

    senaryolar = {
        "model yok - herkese onay": kar_hesapla(y_test, p_xgb_test, tutar_test, 1.01),
        "sabit 0.50 esigi (XGBoost)": kar_hesapla(y_test, p_xgb_test, tutar_test, 0.50),
        "WOE scorecard - optimum": kar_hesapla(y_test, p_sc_test, tutar_test, opt_sc_esik),
        "XGBoost - optimum": kar_hesapla(y_test, p_xgb_test, tutar_test, opt_xgb_esik),
    }

    print(f"{'senaryo':30s} {'onay':>7s} {'temerrut':>9s} {'kar':>14s} {'basvuri basi':>13s}")
    print("-" * 78)
    for ad, s in senaryolar.items():
        print(f"{ad:30s} {100*s['onay_orani']:6.1f}% {100*s['onaylananda_temerrut']:8.2f}% "
              f"{para(s['kar']):>14s} {s['kar_basvuri']:>13,.0f}")

    temel = senaryolar["model yok - herkese onay"]["kar"]
    en_iyi = senaryolar["XGBoost - optimum"]
    sc_opt = senaryolar["WOE scorecard - optimum"]

    print(f"\n  Modelsiz duruma gore kazanc:")
    print(f"    WOE scorecard : {para(sc_opt['kar'] - temel)}  "
          f"(%{100*(sc_opt['kar']-temel)/abs(temel):+.1f})")
    print(f"    XGBoost       : {para(en_iyi['kar'] - temel)}  "
          f"(%{100*(en_iyi['kar']-temel)/abs(temel):+.1f})")
    print(f"\n  XGBoost'un scorecard uzerine kattigi: {para(en_iyi['kar'] - sc_opt['kar'])}")
    print(f"  Onlenen batik kredi (XGBoost): {en_iyi['onlenen_batik']:,}")
    print(f"  Reddedilen iyi musteri        : {en_iyi['kacirilan_iyi']:,}")

    # ------------------------------------------------------------------
    # 3) Duyarlilik analizi
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("3) DUYARLILIK: esik varsayimlara ne kadar bagli?")
    print("=" * 78)
    print("Optimum esik / onay orani, farkli marj ve LGD degerlerinde:\n")

    marjlar = [0.06, 0.09, 0.12, 0.18, 0.25]
    lgdler = [0.45, 0.55, 0.65, 0.75, 0.85]

    print(f"{'':>10s}" + "".join(f"{'LGD %'+str(int(100*l)):>13s}" for l in lgdler))
    for m in marjlar:
        satir = f"marj %{int(100*m):<4d}"
        for l in lgdler:
            e = kar_egrisi(y_valid, p_xgb_valid, tutar_valid, esikler, m, l)
            best = e.loc[e["kar"].idxmax()]
            satir += f"{best['esik']:.3f}/%{100*best['onay_orani']:.0f}".rjust(13)
        print(satir)

    print("\n  Okuma: hucrelerde 'esik / onay orani'.")
    print("  Marj artarsa esik yukselir (daha cok riske girmeye deger),")
    print("  LGD artarsa esik duser (batigin maliyeti agirlasir).")
    print("  Esigin varsayimlara duyarli olmasi modelin degil, ISIN dogasidir -")
    print("  bu yuzden gercek projede bu iki sayi bankanin kendi verisinden gelir.")

    # ------------------------------------------------------------------
    # 4) Grafik
    # ------------------------------------------------------------------
    egri_sc_t = kar_egrisi(y_test, p_sc_test, tutar_test, esikler)
    egri_xgb_t = kar_egrisi(y_test, p_xgb_test, tutar_test, esikler)
    opt_sc_t = kar_hesapla(y_test, p_sc_test, tutar_test, opt_sc_esik)
    opt_xgb_t = kar_hesapla(y_test, p_xgb_test, tutar_test, opt_xgb_esik)

    yol = REPORTS_DIR / "kar_egrisi.png"
    grafik_ciz(egri_sc_t, egri_xgb_t, opt_sc_t, opt_xgb_t, yol)
    print(f"\nGrafik: {yol}")

    (REPORTS_DIR / "kar_optimizasyonu.json").write_text(
        json.dumps(
            {"varsayimlar": {"marj": MARJ, "lgd": LGD},
             "optimum_esik": {"scorecard": opt_sc_esik, "xgboost": opt_xgb_esik},
             "test_sonuclari": senaryolar},
            indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Sonuclar: {REPORTS_DIR / 'kar_optimizasyonu.json'}")


if __name__ == "__main__":
    main()
