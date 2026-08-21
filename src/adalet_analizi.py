"""Model adaleti ve regulasyon uyumu denetimi.

BU DOSYA NEYI DUZELTIYOR:
    Ilk surumde yalnizca CINSIYET incelenmisti - cunku SHAP grafiginde tesadufen
    goze carpmisti. Sonraki denetimde gorulду ki medeni durum (name_family_status)
    her iki modelde de duruyordu ve ECOA bunu cinsiyetle ayni listede sayiyor.
    Ayrica cinsiyet yalnizca XGBoost'tan cikarilmis, scorecard'da birakilmisti.

    Ders: korumali ozellik denetimi tesadufe birakilamaz. Artik once politika
    tanimlaniyor (src/korumali_ozellikler.py), sonra HER modele ayni sekilde
    uygulaniyor ve TUM izlenen boyutlarda olculuyor.

BU DOSYA DORT SEY OLCER:
    1. MALIYET      : politikayi uygulamak her iki model ailesinde ne kaybettiriyor?
    2. VEKIL SIZINTI: cikarilan her ozellik kalan degiskenlerden tahmin edilebiliyor mu?
    3. ADALET       : izlenen TUM boyutlarda grup metrikleri, oncesi/sonrasi.
    4. KAR ETKISI   : politikanin portfoy karina etkisi.

ADALET METRIKLERI:
    - Demographic parity : gruplarin onay oranlari esit mi?
    - Equal opportunity  : ODEYECEK musterilerin reddedilme orani gruplar arasi
      esit mi? Kredi riskinde en savunulabilir olcut budur, cunku kimseye hak
      etmedigi krediyi vermeyi gerektirmez.
    - Predictive parity  : onaylananlarda gerceklesen temerrut orani esit mi?

    Gruplarin gercek temerrut oranlari farkliyken bu ucu AYNI ANDA saglanamaz
    (imkansizlik teoremi - Kleinberg ve ark.; Chouldechova, 2016-17). Bu projede
    equal opportunity tercih edilmistir.

Calistirmak icin (proje kokunden):
    .\\.venv\\Scripts\\python.exe -m src.adalet_analizi
Onkosul: hem denetim hem adil modeller egitilmis olmali.
    python -m src.train_xgboost          /  --adil
    python -m src.train_scorecard        /  --adil
"""

from __future__ import annotations

import json
import warnings

import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from optbinning import Scorecard  # noqa: E402

from src.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TARGET  # noqa: E402
from src.data import kolon_tipleri, model_input_yukle, veri_bol  # noqa: E402
from src.kar_optimizasyonu import kar_egrisi, kar_hesapla, para  # noqa: E402
from src.korumali_ozellikler import (  # noqa: E402
    IZLENEN_BOYUTLAR,
    KADEME_1,
    cikarilanlar,
    politika_yazdir,
    temiz_ozellikler,
)
from src.metrics import degerlendir, rapor_yazdir  # noqa: E402

ESIK_ARAMA = np.linspace(0.005, 1.0, 400)

RENK_1 = "#2a78d6"
RENK_2 = "#eb6834"
YUZEY = "#fcfcfb"
MUREKKEP = "#0b0b0b"
IKINCIL = "#52514e"
SOLUK = "#898781"
IZGARA = "#e1e0d9"
EKSEN = "#c3c2b7"

# Grafik ve tabloda gosterilecek asgari grup buyuklugu. Veri setinde 'XNA'
# cinsiyet grubu 2 kisiden olusuyor; 40.000 kisilik grupla yan yana gostermek
# istatistiksel olarak anlamsiz, gorsel olarak yaniltici olur.
MIN_GRUP = 100


def boyut_degeri(df: pd.DataFrame, boyut: str) -> pd.Series:
    """Izlenen boyutu grup etiketine cevirir (surekli olanlari bantlar)."""
    if boyut == "age_years":
        return pd.cut(df["age_years"], [0, 30, 40, 50, 60, 120],
                      labels=["≤30", "31-40", "41-50", "51-60", "60+"]).astype(str)
    if boyut == "region_rating_client_w_city":
        return df[boyut].astype("Int64").astype(str).radd("bölge ")
    return df[boyut].astype(str)


def adalet_metrikleri(y, olasilik, grup, esik) -> pd.DataFrame:
    onay = olasilik < esik
    d = pd.DataFrame({"y": y, "onay": onay, "grup": np.asarray(grup)})
    satirlar = []
    for g, alt in d.groupby("grup", observed=True):
        iyi = alt[alt["y"] == 0]
        kotu = alt[alt["y"] == 1]
        satirlar.append({
            "grup": str(g),
            "basvuru": len(alt),
            "onay_orani": alt["onay"].mean(),
            "gercek_temerrut": alt["y"].mean(),
            "iyi_musteri_red_orani": 1 - iyi["onay"].mean() if len(iyi) else np.nan,
            "kotu_musteri_red_orani": 1 - kotu["onay"].mean() if len(kotu) else np.nan,
            "onaylananda_temerrut": alt.loc[alt["onay"], "y"].mean()
            if alt["onay"].any() else np.nan,
        })
    return pd.DataFrame(satirlar).sort_values("basvuru", ascending=False)


def fark(tablo: pd.DataFrame, kolon: str) -> float:
    """Gruplar arasi en buyuk fark (yalnizca anlamli buyuklukteki gruplar)."""
    t = tablo[tablo["basvuru"] >= MIN_GRUP]
    if len(t) < 2:
        return np.nan
    return float(t[kolon].max() - t[kolon].min())


def scorecard_hazirla(d: pd.DataFrame, degiskenler: list[str]) -> pd.DataFrame:
    X = d[degiskenler].copy()
    for k in X.columns:
        if pd.api.types.is_bool_dtype(X[k]):
            X[k] = X[k].astype("float64")
        elif isinstance(X[k].dtype, pd.CategoricalDtype):
            X[k] = X[k].astype(object)
    return X


def grafik(ozet: pd.DataFrame, yol) -> None:
    """Izlenen her boyutta equal-opportunity farki: politika oncesi/sonrasi."""
    d = ozet.iloc[::-1]
    y = np.arange(len(d))
    h = 0.36

    fig, eks = plt.subplots(figsize=(10.5, 1.15 * len(d) + 3.2))
    fig.patch.set_facecolor(YUZEY)
    eks.set_facecolor(YUZEY)
    eks.grid(True, axis="x", color=IZGARA, linewidth=0.8, zorder=0)
    eks.set_axisbelow(True)
    for kenar in ("top", "right"):
        eks.spines[kenar].set_visible(False)
    for kenar in ("left", "bottom"):
        eks.spines[kenar].set_color(EKSEN)
        eks.spines[kenar].set_linewidth(1)
    eks.tick_params(colors=SOLUK, labelsize=9.5)

    eks.barh(y + h / 2 + 0.01, 100 * d["eo_once"], h, color=RENK_1,
             zorder=3, label="politika öncesi")
    eks.barh(y - h / 2 - 0.01, 100 * d["eo_sonra"], h, color=RENK_2,
             zorder=3, label="korumalı özellikler çıkarılmış")

    for i, (_, r) in enumerate(d.iterrows()):
        eks.text(100 * r["eo_once"] + 0.25, i + h / 2 + 0.01,
                 f"%{100*r['eo_once']:.1f}", va="center", color=IKINCIL, fontsize=9)
        eks.text(100 * r["eo_sonra"] + 0.25, i - h / 2 - 0.01,
                 f"%{100*r['eo_sonra']:.1f}", va="center", color=IKINCIL, fontsize=9)

    eks.set_yticks(y)
    eks.set_yticklabels(d["boyut_ad"], color=IKINCIL, fontsize=10.5)
    eks.set_xlabel("Ödeyecek müşterinin reddedilme oranında gruplar arası en büyük fark (%)",
                   color=IKINCIL, fontsize=10)
    # Iki satirlik alt basligin baslikla cakismamasi icin pad genis tutuldu.
    eks.set_title("Korumalı özellik politikasının adalete etkisi",
                  color=MUREKKEP, fontsize=15, fontweight="bold", loc="left", pad=62)
    eks.text(0, 1.012,
             "Küçük değer = daha adil. Ölçüt: equal opportunity — ödeyecek bir müşterinin "
             "reddedilme\nolasılığı grubuna bağlı olmamalı. Test kümesi.",
             transform=eks.transAxes, color=SOLUK, fontsize=9.5, va="bottom")
    # Ust taraftaki cubuklar kisa oldugu icin lejant orada bos alan buluyor.
    eks.legend(frameon=False, labelcolor=IKINCIL, fontsize=10, loc="upper right")
    eks.margins(x=0.16)

    fig.savefig(yol, dpi=160, bbox_inches="tight", facecolor=YUZEY)
    plt.close(fig)


def main() -> None:
    print("=" * 78)
    politika_yazdir()

    df = model_input_yukle()
    train, valid, test = veri_bol(df)
    ozellikler = kolon_tipleri(df)["ozellikler"]
    adil_ozellikler = temiz_ozellikler(ozellikler)
    cikan = cikarilanlar(ozellikler)

    y_train, y_valid, y_test = (train[TARGET].values, valid[TARGET].values,
                                test[TARGET].values)
    tutar_valid = valid["amt_credit"].values.astype(float)
    tutar_test = test["amt_credit"].values.astype(float)

    print(f"\ndegisken: {len(ozellikler)} -> {len(adil_ozellikler)} "
          f"({len(cikan)} korumali ozellik cikarildi)")

    # ------------------------------------------------------------------
    # 1) MALIYET - her iki model ailesinde
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("1) POLITIKANIN MALIYETI")
    print("=" * 78)

    modeller = {}

    for ad, dosya in (("denetim", "xgboost.json"), ("adil", "xgboost_adil.json")):
        m = xgb.XGBClassifier()
        m.load_model(str(MODELS_DIR / dosya))
        f = list(m.get_booster().feature_names)
        modeller[f"xgb_{ad}"] = {
            "model": m, "ozellik": f,
            "p_valid": m.predict_proba(valid[f])[:, 1],
            "p_test": m.predict_proba(test[f])[:, 1],
        }

    for ad, dosya, sec_dosya in (
        ("denetim", "scorecard.pkl", "scorecard_secilen_degiskenler.json"),
        ("adil", "scorecard_adil.pkl", "scorecard_secilen_degiskenler_adil.json"),
    ):
        yol = MODELS_DIR / dosya
        if not yol.exists():
            print(f"  UYARI: {dosya} yok, atlaniyor "
                  f"(once 'python -m src.train_scorecard{' --adil' if ad=='adil' else ''}')")
            continue
        sc = Scorecard.load(str(yol))
        f = json.loads((REPORTS_DIR / sec_dosya).read_text(encoding="utf-8"))
        modeller[f"sc_{ad}"] = {
            "model": sc, "ozellik": f,
            "p_valid": sc.predict_proba(scorecard_hazirla(valid, f))[:, 1],
            "p_test": sc.predict_proba(scorecard_hazirla(test, f))[:, 1],
        }

    sonuclar = []
    etiket = {
        "sc_denetim": "scorecard (denetim)", "sc_adil": "scorecard (adil)",
        "xgb_denetim": "XGBoost (denetim)", "xgb_adil": "XGBoost (adil)",
    }
    for k in ("sc_denetim", "sc_adil", "xgb_denetim", "xgb_adil"):
        if k in modeller:
            sonuclar.append(degerlendir(y_valid, modeller[k]["p_valid"], etiket[k]))
    print()
    rapor_yazdir(sonuclar)

    for aile, d_key, a_key in (("Scorecard", "sc_denetim", "sc_adil"),
                               ("XGBoost", "xgb_denetim", "xgb_adil")):
        if d_key in modeller and a_key in modeller:
            d = degerlendir(y_valid, modeller[d_key]["p_valid"], "")
            a = degerlendir(y_valid, modeller[a_key]["p_valid"], "")
            print(f"\n  {aile:10s} AUC kaybi: {d['auc']-a['auc']:+.4f}  "
                  f"| Gini kaybi: {d['gini']-a['gini']:+.4f}  "
                  f"| degisken: {len(modeller[d_key]['ozellik'])} -> "
                  f"{len(modeller[a_key]['ozellik'])}")

    # ------------------------------------------------------------------
    # 2) VEKIL SIZINTI - cikarilan HER ozellik icin
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("2) VEKIL SIZINTI TESTI")
    print("=" * 78)
    print("  Cikarilan ozellik, kalan degiskenlerden tahmin edilebiliyor mu?")
    print("  Yuksek AUC = silmek sorunu COZMEZ, sadece OLCULEMEZ hale getirir.\n")

    sizinti = {}
    for ozel in cikan:
        seri = train[ozel]
        if pd.api.types.is_numeric_dtype(seri):
            hedef = (seri > seri.median()).astype(int)
            hedef_test = (test[ozel] > seri.median()).astype(int)
            aciklama = f"medyan ustu mu ({seri.median():.0f})"
        else:
            en_yaygin = seri.astype(str).mode()[0]
            hedef = (seri.astype(str) == en_yaygin).astype(int)
            hedef_test = (test[ozel].astype(str) == en_yaygin).astype(int)
            aciklama = f"'{en_yaygin}' mi"
        if hedef.nunique() < 2:
            continue

        v = xgb.XGBClassifier(n_estimators=250, learning_rate=0.1, max_depth=5,
                              tree_method="hist", enable_categorical=True,
                              random_state=RANDOM_STATE, n_jobs=-1)
        v.fit(train[adil_ozellikler], hedef)
        auc = roc_auc_score(hedef_test, v.predict_proba(test[adil_ozellikler])[:, 1])
        sizinti[ozel] = float(auc)

        onem = sorted(v.get_booster().get_score(importance_type="gain").items(),
                      key=lambda t: -t[1])[:3]
        durum = "YUKSEK - izleme sart" if auc > 0.75 else "orta" if auc > 0.65 else "dusuk"
        print(f"  {ozel:24s} [{aciklama}]")
        print(f"      tahmin AUC {auc:.4f}  ->  {durum}")
        print(f"      en guclu vekiller: {', '.join(k for k, _ in onem)}")

    # ------------------------------------------------------------------
    # 3) ADALET - izlenen TUM boyutlarda
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("3) ADALET METRIKLERI (test kumesi, izlenen tum boyutlar)")
    print("=" * 78)

    e_denetim = float(kar_egrisi(y_valid, modeller["xgb_denetim"]["p_valid"],
                                 tutar_valid, ESIK_ARAMA)
                      .pipe(lambda d: d.loc[d["kar"].idxmax(), "esik"]))
    e_adil = float(kar_egrisi(y_valid, modeller["xgb_adil"]["p_valid"],
                              tutar_valid, ESIK_ARAMA)
                   .pipe(lambda d: d.loc[d["kar"].idxmax(), "esik"]))
    print(f"  esikler: denetim {e_denetim:.3f} | adil {e_adil:.3f}\n")

    boyut_ad = {
        "code_gender": "Cinsiyet  (çıkarıldı)",
        "name_family_status": "Medeni durum  (çıkarıldı)",
        "age_years": "Yaş bandı  (modelde kalıyor)",
        "region_rating_client_w_city": "Bölge derecesi  (modelde kalıyor)",
    }

    ozet_satir = []
    detay = {}
    for boyut in IZLENEN_BOYUTLAR:
        if boyut not in test.columns:
            continue
        grup = boyut_degeri(test, boyut)
        once = adalet_metrikleri(y_test, modeller["xgb_denetim"]["p_test"], grup, e_denetim)
        sonra = adalet_metrikleri(y_test, modeller["xgb_adil"]["p_test"], grup, e_adil)
        detay[boyut] = {"once": once.to_dict("records"), "sonra": sonra.to_dict("records")}

        print(f"  --- {boyut_ad.get(boyut, boyut)} ---")
        print(f"    {'grup':14s} {'başvuru':>9s} {'onay %':>8s} {'gerçek tem.':>12s} "
              f"{'iyi müş. red %':>15s}")
        for _, r in sonra[sonra["basvuru"] >= MIN_GRUP].iterrows():
            print(f"    {r['grup']:14s} {r['basvuru']:>9,} {100*r['onay_orani']:>7.1f}% "
                  f"{100*r['gercek_temerrut']:>11.2f}% {100*r['iyi_musteri_red_orani']:>14.1f}%")

        satir = {
            "boyut": boyut, "boyut_ad": boyut_ad.get(boyut, boyut),
            "eo_once": fark(once, "iyi_musteri_red_orani"),
            "eo_sonra": fark(sonra, "iyi_musteri_red_orani"),
            "dp_once": fark(once, "onay_orani"),
            "dp_sonra": fark(sonra, "onay_orani"),
            "pp_once": fark(once, "onaylananda_temerrut"),
            "pp_sonra": fark(sonra, "onaylananda_temerrut"),
        }
        ozet_satir.append(satir)
        print(f"    -> equal opportunity farki: %{100*satir['eo_once']:.2f} "
              f"-> %{100*satir['eo_sonra']:.2f}"
              f"  ({'iyilesti' if satir['eo_sonra'] < satir['eo_once'] else 'kotulesti'})\n")

    ozet = pd.DataFrame(ozet_satir)

    print("  OZET - gruplar arasi en buyuk farklar (kucuk = daha adil):")
    print(f"    {'boyut':32s} {'equal opp.':>20s} {'dem. parity':>20s}")
    for _, r in ozet.iterrows():
        print(f"    {r['boyut_ad']:32s} "
              f"{100*r['eo_once']:8.2f}% -> {100*r['eo_sonra']:6.2f}% "
              f"{100*r['dp_once']:9.2f}% -> {100*r['dp_sonra']:6.2f}%")

    # ------------------------------------------------------------------
    # 4) KAR ETKISI
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("4) KAR ETKISI (test kumesi)")
    print("=" * 78)
    k_d = kar_hesapla(y_test, modeller["xgb_denetim"]["p_test"], tutar_test, e_denetim)
    k_a = kar_hesapla(y_test, modeller["xgb_adil"]["p_test"], tutar_test, e_adil)
    print(f"  denetim modeli : {para(k_d['kar'])}  (onay %{100*k_d['onay_orani']:.1f})")
    print(f"  adil model     : {para(k_a['kar'])}  (onay %{100*k_a['onay_orani']:.1f})")
    d = k_a["kar"] - k_d["kar"]
    print(f"  fark           : {para(d)}  (%{100*d/k_d['kar']:+.2f})")

    # ------------------------------------------------------------------
    grafik(ozet, REPORTS_DIR / "adalet_analizi.png")

    (REPORTS_DIR / "adalet_analizi.json").write_text(
        json.dumps({
            "politika": {"kademe_1": list(KADEME_1), "izlenen": IZLENEN_BOYUTLAR},
            "cikarilan": cikan,
            "esikler": {"denetim": e_denetim, "cinsiyetsiz": e_adil, "adil": e_adil},
            "vekil_sizinti_auc": sizinti,
            "kar": {"denetim": k_d["kar"], "adil": k_a["kar"]},
            "adalet_ozeti": ozet.to_dict("records"),
            "grup_detay": detay,
        }, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    print(f"\nGrafik  : {REPORTS_DIR / 'adalet_analizi.png'}")
    print(f"Sonuclar: {REPORTS_DIR / 'adalet_analizi.json'}")


if __name__ == "__main__":
    main()
