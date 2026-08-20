"""Model adaleti ve regulasyon uyumu analizi.

SORUN:
    SHAP analizi, XGBoost modelinin CINSIYETI (code_gender) 4. en onemli
    degisken olarak kullandigini ortaya cikardi (ortalama |SHAP| 0.1154,
    toplam onemin %3'u). Skorkartta da en yuksek katsayili degiskenler
    arasindaydi.

NEDEN KABUL EDILEMEZ:
    - ABD'de Equal Credit Opportunity Act, AB'de 2004/113/EC direktifi
      finansal hizmetlerde cinsiyete dayali ayrimi yasaklar.
    - Turkiye'de de bankalar ayrimcilik yasagina ve adil kredilendirme
      beklentilerine tabidir.
    - Model riski: cinsiyet muhtemelen nedensel bir faktor degil, baska
      seylerin (meslek dagilimi, gelir yapisi) vekili. Vekiller zamanla kayar.

BU DOSYA DORT SEY OLCER:
    1. MALIYET      : cinsiyeti cikarinca AUC/Gini ne kadar duser?
    2. VEKIL SIZINTI: kalan degiskenlerden cinsiyet ne kadar tahmin edilebiliyor?
                      (Yuksekse, degiskeni silmek sorunu cozmez sadece gizler.)
    3. ADALET       : gruplar arasi onay orani ve hata orani farklari.
    4. KAR ETKISI   : adil model portfoy karini ne kadar degistiriyor?

ADALET METRIKLERI (literaturdeki adlariyla):
    - Demographic parity : gruplarin onay oranlari esit mi?
      Elestiri: gruplarin gercek risk dagilimi farkliysa esit onay orani
      dayatmak, dusuk riskli gruba haksizlik olur.
    - Equal opportunity  : ODEYECEK musterilerin reddedilme orani gruplar
      arasi esit mi? Kredi riskinde en savunulabilir olcut budur:
      "iyi musteriysen, cinsiyetin reddedilme sansini degistirmemeli."
    - Predictive parity  : onaylananlarda gerceklesen temerrut orani esit mi?

Calistirmak icin (proje kokunden):
    .\\.venv\\Scripts\\python.exe -m src.adalet_analizi
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TARGET  # noqa: E402
from src.data import kolon_tipleri, model_input_yukle, veri_bol  # noqa: E402
from src.metrics import degerlendir, rapor_yazdir  # noqa: E402
from src.kar_optimizasyonu import MARJ, LGD, kar_egrisi, kar_hesapla, para  # noqa: E402
from src.train_xgboost import PARAMETRELER  # noqa: E402

KORUNAN = "code_gender"
ESIK_ARAMA = np.linspace(0.005, 1.0, 400)

RENK_1 = "#2a78d6"
RENK_2 = "#eb6834"
YUZEY = "#fcfcfb"
MUREKKEP = "#0b0b0b"
IKINCIL = "#52514e"
SOLUK = "#898781"
IZGARA = "#e1e0d9"
EKSEN = "#c3c2b7"


def adalet_metrikleri(y, olasilik, grup, esik) -> pd.DataFrame:
    """Grup bazinda onay orani, temerrut ve hata oranlari."""
    onay = olasilik < esik
    d = pd.DataFrame({"y": y, "onay": onay, "grup": np.asarray(grup)})
    satirlar = []
    for g, alt in d.groupby("grup", observed=True):
        iyi = alt[alt["y"] == 0]      # gercekte odeyenler
        kotu = alt[alt["y"] == 1]     # gercekte batanlar
        satirlar.append({
            "grup": str(g),
            "basvuru": len(alt),
            "onay_orani": alt["onay"].mean(),
            "gercek_temerrut": alt["y"].mean(),
            # Equal opportunity: odeyecek musterinin reddedilme orani
            "iyi_musteri_red_orani": 1 - iyi["onay"].mean() if len(iyi) else np.nan,
            # Kotu musterinin yakalanma orani
            "kotu_musteri_red_orani": 1 - kotu["onay"].mean() if len(kotu) else np.nan,
            # Predictive parity: onaylananlarda gerceklesen temerrut
            "onaylananda_temerrut": alt.loc[alt["onay"], "y"].mean()
            if alt["onay"].any() else np.nan,
        })
    return pd.DataFrame(satirlar).sort_values("basvuru", ascending=False)


def grafik(oncesi: pd.DataFrame, sonrasi: pd.DataFrame, yol, min_basvuru: int = 100) -> None:
    """Cinsiyet degiskeni cikarilmadan once ve sonra grup metrikleri.

    Kucuk gruplar (min_basvuru altinda) grafikten cikarilir: veri setinde
    'XNA' grubu tek kisiden olusuyor ve grafikte 40.000 kisilik grupla ayni
    genislikte gorunuyor - istatistiksel olarak anlamsiz, gorsel olarak
    yanilticidir. Sayisal tabloda yine raporlaniyor.
    """
    oncesi = oncesi[oncesi["basvuru"] >= min_basvuru].reset_index(drop=True)
    sonrasi = sonrasi.set_index("grup").reindex(oncesi["grup"]).reset_index()
    gruplar = oncesi["grup"].tolist()
    x = np.arange(len(gruplar))
    genislik = 0.36

    fig, (sol, sag) = plt.subplots(1, 2, figsize=(12, 5.4))
    fig.patch.set_facecolor(YUZEY)

    for eks, kolon, baslik in (
        (sol, "onay_orani", "Onay oranı"),
        (sag, "iyi_musteri_red_orani", "Ödeyecek müşterinin reddedilme oranı"),
    ):
        eks.set_facecolor(YUZEY)
        eks.grid(True, axis="y", color=IZGARA, linewidth=0.8, zorder=0)
        eks.set_axisbelow(True)
        for kenar in ("top", "right"):
            eks.spines[kenar].set_visible(False)
        for kenar in ("left", "bottom"):
            eks.spines[kenar].set_color(EKSEN)
            eks.spines[kenar].set_linewidth(1)
        eks.tick_params(colors=SOLUK, labelsize=9)

        # 2px yuzey bosluğu icin cubuklar arasi kucuk aralik birakiliyor
        eks.bar(x - genislik / 2 - 0.01, 100 * oncesi[kolon], genislik,
                color=RENK_1, zorder=3, label="cinsiyet değişkeniyle")
        eks.bar(x + genislik / 2 + 0.01, 100 * sonrasi[kolon], genislik,
                color=RENK_2, zorder=3, label="cinsiyet değişkeni çıkarılmış")

        for i in range(len(gruplar)):
            eks.text(x[i] - genislik / 2 - 0.01, 100 * oncesi[kolon].iloc[i] + 0.8,
                     f"%{100*oncesi[kolon].iloc[i]:.1f}", ha="center",
                     color=IKINCIL, fontsize=9)
            eks.text(x[i] + genislik / 2 + 0.01, 100 * sonrasi[kolon].iloc[i] + 0.8,
                     f"%{100*sonrasi[kolon].iloc[i]:.1f}", ha="center",
                     color=IKINCIL, fontsize=9)

        eks.set_xticks(x)
        eks.set_xticklabels([f"{'Kadın' if g == 'F' else 'Erkek' if g == 'M' else g}\n(n={n:,})"
                             for g, n in zip(gruplar, oncesi["basvuru"])],
                            color=IKINCIL, fontsize=10)
        eks.set_ylabel("Yüzde", color=IKINCIL, fontsize=10)
        eks.set_ylim(0, 1.28 * max(100 * oncesi[kolon].max(), 100 * sonrasi[kolon].max()))
        eks.set_title(baslik, color=MUREKKEP, fontsize=12.5, fontweight="bold",
                      loc="left", pad=10)

    # Lejant grafigin USTUNE, cubuklarin uzerine binmeyecek sekilde.
    sol.legend(frameon=False, labelcolor=IKINCIL, fontsize=9.5,
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.suptitle("Cinsiyet değişkeninin model adaletine etkisi",
                 color=MUREKKEP, fontsize=15, fontweight="bold", x=0.075, ha="left", y=1.08)
    fig.text(0.075, 1.015,
             f"test kümesi · eşik her model için kendi doğrulama kümesinde seçildi · "
             f"marj %{100*MARJ:.0f} · LGD %{100*LGD:.0f}",
             color=SOLUK, fontsize=10, ha="left")
    fig.savefig(yol, dpi=160, bbox_inches="tight", facecolor=YUZEY)
    plt.close(fig)


def main() -> None:
    print("=" * 78)
    print("VERI")
    print("=" * 78)
    df = model_input_yukle()
    train, valid, test = veri_bol(df)
    ozellikler = kolon_tipleri(df)["ozellikler"]

    y_train, y_valid, y_test = (train[TARGET].values, valid[TARGET].values,
                                test[TARGET].values)
    tutar_valid = valid["amt_credit"].values.astype(float)
    tutar_test = test["amt_credit"].values.astype(float)
    grup_test = test[KORUNAN].astype(str).values

    print(f"{KORUNAN} dagilimi (test):")
    for g, n in pd.Series(grup_test).value_counts().items():
        alt = test[test[KORUNAN].astype(str) == g]
        print(f"  {g:5s} {n:>7,} basvuru | gercek temerrut %{100*alt[TARGET].mean():.2f}")

    mevcut = xgb.XGBClassifier()
    mevcut.load_model(str(MODELS_DIR / "xgboost.json"))

    # ==================================================================
    # 1) MALIYET: cinsiyeti cikarinca ne kaybediyoruz?
    # ==================================================================
    print("\n" + "=" * 78)
    print("1) MALIYET: cinsiyet degiskeni cikarilmis model egitiliyor")
    print("=" * 78)

    adil_ozellikler = [k for k in ozellikler if k != KORUNAN]
    print(f"  {len(ozellikler)} -> {len(adil_ozellikler)} degisken")

    adil = xgb.XGBClassifier(**PARAMETRELER)
    adil.fit(
        train[adil_ozellikler], y_train,
        eval_set=[(valid[adil_ozellikler], y_valid)],
        verbose=False,
    )
    print(f"  en iyi tur: {adil.best_iteration}")

    p_mevcut_valid = mevcut.predict_proba(valid[ozellikler])[:, 1]
    p_adil_valid = adil.predict_proba(valid[adil_ozellikler])[:, 1]
    p_mevcut_test = mevcut.predict_proba(test[ozellikler])[:, 1]
    p_adil_test = adil.predict_proba(test[adil_ozellikler])[:, 1]

    sonuclar = [
        degerlendir(y_valid, p_mevcut_valid, "XGBoost (cinsiyetli)"),
        degerlendir(y_valid, p_adil_valid, "XGBoost (cinsiyetsiz)"),
    ]
    print()
    rapor_yazdir(sonuclar)
    kayip = sonuclar[0]["auc"] - sonuclar[1]["auc"]
    print(f"\n  AUC kaybi: {kayip:.4f}  "
          f"(Gini kaybi {sonuclar[0]['gini'] - sonuclar[1]['gini']:.4f})")

    # ==================================================================
    # 2) VEKIL SIZINTI: cinsiyet kalan degiskenlerden tahmin edilebiliyor mu?
    # ==================================================================
    print("\n" + "=" * 78)
    print("2) VEKIL SIZINTI TESTI")
    print("=" * 78)
    print("  Kalan degiskenlerden cinsiyeti tahmin etmeye calisiyoruz.")
    print("  Yuksek AUC = degiskeni silmek sorunu COZMEZ, sadece GIZLER.\n")

    # Ikili hale getir: en yaygin iki grup disindakiler analiz disi.
    g_train = train[KORUNAN].astype(str)
    ikili = g_train.isin(["M", "F"])
    vekil = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=5,
        tree_method="hist", enable_categorical=True,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    vekil.fit(train.loc[ikili, adil_ozellikler], (g_train[ikili] == "M").astype(int))

    g_test = test[KORUNAN].astype(str)
    ikili_test = g_test.isin(["M", "F"])
    vekil_auc = roc_auc_score(
        (g_test[ikili_test] == "M").astype(int),
        vekil.predict_proba(test.loc[ikili_test, adil_ozellikler])[:, 1],
    )
    print(f"  Cinsiyet tahmin AUC'si: {vekil_auc:.4f}")
    if vekil_auc > 0.75:
        print("  --> YUKSEK. Cinsiyet kalan degiskenlerden buyuk olcude geri")
        print("      kazanilabiliyor. Sadece kolonu silmek yeterli DEGIL;")
        print("      grup bazli adalet metrikleri izlenmeye devam etmeli.")
    else:
        print("  --> Dusuk/orta. Kolonu cikarmak anlamli bir etki yapiyor.")

    vekil_onem = (
        pd.DataFrame({
            "degisken": adil_ozellikler,
            "onem": [vekil.get_booster().get_score(importance_type="gain").get(k, 0.0)
                     for k in adil_ozellikler],
        })
        .sort_values("onem", ascending=False).head(8)
    )
    print("\n  Cinsiyeti en cok ele veren degiskenler (vekiller):")
    for _, r in vekil_onem.iterrows():
        print(f"    {r['degisken']:38s} {r['onem']:.1f}")

    # ==================================================================
    # 3) ADALET METRIKLERI
    # ==================================================================
    print("\n" + "=" * 78)
    print("3) ADALET METRIKLERI (test kumesi)")
    print("=" * 78)

    # Her model icin esigi kendi dogrulama kumesinde sec (adil karsilastirma).
    e_mevcut = float(kar_egrisi(y_valid, p_mevcut_valid, tutar_valid, ESIK_ARAMA)
                     .pipe(lambda d: d.loc[d["kar"].idxmax(), "esik"]))
    e_adil = float(kar_egrisi(y_valid, p_adil_valid, tutar_valid, ESIK_ARAMA)
                   .pipe(lambda d: d.loc[d["kar"].idxmax(), "esik"]))
    print(f"  optimum esikler: cinsiyetli {e_mevcut:.3f} | cinsiyetsiz {e_adil:.3f}\n")

    onceki = adalet_metrikleri(y_test, p_mevcut_test, grup_test, e_mevcut)
    sonraki = adalet_metrikleri(y_test, p_adil_test, grup_test, e_adil)

    for ad, tablo in (("CINSIYET DEGISKENIYLE", onceki),
                      ("CINSIYET DEGISKENI CIKARILMIS", sonraki)):
        print(f"  --- {ad} ---")
        print(f"    {'grup':6s} {'basvuru':>8s} {'onay':>7s} {'gercek tem.':>12s} "
              f"{'iyi musteri red':>16s} {'onaylananda tem.':>17s}")
        for _, r in tablo.iterrows():
            print(f"    {r['grup']:6s} {r['basvuru']:>8,} {100*r['onay_orani']:>6.1f}% "
                  f"{100*r['gercek_temerrut']:>11.2f}% {100*r['iyi_musteri_red_orani']:>15.1f}% "
                  f"{100*r['onaylananda_temerrut']:>16.2f}%")
        print()

    def fark(tablo, kolon):
        ik = tablo[tablo["grup"].isin(["M", "F"])]
        if len(ik) < 2:
            return np.nan
        return float(abs(ik[kolon].iloc[0] - ik[kolon].iloc[1]))

    print("  KADIN-ERKEK FARKLARI (kucuk = daha adil):")
    print(f"    {'olcut':34s} {'oncesi':>10s} {'sonrasi':>10s}")
    for kolon, ad in (("onay_orani", "onay orani farki (dem. parity)"),
                      ("iyi_musteri_red_orani", "iyi musteri red farki (eq. opp.)"),
                      ("onaylananda_temerrut", "onaylananda temerrut farki")):
        o, s = fark(onceki, kolon), fark(sonraki, kolon)
        ok = "iyilesti" if s < o else "kotulesti" if s > o else "degismedi"
        print(f"    {ad:34s} {100*o:>9.2f}% {100*s:>9.2f}%  {ok}")

    # ==================================================================
    # 4) KAR ETKISI
    # ==================================================================
    print("\n" + "=" * 78)
    print("4) KAR ETKISI (test kumesi)")
    print("=" * 78)
    k_mevcut = kar_hesapla(y_test, p_mevcut_test, tutar_test, e_mevcut)
    k_adil = kar_hesapla(y_test, p_adil_test, tutar_test, e_adil)
    print(f"  cinsiyetli  : {para(k_mevcut['kar'])}  "
          f"(onay %{100*k_mevcut['onay_orani']:.1f})")
    print(f"  cinsiyetsiz : {para(k_adil['kar'])}  "
          f"(onay %{100*k_adil['onay_orani']:.1f})")
    dfark = k_adil["kar"] - k_mevcut["kar"]
    print(f"  fark        : {para(dfark)}  (%{100*dfark/k_mevcut['kar']:+.2f})")

    # ==================================================================
    grafik(onceki, sonraki, REPORTS_DIR / "adalet_analizi.png")
    adil.save_model(str(MODELS_DIR / "xgboost_adil.json"))

    (REPORTS_DIR / "adalet_analizi.json").write_text(
        json.dumps({
            "korunan_ozellik": KORUNAN,
            "auc_kaybi": kayip,
            "vekil_tahmin_auc": float(vekil_auc),
            "esikler": {"cinsiyetli": e_mevcut, "cinsiyetsiz": e_adil},
            "kar": {"cinsiyetli": k_mevcut["kar"], "cinsiyetsiz": k_adil["kar"]},
            "grup_metrikleri_oncesi": onceki.to_dict("records"),
            "grup_metrikleri_sonrasi": sonraki.to_dict("records"),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nKaydedilenler:")
    print("  models/xgboost_adil.json")
    print("  reports/adalet_analizi.png")
    print("  reports/adalet_analizi.json")


if __name__ == "__main__":
    main()
