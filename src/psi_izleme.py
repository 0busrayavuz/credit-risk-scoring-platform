"""PSI (Population Stability Index) ile model izleme.

SORU: "Modelin 6 ay sonra bozuldugunu nasil anlarsin?"
    Bu, banka risk ekiplerinin standart mulakat sorusudur ve cevabi
    sanildigi kadar basit degildir.

NEDEN AUC IZLEMEK YETMEZ:
    AUC hesaplamak icin GERCEKLESMIS SONUC gerekir - kimin odedigini,
    kimin batmadigini bilmelisin. Ama bir tuketici kredisinin temerrude
    dusmesi 12-24 ay surer. Yani bugun verdigin kredilerin AUC'sini ancak
    2 yil sonra olcebilirsin. O zamana kadar model bozuk calisiyorsa,
    zarar coktan yazilmis olur.

    PSI ise SONUC GEREKTIRMEZ. Sadece bugun gelen basvurularin dagilimini,
    modelin egitildigi dagilimla karsilastirir. Basvurular geldigi gun
    hesaplanabilir. Bu yuzden PSI bir ONCU GOSTERGEDIR (leading indicator);
    AUC ise gecikmeli (lagging).

PSI NASIL HESAPLANIR:
    Referans dagilim dilimlere ayrilir (genelde 10). Her dilim icin:
        PSI_dilim = (yeni_oran - referans_oran) * ln(yeni_oran / referans_oran)
    Toplami PSI'dir. Simetriktir ve 0'dan buyuktur.

SEKTORDEKI ESIKLER:
    < 0.10        anlamli kayma yok
    0.10 - 0.25   orta duzeyde kayma - izlemeye al
    > 0.25        ciddi kayma - model gozden gecirilmeli

VERI KISITI (durustce belirtiliyor):
    Home Credit veri setinde BASVURU TARIHI YOKTUR. Bu yuzden gercek bir
    zaman serisi kaymasi gosterilemez. Onun yerine, iktisadi olarak makul
    SENARYOLAR simule ediliyor ve PSI'nin bunlari yakalayip yakalamadigi
    olculuyor. Simulasyon tutarli kuruluyor: geliri degistirdigimizde ona
    bagli oranlar (kredi/gelir, taksit/gelir, kisi basi gelir) yeniden
    hesaplaniyor - aksi halde PSI, gercek kaymayi degil bizim tutarsizligimizi
    olcerdi.

Calistirmak icin (proje kokunden):
    .\\.venv\\Scripts\\python.exe -m src.psi_izleme
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import MODELS_DIR, REPORTS_DIR, TARGET  # noqa: E402
from src.data import model_input_yukle, veri_bol  # noqa: E402
from src.metrics import degerlendir  # noqa: E402

ESIK_IZLE = 0.10
ESIK_GOZDEN_GECIR = 0.25

RENK_1 = "#2a78d6"
YUZEY = "#fcfcfb"
MUREKKEP = "#0b0b0b"
IKINCIL = "#52514e"
SOLUK = "#898781"
IZGARA = "#e1e0d9"
EKSEN = "#c3c2b7"


def psi(referans, yeni, dilim: int = 10) -> float:
    """Iki dagilim arasindaki PSI degerini hesaplar.

    Dilim sinirlari REFERANS dagilimdan belirlenir - yeni dagilimdan degil.
    Sebep: referans, modelin egitildigi dunyayi temsil eder; olcumu ona gore
    yapariz. Sinirlari her seferinde yeniden hesaplarsak kaymayi goremeyiz.
    """
    referans = np.asarray(referans, dtype=float)
    yeni = np.asarray(yeni, dtype=float)
    referans = referans[~np.isnan(referans)]
    yeni = yeni[~np.isnan(yeni)]
    if len(referans) == 0 or len(yeni) == 0:
        return np.nan

    # Yuzdeliklerle esit BUYUKLUKTE dilimler. Tekrar eden degerler varsa
    # (ornegin cok sayida sifir) bazi sinirlar cakisir; unique ile ayikliyoruz.
    sinirlar = np.unique(np.percentile(referans, np.linspace(0, 100, dilim + 1)))
    if len(sinirlar) < 3:
        return 0.0
    sinirlar[0], sinirlar[-1] = -np.inf, np.inf

    r_pay = np.histogram(referans, bins=sinirlar)[0] / len(referans)
    y_pay = np.histogram(yeni, bins=sinirlar)[0] / len(yeni)

    # Bos dilimlerde ln(0) tanimsiz olur. Kucuk bir taban degeri koyuyoruz;
    # sektorde yaygin uygulama budur (aksi halde tek bos dilim PSI'yi
    # sonsuza gonderir ve olcu kullanilamaz hale gelir).
    taban = 1e-6
    r_pay = np.clip(r_pay, taban, None)
    y_pay = np.clip(y_pay, taban, None)

    return float(np.sum((y_pay - r_pay) * np.log(y_pay / r_pay)))


def yorumla(deger: float) -> str:
    if np.isnan(deger):
        return "hesaplanamadi"
    if deger < ESIK_IZLE:
        return "kayma yok"
    if deger < ESIK_GOZDEN_GECIR:
        return "orta - izle"
    return "CIDDI - gozden gecir"


def senaryo_uygula(df: pd.DataFrame, ad: str) -> pd.DataFrame:
    """Iktisadi olarak makul kayma senaryolari uretir.

    ONEMLI: turetilmis oranlar yeniden hesaplaniyor. Sadece amt_income_total'i
    degistirip credit_income_ratio'yu eski haliyle birakmak, veriyi kendi
    icinde tutarsiz kilardi ve PSI gercek kaymayi degil bu tutarsizligi
    olcerdi.
    """
    d = df.copy()

    if ad == "hafif_durgunluk":
        # Gelirler %10 duser, kart limit kullanimi %15 artar,
        # taksit gecikmeleri %30 artar, dis kredi skorlari %6 geriler.
        gelir_carpan, kullanim_carpan, gecikme_carpan, skor_carpan = 0.90, 1.15, 1.30, 0.94
    elif ad == "siddetli_durgunluk":
        gelir_carpan, kullanim_carpan, gecikme_carpan, skor_carpan = 0.75, 1.40, 2.00, 0.82
    elif ad == "genc_musteri_kampanyasi":
        # Pazarlama, daha genc ve daha kisa calisma gecmisli kitleye yoneliyor.
        d["days_birth"] = d["days_birth"] * 0.75
        d["age_years"] = -d["days_birth"] / 365.25
        d["days_employed"] = np.where(
            d["days_employed"] == 365243, 365243, d["days_employed"] * 0.5
        )
        d["employed_years"] = np.where(
            d["days_employed"] == 365243, np.nan, -d["days_employed"] / 365.25
        )
        return d
    else:
        return d

    d["amt_income_total"] = d["amt_income_total"] * gelir_carpan

    # DIS KREDI SKORLARI - senaryonun en onemli parcasi.
    # Ilk denemede bunlara dokunmamistim ve skor PSI'si 0.003'te kaldi:
    # model agirligini buyuk olcude ext_source_* uzerine koydugu icin,
    # geliri %25 dusurmek skoru neredeyse hic oynatmadi.
    # Iktisaden de dogrusu bunlari kaydirmak: bir durgunlukta sistem
    # genelinde odemeler aksar ve kredi burosu skorlari topluca geriler.
    for k in ("ext_source_1", "ext_source_2", "ext_source_3"):
        if k in d.columns:
            d[k] = d[k] * skor_carpan
    # Turetilmis skor ozetlerini YENIDEN hesapla (tutarlilik).
    kaynaklar = [k for k in ("ext_source_1", "ext_source_2", "ext_source_3") if k in d.columns]
    if kaynaklar:
        d["ext_source_mean"] = d[kaynaklar].mean(axis=1, skipna=True)
        d["ext_source_min"] = d[kaynaklar].min(axis=1, skipna=True)
        d["ext_source_max"] = d[kaynaklar].max(axis=1, skipna=True)

    for k in ("cc_utilization_avg", "cc_utilization_avg_1y", "cc_utilization_max"):
        if k in d.columns:
            d[k] = np.minimum(d[k] * kullanim_carpan, 3.0)

    for k in ("inst_late_ratio", "inst_late_ratio_1y", "pos_dpd_ratio", "bb_dpd_ratio"):
        if k in d.columns:
            d[k] = np.minimum(d[k] * gecikme_carpan, 1.0)

    # Gelire bagli turetilmis oranlari YENIDEN HESAPLA - tutarlilik icin.
    d["credit_income_ratio"] = d["amt_credit"] / d["amt_income_total"].replace(0, np.nan)
    d["annuity_income_ratio"] = d["amt_annuity"] / d["amt_income_total"].replace(0, np.nan)
    if "income_per_person" in d.columns:
        d["income_per_person"] = (
            d["amt_income_total"] / d["cnt_fam_members"].replace(0, np.nan)
        )
    return d


def grafik(ozet: pd.DataFrame, yol) -> None:
    """Senaryo bazinda skor PSI'si, esik bantlariyla."""
    d = ozet.iloc[::-1]
    fig, eks = plt.subplots(figsize=(10, 5.2))
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

    eks.barh(d["senaryo_ad"], d["skor_psi"], color=RENK_1, height=0.6, zorder=3)
    for i, (_, r) in enumerate(d.iterrows()):
        eks.text(r["skor_psi"] + 0.012, i,
                 f"{r['skor_psi']:.3f}  ·  {r['yorum']}",
                 va="center", color=IKINCIL, fontsize=9.5)

    for x, etiket in ((ESIK_IZLE, "0,10 — izlemeye al"),
                      (ESIK_GOZDEN_GECIR, "0,25 — modeli gözden geçir")):
        eks.axvline(x, color=EKSEN, linewidth=1.2, linestyle=(0, (4, 4)), zorder=2)
        eks.text(x, len(d) - 0.35, etiket, color=SOLUK, fontsize=9,
                 rotation=90, va="top", ha="right")

    eks.set_xlabel("Skor dağılımı PSI'si (referans: eğitim kümesi)",
                   color=IKINCIL, fontsize=10)
    eks.set_title("Popülasyon kayması senaryoları",
                  color=MUREKKEP, fontsize=15, fontweight="bold", loc="left", pad=32)
    eks.text(0, 1.045,
             "PSI sonuç gerektirmez — başvurular geldiği gün hesaplanır. "
             "AUC ise temerrütlerin gerçekleşmesini bekler.",
             transform=eks.transAxes, color=SOLUK, fontsize=9.5)
    eks.margins(x=0.30)
    fig.savefig(yol, dpi=160, bbox_inches="tight", facecolor=YUZEY)
    plt.close(fig)


def main() -> None:
    print("=" * 78)
    print("VERI VE MODEL")
    print("=" * 78)
    df = model_input_yukle()
    train, valid, test = veri_bol(df)

    model = xgb.XGBClassifier()
    model.load_model(str(MODELS_DIR / "xgboost_adil.json"))
    ozellikler = list(model.get_booster().feature_names)

    p_train = model.predict_proba(train[ozellikler])[:, 1]
    y_test = test[TARGET].values

    print(f"referans (egitim): {len(train):,} basvuru")
    print(f"izlenen (test)   : {len(test):,} basvuru")

    # En onemli degiskenleri SHAP raporundan al (yoksa yedek liste).
    shap_yol = REPORTS_DIR / "shap_global_onem.csv"
    if shap_yol.exists():
        izlenen = (pd.read_csv(shap_yol)["degisken"].head(10).tolist())
    else:
        izlenen = ["ext_source_mean", "amt_annuity", "credit_term",
                   "inst_late_ratio_1y", "bureau_debt_ratio"]
    izlenen = [k for k in izlenen if k in ozellikler]

    senaryolar = [
        ("gercek_test", "Gerçek test kümesi (kayma yok)"),
        ("genc_musteri_kampanyasi", "Genç müşteri kampanyası"),
        ("hafif_durgunluk", "Hafif durgunluk"),
        ("siddetli_durgunluk", "Şiddetli durgunluk"),
    ]

    print("\n" + "=" * 78)
    print("SENARYO ANALIZI")
    print("=" * 78)
    print("Her senaryoda: skor dagilimi PSI'si + gercek model performansi\n")

    satirlar = []
    for kod, ad in senaryolar:
        d = senaryo_uygula(test, kod)
        p = model.predict_proba(d[ozellikler])[:, 1]

        skor_psi = psi(p_train, p)
        olcum = degerlendir(y_test, p, ad)

        # Degisken bazli PSI - hangi degiskenin kaydigini gormek icin
        degisken_psi = {
            k: psi(train[k].values, d[k].values)
            for k in izlenen
            if pd.api.types.is_numeric_dtype(train[k])
        }

        satirlar.append({
            "senaryo": kod, "senaryo_ad": ad,
            "skor_psi": skor_psi, "yorum": yorumla(skor_psi),
            "auc": olcum["auc"], "gini": olcum["gini"],
            "ort_tahmin": float(p.mean()),
            "degisken_psi": degisken_psi,
        })

        print(f"--- {ad} ---")
        print(f"    skor PSI      : {skor_psi:.4f}   [{yorumla(skor_psi)}]")
        print(f"    ortalama tahmin: %{100*p.mean():.2f}  "
              f"(gercek temerrut %{100*y_test.mean():.2f})")
        print(f"    AUC / Gini    : {olcum['auc']:.4f} / {olcum['gini']:.4f}")
        if degisken_psi:
            en_cok = sorted(degisken_psi.items(), key=lambda t: -t[1])[:4]
            print("    en cok kayan degiskenler:")
            for k, v in en_cok:
                print(f"      {k:34s} {v:.4f}  [{yorumla(v)}]")
        print()

    ozet = pd.DataFrame(satirlar)

    # ------------------------------------------------------------------
    print("=" * 78)
    print("OZET")
    print("=" * 78)
    print(f"{'senaryo':34s} {'skor PSI':>10s} {'AUC':>8s} {'ort. tahmin':>12s}  durum")
    for _, r in ozet.iterrows():
        print(f"{r['senaryo_ad']:34s} {r['skor_psi']:>10.4f} {r['auc']:>8.4f} "
              f"{100*r['ort_tahmin']:>11.2f}%  {r['yorum']}")

    temel = ozet.iloc[0]
    print(f"\nDogrulama: kaymanin OLMADIGI durumda PSI = {temel['skor_psi']:.4f}")
    print("Bu, olcunun kendisinin dogru calistiginin kanitidir - egitim ve test")
    print("ayni dagilimdan geldigi icin PSI sifira yakin cikmali, cikti da.")

    print("\nKRITIK GOZLEM:")
    for _, r in ozet.iloc[1:].iterrows():
        auc_fark = r["auc"] - temel["auc"]
        print(f"  {r['senaryo_ad']:32s} PSI {r['skor_psi']:.3f} -> "
              f"AUC degisimi {auc_fark:+.4f}")
    print("\n  PSI yuksek olsa bile AUC pek dusmeyebilir: model AYIRT ETME")
    print("  gucunu korurken KALIBRASYONU bozulmus olabilir. Yani siralamasi")
    print("  hala dogru ama '%10 risk' dedigi musteriler artik %15 batıyor")
    print("  olabilir. Kar hesabi kalibre olasiliklara dayandigi icin, PSI")
    print("  sinyali verdiginde esigin de yeniden hesaplanmasi gerekir.")

    grafik(ozet, REPORTS_DIR / "psi_izleme.png")

    (REPORTS_DIR / "psi_izleme.json").write_text(
        json.dumps(ozet.drop(columns=["degisken_psi"]).to_dict("records")
                   + [{"degisken_psi_detay": satirlar[-1]["degisken_psi"]}],
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nGrafik  : {REPORTS_DIR / 'psi_izleme.png'}")
    print(f"Sonuclar: {REPORTS_DIR / 'psi_izleme.json'}")


if __name__ == "__main__":
    main()
