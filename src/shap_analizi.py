"""SHAP ile model aciklanabilirligi.

NEDEN GEREKLI:
    Skorkart zaten okunabilir bir puan tablosu uretiyor. XGBoost ise 910 agacin
    toplami - hicbir insan bunu okuyarak "bu musteri neden reddedildi?"
    sorusunu cevaplayamaz. SHAP bu bosluğu doldurur.

    Bu akademik bir suslemeden ibaret degil, iki somut ihtiyaci karsilar:
      1. MODEL DENETIMI (global): model hangi degiskenlere dayaniyor? Beklenmedik
         bir degisken one ciktiysa ya veri sizintisi ya da onyargi vardir.
      2. KARAR GEREKCESI (yerel): reddedilen basvuru sahibine sebep bildirmek
         bircok ulkede yasal zorunluluktur (ornegin ABD'de Adverse Action Notice).
         SHAP, TEK BIR basvuru icin "hangi faktor kararı ne yonde etkiledi"
         dokumu uretebilir.

SHAP NEDIR (kisaca):
    Oyun teorisinden gelir (Shapley degerleri). Her degiskenin tahmine kattigi
    katkiyi, o degiskenin TUM olasi degisken kombinasyonlarindaki ortalama
    marjinal katkisi olarak hesaplar. Ozelligi TOPLANABILIR olmasidir:
        temel deger + tum SHAP katkilari = o basvurunun tahmini
    Yani her tahmin, degiskenlere tam olarak paylastirilabilir.

NEDEN shap KUTUPHANESI YERINE XGBOOST'UN KENDISI:
    XGBoost, TreeSHAP algoritmasini kendi icinde barindirir
    (booster.predict(..., pred_contribs=True)). Sonuc matematiksel olarak
    ayni, ama araya bir kutuphane daha girmiyor - bu projede zaten
    sklearn/optbinning surum catismasi yasadik, bagimlilik yuzeyini
    gereksiz genisletmiyoruz.

Calistirmak icin (proje kokunden):
    .\\.venv\\Scripts\\python.exe -m src.shap_analizi
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TARGET  # noqa: E402
from src.data import kolon_tipleri, model_input_yukle, veri_bol  # noqa: E402

# SHAP hesabi tum test kumesinde de yapilabilir ama gereksiz:
# global onem siralamasi birkac bin ornekte kararli hale gelir.
ORNEK = 8000

# --- Gorsel kimlik (dogrulanmis palet) --------------------------------------
RENK_1 = "#2a78d6"     # mavi  - riski AZALTAN katki (diverging pozitif kutup)
RENK_KIRMIZI = "#e34948"  # kirmizi - riski ARTIRAN katki (diverging negatif kutup)
YUZEY = "#fcfcfb"
MUREKKEP = "#0b0b0b"
IKINCIL = "#52514e"
SOLUK = "#898781"
IZGARA = "#e1e0d9"
EKSEN = "#c3c2b7"


def eksen_duzenle(eks, dikey_izgara=True):
    eks.set_facecolor(YUZEY)
    eks.grid(True, axis="x" if dikey_izgara else "y",
             color=IZGARA, linewidth=0.8, zorder=0)
    eks.set_axisbelow(True)
    for kenar in ("top", "right"):
        eks.spines[kenar].set_visible(False)
    for kenar in ("left", "bottom"):
        eks.spines[kenar].set_color(EKSEN)
        eks.spines[kenar].set_linewidth(1)
    eks.tick_params(colors=SOLUK, labelsize=9)


def global_grafik(onem: pd.DataFrame, yol, n=20):
    """Ortalama mutlak SHAP - buyukluk oldugu icin TEK HUE (sequential).

    Kategorik palet kullanmak yanlis olurdu: buradaki cubuklar farkli
    'kimlikler' degil, ayni olcunun farkli buyuklukleri.
    """
    d = onem.head(n).iloc[::-1]
    fig, eks = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(YUZEY)
    eksen_duzenle(eks)

    eks.barh(d["degisken"], d["ortalama_mutlak_shap"], color=RENK_1,
             height=0.72, zorder=3)
    # Degerleri secili olarak dogrudan yaz (her cubuga degil, hepsi kisa)
    for i, (_, r) in enumerate(d.iterrows()):
        eks.text(r["ortalama_mutlak_shap"] * 1.02, i, f"{r['ortalama_mutlak_shap']:.4f}",
                 va="center", color=IKINCIL, fontsize=8.5)

    eks.set_xlabel("Ortalama |SHAP| — tahmine ortalama katkı büyüklüğü (log-odds)",
                   color=IKINCIL, fontsize=10)
    eks.set_title("Modelin en çok dayandığı 20 değişken",
                  color=MUREKKEP, fontsize=15, fontweight="bold", loc="left", pad=32)
    eks.text(0, 1.03, f"XGBoost · {ORNEK:,} test başvurusu üzerinden TreeSHAP",
             transform=eks.transAxes, color=SOLUK, fontsize=10)
    eks.margins(x=0.14)
    fig.savefig(yol, dpi=160, bbox_inches="tight", facecolor=YUZEY)
    plt.close(fig)


def yerel_grafik(katkilar: pd.DataFrame, baslik: str, altbaslik: str, yol):
    """Tek bir basvuru icin karar dokumu.

    Diverging renk: katki riski ARTIRIYOR mu AZALTIYOR mu - iki kutup, aralarinda
    dogal bir sifir noktasi var. Bu, sequential degil diverging bir islir.
    """
    d = katkilar.iloc[::-1]
    renkler = [RENK_KIRMIZI if v > 0 else RENK_1 for v in d["shap"]]

    fig, eks = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(YUZEY)
    eksen_duzenle(eks)

    eks.barh(d["etiket"], d["shap"], color=renkler, height=0.72, zorder=3)
    eks.axvline(0, color=EKSEN, linewidth=1.2, zorder=4)

    for i, (_, r) in enumerate(d.iterrows()):
        kayma = 0.012 * max(abs(d["shap"].min()), abs(d["shap"].max()))
        eks.text(r["shap"] + (kayma if r["shap"] > 0 else -kayma), i,
                 f"{r['shap']:+.3f}", va="center",
                 ha="left" if r["shap"] > 0 else "right",
                 color=IKINCIL, fontsize=8.5)

    eks.set_xlabel("SHAP katkısı (log-odds) — sağ: riski artırır · sol: riski azaltır",
                   color=IKINCIL, fontsize=10)
    eks.set_title(baslik, color=MUREKKEP, fontsize=15, fontweight="bold",
                  loc="left", pad=32)
    eks.text(0, 1.03, altbaslik, transform=eks.transAxes, color=SOLUK, fontsize=10)
    eks.margins(x=0.20)
    fig.savefig(yol, dpi=160, bbox_inches="tight", facecolor=YUZEY)
    plt.close(fig)


def deger_yaz(v) -> str:
    if pd.isna(v):
        return "eksik"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.3f}" if abs(v) < 100 else f"{v:,.0f}"
    return str(v)


def main() -> None:
    print("=" * 78)
    print("VERI VE MODEL")
    print("=" * 78)
    df = model_input_yukle()
    train, valid, test = veri_bol(df)
    ozellikler = kolon_tipleri(df)["ozellikler"]

    model = xgb.XGBClassifier()
    model.load_model(str(MODELS_DIR / "xgboost.json"))

    ornek = test.sample(n=min(ORNEK, len(test)), random_state=RANDOM_STATE)
    X = ornek[ozellikler]
    y = ornek[TARGET].values
    print(f"SHAP ornegi: {len(X):,} test basvurusu x {len(ozellikler)} degisken")

    # ------------------------------------------------------------------
    # TreeSHAP - XGBoost'un kendi uygulamasi
    # ------------------------------------------------------------------
    print("\nTreeSHAP hesaplaniyor...")
    dmat = xgb.DMatrix(X, enable_categorical=True)

    # DIKKAT - erken durdurma tuzagi:
    # Model 910. turda en iyi sonucu buldu ama 1010. turda durdu (100 tur
    # sabir suresi). predict_proba() sadece EN IYI 910 agaci kullanir;
    # ham booster.predict() ise varsayilan olarak TUM agaclari kullanir.
    # Ayni araligi vermezsek iki farkli modeli karsilastirmis oluruz ve
    # SHAP katkilari modelin gercek tahminleriyle ortusmez.
    # (Ilk calistirmada toplanabilirlik sapmasi 5.5e-02 ciktigi icin fark edildi.)
    en_iyi = getattr(model, "best_iteration", None)
    aralik = (0, int(en_iyi) + 1) if en_iyi is not None else None
    print(f"  kullanilan agac araligi: {aralik if aralik else 'tumu'}")

    katkilar = model.get_booster().predict(
        dmat, pred_contribs=True, **({"iteration_range": aralik} if aralik else {})
    )
    # Son kolon 'bias' (temel deger): modelin hicbir bilgi olmadan verdigi tahmin.
    shap_degerleri = katkilar[:, :-1]
    temel_deger = float(katkilar[0, -1])
    print(f"  temel deger (bias): {temel_deger:.4f} log-odds "
          f"= %{100/(1+np.exp(-temel_deger)):.2f} olasilik")

    # Toplanabilirlik dogrulamasi: temel + katkilar = modelin tahmini olmali.
    # Bu kontrol, SHAP'in dogru hesaplandiginin matematiksel kaniti.
    tahmin_log = katkilar.sum(axis=1)
    tahmin_shap = 1 / (1 + np.exp(-tahmin_log))
    tahmin_model = model.predict_proba(X)[:, 1]
    fark = float(np.abs(tahmin_shap - tahmin_model).max())
    print(f"  toplanabilirlik kontrolu: en buyuk sapma {fark:.2e}", end="")
    print("  (sifira yakin = DOGRU)" if fark < 1e-4 else "  <-- SORUN VAR")

    # ------------------------------------------------------------------
    # GLOBAL: model neye dayaniyor?
    # ------------------------------------------------------------------
    onem = (
        pd.DataFrame({
            "degisken": ozellikler,
            "ortalama_mutlak_shap": np.abs(shap_degerleri).mean(axis=0),
        })
        .sort_values("ortalama_mutlak_shap", ascending=False)
        .reset_index(drop=True)
    )
    onem["pay_yuzde"] = (100 * onem["ortalama_mutlak_shap"]
                         / onem["ortalama_mutlak_shap"].sum()).round(2)
    onem.to_csv(REPORTS_DIR / "shap_global_onem.csv", index=False, encoding="utf-8")

    print("\n" + "=" * 78)
    print("GLOBAL: EN ONEMLI 20 DEGISKEN (ortalama |SHAP|)")
    print("=" * 78)
    for _, r in onem.head(20).iterrows():
        print(f"  {r['degisken']:40s} {r['ortalama_mutlak_shap']:.4f}  "
              f"(%{r['pay_yuzde']:.1f})")

    global_grafik(onem, REPORTS_DIR / "shap_global_onem.png")

    # SHAP siralamasi ile 'gain' siralamasini karsilastir.
    # Ikisi farkli sorulara cevap verir: gain "agac kurarken ne kadar ise
    # yaradi", SHAP "tahminleri ne kadar degistiriyor". Buyuk ayrisma,
    # bir degiskenin cok bolunmede kullanildigi ama tahmini az degistirdigi
    # (ya da tersi) anlamina gelir.
    gain_yol = REPORTS_DIR / "xgboost_degisken_onemi.csv"
    if gain_yol.exists():
        gain = pd.read_csv(gain_yol)
        birlesik = (
            onem.assign(shap_sira=lambda d: d.index + 1)
            .merge(gain.assign(gain_sira=lambda d: d.index + 1)[["degisken", "gain_sira"]],
                   on="degisken", how="left")
        )
        birlesik["sira_farki"] = birlesik["gain_sira"] - birlesik["shap_sira"]
        buyuk = birlesik.head(40).reindex(
            birlesik.head(40)["sira_farki"].abs().sort_values(ascending=False).index
        ).head(6)
        print("\n  SHAP ve 'gain' siralamasi en cok ayrisan degiskenler:")
        for _, r in buyuk.iterrows():
            print(f"    {r['degisken']:38s} SHAP #{r['shap_sira']:<3.0f} "
                  f"gain #{r['gain_sira']:<3.0f}  (fark {r['sira_farki']:+.0f})")

    # ------------------------------------------------------------------
    # YEREL: tek basvuru icin karar gerekcesi
    # ------------------------------------------------------------------
    olasilik = tahmin_model
    ESIK = 0.135  # kar optimizasyonundan gelen esik

    # En riskli reddedilen ve guvenli onaylanan birer ornek secelim.
    idx_red = int(np.argmax(olasilik))
    idx_onay = int(np.argmin(olasilik))

    print("\n" + "=" * 78)
    print("YEREL ACIKLAMA: TEK BASVURU ICIN KARAR GEREKCESI")
    print("=" * 78)

    for etiket, idx in (("reddedilen", idx_red), ("onaylanan", idx_onay)):
        s = pd.Series(shap_degerleri[idx], index=ozellikler)
        ust = s.reindex(s.abs().sort_values(ascending=False).index).head(12)
        satirlar = pd.DataFrame({
            "degisken": ust.index,
            "shap": ust.values,
            "deger": [X.iloc[idx][k] for k in ust.index],
        })
        satirlar["etiket"] = [
            f"{k}  =  {deger_yaz(v)}" for k, v in zip(satirlar["degisken"], satirlar["deger"])
        ]

        karar = "RED" if olasilik[idx] >= ESIK else "ONAY"
        print(f"\n--- {etiket.upper()} basvuru (sk_id_curr={ornek.iloc[idx]['sk_id_curr']}) ---")
        print(f"  model tahmini: %{100*olasilik[idx]:.2f} temerrut olasiligi")
        print(f"  esik %{100*ESIK:.1f} -> KARAR: {karar}")
        print(f"  gerceklesen  : {'TEMERRUT' if y[idx] == 1 else 'odedi'}")
        print(f"  {'faktor':40s} {'deger':>16s} {'katki':>9s}")
        for _, r in satirlar.iterrows():
            yon = "riski artirdi" if r["shap"] > 0 else "riski azaltti"
            print(f"    {r['degisken']:38s} {deger_yaz(r['deger']):>16s} "
                  f"{r['shap']:+8.3f}  {yon}")

        yerel_grafik(
            satirlar,
            f"Karar gerekçesi — {'reddedilen' if karar == 'RED' else 'onaylanan'} başvuru",
            f"tahmin %{100*olasilik[idx]:.2f} · eşik %{100*ESIK:.1f} · karar {karar} · "
            f"gerçekleşen: {'temerrüt' if y[idx] == 1 else 'ödedi'}",
            REPORTS_DIR / f"shap_yerel_{etiket}.png",
        )

    (REPORTS_DIR / "shap_ozet.json").write_text(
        json.dumps({
            "ornek_sayisi": int(len(X)),
            "temel_deger_logodds": temel_deger,
            "toplanabilirlik_sapmasi": fark,
            "en_onemli_10": onem.head(10)[["degisken", "ortalama_mutlak_shap"]]
            .to_dict("records"),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nKaydedilenler:")
    for f in ["reports/shap_global_onem.png", "reports/shap_global_onem.csv",
              "reports/shap_yerel_reddedilen.png", "reports/shap_yerel_onaylanan.png",
              "reports/shap_ozet.json"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
