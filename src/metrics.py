"""Kredi riski degerlendirme metrikleri.

Bankacilikta kullanilan metrikler, genel makine ogrenmesi metriklerinden
kismen farklidir. Burada ikisini birlestiriyoruz.

NEDEN ACCURACY YOK:
    Temerrut orani %8. "Herkes oder" diyen bir model %92 dogruluk alir ama
    hicbir ise yaramaz. Dengesiz siniflarda accuracy yanilticidir.

KULLANILAN METRIKLER:
    ROC-AUC : Rastgele secilen bir temerrutlunun, rastgele secilen bir
              iyi musteriden daha yuksek risk skoru alma olasiligi.
              0.5 = rastgele, 1.0 = kusursuz.
    Gini    : 2 * AUC - 1. Ayni bilgiyi tasir ama bankacilik jargonunda
              standart olan budur. Mulakatta "Gini kac?" diye sorulur.
    KS      : Kolmogorov-Smirnov. Iyi ve kotu musterilerin kumulatif
              dagilimlari arasindaki en buyuk fark. Skorkart kalitesinin
              klasik olcusudur; kesim noktasi secerken de kullanilir.
    PR-AUC  : Precision-Recall egrisi altindaki alan. Dengesiz veride
              ROC-AUC'den daha hassastir, cunku cogunluk sinifini odullendirmez.
    Brier   : Olasilik tahminlerinin kare hatasi. Siralamayi degil
              KALIBRASYONU olcer - "%10 dedigin musterilerin gercekten
              %10'u temerrude dusuyor mu?" Kar hesabi yapacaksak sart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)


def gini(y_true, y_skor) -> float:
    """Gini katsayisi = 2 * AUC - 1."""
    return 2 * roc_auc_score(y_true, y_skor) - 1


def ks_istatistigi(y_true, y_skor) -> float:
    """KS istatistigi: TPR ve FPR egrileri arasindaki en buyuk dikey fark.

    Sezgisel anlami: modelin iyi ve kotu musterileri en iyi ayirdigi
    noktadaki ayrim gucu. Bankacilikta 0.30 uzeri "kullanilabilir",
    0.40 uzeri "iyi" kabul edilir (portfoye gore degisir).
    """
    fpr, tpr, _ = roc_curve(y_true, y_skor)
    return float(np.max(tpr - fpr))


def degerlendir(y_true, y_skor, ad: str = "model") -> dict:
    """Tum metrikleri hesaplayip sozluk olarak dondurur.

    SIRALAMA vs KALIBRASYON ayrimi:
        AUC, Gini, KS ve PR-AUC yalnizca SIRALAMAYA bakar; skorun 0-1
        araliginda olmasi gerekmez, herhangi bir monoton donusum sonucu
        degistirmez. Ham bir degiskeni skor olarak kullanabiliriz.

        Brier ise KALIBRASYON olcer: "%10 dedigin musterilerin gercekten
        %10'u temerrude dusuyor mu?" Bunun icin skorun gercek bir olasilik
        olmasi sarttir. Olasilik olmayan skorlarda hesaplamak anlamsizdir,
        o yuzden None donuyoruz.
    """
    y_skor = np.asarray(y_skor, dtype=float)
    auc = roc_auc_score(y_true, y_skor)

    olasilik_mi = bool(np.all((y_skor >= 0) & (y_skor <= 1)))

    return {
        "model": ad,
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_istatistigi(y_true, y_skor),
        "pr_auc": average_precision_score(y_true, y_skor),
        "brier": brier_score_loss(y_true, y_skor) if olasilik_mi else None,
        "n": len(y_true),
        "temerrut_orani": float(np.mean(y_true)),
    }


def rapor_yazdir(sonuclar: list[dict]) -> pd.DataFrame:
    """Degerlendirme sonuclarini okunakli bir tablo olarak yazdirir."""
    df = pd.DataFrame(sonuclar)
    gosterim = df[["model", "auc", "gini", "ks", "pr_auc", "brier"]].copy()
    for k in ["auc", "gini", "ks", "pr_auc", "brier"]:
        # Brier None olabilir (olasilik olmayan skorlarda) - o hucre bos kalir.
        gosterim[k] = gosterim[k].map(lambda v: round(v, 4) if pd.notna(v) else "-")
    print(gosterim.to_string(index=False))
    return df


def dilim_analizi(y_true, y_skor, dilim_sayisi: int = 10) -> pd.DataFrame:
    """Skoru dilimlere ayirip her dilimin gercek temerrut oranini gosterir.

    Bu tablo, metriklerin ozetleyemedigi seyi gosterir: model riski
    MONOTONIK siralayabiliyor mu? En riskli dilimde temerrut orani en
    yuksek, en guvenli dilimde en dusuk olmali. Kirilma varsa model
    bazi bolgelerde guvenilmez demektir.

    Ayrica LIFT hesaplar: "en riskli %10'luk dilim, ortalamanin kac kati
    riskli?" Is birimlerine model degerini anlatmanin en dogrudan yoludur.
    """
    d = pd.DataFrame({"y": np.asarray(y_true), "skor": np.asarray(y_skor)})

    # qcut ile esit BUYUKLUKTE dilimler. duplicates='drop': skorlar cok
    # yiginlaysa bazi sinirlar cakisir, hata vermek yerine dilim sayisini azaltir.
    d["dilim"] = pd.qcut(d["skor"], dilim_sayisi, labels=False, duplicates="drop")

    genel = d["y"].mean()
    ozet = (
        d.groupby("dilim")
        .agg(musteri=("y", "size"), temerrut=("y", "sum"), oran=("y", "mean"),
             ort_skor=("skor", "mean"))
        .reset_index()
        .sort_values("dilim", ascending=False)
    )
    ozet["oran_yuzde"] = (100 * ozet["oran"]).round(2)
    ozet["lift"] = (ozet["oran"] / genel).round(2)
    return ozet[["dilim", "musteri", "temerrut", "oran_yuzde", "lift", "ort_skor"]]
