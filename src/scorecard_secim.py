"""Skorkart degisken secimi: korelasyon budama + isaret duzeltme.

NEDEN GEREKLI:
    IV'ye gore secilen 108 degiskenle egitilen ilk skorkartta, degiskenlerin
    35'inde katsayi isareti TERS cikti. Ornegin kredi karti limit kullanimi
    arttikca musteri DAHA COK puan aliyordu - is mantiginin tam tersi.

    Sebep coklu dogrusal baglanti (multicollinearity): oznitelik uretirken
    hem sayi hem oran versiyonlarini urettik (inst_late_count / inst_late_ratio),
    ustelik age_years ile days_birth gibi birebir ayni bilgiyi tasiyan ciftler var.
    Regresyon aciklama gucunu bu benzer degiskenler arasinda keyfi bolustururken
    bazilarinin isareti tersine doner.

    Bankacilikta bu kabul edilemez: skorkart bir kredi komitesine sunulur ve
    her satirinin is mantigina uygun olmasi beklenir. "Daha cok gecikmis
    musteriye daha cok puan" diyen bir tablo reddedilir - istatistiksel olarak
    ne kadar iyi calisirsa calissin.

WOE ISARET KURALI:
    optbinning'de WOE = ln(iyi orani / kotu orani). Yani YUKSEK WOE = DUSUK RISK.
    Temerrudu tahmin eden lojistik regresyonda, WOE'ye ait TUM katsayilar
    NEGATIF olmalidir. Pozitif katsayi = o degiskenin yonu bozuk demektir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def korelasyon_buda(
    X_woe: pd.DataFrame, iv: pd.Series, esik: float = 0.75, ayrinti: bool = True
) -> list[str]:
    """Birbirine cok benzeyen degiskenlerden IV'si yuksek olani tutar.

    Yontem (acgozlu / greedy):
        Degiskenleri IV'ye gore buyukten kucuge sirala. Sirayla ele al:
        eger halihazirda tutulanlardan herhangi biriyle korelasyonu esigi
        asiyorsa ele, degilse tut.

    Neden IV'si yuksek olani tutuyoruz:
        Ikisi de ayni bilgiyi tasiyorsa, tek basina daha guclu ayirt eden
        hangisiyse onu birakmak mantikli.

    esik=0.75 secimi:
        Sektorde 0.70-0.80 arasi yaygin. Cok dusuk secersen faydali
        degiskenleri kaybedersin, cok yuksek secersen isaret sorunu surer.
    """
    sirali = iv.sort_values(ascending=False).index.tolist()
    sirali = [d for d in sirali if d in X_woe.columns]

    korelasyon = X_woe[sirali].corr().abs()

    tutulan: list[str] = []
    elenen: list[tuple[str, str, float]] = []

    for aday in sirali:
        if not tutulan:
            tutulan.append(aday)
            continue
        r = korelasyon.loc[aday, tutulan]
        if (r > esik).any():
            suclu = r.idxmax()
            elenen.append((aday, suclu, float(r.max())))
        else:
            tutulan.append(aday)

    if ayrinti:
        print(f"  korelasyon budama (esik {esik}): "
              f"{len(sirali)} -> {len(tutulan)} degisken")
        if elenen:
            print("  elenen ilk 12 (elenen <- benzedigi degisken, korelasyon):")
            for a, b, r in elenen[:12]:
                print(f"    {a:34s} <- {b:34s} r={r:.3f}")

    return tutulan


def isaret_duzelt(
    X_woe: pd.DataFrame,
    y: np.ndarray,
    degiskenler: list[str],
    random_state: int = 42,
    max_tur: int = 200,
    ayrinti: bool = True,
) -> tuple[list[str], LogisticRegression]:
    """Katsayi isareti ters olan degiskenleri teker teker eleyerek yeniden egitir.

    Neden TEKER TEKER:
        Ters isaretli degiskenlerin hepsini birden atmak yanlis olur; cogu
        zaman tek bir problemli degiskeni cikarmak, ona bagli digerlerinin
        isaretini kendiliginden duzeltir. Hepsini birden atarsak gereksiz
        yere bilgi kaybederiz.

    Hangisi once atilir:
        En buyuk pozitif katsayili olan - yani bozulmanin merkezindeki.
    """
    kalan = list(degiskenler)
    model = None

    for tur in range(max_tur):
        model = LogisticRegression(
            max_iter=2000, random_state=random_state, solver="lbfgs"
        )
        model.fit(X_woe[kalan], y)

        katsayi = pd.Series(model.coef_[0], index=kalan)
        ters = katsayi[katsayi > 0]

        if ters.empty:
            if ayrinti:
                print(f"  isaret duzeltme: {tur} tur, "
                      f"{len(degiskenler)} -> {len(kalan)} degisken")
                print("  tum katsayilar negatif = tum degiskenler is mantigina uygun")
            return kalan, model

        if len(kalan) <= 2:
            break

        kalan.remove(ters.idxmax())

    if ayrinti:
        print(f"  UYARI: {max_tur} turda tum isaretler duzelmedi.")
    return kalan, model


def kalite_raporu(X_woe: pd.DataFrame, y: np.ndarray, degiskenler: list[str],
                  model: LogisticRegression) -> pd.DataFrame:
    """Secilen degiskenlerin katsayi ve yon ozetini dondurur."""
    katsayi = pd.Series(model.coef_[0], index=degiskenler)
    return (
        pd.DataFrame({
            "degisken": degiskenler,
            "katsayi": katsayi.values,
            "onem": np.abs(katsayi.values),
        })
        .sort_values("onem", ascending=False)
        .reset_index(drop=True)
    )
