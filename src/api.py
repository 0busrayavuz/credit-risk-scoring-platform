"""Kredi risk skorlama servisi (FastAPI).

NE YAPAR:
    Bir basvuruyu alir, temerrut olasiligini tahmin eder, kar bazli esige gore
    ONAY/RED karari verir ve karari SHAP ile GEREKCELENDIRIR.

    Gerekce kismi kritik: bircok ulkede reddedilen basvuru sahibine sebep
    bildirmek yasal zorunluluktur. Sadece skor donen bir servis bu ihtiyaci
    karsilamaz.

IKI SKORLAMA YOLU - VE NEDENI (uretim ML'inin gercek problemi):
    Model 227 degisken istiyor ama bunlarin cogu ham basvuru formunda YOKTUR;
    SQL'de milyonlarca satirdan turetilen ozet metriklerdir (musterinin son
    12 aydaki taksit gecikme orani gibi). Basvuru aninda bunlari kullanicidan
    isteyemezsiniz.

    Bu, "feature store" problemidir. Iki yaklasim sunuyoruz:

    1. /skorla/musteri/{sk_id_curr}
       Ozellikler veritabanindan (features.model_input) okunur.
       Gercek bir bankada bu, onceden hesaplanmis bir oznitelik deposudur.
       AVANTAJ: egitimde kullanilan hesaplamanin AYNISI kullanilir, dolayisiyla
       "training-serving skew" (egitim-servis tutarsizligi) riski yoktur.

    2. /skorla
       Ozellikler dogrudan istekte gonderilir. Eksik birakilanlar NaN olur;
       XGBoost eksik degerleri zaten yerlesik olarak isler.
       AVANTAJ: veritabanina bagimli degil, yeni musteride de calisir.
       RISK: gonderen taraf metrikleri farkli hesaplarsa model sessizce
       yanlis skor uretir. Uretimde bu en sik goz ardi edilen hata kaynagidir.

HANGI MODEL:
    models/xgboost_adil.json - cinsiyet degiskeni CIKARILMIS surum.
    Adalet analizi (src/adalet_analizi.py) bunun maliyetinin ihmal edilebilir
    oldugunu gosterdi (AUC -0.0013). Servise konan model bu olmali.

Calistirmak icin (proje kokunden):
    .\\.venv\\Scripts\\python.exe -m uvicorn src.api:app --reload --port 8000
Sonra: http://localhost:8000/docs
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field

from src.config import MODELS_DIR, REPORTS_DIR
from src.data import model_input_yukle, sorgu

# Servisin durumu. Model ve sema BIR KEZ, acilista yuklenir.
# Her istekte yeniden yuklemek saniyeler surerdi.
DURUM: dict[str, Any] = {}


def _esik_oku() -> float:
    """Kar optimizasyonundan gelen esigi okur."""
    yol = REPORTS_DIR / "adalet_analizi.json"
    if yol.exists():
        return float(json.loads(yol.read_text(encoding="utf-8"))["esikler"]["cinsiyetsiz"])
    return 0.145  # yedek deger


@asynccontextmanager
async def yasam_dongusu(app: FastAPI):
    """Acilis ve kapanis islemleri."""
    model = xgb.XGBClassifier()
    model.load_model(str(MODELS_DIR / "xgboost_adil.json"))

    # Kategorik kolonlarin hangi degerleri alabilecegini bilmemiz gerekiyor:
    # XGBoost, egitimde gordugu kategori tanimlariyla ayni tanimi bekler.
    # Kucuk bir ornek cekip dtype sablonu olusturuyoruz.
    sablon = model_input_yukle(limit=2000)

    DURUM["model"] = model
    DURUM["ozellikler"] = list(model.get_booster().feature_names)
    DURUM["sablon"] = sablon
    DURUM["esik"] = _esik_oku()

    # EGITIMDEKI EKSIKLIK ORANLARI - veri kalitesi denetimi icin.
    # Neden gerekli: XGBoost eksik degerleri "isler" ama bu, her eksigin
    # guvenli oldugu anlamina GELMEZ. Egitimde neredeyse hic bos gormedigi
    # bir alan servis sirasinda bos gelirse, model o dala dair gucsuz bir
    # ogrenmeye dayanarak uc bir skor uretebilir.
    # Olculdu: sadece 10 ozellik gonderilen bir istekte, bos birakilan
    # organization_type tek basina +2.75 log-odds katki uretti ve tahmini
    # %88'e cikardi - gercek bir risk sinyali degil, bir yapaylik.
    DURUM["egitim_eksiklik"] = sablon[DURUM["ozellikler"]].isna().mean().to_dict()
    # Erken durdurma nedeniyle SHAP hesabinda ayni agac araligi kullanilmali.
    en_iyi = getattr(model, "best_iteration", None)
    DURUM["aralik"] = (0, int(en_iyi) + 1) if en_iyi is not None else None

    print(f"[servis] model yuklendi: {len(DURUM['ozellikler'])} degisken | "
          f"esik {DURUM['esik']:.3f} | agac araligi {DURUM['aralik']}")
    yield
    DURUM.clear()


app = FastAPI(
    title="Kredi Risk Skorlama Servisi",
    description=(
        "Home Credit veri seti üzerinde eğitilmiş XGBoost modeliyle temerrüt "
        "olasılığı tahmini, kâr bazlı karar ve SHAP tabanlı gerekçelendirme."
    ),
    version="1.0.0",
    lifespan=yasam_dongusu,
)


# ---------------------------------------------------------------------------
# Şema tanımları
# ---------------------------------------------------------------------------
class Faktor(BaseModel):
    degisken: str = Field(description="Değişken adı")
    deger: str | None = Field(description="Bu başvurudaki değeri")
    katki: float = Field(description="SHAP katkısı (log-odds). + riski artırır")
    yon: str = Field(description="'riski artırdı' veya 'riski azalttı'")


class SkorYaniti(BaseModel):
    sk_id_curr: int | None = None
    temerrut_olasiligi: float = Field(description="0-1 arası tahmini temerrüt olasılığı")
    karar: str = Field(description="ONAY, RED veya İNCELE")
    esik: float = Field(description="Kâr optimizasyonundan gelen kesim noktası")
    risk_bandi: str
    guvenilirlik: str = Field(
        description="YUKSEK / ORTA / DUSUK — girdi verisinin eğitim verisine benzerliği"
    )
    uyarilar: list[str] = Field(
        default_factory=list,
        description="Veri kalitesi uyarıları. Doluysa skor temkinli yorumlanmalı.",
    )
    gerekce: list[Faktor] = Field(description="Karara en çok etki eden faktörler")
    kullanilan_degisken: int
    eksik_degisken: int


class SkorIstegi(BaseModel):
    ozellikler: dict[str, float | str | None] = Field(
        description=(
            "Değişken adı → değer. Eksik bırakılanlar NaN olarak işlenir; "
            "XGBoost eksik değerleri yerleşik olarak ele alır."
        ),
        examples=[{
            "ext_source_mean": 0.42,
            "ext_source_min": 0.31,
            "amt_credit": 450000,
            "amt_annuity": 24700,
            "amt_income_total": 180000,
            "days_birth": -14500,
            "credit_income_ratio": 2.5,
            "annuity_income_ratio": 0.137,
            "inst_late_ratio_1y": 0.05,
            "prev_refused_ratio": 0.2,
        }],
    )


def _risk_bandi(p: float) -> str:
    if p < 0.02:
        return "çok düşük"
    if p < 0.05:
        return "düşük"
    if p < 0.10:
        return "orta"
    if p < 0.20:
        return "yüksek"
    return "çok yüksek"


def _deger_yaz(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):,.4f}" if abs(v) < 100 else f"{float(v):,.0f}"
    return str(v)


def _cerceve_kur(degerler: dict) -> pd.DataFrame:
    """Gelen degerlerden, modelin bekledigi tam sema ile bir satir olusturur.

    Kritik nokta: kategorik kolonlarin dtype'i egitimdekiyle AYNI olmali.
    Sablondan kategori tanimlarini aliyoruz; boyle yapmazsak XGBoost
    'categorical data mismatch' hatasi verir ya da daha kotusu, kategorileri
    farkli kodlayip sessizce yanlis skor uretir.
    """
    ozellikler = DURUM["ozellikler"]
    sablon = DURUM["sablon"]

    satir = {}
    for k in ozellikler:
        v = degerler.get(k, None)
        satir[k] = np.nan if v is None else v

    df = pd.DataFrame([satir], columns=ozellikler)
    for k in ozellikler:
        s = sablon[k]
        if isinstance(s.dtype, pd.CategoricalDtype):
            df[k] = pd.Categorical(df[k], categories=s.cat.categories)
        elif pd.api.types.is_bool_dtype(s):
            df[k] = df[k].astype("float32")
        else:
            df[k] = pd.to_numeric(df[k], errors="coerce").astype("float32")
    return df


def _veri_kalitesi(df: pd.DataFrame) -> tuple[str, list[str]]:
    """Girdinin egitim verisine ne kadar benzedigini denetler.

    Odak nokta BEKLENMEDIK eksikler: egitimde neredeyse hic bos olmayan bir
    alanin serviste bos gelmesi. Bunlar modelin zayif ogrendigi dallardir ve
    uc skorlar uretebilir. Egitimde zaten cogunlukla bos olan alanlarin
    (ornegin kredi karti metrikleri, musterilerin %72'sinde yok) bos gelmesi
    normaldir ve uyari uretmez.
    """
    egitim = DURUM["egitim_eksiklik"]
    satir = df.iloc[0]

    beklenmedik = [
        k for k in DURUM["ozellikler"]
        if pd.isna(satir[k]) and egitim.get(k, 1.0) < 0.02
    ]
    uyarilar: list[str] = []

    if beklenmedik:
        uyarilar.append(
            f"{len(beklenmedik)} değişken beklenmedik şekilde boş "
            f"(eğitimde neredeyse hiç boş değildi): "
            + ", ".join(beklenmedik[:6]) + (" ..." if len(beklenmedik) > 6 else "")
        )

    if len(beklenmedik) > 30:
        guven = "DUSUK"
        uyarilar.append(
            "Skor güvenilir değil. Bu kadar çok temel alan eksikken model, "
            "eğitimde nadir gördüğü dallara düşer ve uç değerler üretebilir. "
            "Karar için /skorla/musteri uç noktasını kullanın veya eksik "
            "alanları tamamlayın."
        )
    elif len(beklenmedik) > 8:
        guven = "ORTA"
    else:
        guven = "YUKSEK"

    return guven, uyarilar


def _skorla(df: pd.DataFrame, sk_id: int | None, ham_degerler: dict) -> SkorYaniti:
    model = DURUM["model"]
    ozellikler = DURUM["ozellikler"]
    esik = DURUM["esik"]

    olasilik = float(model.predict_proba(df)[0, 1])
    guven, uyarilar = _veri_kalitesi(df)

    # SHAP ile gerekce - erken durdurma nedeniyle ayni agac araligi kullanilir.
    dmat = xgb.DMatrix(df, enable_categorical=True)
    ek = {"iteration_range": DURUM["aralik"]} if DURUM["aralik"] else {}
    katkilar = model.get_booster().predict(dmat, pred_contribs=True, **ek)[0, :-1]

    s = pd.Series(katkilar, index=ozellikler)
    ust = s.reindex(s.abs().sort_values(ascending=False).index).head(8)

    gerekce = [
        Faktor(
            degisken=k,
            deger=_deger_yaz(ham_degerler.get(k, df.iloc[0][k])),
            katki=round(float(v), 4),
            yon="riski artırdı" if v > 0 else "riski azalttı",
        )
        for k, v in ust.items()
    ]

    # Guvenilirlik dusukse otomatik ONAY/RED vermiyoruz.
    # Bir kredi kararinda "emin degilim" demek, sessizce yanlis cevap
    # vermekten iyidir; bu tur basvurular insan incelemesine yonlendirilir.
    if guven == "DUSUK":
        karar = "İNCELE"
    else:
        karar = "ONAY" if olasilik < esik else "RED"

    eksik = int(df.iloc[0].isna().sum())
    return SkorYaniti(
        sk_id_curr=sk_id,
        temerrut_olasiligi=round(olasilik, 6),
        karar=karar,
        esik=round(esik, 4),
        risk_bandi=_risk_bandi(olasilik),
        guvenilirlik=guven,
        uyarilar=uyarilar,
        gerekce=gerekce,
        kullanilan_degisken=len(ozellikler) - eksik,
        eksik_degisken=eksik,
    )


# ---------------------------------------------------------------------------
# Uç noktalar
# ---------------------------------------------------------------------------
@app.get("/saglik", tags=["servis"], summary="Servis ve model durumu")
def saglik() -> dict:
    """Servisin ayakta olup olmadigini ve modelin yuklendigini dogrular."""
    return {
        "durum": "calisiyor" if DURUM.get("model") is not None else "model yuklenmedi",
        "degisken_sayisi": len(DURUM.get("ozellikler", [])),
        "esik": DURUM.get("esik"),
    }


@app.get("/model", tags=["servis"], summary="Model bilgileri")
def model_bilgi() -> dict:
    """Modelin kimligi ve karar esigi hakkinda bilgi."""
    return {
        "model": "XGBoost (cinsiyet değişkeni çıkarılmış)",
        "dosya": "models/xgboost_adil.json",
        "degisken_sayisi": len(DURUM["ozellikler"]),
        "karar_esigi": DURUM["esik"],
        "esik_nasil_secildi": (
            "Doğrulama kümesinde beklenen portföy kârı maksimize edilerek "
            "(marj %12, LGD %65 varsayımıyla). 0,5 gibi varsayılan bir eşik "
            "iki hata türünün eşit maliyetli olduğunu varsayar; kredi riskinde "
            "bu yanlıştır."
        ),
        "adalet_notu": (
            "code_gender modelden çıkarıldı (AUC maliyeti 0,0013). Ancak kalan "
            "değişkenlerden cinsiyet AUC 0,911 ile tahmin edilebiliyor, bu yüzden "
            "grup bazlı adalet metrikleri izlenmeye devam etmelidir."
        ),
    }


@app.post("/skorla", response_model=SkorYaniti, tags=["skorlama"],
          summary="Özellikleri doğrudan vererek skorla")
def skorla(istek: SkorIstegi) -> SkorYaniti:
    """Gönderilen özelliklerle skorlar. Eksik özellikler NaN olarak işlenir.

    Yeni müşteriler için uygundur (veritabanında kaydı yoktur). Karşılığında
    gönderen tarafın metrikleri eğitimdekiyle aynı şekilde hesaplaması gerekir.
    """
    if not istek.ozellikler:
        raise HTTPException(400, "En az bir özellik gönderilmeli.")

    bilinmeyen = set(istek.ozellikler) - set(DURUM["ozellikler"])
    if bilinmeyen:
        raise HTTPException(
            422,
            f"Model bu değişkenleri tanımıyor: {sorted(bilinmeyen)[:10]}"
            + (" ..." if len(bilinmeyen) > 10 else ""),
        )

    df = _cerceve_kur(istek.ozellikler)
    return _skorla(df, None, istek.ozellikler)


@app.post("/skorla/musteri/{sk_id_curr}", response_model=SkorYaniti, tags=["skorlama"],
          summary="Mevcut müşteriyi veritabanından okuyarak skorla")
def skorla_musteri(
    sk_id_curr: int = Path(description="Müşteri kimliği", examples=[100002]),
) -> SkorYaniti:
    """Özellikleri features.model_input tablosundan okur ve skorlar.

    Eğitimde kullanılan hesaplamanın aynısı kullanıldığı için
    eğitim-servis tutarsızlığı (training-serving skew) riski yoktur.
    """
    df_ham = sorgu(
        f"SELECT * FROM features.model_input WHERE sk_id_curr = {int(sk_id_curr)}"
    )
    if df_ham.empty:
        raise HTTPException(404, f"Müşteri bulunamadı: {sk_id_curr}")

    ham = df_ham.iloc[0].to_dict()
    df = _cerceve_kur(ham)
    return _skorla(df, int(sk_id_curr), ham)
