"""Veri erisim katmani: model tablosunu Postgres'ten cekmek ve bolmek.

Modelleme kodunun veritabani detaylarini bilmemesi icin her sey burada toplanir.
Ileride veri kaynagi degisirse (ornegin bir veri ambari), sadece bu dosya degisir.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split

from src.config import (
    DB_URL,
    ID_COL,
    RANDOM_STATE,
    TARGET,
    TEST_ORAN,
    VALID_ORAN,
)

_engine = None


def motor():
    """SQLAlchemy motorunu dondurur (tek kez olusturulur).

    Her sorguda yeni motor acmak baglanti havuzunu bosa harcar; modul
    seviyesinde tek bir motor tutup yeniden kullaniyoruz.
    """
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URL, pool_pre_ping=True)
    return _engine


def sorgu(sql: str) -> pd.DataFrame:
    """Serbest SQL calistirip DataFrame dondurur (kesif icin)."""
    return pd.read_sql(text(sql), motor())


def model_input_yukle(limit: int | None = None, bellek_optimize: bool = True) -> pd.DataFrame:
    """features.model_input tablosunu yukler.

    Args:
        limit: sadece ilk N satiri cek (hizli deneme icin). None = hepsi.
        bellek_optimize: float64 -> float32 donusumu ve metin -> category.

    Bellek notu:
        307.511 satir x 230 kolon, float64 ile ~570 MB tutar. float32'ye
        dusurmek bunu yariya indirir ve model dogrulugunu pratikte etkilemez -
        kredi skorlamada 7 ondalik basamak hassasiyete ihtiyac yoktur.
    """
    sql = "SELECT * FROM features.model_input"
    if limit:
        sql += f" LIMIT {int(limit)}"

    df = pd.read_sql(text(sql), motor())

    if bellek_optimize:
        for kolon in df.columns:
            if kolon in (ID_COL, TARGET):
                continue
            if pd.api.types.is_float_dtype(df[kolon]):
                df[kolon] = df[kolon].astype("float32")
            elif pd.api.types.is_string_dtype(df[kolon]) or df[kolon].dtype == object:
                # Kategorik kolonlar: hem bellek kazanci hem de XGBoost'un
                # yerlesik kategorik destegini kullanabilmek icin.
                df[kolon] = df[kolon].astype("category")

    return df


def kolon_tipleri(df: pd.DataFrame) -> dict[str, list[str]]:
    """Kolonlari rol ve tipe gore ayirir.

    Modelleme kodunun "hangi kolon sayisal, hangisi kategorik, hangisi
    kullanilmamali" sorusunu tek yerden cevaplar.
    """
    haric = {ID_COL, TARGET}
    ozellikler = [k for k in df.columns if k not in haric]

    kategorik = [k for k in ozellikler if isinstance(df[k].dtype, pd.CategoricalDtype)]
    bool_kolon = [k for k in ozellikler if pd.api.types.is_bool_dtype(df[k])]
    sayisal = [k for k in ozellikler if k not in kategorik and k not in bool_kolon]

    return {
        "ozellikler": ozellikler,
        "sayisal": sayisal,
        "kategorik": kategorik,
        "bool": bool_kolon,
    }


def veri_bol(df: pd.DataFrame, random_state: int = RANDOM_STATE):
    """Veriyi egitim / dogrulama / test olarak boler.

    Neden UC parca:
        train : model bunun uzerinde ogrenir
        valid : model secimi ve hiperparametre ayari burada yapilir
        test  : sadece EN SONDA, bir kez bakilir

    Eger hiperparametreleri valid uzerinde ayarlayip yine valid uzerinde
    rapor verirsen, o skor iyimser cikar - modeli o kumeye gore secmis
    olursun. Test kumesine dokunmamak, "gercek hayatta ne olur" sorusunun
    tek durust cevabidir.

    stratify=y:
        Temerrut orani %8. Rastgele bolmede bir parcaya %6, digerine %10
        dusebilir ve karsilastirmalar anlamsizlasir. stratify, hedef
        dagilimini uc parcada da ayni tutar.
    """
    y = df[TARGET]

    # Once test'i ayir.
    gecici, test = train_test_split(
        df, test_size=TEST_ORAN, stratify=y, random_state=random_state
    )

    # Kalan kisimdan valid ayir. TOPLAMIN %20'si olmasi icin oran duzeltilir:
    # kalan %80 icinde %25 = toplamin %20'si.
    valid_oran_ic = VALID_ORAN / (1 - TEST_ORAN)
    train, valid = train_test_split(
        gecici,
        test_size=valid_oran_ic,
        stratify=gecici[TARGET],
        random_state=random_state,
    )

    return train, valid, test


def ozet(df: pd.DataFrame, ad: str = "veri") -> None:
    """Bir veri kumesinin ozetini yazdirir."""
    n = len(df)
    pozitif = int(df[TARGET].sum())
    print(
        f"{ad:8s} {n:>7,} satir | {pozitif:>6,} temerrut "
        f"({100 * pozitif / n:.2f}%) | {df.shape[1]} kolon"
    )
