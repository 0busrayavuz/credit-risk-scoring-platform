"""
data/raw altindaki CSV dosyalarinin basliklarindan PostgreSQL CREATE TABLE
ifadeleri uretir ve sql/00_init/01_create_raw_tables.sql dosyasina yazar.

Neden: application_train.csv'nin 122 kolonu var, 8 dosyada toplam ~350 kolon.
Bunu elle yazmak saatler surer ve mutlaka yazim hatasi girer. Semayi veriden
turetmek hem hizli hem de veri degisirse tekrar calistirilabilir.

Calistirmak icin:
    .\.venv\Scripts\python.exe src\generate_ddl.py
"""

from pathlib import Path

import pandas as pd

# --- Yollar -----------------------------------------------------------------
# __file__ = bu dosyanin kendi yolu. resolve() mutlak yola cevirir,
# parent.parent ise src/ klasorunden bir ust seviyeye, proje kokune cikar.
# Boylece script'i hangi klasorden calistirirsan calistir yollar dogru olur.
PROJE_KOK = Path(__file__).resolve().parent.parent
RAW_DIR = PROJE_KOK / "data" / "raw"
CIKTI = PROJE_KOK / "sql" / "00_init" / "01_create_raw_tables.sql"

# --- Ayarlar ----------------------------------------------------------------
# Bu iki dosya veri tablosu degil: biri kolon aciklamalari (dokuman),
# digeri Kaggle'in ornek gonderim sablonu. Ham semaya girmelerine gerek yok.
ATLA = {"HomeCredit_columns_description.csv", "sample_submission.csv"}

# Tip tahmini icin dosyanin tamamini okumak israf (2,5 GB).
# Ilk 200 bin satir tip cikarimi icin fazlasiyla yeterli.
ORNEK_SATIR = 200_000

# Semayi 'raw' altinda topluyoruz. Ilerideki turetilmis tablolar
# 'features' semasina gidecek. Ham ve islenmis veriyi ayirmak,
# "bu tabloya guvenebilir miyim?" sorusunu bastan cevaplar.
SEMA = "raw"


def postgres_tipi(seri: pd.Series) -> str:
    """Bir pandas kolonunun PostgreSQL tip karsiligini dondurur.

    Ham katmanda amac veriyi kaybetmemek, guzellestirmek degil.
    O yuzden genis tipler seciyoruz: BIGINT (INTEGER degil),
    DOUBLE PRECISION (NUMERIC degil), TEXT (VARCHAR(n) degil).
    """
    if pd.api.types.is_bool_dtype(seri):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(seri):
        return "BIGINT"
    if pd.api.types.is_float_dtype(seri):
        # Dikkat: float64 her zaman "ondalikli sayi" demek degil.
        # Home Credit'te bircok tamsayi kolon bos deger icerdigi icin
        # pandas onlari float64 okur (ornek: CNT_FAM_MEMBERS).
        # DOUBLE PRECISION ikisini de sorunsuz tutar.
        return "DOUBLE PRECISION"
    return "TEXT"


def kolon_adini_duzelt(ad: str) -> str:
    """CSV baslgini PostgreSQL'de rahat kullanilacak bir isme cevirir.

    ONEMLI - Oracle'dan gelenlerin sik dustugu tuzak:
    Oracle tirnaksiz tanimlayicilari BUYUK harfe katlar,
    PostgreSQL ise KUCUK harfe katlar.
    Tabloyu "SK_ID_CURR" diye tirnakli olusturursan, o gunden sonra
    her sorguda tirnak yazmak zorunda kalirsin. Bastan kucuk harf yap.
    """
    temiz = ad.strip().lower()
    # Harf, rakam ve alt cizgi disindaki her seyi alt cizgiye cevir.
    # (Home Credit basliklari zaten temiz, ama bu script'i baska bir
    #  veri setinde de kullanabilmek icin savunmaci davraniyoruz.)
    temiz = "".join(k if (k.isalnum() or k == "_") else "_" for k in temiz)
    return temiz


def tablo_ddl_uret(csv_yolu: Path) -> tuple[str, int]:
    """Tek bir CSV icin DROP + CREATE TABLE metnini ve kolon sayisini dondurur."""
    ornek = pd.read_csv(csv_yolu, nrows=ORNEK_SATIR)

    # Dosya adi -> tablo adi. POS_CASH_balance.csv -> pos_cash_balance
    # .stem uzantisiz dosya adini verir.
    tablo = csv_yolu.stem.lower()

    # Her kolon icin "    ad    TIP" satiri hazirla.
    # En uzun kolon adina gore hizalayarak okunakli bir SQL uretiyoruz.
    adlar = [kolon_adini_duzelt(k) for k in ornek.columns]
    genislik = max(len(a) for a in adlar)

    satirlar = [
        f"    {ad:<{genislik}}  {postgres_tipi(ornek[orijinal])}"
        for ad, orijinal in zip(adlar, ornek.columns)
    ]

    ddl = (
        f"-- {csv_yolu.name} ({len(adlar)} kolon)\n"
        f"DROP TABLE IF EXISTS {SEMA}.{tablo};\n"
        f"CREATE TABLE {SEMA}.{tablo} (\n"
        + ",\n".join(satirlar)
        + "\n);\n"
    )
    return ddl, len(adlar)


def main() -> None:
    """RAW_DIR'deki her CSV icin DDL uretip tek bir .sql dosyasina yazar."""
    if not RAW_DIR.exists():
        raise SystemExit(f"Ham veri klasoru bulunamadi: {RAW_DIR}")

    # sorted() ile alfabetik sira: cikti dosyasi her calistirmada ayni olsun,
    # boylece git diff'te gereksiz degisiklik gorunmez.
    csv_dosyalari = sorted(p for p in RAW_DIR.glob("*.csv") if p.name not in ATLA)

    if not csv_dosyalari:
        raise SystemExit(f"{RAW_DIR} icinde islenecek CSV yok.")

    parcalar = [
        "-- OTOMATIK URETILDI - src/generate_ddl.py\n"
        "-- Bu dosyayi elle duzenleme; script'i tekrar calistir.\n\n"
        f"CREATE SCHEMA IF NOT EXISTS {SEMA};\n"
    ]

    toplam_kolon = 0
    for yol in csv_dosyalari:
        ddl, kolon_sayisi = tablo_ddl_uret(yol)
        parcalar.append(ddl)
        toplam_kolon += kolon_sayisi
        print(f"  {yol.stem.lower():28s} {kolon_sayisi:>4} kolon")

    # parents=True: sql/00_init yoksa olustur. exist_ok=True: varsa hata verme.
    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    CIKTI.write_text("\n".join(parcalar), encoding="utf-8")

    print(f"\n{len(csv_dosyalari)} tablo, {toplam_kolon} kolon yazildi:")
    print(f"  {CIKTI}")


if __name__ == "__main__":
    main()
