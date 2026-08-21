"""Merkezi yapilandirma: yollar, veritabani baglantisi, sabitler.

Neden ayri bir dosya: veritabani sifresi, rastgelelik tohumu, hedef kolon adi
gibi degerler bircok yerde kullanilir. Tek yerde tutulmazsa, birini
degistirdiginde digerlerini unutursun ve sessizce tutarsiz sonuclar uretirsin.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Yollar -----------------------------------------------------------------
PROJE_KOK = Path(__file__).resolve().parent.parent
DATA_DIR = PROJE_KOK / "data"
MODELS_DIR = PROJE_KOK / "models"
REPORTS_DIR = PROJE_KOK / "reports"

for _d in (MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Veritabani -------------------------------------------------------------
# .env dosyasini acikca yol vererek yukluyoruz. load_dotenv() varsayilan olarak
# CALISMA DIZININDEN yukari dogru arar; script'i baska bir klasorden
# calistirdiginda dosyayi bulamaz ve sessizce bos degerlerle devam eder.
load_dotenv(PROJE_KOK / ".env", override=True)

DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5433")
DB_NAME = os.getenv("POSTGRES_DB")

if not all([DB_USER, DB_PASSWORD, DB_NAME]):
    raise RuntimeError(
        f".env okunamadi veya eksik: {PROJE_KOK / '.env'}\n"
        ".env.example dosyasini .env olarak kopyalayip doldurun."
    )

DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- Modelleme sabitleri ----------------------------------------------------
RANDOM_STATE = 42          # tekrarlanabilirlik icin sabit tohum
TARGET = "target"
ID_COL = "sk_id_curr"

# Veri bolme oranlari.
# test  : final degerlendirme icin ayrilir, egitim sirasinda ASLA kullanilmaz
# valid : model secimi ve hiperparametre ayari icin
TEST_ORAN = 0.20
VALID_ORAN = 0.20          # kalan %80'in icinden degil, TOPLAMIN %20'si
