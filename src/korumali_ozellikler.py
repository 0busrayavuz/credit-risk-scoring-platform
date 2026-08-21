"""Korumali ozellik politikasi - hangi degisken modele girer, hangisi girmez.

NEDEN AYRI BIR MODUL:
    Ilk denemede cinsiyeti SHAP grafiginde TESADUFEN fark ettik, cikardik ve isi
    bitmis saydik. Sonraki denetimde gorulду ki medeni durum (name_family_status)
    her iki modelde de duruyordu ve ABD'de Equal Credit Opportunity Act bunu
    cinsiyetle AYNI listede, acikca korumali ozellik olarak sayiyor.

    Ders: korumali ozellik denetimi tesadufe birakilamaz. Politika onceden ve
    acikca tanimlanmali, sonra her modele ayni sekilde uygulanmalidir. Bu dosya
    o politikadir.

YASAL CERCEVE (ozet):
    ABD - Equal Credit Opportunity Act korumali ozellikleri tek tek sayar:
        irk, renk, din, ulusal koken, CINSIYET, MEDENI DURUM, YAS,
        kamu yardimi alma durumu.
    AB  - 2004/113/EC direktifi finansal hizmetlerde cinsiyete dayali ayrimi yasaklar.
    TR  - Bankalar ayrimcilik yasagina ve adil kredilendirme beklentilerine tabidir.

    Not: ECOA yasi mutlak olarak yasaklamaz; "ampirik olarak turetilmis,
    istatistiksel gecerliligi gosterilmis" skorlama sistemlerinde yasin
    kullanimina, yaslilari dezavantajli duruma dusurmemek kosuluyla izin verir.
    Cinsiyet ve medeni durumda boyle bir istisna yoktur.

BU PROJEDEKI POLITIKA - iki kademe:

  KADEME 1 - MODELDEN CIKARILIR
    Dogrudan korumali ozellikler ve onlarin yakin ikameleri. Bunlarin risk
    tahminine katkisi olsa bile kullanilmaz; olculdu ki maliyet ihmal edilebilir.

  KADEME 2 - KULLANILIR AMA IZLENIR
    Mesru risk anlami olan, ancak ayrimcilik vekili haline gelebilen degiskenler.
    Cikarilmalari modeli belirgin zayiflatir; bu yuzden kullanilir, fakat grup
    bazli adalet metrikleri panelde SUREKLI izlenir.

  Kademe 2'nin gerekcesi sadece performans degil, olculmus bir gercek:
  cinsiyet modelden cikarildiktan sonra bile kalan degiskenlerden AUC 0,909 ile
  tahmin edilebiliyor. Yani "hassas her seyi sil" stratejisi ayrimciligi yok
  etmez, yalnizca OLCULEMEZ hale getirir - ki bu daha tehlikelidir.
"""

from __future__ import annotations

# --- KADEME 1: modelden cikarilir ------------------------------------------
KADEME_1: dict[str, str] = {
    "code_gender":
        "Cinsiyet. ECOA ve AB 2004/113/EC kapsaminda dogrudan korumali.",
    "name_family_status":
        "Medeni durum. ECOA'da cinsiyetle ayni listede, acikca korumali.",
    "cnt_children":
        "Cocuk sayisi. Medeni durum ve aile yapisinin yakin vekili; "
        "modeldeki katkisi zaten ihmal edilebilir (SHAP siralamasinda ~#149).",
    "cnt_fam_members":
        "Hane buyuklugu. Ayni gerekce (SHAP ~#118).",
}

# --- KADEME 2: kullanilir, izlenir -----------------------------------------
KADEME_2: dict[str, str] = {
    "days_birth":
        "Yas. ECOA, istatistiksel gecerliligi gosterilmis skorlama sistemlerinde "
        "yasin kullanimina izin verir; kredi riskinde mesru ve guclu bir "
        "degiskendir. Yas bandi bazli adalet metrikleri izlenir.",
    "age_years":
        "Yas (yil cinsinden turetilmis). days_birth ile ayni gerekce.",
    "region_rating_client":
        "Bolge derecelendirmesi. Bolgesel iktisadi kosullar mesru risk "
        "bilgisidir; ancak cografya, sosyoekonomik ve etnik ayrismanin vekili "
        "olabilir (adil kredilendirmede 'redlining' endisesi). Izlenir.",
    "region_rating_client_w_city":
        "Sehir duzeltmeli bolge derecelendirmesi. Ayni gerekce.",
}

# --- Panelde ve adalet raporunda izlenen boyutlar ---------------------------
# Bir degiskeni modelden cikarmak, o boyutta ayrimcilik OLMADIGI anlamina
# gelmez - vekiller uzerinden yeniden dogabilir. Bu yuzden KADEME 1'dekiler de
# izlenmeye devam eder; sadece modele girdi olarak verilmez.
IZLENEN_BOYUTLAR: list[str] = [
    "code_gender",
    "name_family_status",
    "age_years",
    "region_rating_client_w_city",
]


def temiz_ozellikler(ozellikler: list[str]) -> list[str]:
    """Kademe 1 korumali ozellikleri listeden cikarir."""
    return [k for k in ozellikler if k not in KADEME_1]


def cikarilanlar(ozellikler: list[str]) -> list[str]:
    """Verilen listede fiilen bulunan Kademe 1 ozelliklerini dondurur."""
    return [k for k in ozellikler if k in KADEME_1]


def politika_yazdir(ozellikler: list[str] | None = None) -> None:
    """Politikayi okunakli bicimde yazdirir."""
    print("KORUMALI OZELLIK POLITIKASI")
    print("-" * 74)
    print("KADEME 1 - modelden cikarilir:")
    for k, gerekce in KADEME_1.items():
        var = "" if ozellikler is None else ("  [veride VAR]" if k in ozellikler
                                             else "  [veride yok]")
        print(f"  {k}{var}")
        print(f"      {gerekce}")
    print("\nKADEME 2 - kullanilir, grup metrikleri izlenir:")
    for k, gerekce in KADEME_2.items():
        print(f"  {k}")
        print(f"      {gerekce}")
    print("-" * 74)
