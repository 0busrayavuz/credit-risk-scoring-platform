"""Skorlama servisi testleri.

FastAPI'nin TestClient'i uygulamayi SUREC ICINDE calistirir - ayrica bir
sunucu baslatmak gerekmez. Bu, testleri CI ortaminda da calistirilabilir kilar.

Not: testler gercek veritabanina ve egitilmis modele ihtiyac duyar; bu bir
BUTUNLESME (integration) testi setidir, birim testi degil. Kredi skorlama
gibi bir sistemde asil risk, parcalarin tek tek degil BIRLIKTE yanlis
calismasidir - orneklerle: model ile veritabani semasinin ayrisması, ya da
kategorik kolon tanimlarinin egitimdekiyle eslesmemesi.

Calistirmak icin (proje kokunden):
    .\\.venv\\Scripts\\python.exe -m pytest tests -v
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import app

# Bu dosyadaki her test calisan bir PostgreSQL ve egitilmis modeller ister.
# CI'da atlanir; yerelde calisir. Gerekce icin pytest.ini'ye bakin.
pytestmark = pytest.mark.veritabani


@pytest.fixture(scope="module")
def istemci():
    # 'with' blogu lifespan'i tetikler: model ve sema yuklenir.
    with TestClient(app) as c:
        yield c


def test_saglik(istemci):
    y = istemci.get("/saglik")
    assert y.status_code == 200
    veri = y.json()
    assert veri["durum"] == "calisiyor"
    assert veri["uyum"] == "gecti"
    # Degisken sayisini SABIT yazmiyoruz.
    # Ilk surumde 227 diye sabitlenmisti; korumali ozellik politikasi
    # genisleyince (cinsiyete medeni durum ve aile buyuklugu eklenince) test
    # kirildi - oysa DAVRANIS dogruydu, degisen yalnizca bir sayiydi.
    # Anlamli olan sayi degil, korumali ozelligin modelde OLMAMASI.
    assert 150 < veri["degisken_sayisi"] < 230, "degisken sayisi makul aralikta olmali"
    assert 0 < veri["esik"] < 0.5, "esik 0-0.5 araliginda olmali"


def test_korumali_ozellik_kullanilmiyor(istemci):
    """Servisteki model hicbir Kademe 1 korumali ozelligi kullanmamali.

    Bu, projenin en kritik uyum kontrolu. Servis zaten acilista ayni denetimi
    yapip ihlal varsa BASLAMIYOR; bu test o mekanizmanin calistigini dogrular.
    Yanlis model dosyasini gostermek (xgboost_adil.json yerine xgboost.json)
    tek satirlik bir hatadir, ama sonucu cinsiyete gore kredi karari veren
    canli bir sistemdir.
    """
    from src.api import DURUM
    from src.korumali_ozellikler import KADEME_1

    ihlal = [k for k in DURUM["ozellikler"] if k in KADEME_1]
    assert ihlal == [], f"servisteki model korumali ozellik iceriyor: {ihlal}"


def test_model_bilgisi_politikayi_bildiriyor(istemci):
    """/model uc noktasi korumali ozellik politikasini raporlamali.

    Bir denetci, kodu okumadan servisin hangi politikayi uyguladigini
    gorebilmeli - model yonetisiminin temel gereklerinden biri.
    """
    veri = istemci.get("/model").json()
    assert veri["karar_esigi"] < 0.5
    politika = veri["korumali_ozellik_politikasi"]
    assert "code_gender" in politika["cikarilan"]
    assert "name_family_status" in politika["cikarilan"]
    assert politika["uyum_denetimi"].startswith("geçti")


def test_mevcut_musteri_skorlanir(istemci):
    y = istemci.post("/skorla/musteri/100002")
    assert y.status_code == 200
    veri = y.json()
    assert veri["sk_id_curr"] == 100002
    assert 0 <= veri["temerrut_olasiligi"] <= 1
    assert veri["karar"] in {"ONAY", "RED", "İNCELE"}
    assert veri["guvenilirlik"] == "YUKSEK", "tam ozellikli kayit yuksek guvenilir olmali"
    assert len(veri["gerekce"]) > 0, "her karar gerekcelendirilmeli"


def test_gerekce_shap_toplanabilirligi(istemci):
    """Gerekce faktorleri gercekten en etkili olanlar mi?

    SHAP katkilari mutlak degere gore siralanmis olmali.
    """
    veri = istemci.post("/skorla/musteri/100002").json()
    katkilar = [abs(f["katki"]) for f in veri["gerekce"]]
    assert katkilar == sorted(katkilar, reverse=True), "gerekce etkiye gore sirali degil"


def test_olmayan_musteri_404(istemci):
    y = istemci.post("/skorla/musteri/999999999")
    assert y.status_code == 404


def test_tanimsiz_degisken_reddedilir(istemci):
    """Modelin tanimadigi bir alan sessizce yok sayilmamali.

    Sessizce yok saymak, gonderen tarafin yazim hatasini fark etmemesine
    ve modelin o bilgiyi hic gormedigini bilmemesine yol acar.
    """
    y = istemci.post("/skorla", json={"ozellikler": {"uydurma_kolon": 1}})
    assert y.status_code == 422


def test_bos_istek_reddedilir(istemci):
    assert istemci.post("/skorla", json={"ozellikler": {}}).status_code == 400


def test_veri_kalitesi_korumasi(istemci):
    """Cok fazla temel alan eksikken servis kesin karar VERMEMELI.

    Olculdu: sadece 10 ozellik gonderildiginde model %88 temerrut tahmin
    ediyor, ama bunun buyuk kismi eksik alanlarin yarattigi yapaylik
    (organization_type'in bos olmasi tek basina +2.75 log-odds katki uretti).
    Boyle bir skora dayanarak RED demek, egitim-servis tutarsizligini
    musteriye fatura etmek olurdu.
    """
    istek = {"ozellikler": {
        "ext_source_mean": 0.18, "ext_source_min": 0.09, "amt_credit": 900000,
        "amt_annuity": 48000, "amt_income_total": 135000, "days_birth": -9500,
        "credit_income_ratio": 6.7, "annuity_income_ratio": 0.356,
        "inst_late_ratio_1y": 0.42, "prev_refused_ratio": 0.6,
    }}
    veri = istemci.post("/skorla", json=istek).json()
    assert veri["guvenilirlik"] == "DUSUK"
    assert veri["karar"] == "İNCELE", "guvenilirlik dusukken otomatik karar verilmemeli"
    assert veri["uyarilar"], "kullanici neden incelemeye dustugunu gormeli"


def test_beklenen_eksikler_uyari_uretmez(istemci):
    """Kredi karti gecmisi olmayan musteri normaldir, uyari uretmemeli.

    Musterilerin %72'sinde kredi karti metrikleri yok. Bunlari 'veri kalitesi
    sorunu' saymak, servisi surekli uyari ureten ve bu yuzden goz ardi edilen
    bir hale getirirdi.
    """
    veri = istemci.post("/skorla/musteri/100002").json()
    assert veri["eksik_degisken"] > 0, "bu musteride beklenen eksikler var"
    assert veri["uyarilar"] == [], "beklenen eksikler uyari uretmemeli"
