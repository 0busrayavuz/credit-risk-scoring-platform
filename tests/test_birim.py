"""Birim testleri - veritabani veya egitilmis model GEREKTIRMEZ.

NEDEN AYRI BIR DOSYA:
    Projedeki diger testler butunlesme testidir: calisan bir PostgreSQL ve
    egitilmis modeller ister. Bu, yerel gelistirmede dogru tercihtir - asil
    risk parcalarin birlikte yanlis calismasidir.

    Ama sürekli entegrasyonda (CI) 2,5 GB veriyi yukleyip model egitmek
    pratik degildir. Bu dosya, matematiksel cekirdegi bagimsiz olarak test
    eder: PSI hesabi, Gini, KS ve korumali ozellik politikasi. Boylece her
    itmede (push) en azindan bu mantik dogrulanir.

    Ayrica birim testinin kendi degeri var: PSI'nin dogru hesaplandigini
    307 bin satirlik gercek veriyle degil, sonucunu ONCEDEN BILDIGIMIZ
    yapay verilerle kanitlamak gerekir.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.korumali_ozellikler import (
    IZLENEN_BOYUTLAR,
    KADEME_1,
    KADEME_2,
    cikarilanlar,
    temiz_ozellikler,
)
from src.metrics import degerlendir, gini, ks_istatistigi
from src.psi_izleme import psi, yorumla


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------
def test_psi_ayni_dagilimda_sifira_yakin():
    """Kayma yoksa PSI ~0 olmali - olcunun kendisinin dogrulugu."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=20000)
    b = rng.normal(size=20000)
    assert psi(a, b) < 0.01


def test_psi_kayma_arttikca_buyur():
    """PSI monotonik olmali: kayma buyudukce deger buyumeli."""
    rng = np.random.default_rng(1)
    referans = rng.normal(size=20000)
    kucuk = psi(referans, rng.normal(loc=0.2, size=20000))
    orta = psi(referans, rng.normal(loc=0.6, size=20000))
    buyuk = psi(referans, rng.normal(loc=1.5, size=20000))
    assert kucuk < orta < buyuk, "PSI kayma ile birlikte artmali"
    assert buyuk > 0.25, "belirgin kayma esigi asmali"


def test_psi_bos_dilimde_patlamiyor():
    """Yeni dagilim referansin disina tamamen kaysa bile sonlu deger donmeli.

    ln(0) tanimsizdir; taban degeri konmasaydi PSI sonsuz olurdu ve olcu
    kullanilamaz hale gelirdi.
    """
    rng = np.random.default_rng(2)
    d = psi(rng.normal(size=5000), rng.normal(loc=50, size=5000))
    assert np.isfinite(d) and d > 0


def test_psi_eksik_degerleri_atliyor():
    rng = np.random.default_rng(3)
    a = rng.normal(size=5000)
    b = np.concatenate([rng.normal(size=4000), np.full(1000, np.nan)])
    assert np.isfinite(psi(a, b))


@pytest.mark.parametrize("deger,beklenen", [
    (0.02, "kayma yok"), (0.15, "orta - izle"), (0.40, "CIDDI - gozden gecir"),
])
def test_psi_yorumu_esiklere_uyuyor(deger, beklenen):
    assert yorumla(deger) == beklenen


# ---------------------------------------------------------------------------
# Metrikler
# ---------------------------------------------------------------------------
def test_gini_kusursuz_ayrimda_bir():
    y = np.array([0] * 500 + [1] * 500)
    skor = np.array([0.1] * 500 + [0.9] * 500)
    assert gini(y, skor) == pytest.approx(1.0)


def test_gini_rastgele_skorda_sifira_yakin():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 20000)
    assert abs(gini(y, rng.random(20000))) < 0.05


def test_ks_kusursuz_ayrimda_bir():
    y = np.array([0] * 500 + [1] * 500)
    skor = np.array([0.1] * 500 + [0.9] * 500)
    assert ks_istatistigi(y, skor) == pytest.approx(1.0)


def test_brier_yalnizca_olasilikta_hesaplanir():
    """Brier kalibrasyon olcer; olasilik olmayan skorda None donmeli.

    AUC/Gini/KS yalnizca SIRALAMAYA bakar, herhangi bir monoton skorla
    calisir. Brier ise skorun gercek bir olasilik olmasini gerektirir.
    """
    y = np.array([0, 0, 1, 1, 0, 1])
    assert degerlendir(y, np.array([.1, .2, .8, .9, .3, .7]))["brier"] is not None
    # 0-1 disindaki ham skorlar (ornegin -ext_source_mean) icin None
    assert degerlendir(y, np.array([-5.0, -3.0, 2.0, 4.0, -1.0, 1.0]))["brier"] is None


def test_degerlendir_gini_auc_iliskisi():
    """Gini = 2*AUC - 1 kimligi her zaman saglanmali."""
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 5000)
    skor = rng.random(5000) * 0.4 + y * 0.3
    s = degerlendir(y, skor)
    assert s["gini"] == pytest.approx(2 * s["auc"] - 1)


# ---------------------------------------------------------------------------
# Korumali ozellik politikasi
# ---------------------------------------------------------------------------
def test_kademe1_ozellikleri_cikariliyor():
    ozellikler = ["ext_source_mean", "code_gender", "amt_credit",
                  "name_family_status", "cnt_children", "cnt_fam_members"]
    temiz = temiz_ozellikler(ozellikler)
    assert temiz == ["ext_source_mean", "amt_credit"]
    assert set(cikarilanlar(ozellikler)) == {
        "code_gender", "name_family_status", "cnt_children", "cnt_fam_members"}


def test_kademe2_ozellikleri_cikarilmiyor():
    """Yas ve bolge bilincli olarak modelde birakiliyor."""
    ozellikler = ["age_years", "days_birth", "region_rating_client_w_city"]
    assert temiz_ozellikler(ozellikler) == ozellikler


def test_kademe_listeleri_cakismıyor():
    assert not (set(KADEME_1) & set(KADEME_2)), "bir ozellik iki kademede olamaz"


def test_her_ozelligin_gerekcesi_var():
    """Politika kod olarak duruyorsa, gerekcesi de kod olarak durmali."""
    for k, gerekce in {**KADEME_1, **KADEME_2}.items():
        assert len(gerekce) > 30, f"{k} icin gerekce cok kisa veya yok"


def test_cikarilanlar_izlenmeye_devam_ediyor():
    """Modelden cikarilan bir ozellik, izleme listesinden DUSMEMELI.

    Cikarmak, o boyutta ayrimcilik olmadigi anlamina gelmez - vekiller
    uzerinden yeniden dogabilir (olculdu: cinsiyet AUC 0,905 ile geri
    kazanilabiliyor). Bu yuzden izleme devam etmeli.
    """
    for k in ("code_gender", "name_family_status"):
        assert k in IZLENEN_BOYUTLAR, f"{k} cikarildi ama izlenmiyor"
