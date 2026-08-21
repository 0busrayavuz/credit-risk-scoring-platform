"""Veri bolmesinin tekrarlanabilirligi testleri.

NEDEN BU TESTLER VAR:
    Bu projede gercek bir hata yasandi. model_input_yukle() sorgusunda
    ORDER BY yoktu. SQL, ORDER BY olmadan satir sirasini garanti etmez;
    indeks eklenmesi ve ANALYZE calistirilmasi sorgu planini degistirdi ve
    satirlar farkli sirada gelmeye basladi.

    train_test_split sabit tohumla ayni GIRDI SIRASINI ayni sekilde boler.
    Sira degisince bolme degisti, egitim verisi test kumesine sizdi ve
    test AUC'si 0.78'den 0.84'e "yukseldi". Bu bir iyilesme degil, sizintiydi.

    Sinsi olan yani: hicbir hata mesaji yok. Sadece sonuclar oldugundan iyi
    gorunur - ki bu, makine ogrenmesindeki en tehlikeli hata turudur.

    Asagidaki testler bunun tekrarlanmasini engeller.

Calistirmak icin (proje kokunden):
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_veri_bolme.py -v
"""

from __future__ import annotations

import pytest

from src.config import ID_COL, TARGET
from src.data import model_input_yukle, veri_bol

# Bu testler features.model_input tablosunu okur; PostgreSQL gerektirir.
pytestmark = pytest.mark.veritabani


@pytest.fixture(scope="module")
def veri():
    return model_input_yukle()


def test_bolme_ayni_veride_tekrarlanabilir(veri):
    """Ayni DataFrame iki kez bolununce ayni kumeler cikmali."""
    a1, b1, c1 = veri_bol(veri)
    a2, b2, c2 = veri_bol(veri)
    for x, y, ad in ((a1, a2, "egitim"), (b1, b2, "dogrulama"), (c1, c2, "test")):
        assert set(x[ID_COL]) == set(y[ID_COL]), f"{ad} kumesi degisti"


def test_bolme_satir_sirasindan_bagimsiz(veri):
    """ASIL TEST: veri karistirilmis gelse bile bolme AYNI olmali.

    Veritabanindan gelen sira degisebilir. Bolme buna dayanmamali.
    """
    karisik = veri.sample(frac=1.0, random_state=999).reset_index(drop=True)
    a1, b1, c1 = veri_bol(veri)
    a2, b2, c2 = veri_bol(karisik)
    assert set(c1[ID_COL]) == set(c2[ID_COL]), (
        "Test kumesi satir sirasina bagli! Bu, egitim verisinin test kumesine "
        "sizmasina yol acar."
    )
    assert set(a1[ID_COL]) == set(a2[ID_COL])


def test_kumeler_kesismiyor(veri):
    """Bir musteri yalnizca tek bir kumede olmali."""
    tr, va, te = veri_bol(veri)
    s_tr, s_va, s_te = set(tr[ID_COL]), set(va[ID_COL]), set(te[ID_COL])
    assert not (s_tr & s_va), "egitim ve dogrulama kesisiyor"
    assert not (s_tr & s_te), "egitim ve TEST kesisiyor - veri sizintisi"
    assert not (s_va & s_te), "dogrulama ve test kesisiyor"
    assert len(s_tr) + len(s_va) + len(s_te) == len(veri), "satir kaybi veya tekrari"


def test_hedef_dagilimi_korunuyor(veri):
    """stratify calisiyor mu: uc kumede de temerrut orani ayni olmali."""
    genel = veri[TARGET].mean()
    for kume, ad in zip(veri_bol(veri), ("egitim", "dogrulama", "test")):
        assert abs(kume[TARGET].mean() - genel) < 0.002, f"{ad} dagilimi kaymis"


def test_sorgu_order_by_iceriyor():
    """Yukleme sorgusu ORDER BY icermeli - regresyon korumasi."""
    import inspect

    from src import data

    kaynak = inspect.getsource(data.model_input_yukle)
    assert "ORDER BY" in kaynak.upper(), (
        "model_input_yukle sorgusunda ORDER BY yok. Satir sirasi garanti "
        "edilmezse bolme degisir ve veri sizar."
    )
