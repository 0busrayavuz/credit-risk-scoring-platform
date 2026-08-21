"""MLflow ile deney takibi - ince bir sarmalayici.

NEDEN DENEY TAKIBI:
    Bu projede su ana kadar sonuclari JSON dosyalarina yazdik. Calisiyor, ama
    olcegi yok: "ucuncu denemede learning_rate neydi?", "hangi degisken seti
    0.78 vermisti?", "gecen haftaki model bu modelden iyi miydi?" sorularini
    cevaplayamazsin. Gercek projelerde onlarca deneme yapilir ve hangisinin
    ne oldugunu hatirlamak imkansizlasir.

    MLflow her egitimde parametreleri, metrikleri, uretilen dosyalari ve
    modelin kendisini tek bir kayda baglar. Sonradan karsilastirilabilir,
    geri donulebilir hale gelir.

    Bu ayrica MODEL YONETISIMI (governance) gereksinimidir: bir banka,
    kullandigi modelin hangi veriyle, hangi parametrelerle ve ne zaman
    egitildigini belgeleyebilmek zorundadir.

NEDEN SQLITE DEPO (ve neden mlruns/ klasoru DEGIL):
    MLflow 3.x, klasik dosya tabanli depoyu (mlruns/) kullanimdan kaldirdi:
        "The filesystem tracking backend is in maintenance mode and will not
         receive further updates. Please migrate to a database backend."
    MLFLOW_ALLOW_FILE_STORE=true ile bastirilabilir ama bu, kaldirilmis bir
    yolu zorlamak olur. Onerilen yol SQLite arka ucudur; tek dosya, sunucu
    gerektirmez ve MLflow arayuzu ile sorunsuz calisir.

    Gercek bir ekipte ortak bir MLflow sunucusu (PostgreSQL arka uclu)
    kullanilir. Bu kodda degisecek tek sey tracking URI'sidir - geri kalan
    her sey aynen calisir. Tasarimin amaci zaten bu: depo secimi tek yerde.

Kullanim:
    from src.takip import deney

    with deney("xgboost", {"max_depth": 5}) as kosu:
        ...
        kosu.metrikler({"auc": 0.78})
        kosu.dosya("reports/grafik.png")
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.config import PROJE_KOK

DENEY_ADI = "kredi-risk-skorlama"

# Metrik ve parametreler SQLite'ta, uretilen dosyalar ayri bir klasorde.
# MLflow bu ikisini ayirir: veritabani "ne oldu"yu, artefakt deposu
# "ne uretildi"yi tutar.
MLFLOW_DB = PROJE_KOK / "mlflow.db"
MLARTIFACTS = PROJE_KOK / "mlartifacts"

# as_posix(): Windows'ta C:\... yolunu C:/... bicimine cevirir.
# sqlite:///C:\Users\... yazarsan ters bolu isaretleri URI'yi bozar.
TAKIP_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"


class Kosu:
    """Tek bir egitim kosusuna yazma arayuzu."""

    def __init__(self, run) -> None:
        self.run = run
        self.id = run.info.run_id

    def parametreler(self, d: dict[str, Any]) -> None:
        # MLflow parametreleri metin olarak saklar; karmasik degerleri
        # kisaltarak yaziyoruz ki kayit okunabilir kalsin.
        mlflow.log_params({k: str(v)[:250] for k, v in d.items()})

    def metrikler(self, d: dict[str, Any]) -> None:
        # Yalnizca sayisal degerler metrik olabilir; model adi gibi metin
        # alanlarini sessizce atliyoruz.
        sayisal = {
            k: float(v) for k, v in d.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if sayisal:
            mlflow.log_metrics(sayisal)

    def etiket(self, ad: str, deger: str) -> None:
        mlflow.set_tag(ad, deger)

    def dosya(self, yol: str | Path) -> None:
        """Uretilen bir dosyayi (grafik, tablo) kosuya baglar."""
        p = Path(yol)
        if p.exists():
            mlflow.log_artifact(str(p))

    def model_xgboost(self, model, ad: str = "model") -> None:
        import mlflow.xgboost

        mlflow.xgboost.log_model(model, name=ad)

    def model_sklearn(self, model, ad: str = "model") -> None:
        import mlflow.sklearn

        mlflow.sklearn.log_model(model, name=ad)


@contextmanager
def deney(kosu_adi: str, parametreler: dict[str, Any] | None = None,
          deney_adi: str = DENEY_ADI):
    """Bir MLflow kosusu acar ve kapanista otomatik sonlandirir.

    Hata durumunda bile kosu duzgun kapanir; boylece yarim kalmis
    kayitlar birikmez.
    """
    MLARTIFACTS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TAKIP_URI)

    # Deney ilk kez olusturuluyorsa artefakt konumunu acikca belirtiyoruz.
    # Belirtmezsek MLflow calisma dizinine gore goreli bir yol secer ve
    # script'i baska bir klasorden calistirdiginda dosyalar dagilir.
    istemci = MlflowClient()
    if istemci.get_experiment_by_name(deney_adi) is None:
        istemci.create_experiment(deney_adi, artifact_location=MLARTIFACTS.as_uri())
    mlflow.set_experiment(deney_adi)

    with mlflow.start_run(run_name=kosu_adi) as run:
        k = Kosu(run)
        if parametreler:
            k.parametreler(parametreler)
        yield k
