"""XGBoost modeli - skorkartin performans tarafindaki rakibi.

NEDEN IKI MODEL BIRDEN:
    Skorkart aciklanabilir ama dogrusaldir. XGBoost daha guclu ama bir kredi
    komitesine "neden bu karar?" sorusunu tablo halinde sunamaz.
    Bankalarda ikisi bir arada kullanilir:
      - Skorkart: nihai kredi karari, regulatore sunulan model
      - XGBoost : on eleme, erken uyari, portfoy izleme, sampiyon-rakip testi
    Bu projede ikisini de kurup FARKI OLCUYORUZ. Karar, olculen farkin
    aciklanabilirlik kaybina degip degmedigine gore verilir.

XGBOOST'UN BU VERIDEKI IKI DOGAL AVANTAJI:
    1. Eksik degerleri kendisi ogrenir. Her bolunmede "bu deger yoksa hangi
       dala gitsin?" karari veriden ogrenilir. Onceki asamada olctuk ki
       eksikligin kendisi bilgi tasiyor (dis kredi gecmisi yok -> %10,12
       temerrut). XGBoost bunu doldurma yapmadan dogrudan kullanir.
    2. Dogrusal olmayan iliskiler icin ek islem gerekmez. credit_income_ratio'nun
       ters U davranisini agaclar bolerek zaten ogrenir; WOE'ye ihtiyaci yoktur.

DENGESIZ SINIF (scale_pos_weight) KULLANMIYORUZ:
    Temerrut orani %8. scale_pos_weight ile agirliklandirmak siralamayi
    (AUC/Gini) neredeyse hic degistirmez ama OLASILIKLARI bozar: model
    gercekte %8 olan riski %40 gibi tahmin etmeye baslar. Bir sonraki asamada
    kar hesabi yapacagiz ve o hesap dogru kalibre edilmis olasiliklara dayanir.
    Bu yuzden siniflari oldugu gibi birakiyoruz.

Calistirmak icin (proje kokunden):
    .\\.venv\\Scripts\\python.exe -m src.train_xgboost
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TARGET
from src.data import kolon_tipleri, model_input_yukle, ozet, veri_bol
from src.metrics import degerlendir, dilim_analizi, rapor_yazdir

# Hiperparametreler.
# Buyuk bir arama yapmadik; kredi riski verisinde iyi bilinen, muhafazakar
# degerler secildi. Asiri ogrenmeyi (overfitting) engellemek icin:
#   - sig agaclar (max_depth=5): derin agaclar gurultuyu ezberler
#   - dusuk ogrenme hizi + erken durdurma: kac agac gerektigine veri karar verir
#   - satir ve kolon ornekleme: her agac verinin farkli bir kismini gorur
PARAMETRELER = dict(
    n_estimators=3000,          # ust sinir; gercek sayiyi erken durdurma belirler
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=20,        # yaprakta en az bu kadar agirlik olsun (gurultu freni)
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=2.0,
    eval_metric="auc",
    early_stopping_rounds=100,  # 100 turda dogrulama AUC'si iyilesmezse dur
    tree_method="hist",
    enable_categorical=True,    # kategorik kolonlari yerlesik olarak isler
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


def main() -> None:
    baslangic = time.time()

    print("=" * 78)
    print("VERI")
    print("=" * 78)
    df = model_input_yukle()
    train, valid, test = veri_bol(df)
    ozet(train, "egitim")
    ozet(valid, "dogrulama")

    tipler = kolon_tipleri(df)
    ozellikler = tipler["ozellikler"]
    print(f"\nkullanilan degisken: {len(ozellikler)} (TAMAMI)")
    print("Skorkartta 54'e indirmistik cunku her satirin denetlenebilir olmasi")
    print("gerekiyordu. XGBoost'ta boyle bir kisit yok: coklu dogrusal baglanti")
    print("agac modellerinde katsayi isareti bozmaz, sadece onemi paylastirir.")

    X_train, X_valid = train[ozellikler], valid[ozellikler]
    y_train, y_valid = train[TARGET].values, valid[TARGET].values

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("EGITIM")
    print("=" * 78)
    print("Erken durdurma ile: dogrulama AUC'si 100 turda iyilesmezse durur.")
    print("Boylece agac sayisini biz degil veri belirler.\n")

    model = xgb.XGBClassifier(**PARAMETRELER)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=200,
    )

    print(f"\nen iyi tur: {model.best_iteration} "
          f"(ust sinir {PARAMETRELER['n_estimators']})")
    print(f"en iyi dogrulama AUC: {model.best_score:.4f}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("DOGRULAMA KUMESI SONUCLARI")
    print("=" * 78)

    p_valid = model.predict_proba(X_valid)[:, 1]
    sonuclar = [degerlendir(y_valid, p_valid, f"XGBoost ({len(ozellikler)} degisken)")]

    onceki = REPORTS_DIR / "model_karsilastirma.json"
    if onceki.exists():
        sonuclar = json.loads(onceki.read_text(encoding="utf-8")) + sonuclar
    ozet_df = rapor_yazdir(sonuclar)

    # Asiri ogrenme kontrolu: egitim ile dogrulama arasindaki fark.
    p_train = model.predict_proba(X_train)[:, 1]
    egitim_auc = degerlendir(y_train, p_train, "egitim")["auc"]
    fark = egitim_auc - sonuclar[-1]["auc"]
    print(f"\nAsiri ogrenme kontrolu:")
    print(f"  egitim AUC    : {egitim_auc:.4f}")
    print(f"  dogrulama AUC : {sonuclar[-1]['auc']:.4f}")
    print(f"  fark          : {fark:.4f}", end="")
    print("  (0.03 alti saglikli, 0.10 ustu asiri ogrenme)" if fark < 0.10
          else "  <-- DIKKAT: asiri ogrenme belirtisi")

    print("\n" + "=" * 78)
    print("SKOR DILIMLERI (en riskli ustte)")
    print("=" * 78)
    print(dilim_analizi(y_valid, p_valid, 10).to_string(index=False))

    # ------------------------------------------------------------------
    # Degisken onemi
    # ------------------------------------------------------------------
    # 'gain' kullaniyoruz: bir degiskenin bolunmelerde sagladigi toplam
    # iyilestirme. 'weight' (kac kez kullanildi) yaniltir - cok kategorili
    # degiskenler sirf cok bolundugu icin one cikar.
    onem = (
        pd.DataFrame({
            "degisken": ozellikler,
            "onem": model.get_booster().get_score(importance_type="gain").get,
        })
        .assign(onem=lambda d: d["degisken"].map(
            lambda k: model.get_booster().get_score(importance_type="gain").get(k, 0.0)))
        .sort_values("onem", ascending=False)
        .reset_index(drop=True)
    )
    onem["pay_yuzde"] = (100 * onem["onem"] / onem["onem"].sum()).round(2)
    onem.to_csv(REPORTS_DIR / "xgboost_degisken_onemi.csv",
                index=False, encoding="utf-8")

    print("\n" + "=" * 78)
    print("EN ONEMLI 20 DEGISKEN (gain)")
    print("=" * 78)
    for _, r in onem.head(20).iterrows():
        print(f"  {r['degisken']:40s} {r['pay_yuzde']:5.2f}%")

    kullanilan = int((onem["onem"] > 0).sum())
    print(f"\nModelin fiilen kullandigi degisken: {kullanilan} / {len(ozellikler)}")

    model.save_model(str(MODELS_DIR / "xgboost.json"))
    ozet_df.to_json(REPORTS_DIR / "model_karsilastirma.json",
                    orient="records", indent=2, force_ascii=False)

    print(f"\nKaydedilenler:")
    print(f"  models/xgboost.json")
    print(f"  reports/xgboost_degisken_onemi.csv")
    print(f"  reports/model_karsilastirma.json")
    print(f"\nToplam sure: {time.time() - baslangic:.0f} sn")


if __name__ == "__main__":
    main()
