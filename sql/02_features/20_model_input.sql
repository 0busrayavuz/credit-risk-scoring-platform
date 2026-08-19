-- =============================================================================
-- features.model_input
-- Modelin dogrudan tuketecegi tablo: musteri basina TEK satir.
-- =============================================================================
--
-- NE YAPIYOR:
--   1. application_train'in 122 ham kolonunu alir
--   2. Uzerine bankacilikta klasik olan ORAN feature'larini ekler
--   3. Daha once urettigimiz 5 ozet tabloyu LEFT JOIN ile baglar
--
--   Sonuc: 307.511 satir, ~200 kolon, musteri basina tek satir.
--
-- NEDEN LEFT JOIN (INNER degil):
--   Ozet tablolarin hepsi tum musterileri kapsamaz - ornegin kredi karti
--   gecmisi olan sadece 103.558 musteri var. INNER JOIN kullansaydik
--   musterilerin ucte ikisini kaybederdik. LEFT JOIN ile eksikler NULL kalir,
--   ve olcumlerimiz gosterdi ki BU BOSLUKLAR BILGI TASIYOR:
--     - dis kredi gecmisi yok  -> %10,12 temerrut (ortalamanin USTU)
--     - kuruma hic basvurmamis -> %5,96  temerrut (ortalamanin ALTI)
--   Bu yuzden bosluklari doldurmuyoruz; modelleme asamasinda ayri kategori olarak ele alinacak.
--
-- NEDEN USING (sk_id_curr):
--   ON a.sk_id_curr = b.sk_id_curr yazarsak sonucta sk_id_curr kolonu
--   6 kez tekrarlanir ve SELECT * patlar. USING, kolonu tek kopya olarak birlestirir.
--
-- KAPSAM NOTU:
--   Bu tablo yalnizca application_train'i kapsar. application_test'te TARGET
--   yoktur ve Kaggle gonderimi bu projenin amaci degil; degerlendirmeyi
--   train icinden ayiracagimiz dogrulama kumesiyle yapacagiz.
--
-- Calistirmak icin:
--   docker exec credit_risk_pg psql -U credit -d credit_risk -v ON_ERROR_STOP=1 -f /sql/02_features/20_model_input.sql
-- =============================================================================

\timing on

DROP TABLE IF EXISTS features.model_input;

CREATE TABLE features.model_input AS
SELECT
    -- Ham basvuru kolonlarinin tamami (122 kolon).
    -- USING sayesinde sk_id_curr tek kopya gelir.
    *,

    -- =========================================================================
    -- TURETILMIS ORANLAR - bankaciligin klasik rasyolari
    -- =========================================================================
    -- Ham tutarlar tek basina yaniltir: 500.000 TL kredi, 2 milyon geliri olan
    -- icin kucuk, 50.000 geliri olan icin imkansizdir. Oran bu baglami tasir.

    -- Kredi / yillik gelir. "Kac yillik gelirim kadar borclaniyorum?"
    round((amt_credit / NULLIF(amt_income_total, 0))::numeric, 4)
        AS credit_income_ratio,

    -- Yillik taksit / yillik gelir. Bankaciliktaki adi DTI (borc-gelir orani);
    -- kredi tahsisinde en cok bakilan tek metriktir.
    round((amt_annuity / NULLIF(amt_income_total, 0))::numeric, 4)
        AS annuity_income_ratio,

    -- Kredi / satin alinan malin bedeli. LTV (loan-to-value) benzeri.
    -- 1'e yakin = pesinat yok, kurum tum riski ustleniyor.
    round((amt_credit / NULLIF(amt_goods_price, 0))::numeric, 4)
        AS credit_goods_ratio,

    -- Kredi / taksit = ima edilen vade (kac taksit).
    round((amt_credit / NULLIF(amt_annuity, 0))::numeric, 2)
        AS credit_term,

    -- Kisi basina dusen gelir. 5 kisilik ailede 30.000 TL ile
    -- bekar birinin 30.000 TL'si ayni sey degildir.
    round((amt_income_total / NULLIF(cnt_fam_members, 0))::numeric, 2)
        AS income_per_person,

    -- =========================================================================
    -- YAS VE CALISMA SURESI
    -- =========================================================================
    round((-days_birth / 365.25)::numeric, 1) AS age_years,

    -- DIKKAT - bu veri setinin en meshur tuzagi:
    -- days_employed = 365243 (1000 yil) gercek bir sure degil, "calismiyor"
    -- anlaminda bir YER TUTUCU. Musterilerin %18'inde var.
    -- Naif hesap bu kisilere -1000 yil verir ve lineer modeli komple bozar.
    -- Cozum: degeri NULL yap, durumu ayri bir bayrakla tasi.
    CASE WHEN days_employed = 365243 THEN NULL
         ELSE round((-days_employed / 365.25)::numeric, 1)
    END AS employed_years,

    -- Bayrak. Bu grubun temerrut orani %5,40 - normal grubun %8,66'sindan
    -- DUSUK, cunku buyuk cogunlugu emekli (duzenli, garantili gelir).
    -- Yani "calismiyor" burada OLUMLU bir sinyal; silmek bilgi kaybi olurdu.
    (days_employed = 365243) AS flag_not_employed,

    -- Hayatinin ne kadarini calisarak gecirmis? Istikrar gostergesi.
    CASE WHEN days_employed = 365243 THEN NULL
         ELSE round((days_employed::numeric / NULLIF(days_birth, 0)), 4)
    END AS employed_life_ratio,

    -- =========================================================================
    -- DIS SKORLAR (ext_source_1/2/3)
    -- =========================================================================
    -- Bunlar disaridan alinan hazir kredi skorlaridir ve bu veri setinin
    -- en guclu tekil degiskenleridir. Ama cok eksikleri var:
    -- ext_source_1 %56 bos, ext_source_3 %20 bos, ext_source_2 neredeyse tam.
    -- Ucunu birlestiren ozet metrikler, tek tek kullanmaktan daha kararlidir.
    round(((COALESCE(ext_source_1,0) + COALESCE(ext_source_2,0) + COALESCE(ext_source_3,0))
           / NULLIF( (ext_source_1 IS NOT NULL)::int
                   + (ext_source_2 IS NOT NULL)::int
                   + (ext_source_3 IS NOT NULL)::int, 0))::numeric, 4)
        AS ext_source_mean,

    LEAST(ext_source_1, ext_source_2, ext_source_3)    AS ext_source_min,
    GREATEST(ext_source_1, ext_source_2, ext_source_3) AS ext_source_max,

    -- Kac tanesi dolu? Eksikligin kendisi de bir sinyal olabilir.
    ( (ext_source_1 IS NOT NULL)::int
    + (ext_source_2 IS NOT NULL)::int
    + (ext_source_3 IS NOT NULL)::int ) AS ext_source_count

FROM raw.application_train
LEFT JOIN features.bureau_agg       USING (sk_id_curr)
LEFT JOIN features.previous_app_agg USING (sk_id_curr)
LEFT JOIN features.installments_agg USING (sk_id_curr)
LEFT JOIN features.pos_cash_agg     USING (sk_id_curr)
LEFT JOIN features.credit_card_agg  USING (sk_id_curr);

ALTER TABLE features.model_input ADD PRIMARY KEY (sk_id_curr);

ANALYZE features.model_input;
