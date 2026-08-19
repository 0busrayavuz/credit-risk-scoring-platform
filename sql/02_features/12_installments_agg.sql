-- =============================================================================
-- features.installments_agg
-- Gecmis kredilerin TAKSIT ODEME DAVRANISI - musteri basina ozet.
-- =============================================================================
--
-- IS BAGLAMI:
--   Onceki iki tablo "baskalari ne yasadi" (bureau) ve "biz ne karar verdik"
--   (previous_application) bilgisiydi. Bu tablo ise musterinin FIILI DAVRANISI:
--   taksidini gununde odemis mi, tam odemis mi?
--
--   Kredi riskinde davranissal veri, beyan edilen veriden (gelir, meslek)
--   genellikle daha guclu tahmin gucu tasir. Cunku beyan degistirilebilir,
--   davranis degistirilemez.
--
-- DOGRULANMIS DEGERLER:
--   13.605.401 taksit kaydi
--   %91,55 zamaninda/erken | %8,14 1-30 gun gec | %0,28 30+ gun gec
--   2.905 kayitta days_entry_payment NULL = odeme HIC yapilmamis
--   Ortalama fark: -8,79 gun (yani ortalama 8,8 gun ERKEN odeniyor)
--   Uc degerler: en erken -3189 gun, en gec +2884 gun (8 yil!)
--
-- TASARIM NOTU - neden hem sayi hem ORAN uretiyoruz:
--   "5 kez gecikmis" ifadesi, 10 taksitten 5'i ile 200 taksitten 5'i icin
--   ayni gorunur ama cok farkli risklerdir. Oranlar kredi gecmisi uzunlugundan
--   bagimsiz oldugu icin modelde daha kararli davranir.
--   Ayrica +2884 gun gibi uc degerler ham ortalamalari bozar; oran metrikleri
--   bu bozulmadan etkilenmez.
--
-- ONEMLI: days_instalment ve days_entry_payment NEGATIFTIR.
--   gecikme = days_entry_payment - days_instalment
--   POZITIF sonuc = GEC odeme. Negatif = erken odeme.
--
-- Calistirmak icin:
--   docker exec credit_risk_pg psql -U credit -d credit_risk -v ON_ERROR_STOP=1 -f /sql/02_features/12_installments_agg.sql
-- =============================================================================

\timing on

DROP TABLE IF EXISTS features.installments_agg;

CREATE TABLE features.installments_agg AS
WITH t AS (
    -- Once satir bazinda tureti̇lmis alanlari hesapla, sonra topla.
    -- Bu ara adim olmadan ayni ifadeyi 10 kez tekrar yazmak gerekirdi.
    SELECT
        sk_id_curr,
        days_instalment,
        amt_instalment,
        amt_payment,
        -- Gecikme gun sayisi. GREATEST(...,0) ile erken odemeleri sifirliyoruz:
        -- "5 gun erken odedi" ile "gecikmedi" ayni seydir, negatif gecikme
        -- diye bir kavram yoktur.
        GREATEST(days_entry_payment - days_instalment, 0) AS dpd,
        -- Odenmemis taksit: odeme tarihi hic girilmemis.
        (days_entry_payment IS NULL)                      AS odenmemis,
        -- Eksik odeme: taksit tutarindan az odenmis.
        -- 1 birimlik tolerans birakiyoruz; kurus farklarini "eksik odeme"
        -- saymak gurultu uretir.
        (amt_payment < amt_instalment - 1)                AS eksik_odeme
    FROM raw.installments_payments
)
SELECT
    sk_id_curr,

    -- --- Hacim ---
    count(*)                                                     AS inst_count,

    -- --- Gecikme davranisi ---
    count(*) FILTER (WHERE dpd > 0)                              AS inst_late_count,
    count(*) FILTER (WHERE dpd > 30)                             AS inst_late30_count,
    round((count(*) FILTER (WHERE dpd > 0))::numeric
          / NULLIF(count(*), 0), 4)                              AS inst_late_ratio,
    round((count(*) FILTER (WHERE dpd > 30))::numeric
          / NULLIF(count(*), 0), 4)                              AS inst_late30_ratio,
    max(dpd)                                                     AS inst_dpd_max,
    round(avg(dpd)::numeric, 2)                                  AS inst_dpd_avg,

    -- --- Odeme tamligi ---
    count(*) FILTER (WHERE odenmemis)                            AS inst_unpaid_count,
    count(*) FILTER (WHERE eksik_odeme)                          AS inst_underpaid_count,
    round((count(*) FILTER (WHERE eksik_odeme))::numeric
          / NULLIF(count(*), 0), 4)                              AS inst_underpaid_ratio,

    -- --- Tutar ---
    sum(amt_instalment)                                          AS inst_amt_due_sum,
    sum(amt_payment)                                             AS inst_amt_paid_sum,
    -- Odenen / odenmesi gereken. 1'e yakin = saglikli.
    -- 1'in ustu de olabilir (erken kapatma, fazla odeme) - bu olumlu sinyaldir.
    round((sum(amt_payment) / NULLIF(sum(amt_instalment), 0))::numeric, 4)
                                                                 AS inst_payment_ratio,

    -- --- Yakin gecmis (son 1 yil) ---
    -- Uc yil onceki bir gecikme ile gecen ayki gecikme ayni agirlikta degildir.
    -- Modelin "guncel durum" ile "eski gecmis"i ayirt edebilmesi icin
    -- ayni metrikleri bir de son 12 ay icin uretiyoruz.
    count(*) FILTER (WHERE days_instalment > -365)               AS inst_count_1y,
    count(*) FILTER (WHERE days_instalment > -365 AND dpd > 0)   AS inst_late_count_1y,
    round((count(*) FILTER (WHERE days_instalment > -365 AND dpd > 0))::numeric
          / NULLIF(count(*) FILTER (WHERE days_instalment > -365), 0), 4)
                                                                 AS inst_late_ratio_1y

FROM t
GROUP BY sk_id_curr;

ALTER TABLE features.installments_agg ADD PRIMARY KEY (sk_id_curr);

ANALYZE features.installments_agg;
