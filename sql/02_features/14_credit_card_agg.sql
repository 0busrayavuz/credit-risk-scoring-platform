-- =============================================================================
-- features.credit_card_agg
-- Kredi karti / rotatif kredi kullanim davranisi.
-- =============================================================================
--
-- IS BAGLAMI:
--   Bu tablonun en degerli metrigi LIMIT KULLANIM ORANI (utilization):
--   bakiye / limit. Kredi riskinde klasik ve guclu bir gostergedir.
--   Limitinin %90'ini surekli kullanan biri, nakit sikisikligi yasiyor demektir;
--   %10'unda gezen biri karti kolaylik icin kullaniyordur.
--
--   Ikinci degerli metrik: ATM'den NAKIT CEKME orani.
--   Kredi kartindan nakit avans, en pahali borclanma bicimlerinden biridir.
--   Buna basvuran musteri genellikle baska secenegi kalmamis musteridir -
--   bankacilikta "distress signal" (sikinti sinyali) olarak bilinir.
--
-- DOGRULANMIS DEGERLER:
--   3.840.312 aylik kayit
--   name_contract_status: Active 3,70M | Completed 129K | Signed 11K | Demand 1.365
--   sk_dpd: 3,69M gecikme yok | 98K'da 1-30 gun | 55K'da 30+ gun | maks 3260
--
-- NOT: Bu veri setinde 'amt_recivable' kolonu YAZIM HATALI (receivable degil).
--   Kaynak veride boyle; duzeltmiyoruz cunku ham katman veriyi oldugu gibi
--   yansitmali. Kullanmiyoruz da - amt_balance zaten ihtiyacimizi karsiliyor.
--
-- Calistirmak icin:
--   docker exec credit_risk_pg psql -U credit -d credit_risk -v ON_ERROR_STOP=1 -f /sql/02_features/14_credit_card_agg.sql
-- =============================================================================

\timing on

DROP TABLE IF EXISTS features.credit_card_agg;

CREATE TABLE features.credit_card_agg AS
WITH t AS (
    SELECT
        sk_id_curr,
        sk_id_prev,
        months_balance,
        sk_dpd,
        sk_dpd_def,
        name_contract_status,
        amt_balance,
        amt_credit_limit_actual,
        amt_drawings_atm_current,
        amt_drawings_current,
        amt_inst_min_regularity,
        amt_payment_current,
        -- Limit kullanim orani. Limit 0 veya NULL ise oran anlamsizdir,
        -- NULL birakiyoruz - sifir yazmak "hic kullanmamis" yalanini uretirdi.
        CASE WHEN amt_credit_limit_actual > 0
             THEN amt_balance / amt_credit_limit_actual
        END AS utilization
    FROM raw.credit_card_balance
)
SELECT
    sk_id_curr,

    -- --- Hacim ---
    count(*)                                                      AS cc_month_count,
    count(DISTINCT sk_id_prev)                                    AS cc_card_count,

    -- --- Limit kullanimi: en guclu metrik adayi ---
    round(avg(utilization)::numeric, 4)                           AS cc_utilization_avg,
    round(max(utilization)::numeric, 4)                           AS cc_utilization_max,
    -- Limit asimi yasanan ay sayisi. Acik bir zorlanma gostergesi.
    count(*) FILTER (WHERE amt_balance > amt_credit_limit_actual) AS cc_over_limit_months,

    -- --- Nakit avans kullanimi (sikinti sinyali) ---
    sum(amt_drawings_atm_current)                                 AS cc_atm_drawings_sum,
    sum(amt_drawings_current)                                     AS cc_drawings_sum,
    round((sum(amt_drawings_atm_current)
           / NULLIF(sum(amt_drawings_current), 0))::numeric, 4)   AS cc_atm_drawings_ratio,

    -- --- Odeme davranisi ---
    -- Odenen / asgari odeme. 1'e yakin = surekli asgari odeyen musteri,
    -- borcu cevirmekten ileri gidemiyor demektir. Yuksek deger = saglikli.
    round((sum(amt_payment_current)
           / NULLIF(sum(amt_inst_min_regularity), 0))::numeric, 4) AS cc_payment_to_min_ratio,
    -- Hic odeme yapilmayan aylar
    count(*) FILTER (WHERE amt_payment_current = 0
                        OR amt_payment_current IS NULL)           AS cc_zero_payment_months,

    -- --- Bakiye / limit seviyeleri ---
    avg(amt_balance)                                              AS cc_balance_avg,
    max(amt_balance)                                              AS cc_balance_max,
    avg(amt_credit_limit_actual)                                  AS cc_limit_avg,

    -- --- Gecikme ---
    count(*) FILTER (WHERE sk_dpd > 0)                            AS cc_dpd_months,
    round((count(*) FILTER (WHERE sk_dpd > 0))::numeric
          / NULLIF(count(*), 0), 4)                               AS cc_dpd_ratio,
    max(sk_dpd)                                                   AS cc_dpd_max,
    max(sk_dpd_def)                                               AS cc_dpd_def_max,

    -- --- Durum ---
    count(*) FILTER (WHERE name_contract_status = 'Active')       AS cc_active_months,
    max(months_balance)                                           AS cc_months_since_last,

    -- --- Son 12 ay ---
    round(avg(utilization) FILTER (WHERE months_balance > -12)::numeric, 4)
                                                                  AS cc_utilization_avg_1y,
    count(*) FILTER (WHERE months_balance > -12 AND sk_dpd > 0)   AS cc_dpd_months_1y

FROM t
GROUP BY sk_id_curr;

ALTER TABLE features.credit_card_agg ADD PRIMARY KEY (sk_id_curr);

ANALYZE features.credit_card_agg;
