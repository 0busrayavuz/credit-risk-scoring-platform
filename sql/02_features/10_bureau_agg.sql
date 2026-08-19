-- =============================================================================
-- features.bureau_agg
-- Diger finans kuruluslarindaki kredi gecmisinden musteri basina ozet metrikler.
-- =============================================================================
--
-- IS BAGLAMI (mulakatta bu dille anlat):
--   bureau tablosu, basvuru sahibinin BASKA kuruluslardaki kredileridir -
--   Turkiye'deki karsiligi KKB / Findeks kredi notu raporudur. bureau_balance
--   ise o kredilerin AYLIK odeme gecmisidir. Kredi riskinde en guclu sinyal
--   genellikle buradan gelir: "bu kisi baskalarina borcunu oduyor mu?"
--
-- VERI YAPISI:
--   application_train (1 satir/musteri)
--        |  sk_id_curr
--   bureau (musteri basina ort. 5.6 kredi, maks 116)
--        |  sk_id_bureau
--   bureau_balance (kredi basina aylik kayit - toplam 27 milyon satir)
--
--   Modelin ihtiyaci: musteri basina TEK satir. Yani iki seviye yukari toplamak.
--
-- STATUS KODLARI (veriden dogrulandi, varsayilmadi):
--   C = kapali, X = bilinmiyor, 0 = gecikme yok,
--   1..5 = gecikme kademeleri (1: 1-30 gun ... 5: 120+ gun / zarar yazilmis)
--
-- ISIMLENDIRME: kolon adlari Ingilizce, yorumlar Turkce. Sebep: bu kolon adlari
--   ileride SHAP grafiklerinde ve model ciktilarinda gorunecek; Ingilizce isimler
--   portfolyoda daha standart durur.
--
-- Calistirmak icin:
--   docker exec credit_risk_pg psql -U credit -d credit_risk -v ON_ERROR_STOP=1 -f /sql/02_features/10_bureau_agg.sql
-- =============================================================================

\timing on

CREATE SCHEMA IF NOT EXISTS features;

DROP TABLE IF EXISTS features.bureau_agg;

CREATE TABLE features.bureau_agg AS
WITH bb AS (
    -- 1. SEVIYE: 27 milyon aylik kayidi kredi basina (sk_id_bureau) ozetle.
    -- FILTER sozdizimi, Oracle'daki CASE WHEN ... END sayma numarasinin
    -- standart SQL karsiligidir; daha okunakli ve daha hizlidir.
    SELECT
        sk_id_bureau,
        count(*)                                                    AS bb_months,
        count(*) FILTER (WHERE status IN ('1','2','3','4','5'))      AS bb_dpd_months,
        count(*) FILTER (WHERE status IN ('3','4','5'))              AS bb_severe_dpd_months,
        -- En kotu gecikme kademesi. status metin oldugu icin sayisal
        -- olanlari int'e cevirip maksimumunu aliyoruz.
        max(CASE WHEN status ~ '^[0-9]$' THEN status::int END)       AS bb_worst_status
    FROM raw.bureau_balance
    GROUP BY sk_id_bureau
),
b AS (
    -- bureau'ya aylik ozeti ekle.
    -- LEFT JOIN kritik: her kredinin bureau_balance kaydi YOK. INNER JOIN
    -- kullansaydik odeme gecmisi olmayan kredileri sessizce kaybederdik.
    SELECT
        bu.sk_id_curr,
        bu.sk_id_bureau,
        bu.credit_active,
        bu.credit_type,
        bu.days_credit,
        bu.days_credit_enddate,
        bu.credit_day_overdue,
        bu.amt_credit_sum,
        bu.amt_credit_sum_debt,
        bu.amt_credit_sum_overdue,
        bu.cnt_credit_prolong,
        bb.bb_months,
        bb.bb_dpd_months,
        bb.bb_severe_dpd_months,
        bb.bb_worst_status
    FROM raw.bureau bu
    LEFT JOIN bb ON bb.sk_id_bureau = bu.sk_id_bureau
)
-- 2. SEVIYE: kredi bazindan musteri bazina (sk_id_curr) topla.
SELECT
    sk_id_curr,

    -- --- Hacim: musterinin kredi gecmisi ne kadar yogun? ---
    count(*)                                                        AS bureau_credit_count,
    count(*) FILTER (WHERE credit_active = 'Active')                AS bureau_active_count,
    count(*) FILTER (WHERE credit_active = 'Closed')                AS bureau_closed_count,
    -- 'Sold' (alacak devri) ve 'Bad debt' (zarar yazma) ikisi de olumsuz sinyal.
    -- Ayri ayri cok seyrek oldugu icin tek metrikte birlestiriyoruz.
    count(*) FILTER (WHERE credit_active IN ('Sold','Bad debt'))     AS bureau_bad_count,
    count(DISTINCT credit_type)                                      AS bureau_type_count,

    -- --- Tutar: ne kadar borclu? ---
    sum(amt_credit_sum)                                              AS bureau_amt_sum,
    sum(amt_credit_sum_debt)                                         AS bureau_debt_sum,
    sum(amt_credit_sum_overdue)                                      AS bureau_overdue_sum,
    -- Borcun limite orani (kredi kullanim yogunlugu). NULLIF ile sifira
    -- bolmeyi engelliyoruz - SQL'de sifira bolme sorguyu komple dusurur.
    round(
        (sum(amt_credit_sum_debt) / NULLIF(sum(amt_credit_sum), 0))::numeric, 4
    )                                                                AS bureau_debt_ratio,

    -- --- Gecikme: en guclu risk sinyali ---
    max(credit_day_overdue)                                          AS bureau_max_overdue_days,
    count(*) FILTER (WHERE credit_day_overdue > 0)                   AS bureau_overdue_credit_count,
    sum(cnt_credit_prolong)                                          AS bureau_prolong_sum,

    -- --- Zamanlama: gecmis ne kadar taze? ---
    -- days_credit negatiftir (basvuru gunune gore geriye dogru gun sayisi).
    -- max = sifira en yakin = EN YENI kredi. Yeni kredi acmis olmak,
    -- nakit sikisikliginin gostergesi olabilir.
    max(days_credit)                                                 AS bureau_days_since_last,
    min(days_credit)                                                 AS bureau_days_since_first,

    -- --- Odeme davranisi (bureau_balance'tan gelen) ---
    sum(bb_months)                                                   AS bb_total_months,
    sum(bb_dpd_months)                                               AS bb_dpd_months,
    sum(bb_severe_dpd_months)                                        AS bb_severe_dpd_months,
    max(bb_worst_status)                                             AS bb_worst_status,
    -- Gecikmeli aylarin toplam aya orani: "her 100 ayin kacinda gecikti?"
    -- Ham sayidan daha iyi bir sinyal, cunku kredi gecmisi uzunlugundan bagimsiz.
    round(
        (sum(bb_dpd_months)::numeric / NULLIF(sum(bb_months), 0)), 4
    )                                                                AS bb_dpd_ratio

FROM b
GROUP BY sk_id_curr;

-- Bu tablo ileride application ile birlestirilecek, indeks sart.
ALTER TABLE features.bureau_agg ADD PRIMARY KEY (sk_id_curr);

ANALYZE features.bureau_agg;
