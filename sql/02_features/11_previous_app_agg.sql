-- =============================================================================
-- features.previous_app_agg
-- Kurumun KENDI gecmisinden musteri basina ozet metrikler.
-- =============================================================================
--
-- IS BAGLAMI:
--   bureau tablosu "baskalari bu kisiye ne yasadi" bilgisiydi (KKB benzeri).
--   previous_application ise "BIZ bu kisiyle ne yasadik" bilgisidir.
--   Kredi riskinde bu ikinci grup genellikle daha guclu sinyal tasir, cunku
--   kurumun kendi musteri deneyimini yansitir.
--
--   En degerli sinyal: DAHA ONCE REDDEDILMIS OLMAK.
--   Kurum bu kisiye bir kez hayir dediyse, o karari verdiren sebep
--   (dusuk gelir, yetersiz teminat, kotu skor) buyuk ihtimalle hala gecerlidir.
--
-- DOGRULANMIS DEGERLER (veriden okundu, varsayilmadi):
--   name_contract_status: Approved %62,1 | Canceled %18,9 | Refused %17,4 | Unused offer %1,6
--   name_contract_type  : Cash loans | Consumer loans | Revolving loans | XNA
--   code_reject_reason  : HC (en yaygin) | LIMIT | SCO | SCOFR | VERIF | XNA
--
-- ONEMLI: days_decision NEGATIFTIR (basvuru gunune gore geriye dogru gun).
--   Yani -30 = 30 gun once. max(days_decision) = sifira en yakin = EN YENI karar.
--
-- Calistirmak icin:
--   docker exec credit_risk_pg psql -U credit -d credit_risk -v ON_ERROR_STOP=1 -f /sql/02_features/11_previous_app_agg.sql
-- =============================================================================

\timing on

DROP TABLE IF EXISTS features.previous_app_agg;

CREATE TABLE features.previous_app_agg AS
SELECT
    sk_id_curr,

    -- --- Hacim: kurumla ne kadar gecmisi var? ---
    count(*)                                                          AS prev_app_count,
    count(*) FILTER (WHERE name_contract_status = 'Approved')          AS prev_approved_count,
    count(*) FILTER (WHERE name_contract_status = 'Refused')           AS prev_refused_count,
    count(*) FILTER (WHERE name_contract_status = 'Canceled')          AS prev_canceled_count,
    count(*) FILTER (WHERE name_contract_status = 'Unused offer')      AS prev_unused_count,

    -- --- Red orani: en guclu tekil sinyal adayi ---
    -- Ham red SAYISI yerine ORANI kullaniyoruz. Sebep: 10 basvurudan 2'si
    -- reddedilmis biri ile 2 basvurudan 2'si reddedilmis biri cok farkli
    -- risklerdir, ama ham sayilari ayni (2) gorunur.
    round(
        (count(*) FILTER (WHERE name_contract_status = 'Refused')::numeric
         / NULLIF(count(*), 0)), 4
    )                                                                  AS prev_refused_ratio,

    -- Son 1 yildaki redler. Eski bir red ile gecen ay alinan red
    -- ayni agirlikta degildir; yakin gecmis daha bilgilendiricidir.
    count(*) FILTER (
        WHERE name_contract_status = 'Refused' AND days_decision > -365
    )                                                                  AS prev_refused_last_year,

    -- --- Zamanlama ---
    max(days_decision)                                                 AS prev_days_since_last,
    min(days_decision)                                                 AS prev_days_since_first,
    -- Son reddin uzerinden ne kadar gecti? (red yoksa NULL kalir - bu da bilgi)
    max(days_decision) FILTER (WHERE name_contract_status = 'Refused')  AS prev_days_since_last_refusal,

    -- --- Tutarlar ---
    sum(amt_application)                                               AS prev_amt_application_sum,
    sum(amt_credit)                                                    AS prev_amt_credit_sum,
    avg(amt_annuity)                                                   AS prev_amt_annuity_avg,

    -- Istenen ile verilen arasindaki oran.
    -- 1'den kucukse: musteri istedigi tutari alamamis, kurum temkinli davranmis.
    -- Bu, gecmis kredi kararlarinin "gizli skoru" gibidir.
    round(
        (sum(amt_credit) / NULLIF(sum(amt_application), 0))::numeric, 4
    )                                                                  AS prev_credit_to_application_ratio,

    avg(rate_down_payment)                                             AS prev_down_payment_rate_avg,
    avg(cnt_payment)                                                   AS prev_term_avg,

    -- --- Urun karmasi ---
    -- Revolving (rotatif/kart benzeri) urun kullanimi, nakit ihtiyaci
    -- suregen olan musteri profiline isaret edebilir.
    count(*) FILTER (WHERE name_contract_type = 'Cash loans')          AS prev_cash_count,
    count(*) FILTER (WHERE name_contract_type = 'Consumer loans')      AS prev_consumer_count,
    count(*) FILTER (WHERE name_contract_type = 'Revolving loans')     AS prev_revolving_count,

    -- --- Red sebepleri (en yaygin uc kod) ---
    -- Farkli red sebepleri farkli riskler anlatir; hepsini tek torbada
    -- toplamak bilgi kaybidir.
    count(*) FILTER (WHERE code_reject_reason = 'HC')                  AS prev_reject_hc,
    count(*) FILTER (WHERE code_reject_reason = 'LIMIT')               AS prev_reject_limit,
    count(*) FILTER (WHERE code_reject_reason = 'SCO')                 AS prev_reject_sco,

    -- Sigortali kredi orani: sigorta yaptirmak temkinli davranisin gostergesi olabilir
    avg(nflag_insured_on_approval)                                     AS prev_insured_ratio

FROM raw.previous_application
GROUP BY sk_id_curr;

ALTER TABLE features.previous_app_agg ADD PRIMARY KEY (sk_id_curr);

ANALYZE features.previous_app_agg;
