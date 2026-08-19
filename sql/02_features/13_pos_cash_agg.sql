-- =============================================================================
-- features.pos_cash_agg
-- Taksitli satis (POS) ve nakit kredilerin AYLIK durum gecmisi.
-- =============================================================================
--
-- IS BAGLAMI:
--   Bu tablo, gecmis kredilerin her ayki durumunu tutar: kac taksit kaldi,
--   o ay gecikme var miydi? installments_payments tek tek odemelere bakarken
--   bu tablo AYLIK FOTOGRAF cekiyor - ikisi birbirini tamamlar.
--
-- DOGRULANMIS DEGERLER:
--   10.001.358 aylik kayit
--   name_contract_status: Active 9,15M | Completed 745K | Signed 87K | Demand 7K | ...
--   sk_dpd: 9,71M kayitta gecikme yok | 163K'da 1-30 gun | 132K'da 30+ gun | maks 4231
--
-- sk_dpd vs sk_dpd_def farki:
--   sk_dpd     = ham gecikme gun sayisi
--   sk_dpd_def = tolerans esigi asildiktan sonraki gecikme (kucuk tutarli
--                gecikmeler sayilmaz). Ikisini de tutuyoruz: sk_dpd hassas,
--                sk_dpd_def ise "gercekten sorunlu mu" sorusunu daha iyi yanitlar.
--
-- Calistirmak icin:
--   docker exec credit_risk_pg psql -U credit -d credit_risk -v ON_ERROR_STOP=1 -f /sql/02_features/13_pos_cash_agg.sql
-- =============================================================================

\timing on

DROP TABLE IF EXISTS features.pos_cash_agg;

CREATE TABLE features.pos_cash_agg AS
SELECT
    sk_id_curr,

    -- --- Hacim ---
    count(*)                                                      AS pos_month_count,
    count(DISTINCT sk_id_prev)                                    AS pos_contract_count,

    -- --- Gecikme ---
    count(*) FILTER (WHERE sk_dpd > 0)                            AS pos_dpd_months,
    count(*) FILTER (WHERE sk_dpd > 30)                           AS pos_dpd30_months,
    round((count(*) FILTER (WHERE sk_dpd > 0))::numeric
          / NULLIF(count(*), 0), 4)                               AS pos_dpd_ratio,
    max(sk_dpd)                                                   AS pos_dpd_max,
    max(sk_dpd_def)                                               AS pos_dpd_def_max,

    -- --- Sozlesme durumu ---
    count(*) FILTER (WHERE name_contract_status = 'Active')       AS pos_active_months,
    count(*) FILTER (WHERE name_contract_status = 'Completed')    AS pos_completed_months,
    -- 'Demand' = borcun tamami muaccel kilinmis (kurum "hemen ode" demis).
    -- Seyrek ama cok guclu bir olumsuz sinyal olabilir.
    count(*) FILTER (WHERE name_contract_status = 'Demand')       AS pos_demand_months,

    -- --- Vade bilgisi ---
    avg(cnt_instalment)                                           AS pos_term_avg,
    avg(cnt_instalment_future)                                    AS pos_remaining_avg,

    -- months_balance negatiftir; max = sifira en yakin = en guncel kayit.
    max(months_balance)                                           AS pos_months_since_last,

    -- --- Son 12 ay ---
    count(*) FILTER (WHERE months_balance > -12)                  AS pos_months_1y,
    count(*) FILTER (WHERE months_balance > -12 AND sk_dpd > 0)   AS pos_dpd_months_1y,
    round((count(*) FILTER (WHERE months_balance > -12 AND sk_dpd > 0))::numeric
          / NULLIF(count(*) FILTER (WHERE months_balance > -12), 0), 4)
                                                                  AS pos_dpd_ratio_1y

FROM raw.pos_cash_balance
GROUP BY sk_id_curr;

ALTER TABLE features.pos_cash_agg ADD PRIMARY KEY (sk_id_curr);

ANALYZE features.pos_cash_agg;
