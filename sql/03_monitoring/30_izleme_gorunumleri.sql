-- =============================================================================
-- monitoring semasi: BI panelinin (Power BI) bagalanacagi gorunumler
-- =============================================================================
--
-- TASARIM ILKESI:
--   Panel, ham veriye degil ONCEDEN ANLAMLANDIRILMIS gorunumlere baglanir.
--   Sebepleri:
--     1. Is birimi 230 kolonluk bir tabloyla calisamaz; "onay orani",
--        "risk dilimi", "beklenen kar" gibi kavramlarla calisir.
--     2. Is mantigi tek yerde tanimli olur. "Onay orani" tanimini panelde
--        DAX ile yazarsan, ikinci bir panelde farkli yazilir ve iki rapor
--        birbirini tutmaz. Burada tanim SQL'de bir kez yapilir.
--     3. Panel her yenilendiginde model calistirmak gerekmez.
--
-- Kaynak tablo: monitoring.skor_portfoy (src/izleme_tablosu.py uretir)
--
-- Calistirmak icin:
--   docker exec credit_risk_pg psql -U credit -d credit_risk -v ON_ERROR_STOP=1 -f /sql/03_monitoring/30_izleme_gorunumleri.sql
-- =============================================================================

\timing on

-- -----------------------------------------------------------------------------
-- 1) Ust duzey gostergeler - panelin en ust seridi
-- -----------------------------------------------------------------------------
DROP VIEW IF EXISTS monitoring.v_kpi;
CREATE VIEW monitoring.v_kpi AS
SELECT
    kume,
    count(*)                                                    AS basvuru,
    count(*) FILTER (WHERE karar = 'ONAY')                      AS onaylanan,
    round(100.0 * avg((karar = 'ONAY')::int), 2)                AS onay_orani,
    round(100.0 * avg(gercek_temerrut), 2)                      AS portfoy_temerrut_orani,
    -- Onaylananlar arasindaki temerrut: modelin ASIL basari olcusu.
    -- Portfoyun geneli degil, modelin KABUL ETTIKLERI onemlidir.
    round(100.0 * avg(gercek_temerrut) FILTER (WHERE karar = 'ONAY'), 2)
                                                                AS onaylananda_temerrut,
    -- Reddedilenler arasindaki temerrut: yuksekse model dogru elemis demektir.
    round(100.0 * avg(gercek_temerrut) FILTER (WHERE karar = 'RED'), 2)
                                                                AS reddedilende_temerrut,
    sum(beklenen_kar)                                           AS beklenen_kar,
    round((sum(beklenen_kar) / count(*))::numeric, 0)           AS basvuri_basi_kar,
    -- Onlenen batik: reddedilen ve gercekten batan krediler
    count(*) FILTER (WHERE karar = 'RED' AND gercek_temerrut = 1) AS onlenen_batik,
    -- Kacirilan is: reddedilen ama aslinda odeyecek olan musteriler
    count(*) FILTER (WHERE karar = 'RED' AND gercek_temerrut = 0) AS kacirilan_iyi_musteri
FROM monitoring.skor_portfoy
GROUP BY kume;


-- -----------------------------------------------------------------------------
-- 2) Risk dilimi performansi - modelin ayirt etme gucunun gorsel kaniti
-- -----------------------------------------------------------------------------
DROP VIEW IF EXISTS monitoring.v_dilim_performans;
CREATE VIEW monitoring.v_dilim_performans AS
SELECT
    kume,
    risk_dilimi,
    count(*)                                        AS basvuru,
    -- NOT: PostgreSQL'de iki argumanli round() yalnizca numeric ile calisir.
    -- temerrut_olasiligi DOUBLE PRECISION oldugu icin acik donusum sart.
    -- (Oracle'da ROUND her sayisal tipte calisir; gecis yaparken sik yanilinan yer.)
    round((100.0 * avg(temerrut_olasiligi))::numeric, 2) AS ortalama_tahmin,
    round(100.0 * avg(gercek_temerrut), 2)          AS gerceklesen_temerrut,
    -- LIFT: bu dilim, portfoy ortalamasinin kac kati riskli?
    -- Is birimlerine model degerini anlatmanin en dogrudan yolu.
    round(avg(gercek_temerrut)::numeric
          / NULLIF((SELECT avg(gercek_temerrut) FROM monitoring.skor_portfoy p2
                    WHERE p2.kume = p.kume), 0), 2) AS lift,
    -- Kalibrasyon farki: tahmin ile gerceklesme arasindaki sapma.
    -- Sifira yakin olmali; buyukse model iyi siraliyor ama yanlis olasilik veriyor.
    round((100.0 * (avg(temerrut_olasiligi) - avg(gercek_temerrut)))::numeric, 2)
                                                    AS kalibrasyon_farki,
    sum(amt_credit)                                 AS toplam_kredi_tutari,
    sum(beklenen_kar)                               AS beklenen_kar
FROM monitoring.skor_portfoy p
GROUP BY kume, risk_dilimi;


-- -----------------------------------------------------------------------------
-- 3) Segment performansi - tum segment boyutlari TEK gorunumde
-- -----------------------------------------------------------------------------
-- Neden tek gorunum: Power BI'da her segment icin ayri gorsel yapmak yerine,
-- 'boyut' alanini dilimleyici (slicer) olarak kullanip tek grafikle
-- tum segmentler arasinda gezinebilirsin.
DROP VIEW IF EXISTS monitoring.v_segment_performans;
CREATE VIEW monitoring.v_segment_performans AS
WITH uzun AS (
    SELECT kume, karar, gercek_temerrut, beklenen_kar, temerrut_olasiligi,
           'yaş bandı' AS boyut, yas_bandi AS deger FROM monitoring.skor_portfoy
    UNION ALL SELECT kume, karar, gercek_temerrut, beklenen_kar, temerrut_olasiligi,
           'gelir bandı', gelir_bandi FROM monitoring.skor_portfoy
    UNION ALL SELECT kume, karar, gercek_temerrut, beklenen_kar, temerrut_olasiligi,
           'eğitim', egitim FROM monitoring.skor_portfoy
    UNION ALL SELECT kume, karar, gercek_temerrut, beklenen_kar, temerrut_olasiligi,
           'aile durumu', aile_durumu FROM monitoring.skor_portfoy
    UNION ALL SELECT kume, karar, gercek_temerrut, beklenen_kar, temerrut_olasiligi,
           'meslek', meslek FROM monitoring.skor_portfoy
    UNION ALL SELECT kume, karar, gercek_temerrut, beklenen_kar, temerrut_olasiligi,
           'sözleşme türü', sozlesme_turu FROM monitoring.skor_portfoy
    UNION ALL SELECT kume, karar, gercek_temerrut, beklenen_kar, temerrut_olasiligi,
           'cinsiyet', cinsiyet FROM monitoring.skor_portfoy
)
SELECT
    kume, boyut, deger,
    count(*)                                                 AS basvuru,
    round(100.0 * avg((karar = 'ONAY')::int), 2)             AS onay_orani,
    round(100.0 * avg(gercek_temerrut), 2)                   AS gerceklesen_temerrut,
    round((100.0 * avg(temerrut_olasiligi))::numeric, 2)     AS ortalama_tahmin,
    -- ADALET IZLEME: odeyecek musterinin reddedilme orani.
    -- Bu satir sadece bir metrik degil, projenin adalet bulgusunun
    -- panelde SUREKLI izlenmesini saglayan mekanizmadir. Cinsiyet
    -- degiskeni modelden cikarildi ama vekiller uzerinden geri sizabiliyor
    -- (olculdu: AUC 0,909), bu yuzden izleme sart.
    round(100.0 * avg((karar = 'RED')::int) FILTER (WHERE gercek_temerrut = 0), 2)
                                                             AS iyi_musteri_red_orani,
    sum(beklenen_kar)                                        AS beklenen_kar
FROM uzun
GROUP BY kume, boyut, deger;


-- -----------------------------------------------------------------------------
-- 4) Karar matrisi - dogru ve yanlis kararlarin dagilimi
-- -----------------------------------------------------------------------------
DROP VIEW IF EXISTS monitoring.v_karar_matrisi;
CREATE VIEW monitoring.v_karar_matrisi AS
SELECT
    kume,
    karar,
    CASE WHEN gercek_temerrut = 1 THEN 'battı' ELSE 'ödedi' END AS gerceklesme,
    CASE
        WHEN karar = 'ONAY' AND gercek_temerrut = 0 THEN 'doğru onay'
        WHEN karar = 'ONAY' AND gercek_temerrut = 1 THEN 'hatalı onay (zarar)'
        WHEN karar = 'RED'  AND gercek_temerrut = 1 THEN 'doğru red (önlenen zarar)'
        ELSE                                             'hatalı red (kaçan gelir)'
    END                                                         AS sonuc,
    count(*)                                                    AS basvuru,
    sum(amt_credit)                                             AS kredi_tutari,
    sum(beklenen_kar)                                           AS kar_etkisi
FROM monitoring.skor_portfoy
GROUP BY kume, karar, gercek_temerrut;


-- -----------------------------------------------------------------------------
-- 5) Risk bandi ozeti - is birimi diliyle
-- -----------------------------------------------------------------------------
DROP VIEW IF EXISTS monitoring.v_risk_bandi;
CREATE VIEW monitoring.v_risk_bandi AS
SELECT
    kume,
    risk_bandi,
    count(*)                                        AS basvuru,
    round(100.0 * count(*) / sum(count(*)) OVER (PARTITION BY kume), 2) AS pay_yuzde,
    round(100.0 * avg(gercek_temerrut), 2)          AS gerceklesen_temerrut,
    round(100.0 * avg((karar = 'ONAY')::int), 2)    AS onay_orani,
    round(avg(amt_credit)::numeric, 0)              AS ortalama_kredi_tutari,
    sum(beklenen_kar)                               AS beklenen_kar
FROM monitoring.skor_portfoy
GROUP BY kume, risk_bandi;


-- Dogrulama: gorunumler bos donmemeli
SELECT 'v_kpi' AS gorunum, count(*) AS satir FROM monitoring.v_kpi
UNION ALL SELECT 'v_dilim_performans', count(*) FROM monitoring.v_dilim_performans
UNION ALL SELECT 'v_segment_performans', count(*) FROM monitoring.v_segment_performans
UNION ALL SELECT 'v_karar_matrisi', count(*) FROM monitoring.v_karar_matrisi
UNION ALL SELECT 'v_risk_bandi', count(*) FROM monitoring.v_risk_bandi;
