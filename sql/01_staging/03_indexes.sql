-- Ham tablolara birincil anahtar ve join indeksleri ekler.
--
-- Neden gerekli:
--   Feature engineering asamasinda bu tablolari sk_id_curr / sk_id_prev /
--   sk_id_bureau uzerinden surekli birlestirecegiz. Indekssiz her join tam
--   tablo taramasi yapar; bureau_balance 27 milyon satir oldugu icin bu
--   dakikalar demek. Indeksle ayni sorgu saniyeler surer.
--
-- Neden yuklemeden SONRA:
--   Indeks varken COPY yapmak yavastir, cunku her satirda indeks de guncellenir.
--   Once yukle, sonra indeksle - toplu yuklemenin standart kuralidir.
--
-- Calistirmak icin:
--   docker exec credit_risk_pg psql -U credit -d credit_risk -v ON_ERROR_STOP=1 -f /sql/01_staging/03_indexes.sql

\timing on

-- --- Birincil anahtarlar --------------------------------------------------
-- sk_id_curr'in bu iki tabloda tekil oldugunu dogruladik (307511 satir = 307511 tekil).
-- PRIMARY KEY hem indeks olusturur hem de "bu kolon tekil ve bos olamaz"
-- kuralini veritabani seviyesinde garanti eder.
ALTER TABLE raw.application_train ADD PRIMARY KEY (sk_id_curr);
ALTER TABLE raw.application_test  ADD PRIMARY KEY (sk_id_curr);

-- --- Join indeksleri -------------------------------------------------------
-- Bu tablolarda sk_id_curr tekil DEGIL (bir musterinin birden cok kredisi var),
-- o yuzden PRIMARY KEY degil, normal indeks.

-- bureau: diger bankalardaki krediler (KKB muadili)
CREATE INDEX idx_bureau_curr   ON raw.bureau (sk_id_curr);
CREATE INDEX idx_bureau_bureau ON raw.bureau (sk_id_bureau);

-- bureau_balance: o kredilerin aylik odeme gecmisi - en buyuk tablo (27M)
CREATE INDEX idx_bureau_balance_bureau ON raw.bureau_balance (sk_id_bureau);

-- previous_application: Home Credit'teki gecmis basvurular
CREATE INDEX idx_prev_curr ON raw.previous_application (sk_id_curr);
CREATE INDEX idx_prev_prev ON raw.previous_application (sk_id_prev);

-- Gecmis basvurularin aylik detaylari
CREATE INDEX idx_pos_curr ON raw.pos_cash_balance (sk_id_curr);
CREATE INDEX idx_pos_prev ON raw.pos_cash_balance (sk_id_prev);

CREATE INDEX idx_inst_curr ON raw.installments_payments (sk_id_curr);
CREATE INDEX idx_inst_prev ON raw.installments_payments (sk_id_prev);

CREATE INDEX idx_cc_curr ON raw.credit_card_balance (sk_id_curr);
CREATE INDEX idx_cc_prev ON raw.credit_card_balance (sk_id_prev);

-- --- Istatistikleri guncelle ----------------------------------------------
-- ANALYZE, sorgu planlayicisinin tablolar hakkindaki istatistiklerini yeniler.
-- Bu olmadan planlayici satir sayilarini yanlis tahmin eder ve kotu plan secer
-- (ornegin indeks yerine tam tarama). Toplu yuklemeden sonra HER ZAMAN calistir.
ANALYZE raw.application_train;
ANALYZE raw.application_test;
ANALYZE raw.bureau;
ANALYZE raw.bureau_balance;
ANALYZE raw.credit_card_balance;
ANALYZE raw.installments_payments;
ANALYZE raw.pos_cash_balance;
ANALYZE raw.previous_application;
