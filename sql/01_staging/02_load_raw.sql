-- Ham CSV'leri raw semasina yukler.
--
-- Neden COPY, neden pandas.to_sql degil?
--   to_sql satir satir INSERT uretir; 2,5 GB / ~28 milyon satir icin saatler surer.
--   COPY ise Postgres'in kendi toplu yukleme yolu: dosyayi sunucu surecinin
--   kendisi okur, araya Python girmez. Ayni is dakikalar icinde biter.
--
-- Dosya yollari CONTAINER icindeki yollardir (/data/raw), Windows yollari degil.
-- docker-compose.yml bu klasoru salt-okunur olarak bagliyor.
--
-- NULL '' : Home Credit CSV'lerinde eksik degerler bos alan olarak yazilmis.
--           Bunu soylemezsek Postgres bos metni TEXT kolonlarda '' olarak,
--           sayisal kolonlarda ise hata olarak degerlendirir.
--
-- Calistirmak icin:
--   docker exec credit_risk_pg psql -U credit -d credit_risk -v ON_ERROR_STOP=1 -f /sql/01_staging/02_load_raw.sql

\timing on

TRUNCATE raw.application_train;
COPY raw.application_train FROM '/data/raw/application_train.csv' WITH (FORMAT csv, HEADER true, NULL '');

TRUNCATE raw.application_test;
COPY raw.application_test FROM '/data/raw/application_test.csv' WITH (FORMAT csv, HEADER true, NULL '');

TRUNCATE raw.bureau;
COPY raw.bureau FROM '/data/raw/bureau.csv' WITH (FORMAT csv, HEADER true, NULL '');

TRUNCATE raw.bureau_balance;
COPY raw.bureau_balance FROM '/data/raw/bureau_balance.csv' WITH (FORMAT csv, HEADER true, NULL '');

TRUNCATE raw.credit_card_balance;
COPY raw.credit_card_balance FROM '/data/raw/credit_card_balance.csv' WITH (FORMAT csv, HEADER true, NULL '');

TRUNCATE raw.installments_payments;
COPY raw.installments_payments FROM '/data/raw/installments_payments.csv' WITH (FORMAT csv, HEADER true, NULL '');

-- Dikkat: dosya adi buyuk harfli (POS_CASH_balance.csv), tablo adi kucuk harfli.
TRUNCATE raw.pos_cash_balance;
COPY raw.pos_cash_balance FROM '/data/raw/POS_CASH_balance.csv' WITH (FORMAT csv, HEADER true, NULL '');

TRUNCATE raw.previous_application;
COPY raw.previous_application FROM '/data/raw/previous_application.csv' WITH (FORMAT csv, HEADER true, NULL '');
