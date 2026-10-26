-- PostgreSQL 15+ with TimescaleDB 2.x. Idempotent: safe to re-apply on every deploy.
-- Storage invariants:
--   * raw_market_bars / ticks_raw hold prints exactly as traded and are never updated.
--   * corporate_actions holds adjustment parameters only; adjusted series are computed at query time.
--   * financial_metrics_pit and macro_observations are bitemporal and append-only;
--     a restatement is a new revision, never an overwrite.

CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS securities (
    ticker          TEXT        NOT NULL,
    exchange        TEXT        NOT NULL CHECK (exchange IN ('NYSE', 'NASDAQ', 'NSE', 'BSE')),
    isin            CHAR(12),
    currency        CHAR(3)     NOT NULL,
    sector_class    TEXT        NOT NULL DEFAULT 'GENERAL'
                                CHECK (sector_class IN ('GENERAL', 'BANK', 'NBFC', 'INSURANCE')),
    listed_on       DATE,
    delisted_on     DATE,
    PRIMARY KEY (ticker, exchange)
);
-- Peer-group key for RV (e.g. a BICS/GICS industry name); added after the first deployments.
ALTER TABLE securities ADD COLUMN IF NOT EXISTS industry TEXT;
CREATE INDEX IF NOT EXISTS ix_securities_industry ON securities (industry);

CREATE OR REPLACE FUNCTION forbid_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'Table % is append-only; % is forbidden', TG_TABLE_NAME, TG_OP;
END;
$$ LANGUAGE plpgsql;

-- ---------------------------------------------------------------- raw market bars (immutable)
CREATE TABLE IF NOT EXISTS raw_market_bars (
    ticker          TEXT             NOT NULL,
    exchange        TEXT             NOT NULL,
    bar_interval    INTERVAL         NOT NULL,         -- '1 day' EOD, '1 minute' live candles, ...
    "timestamp"     TIMESTAMPTZ      NOT NULL,         -- bar open; EOD bars use local midnight of the session
    open            DOUBLE PRECISION NOT NULL,
    high            DOUBLE PRECISION NOT NULL,
    low             DOUBLE PRECISION NOT NULL,
    close           DOUBLE PRECISION NOT NULL,
    volume          BIGINT           NOT NULL CHECK (volume >= 0),
    vwap            DOUBLE PRECISION,
    source          TEXT             NOT NULL,
    ingested_at     TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, exchange, bar_interval, "timestamp"),
    FOREIGN KEY (ticker, exchange) REFERENCES securities (ticker, exchange),
    CHECK (low > 0 AND low <= LEAST(open, close) AND high >= GREATEST(open, close)),
    -- 1e-6 relative slack: venue VWAP = turnover / volume is rounded independently of high/low.
    CHECK (vwap IS NULL OR (vwap >= low * (1 - 1e-6) AND vwap <= high * (1 + 1e-6)))
);
SELECT create_hypertable('raw_market_bars', 'timestamp', if_not_exists => TRUE,
                         chunk_time_interval => INTERVAL '30 days');

CREATE OR REPLACE TRIGGER raw_market_bars_no_update
    BEFORE UPDATE ON raw_market_bars
    FOR EACH ROW EXECUTE FUNCTION forbid_mutation();

-- ---------------------------------------------------------------- corporate actions
-- adjustment_factor multiplies every price strictly before ex_date (capital changes):
--   SPLIT n-for-m -> m/n, BONUS a:b -> b/(a+b), RIGHTS -> TERP / cum-rights close.
-- Cash dividends store only the raw amount; their factor (P - D) / P needs the prior raw close
-- and is derived at query time.
CREATE TABLE IF NOT EXISTS corporate_actions (
    ticker              TEXT            NOT NULL,
    exchange            TEXT            NOT NULL,
    ex_date             DATE            NOT NULL,
    action_type         TEXT            NOT NULL
                                        CHECK (action_type IN ('SPLIT', 'BONUS', 'RIGHTS', 'CASH_DIVIDEND')),
    adjustment_factor   NUMERIC(24, 12),
    raw_dividend_amount NUMERIC(18, 8),                -- per share, trading currency, unadjusted
    announced_at        TIMESTAMPTZ     NOT NULL,      -- knowledge time; invisible to earlier as-of queries
    source              TEXT            NOT NULL,
    ingested_at         TIMESTAMPTZ     NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, exchange, action_type, ex_date),
    FOREIGN KEY (ticker, exchange) REFERENCES securities (ticker, exchange),
    CHECK (
        (action_type IN ('SPLIT', 'BONUS', 'RIGHTS') AND adjustment_factor > 0 AND raw_dividend_amount IS NULL)
        OR (action_type = 'CASH_DIVIDEND' AND raw_dividend_amount > 0 AND adjustment_factor IS NULL)
    ),
    CHECK (action_type NOT IN ('BONUS', 'RIGHTS') OR adjustment_factor <= 1)
);

-- ---------------------------------------------------------------- fundamentals (bitemporal PIT)
CREATE TABLE IF NOT EXISTS financial_metrics_pit (
    ticker              TEXT            NOT NULL,
    exchange            TEXT            NOT NULL,
    metric_name         TEXT            NOT NULL,
    period_type         TEXT            NOT NULL CHECK (period_type IN ('Q', 'H', 'FY', 'TTM')),
    period_end_date     DATE            NOT NULL,      -- valid time
    filing_timestamp    TIMESTAMPTZ     NOT NULL,      -- knowledge time (public dissemination)
    revision_number     INTEGER         NOT NULL DEFAULT 0,  -- assigned by trigger; 1 = first vintage
    metric_value        NUMERIC(28, 6)  NOT NULL,
    currency            CHAR(3)         NOT NULL,
    source_authority    TEXT            NOT NULL,      -- SEC_EDGAR, NSE_XBRL, BSE_XBRL, FMP, POLYGON
    source_document_id  TEXT,                          -- accession number / exchange filing id
    confidence_score    NUMERIC(4, 3)   NOT NULL CHECK (confidence_score BETWEEN 0 AND 1),
    ingested_at         TIMESTAMPTZ     NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, exchange, metric_name, period_type, period_end_date, source_authority, revision_number),
    UNIQUE (ticker, exchange, metric_name, period_type, period_end_date, source_authority, filing_timestamp),
    FOREIGN KEY (ticker, exchange) REFERENCES securities (ticker, exchange),
    CHECK (revision_number >= 1),
    CHECK (filing_timestamp::date >= period_end_date)
);

CREATE INDEX IF NOT EXISTS ix_metrics_pit_asof
    ON financial_metrics_pit (ticker, exchange, metric_name, period_type, filing_timestamp);

-- Revisions are numbered in knowledge-time order so "highest revision with filing_timestamp <= as_of"
-- is always the latest vintage public at as_of. Re-ingesting a known vintage is a silent no-op.
CREATE OR REPLACE FUNCTION assign_metric_revision() RETURNS trigger AS $$
DECLARE
    prior_revision INTEGER;
    prior_filed    TIMESTAMPTZ;
BEGIN
    PERFORM pg_advisory_xact_lock(hashtextextended(concat_ws('|', NEW.ticker, NEW.exchange, NEW.metric_name,
        NEW.period_type, NEW.period_end_date::text, NEW.source_authority), 0));

    IF EXISTS (
        SELECT 1 FROM financial_metrics_pit p
        WHERE p.ticker = NEW.ticker AND p.exchange = NEW.exchange AND p.metric_name = NEW.metric_name
          AND p.period_type = NEW.period_type AND p.period_end_date = NEW.period_end_date
          AND p.source_authority = NEW.source_authority AND p.filing_timestamp = NEW.filing_timestamp
    ) THEN
        RETURN NULL;
    END IF;

    SELECT p.revision_number, p.filing_timestamp INTO prior_revision, prior_filed
    FROM financial_metrics_pit p
    WHERE p.ticker = NEW.ticker AND p.exchange = NEW.exchange AND p.metric_name = NEW.metric_name
      AND p.period_type = NEW.period_type AND p.period_end_date = NEW.period_end_date
      AND p.source_authority = NEW.source_authority
    ORDER BY p.revision_number DESC
    LIMIT 1;

    IF prior_filed IS NOT NULL AND NEW.filing_timestamp < prior_filed THEN
        RAISE EXCEPTION USING
            ERRCODE = 'check_violation',
            MESSAGE = format('out-of-order vintage for %s:%s %s %s %s [%s]: filed %s precedes revision %s filed %s',
                             NEW.exchange, NEW.ticker, NEW.metric_name, NEW.period_type, NEW.period_end_date,
                             NEW.source_authority, NEW.filing_timestamp, prior_revision, prior_filed);
    END IF;

    NEW.revision_number := COALESCE(prior_revision, 0) + 1;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER financial_metrics_pit_revision
    BEFORE INSERT ON financial_metrics_pit
    FOR EACH ROW EXECUTE FUNCTION assign_metric_revision();

CREATE OR REPLACE TRIGGER financial_metrics_pit_append_only
    BEFORE UPDATE OR DELETE ON financial_metrics_pit
    FOR EACH ROW EXECUTE FUNCTION forbid_mutation();

-- ---------------------------------------------------------------- macro (bitemporal, vintage-aware)
CREATE TABLE IF NOT EXISTS macro_observations (
    series_id           TEXT        NOT NULL,          -- e.g. FRED:CPIAUCSL, RBI:REPO
    observation_date    DATE        NOT NULL,
    release_timestamp   TIMESTAMPTZ NOT NULL,
    value               NUMERIC(28, 8) NOT NULL,
    source              TEXT        NOT NULL,
    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (series_id, observation_date, release_timestamp)
);

CREATE OR REPLACE TRIGGER macro_observations_append_only
    BEFORE UPDATE OR DELETE ON macro_observations
    FOR EACH ROW EXECUTE FUNCTION forbid_mutation();

-- ---------------------------------------------------------------- raw ticks (immutable)
CREATE TABLE IF NOT EXISTS ticks_raw (
    ticker          TEXT             NOT NULL,
    exchange        TEXT             NOT NULL,
    "timestamp"     TIMESTAMPTZ      NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    size            BIGINT           NOT NULL,
    accepted        BOOLEAN          NOT NULL,         -- MAD filter verdict, raw print kept either way
    reject_reason   TEXT,
    source          TEXT             NOT NULL
);
SELECT create_hypertable('ticks_raw', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');
CREATE INDEX IF NOT EXISTS ix_ticks_listing_ts ON ticks_raw (ticker, exchange, "timestamp" DESC);

CREATE OR REPLACE TRIGGER ticks_raw_no_update
    BEFORE UPDATE ON ticks_raw
    FOR EACH ROW EXECUTE FUNCTION forbid_mutation();

-- ---------------------------------------------------------------- tick quarantine (immutable)
-- Prints the hygiene filter kept away from bars/indicators, with the statistics that condemned them.
CREATE TABLE IF NOT EXISTS outlier_ticks (
    ticker          TEXT             NOT NULL,
    exchange        TEXT             NOT NULL,
    "timestamp"     TIMESTAMPTZ      NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    size            BIGINT           NOT NULL,
    reason          TEXT             NOT NULL,
    rolling_median  DOUBLE PRECISION,
    rolling_mad     DOUBLE PRECISION,
    deviation_mads  DOUBLE PRECISION,
    best_bid        DOUBLE PRECISION,
    best_ask        DOUBLE PRECISION,
    book_age_ms     DOUBLE PRECISION,
    source          TEXT             NOT NULL,
    quarantined_at  TIMESTAMPTZ      NOT NULL DEFAULT now()
);
SELECT create_hypertable('outlier_ticks', 'timestamp', if_not_exists => TRUE,
                         chunk_time_interval => INTERVAL '7 days');
CREATE INDEX IF NOT EXISTS ix_outliers_listing_ts ON outlier_ticks (ticker, exchange, "timestamp" DESC);

CREATE OR REPLACE TRIGGER outlier_ticks_no_update
    BEFORE UPDATE ON outlier_ticks
    FOR EACH ROW EXECUTE FUNCTION forbid_mutation();

-- ---------------------------------------------------------------- reconciliation audit
CREATE TABLE IF NOT EXISTS reconciliation_breaks (
    break_id            BIGSERIAL   PRIMARY KEY,
    ticker              TEXT        NOT NULL,
    exchange            TEXT        NOT NULL,
    metric_name         TEXT        NOT NULL,
    period_type         TEXT        NOT NULL,
    period_end_date     DATE        NOT NULL,
    as_of               TIMESTAMPTZ NOT NULL,
    primary_source      TEXT        NOT NULL,
    secondary_source    TEXT        NOT NULL,
    primary_value       NUMERIC(28, 6),
    secondary_value     NUMERIC(28, 6),
    relative_variance   DOUBLE PRECISION,              -- unbounded: scale errors produce ratios ~1e9
    status              TEXT        NOT NULL,
    detected_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (ticker, exchange) REFERENCES securities (ticker, exchange)
);
CREATE INDEX IF NOT EXISTS ix_recon_listing_detected ON reconciliation_breaks (ticker, exchange, detected_at DESC);

-- ---------------------------------------------------------------- listed option quotes (immutable)
-- Raw venue quotes as published; implied volatilities and Greeks are derived at query time by OVME.
CREATE TABLE IF NOT EXISTS option_quotes (
    ticker          TEXT             NOT NULL,         -- underlying
    exchange        TEXT             NOT NULL,
    expiry          DATE             NOT NULL,         -- contract expiry as listed
    strike          NUMERIC(18, 6)   NOT NULL CHECK (strike > 0),
    option_type     TEXT             NOT NULL CHECK (option_type IN ('CALL', 'PUT')),
    "timestamp"     TIMESTAMPTZ      NOT NULL,         -- quote time (knowledge time)
    bid             DOUBLE PRECISION NOT NULL CHECK (bid >= 0),
    ask             DOUBLE PRECISION NOT NULL,
    last            DOUBLE PRECISION CHECK (last IS NULL OR last >= 0),
    volume          BIGINT           CHECK (volume IS NULL OR volume >= 0),
    open_interest   BIGINT           CHECK (open_interest IS NULL OR open_interest >= 0),
    source          TEXT             NOT NULL,
    ingested_at     TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, exchange, expiry, strike, option_type, "timestamp"),
    FOREIGN KEY (ticker, exchange) REFERENCES securities (ticker, exchange),
    CHECK (ask > 0 AND ask >= bid)
);
SELECT create_hypertable('option_quotes', 'timestamp', if_not_exists => TRUE,
                         chunk_time_interval => INTERVAL '7 days');

CREATE OR REPLACE TRIGGER option_quotes_no_update
    BEFORE UPDATE ON option_quotes
    FOR EACH ROW EXECUTE FUNCTION forbid_mutation();
