CREATE TABLE IF NOT EXISTS public.tourism_raw (
    date         DATE        NOT NULL,
    state        TEXT        NOT NULL,
    arrivals     NUMERIC,
    overnight    NUMERIC,
    ingested_at  TIMESTAMP   NOT NULL DEFAULT NOW(),
    PRIMARY KEY (date, state)
);

CREATE INDEX IF NOT EXISTS idx_tourism_raw_state ON public.tourism_raw (state);
CREATE INDEX IF NOT EXISTS idx_tourism_raw_date  ON public.tourism_raw (date);