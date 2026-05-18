{{ config(materialized='view') }}

select
    cast(date as date)         as date,
    cast(state as text)        as state,
    cast(arrivals as numeric)  as arrivals,
    cast(overnight as numeric) as overnight,
    ingested_at
from {{ source('raw', 'tourism_raw') }}
where date is not null
  and state is not null