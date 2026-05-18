{{ config(materialized='table') }}

select
    date,
    state,
    arrivals,
    overnight,
    extract(year  from date)::int as year,
    extract(month from date)::int as month
from {{ ref('stg_tourism') }}