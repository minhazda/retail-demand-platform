-- Cleaning rules mirror real-retail-forecasting's documented decisions:
-- drop cancellations (credit invoices), non-positive quantities/prices, and
-- non-product stock codes (postage, fees, adjustments).
with source as (
    select * from {{ source('raw', 'transactions') }}
)

select
    invoice,
    stock_code,
    trim(description)                as description,
    quantity,
    invoice_ts,
    cast(invoice_ts as date)         as invoice_date,
    price,
    quantity * price                 as revenue,
    customer_id,
    country
from source
where quantity > 0
  and price > 0
  and invoice not like 'C%'
  and regexp_matches(stock_code, '^[0-9]{5}')
