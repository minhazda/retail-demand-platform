-- Daily demand per product for the top-50 products by total quantity:
-- the training table for the forecasting model.
with ranked_products as (
    select
        stock_code,
        sum(quantity) as total_quantity
    from {{ ref('stg_transactions') }}
    group by stock_code
    order by total_quantity desc
    limit 50
)

select
    t.stock_code,
    t.invoice_date,
    sum(t.quantity)              as units_sold,
    sum(t.revenue)               as revenue,
    count(distinct t.invoice)    as n_orders
from {{ ref('stg_transactions') }} t
join ranked_products r using (stock_code)
group by t.stock_code, t.invoice_date
