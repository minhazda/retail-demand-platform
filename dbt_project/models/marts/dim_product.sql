-- Most-frequent description per stock code (descriptions vary across rows).
with counted as (
    select
        stock_code,
        description,
        count(*) as n,
        row_number() over (partition by stock_code order by count(*) desc) as rn
    from {{ ref('stg_transactions') }}
    group by stock_code, description
)

select stock_code, description
from counted
where rn = 1
