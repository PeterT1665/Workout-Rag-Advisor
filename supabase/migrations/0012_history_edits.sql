-- Sessions can be edited after the fact now (sets, weights, effort: Record → tap a session), so when two copies
-- of the same session disagree the latest edit wins (`at`, ms). Before, the copy with more fields won, so an
-- edited weight could come back as the old one on the next sync. A deleted session stays deleted.
create or replace function public.merge_history(stored jsonb, incoming jsonb)
returns jsonb
language sql
immutable
set search_path = ''
as $$
  select coalesce(jsonb_agg(e order by e ->> 'date', ord, n), '[]'::jsonb)
  from (
    select distinct on (k) e, ord, n
    from (
      select e, ord, n,
        concat_ws('|', e ->> 'date', e ->> 'id', e ->> 'minutes', e ->> 'title') as k,
        e ? 'deleted' as deleted,
        (e ->> 'load') is not null as checked,
        case when jsonb_typeof(e -> 'at') = 'number' then (e ->> 'at')::numeric else 0 end as edited,
        (select count(*) from jsonb_object_keys(e)) as fields
      from (
        select e, 0 as ord, n from jsonb_array_elements(coalesce(stored, '[]'::jsonb)) with ordinality as t(e, n)
        union all
        select e, 1 as ord, n from jsonb_array_elements(coalesce(incoming, '[]'::jsonb)) with ordinality as t(e, n)
      ) x
    ) y
    order by k, deleted desc, checked desc, edited desc, fields desc, ord
  ) z
$$;
revoke execute on function public.merge_history(jsonb, jsonb) from public, anon;
grant execute on function public.merge_history(jsonb, jsonb) to authenticated;
