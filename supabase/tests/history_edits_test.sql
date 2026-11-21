-- History merge (0012): the latest edit wins, deleted stays deleted, a checked session beats an unchecked one.
select e ->> 'rpe' as rpe, e -> 'lifts' -> 0 -> 'sets' -> 0 ->> 'kg' as kg from jsonb_array_elements(public.merge_history(
  '[{"date":"2026-11-20","id":"g","minutes":50,"title":"Gym","load":350,"rpe":7,"lifts":[{"name":"Bench press","sets":[{"kg":60,"reps":6}]}],"at":100}]',
  '[{"date":"2026-11-20","id":"g","minutes":50,"title":"Gym","load":400,"rpe":8,"lifts":[{"name":"Bench press","sets":[{"kg":62.5,"reps":6}]}],"at":200}]')) e;  -- 8 | 62.5
select e ->> 'rpe' as rpe from jsonb_array_elements(public.merge_history(
  '[{"date":"2026-11-20","id":"g","minutes":50,"title":"Gym","load":400,"rpe":8,"at":200}]',
  '[{"date":"2026-11-20","id":"g","minutes":50,"title":"Gym","load":350,"rpe":7,"at":100}]')) e;  -- 8 (an older edit arriving late loses)
select e ? 'deleted' as deleted from jsonb_array_elements(public.merge_history(
  '[{"date":"2026-11-20","id":"g","minutes":50,"title":"Gym","load":350,"rpe":7,"deleted":true,"at":100}]',
  '[{"date":"2026-11-20","id":"g","minutes":50,"title":"Gym","load":400,"rpe":8,"at":200}]')) e;  -- t
select e ->> 'load' as load from jsonb_array_elements(public.merge_history(
  '[{"date":"2026-11-20","id":"g","minutes":50,"title":"Gym"}]',
  '[{"date":"2026-11-20","id":"g","minutes":50,"title":"Gym","load":350,"rpe":7}]')) e;  -- 350 (old entries without at)
select count(*) from jsonb_array_elements(public.merge_history('[{"date":"1","id":"a","at":"x"}]', '[{"date":"2","id":"b"}]'));  -- 2 (a bad at is ignored)
