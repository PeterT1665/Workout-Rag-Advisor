-- Friend notifications: the other player hears about it straight away, in their language, and a tap opens Team.
--   someone asks you              "Friend request"  "Morris wants to be friends. Tap to accept."
--   someone accepts your request  "New friend"      "Morris accepted your friend request."
--   someone scans your QR         "New friend"      "Morris scanned your QR. You're friends now."
-- Sent by the database itself when a friendship row changes, so it works from any version of the app.

-- its own switch in Settings → Notifications
alter table public.push_subscriptions add column if not exists friends boolean not null default true;

-- a fourth kind of notification
alter table public.scheduled_pushes drop constraint if exists scheduled_pushes_kind_check;
alter table public.scheduled_pushes add constraint scheduled_pushes_kind_check check (kind in ('rest', 'weekly', 'match', 'friend'));

create or replace function public.notify_friendship()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
declare
  what text;     -- request | accepted | qr
  actor uuid;    -- who did it
  target uuid;   -- who hears about it
  who text;
  lng text;
  title text;
  body text;
begin
  if tg_op = 'INSERT' and new.status = 'pending' then what := 'request';
  elsif tg_op = 'INSERT' and new.status = 'accepted' then what := 'qr';           -- accept_invite: friends at once
  elsif tg_op = 'UPDATE' and old.status = 'pending' and new.status = 'accepted' then what := 'accepted';
  else return new;
  end if;

  begin -- a notification problem must never stop the friendship itself
    actor := coalesce(auth.uid(), case when what = 'request' then new.requester else new.addressee end);
    target := case when actor = new.requester then new.addressee else new.requester end;

    -- the target's language, from their newest device that wants friend notifications; none: nothing to send
    select p.lang into lng from public.push_subscriptions p
    where p.user_id = target and p.friends order by p.created_at desc limit 1;
    if lng is null then return new; end if;

    select coalesce(nullif(trim(f.display_name), ''), f.username) into who from public.profiles f where f.user_id = actor;
    who := coalesce(nullif(who, ''), case lng when 'vi' then 'Một người bạn' when 'es' then 'Alguien' when 'pt' then 'Alguém'
      when 'fr' then 'Quelqu''un' when 'id' then 'Seseorang' else 'A friend' end);

    title := case
      when what = 'request' then case lng when 'vi' then 'Lời mời kết bạn' when 'es' then 'Solicitud de amistad' when 'pt' then 'Pedido de amizade'
        when 'fr' then 'Demande d''ami' when 'id' then 'Permintaan teman' else 'Friend request' end
      else case lng when 'vi' then 'Bạn mới' when 'es' then 'Nuevo amigo' when 'pt' then 'Novo amigo'
        when 'fr' then 'Nouvel ami' when 'id' then 'Teman baru' else 'New friend' end
    end;
    body := case what
      when 'request' then case lng
        when 'vi' then '{who} muốn kết bạn với bạn. Chạm để chấp nhận.'
        when 'es' then '{who} quiere ser tu amigo. Toca para aceptar.'
        when 'pt' then '{who} quer ser seu amigo. Toque para aceitar.'
        when 'fr' then '{who} veut devenir ton ami. Touche pour accepter.'
        when 'id' then '{who} ingin berteman. Ketuk untuk menerima.'
        else '{who} wants to be friends. Tap to accept.' end
      when 'accepted' then case lng
        when 'vi' then '{who} đã chấp nhận lời mời kết bạn.'
        when 'es' then '{who} aceptó tu solicitud de amistad.'
        when 'pt' then '{who} aceitou seu pedido de amizade.'
        when 'fr' then '{who} a accepté ta demande d''ami.'
        when 'id' then '{who} menerima permintaan pertemananmu.'
        else '{who} accepted your friend request.' end
      else case lng
        when 'vi' then '{who} đã quét mã QR của bạn. Giờ hai bạn là bạn bè.'
        when 'es' then '{who} escaneó tu QR. Ya sois amigos.'
        when 'pt' then '{who} escaneou seu QR. Agora vocês são amigos.'
        when 'fr' then '{who} a scanné ton QR. Vous êtes amis maintenant.'
        when 'id' then '{who} memindai QR-mu. Sekarang kalian berteman.'
        else '{who} scanned your QR. You''re friends now.' end
    end;

    insert into public.scheduled_pushes (user_id, send_at, kind, title, body, url, tag)
    values (target, now(), 'friend', left(title, 80), left(replace(body, '{who}', who), 200), '/?tab=team', null);
  exception when others then
    raise warning 'friend notification skipped: %', sqlerrm;
  end;
  return new;
end;
$$;
revoke execute on function public.notify_friendship() from public, anon, authenticated;

drop trigger if exists friendship_notify on public.friendships;
create trigger friendship_notify after insert or update of status on public.friendships
  for each row execute function public.notify_friendship();
