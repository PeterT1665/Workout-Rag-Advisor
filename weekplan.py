"""Week skeleton builder: decides WHAT kind of session goes on WHICH day.

Pure rules, no LLM. The agent later fills each slot with concrete content;
this module guarantees the load logic (games first, recovery in hard weeks).
"""
from dataclasses import dataclass, field

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Football skill sessions / gym power sessions / recovery sessions per game count.
# Peter's rule (2026-10-23): 1 game -> 2 football, 2 games -> 1, 3 games -> 0.
# Gym and recovery counts are first drafts; edit freely.
LOAD_TABLE = {
    0: {"football": 3, "gym": 1, "recovery": 1},
    1: {"football": 2, "gym": 1, "recovery": 1},
    2: {"football": 1, "gym": 1, "recovery": 1},
    3: {"football": 0, "gym": 1, "recovery": 2},
}

CHEER_DAY = 5  # Saturday morning


@dataclass
class Game:
    day: int          # 0 = Monday ... 6 = Sunday
    kind: str = "casual"   # "casual" (90+ min) or "competitive" (40 min)


@dataclass
class Slot:
    day: int
    kind: str         # game | football | gym | recovery | cheer | rest
    note: str = ""


@dataclass
class WeekPlan:
    slots: list = field(default_factory=list)
    reasons: list = field(default_factory=list)

    def by_day(self):
        out = {d: [] for d in range(7)}
        for s in self.slots:
            out[s.day].append(s)
        return out

    def render(self):
        lines = []
        for d, slots in self.by_day().items():
            text = ", ".join(f"{s.kind}" + (f" ({s.note})" if s.note else "") for s in slots) or "rest"
            lines.append(f"{DAYS[d]:<10} {text}")
        return "\n".join(lines)


def build_week(games, cheer=True):
    """Place sessions around the week's games. Returns a WeekPlan."""
    games = list(games)
    n = min(len(games), 3)
    counts = LOAD_TABLE[n]
    plan = WeekPlan()
    game_days = {g.day for g in games}
    busy = set()  # days already holding a main session

    for g in games:
        plan.slots.append(Slot(g.day, "game", g.kind))
        busy.add(g.day)
    if cheer and CHEER_DAY not in game_days:
        plan.slots.append(Slot(CHEER_DAY, "cheer", "Saturday morning"))
        busy.add(CHEER_DAY)

    cheer_day = CHEER_DAY if cheer else None

    def penalty(d):
        """Lower is better. Rest before a game matters most."""
        pen = 0
        if (d + 1) in game_days:
            pen += 10   # hard session the day before a game
        if (d - 1) in game_days:
            pen += 4    # still tired from yesterday's game
        if cheer_day is not None:
            if (d + 1) == cheer_day:
                pen += 3  # day before cheerleading
            if (d - 1) == cheer_day:
                pen += 2  # day after cheerleading
        if (d + 1) in busy_hard or (d - 1) in busy_hard:
            pen += 1      # back-to-back hard days
        return pen

    busy_hard = set()  # football/gym days (not games): used to spread sessions out

    def pick(kind):
        cands = [d for d in range(7) if d not in busy]
        if not cands:
            return None
        best = min(cands, key=lambda d: (penalty(d), d))
        if penalty(best) >= 4:
            plan.reasons.append(
                f"{kind} placed on {DAYS[best]}, close to a game/cheer day: no cleaner day was free.")
        return best

    # hardest first: football skill, then gym power, then recovery
    for kind in ("football", "gym"):
        for _ in range(counts[kind]):
            d = pick(kind)
            if d is None:
                plan.reasons.append(f"No free day for a {kind} session; skipped to protect recovery.")
                continue
            plan.slots.append(Slot(d, kind))
            busy.add(d)
            busy_hard.add(d)

    # recovery sessions: day after a game first (flush the legs), then any free day
    for _ in range(counts["recovery"]):
        after_game = [g + 1 for g in sorted(game_days) if g + 1 < 7 and g + 1 not in busy]
        d = after_game[0] if after_game else pick("recovery")
        if d is None:
            continue
        plan.slots.append(Slot(d, "recovery", "hips, hamstrings, quads, light mobility"))
        busy.add(d)

    plan.slots.sort(key=lambda s: s.day)
    plan.reasons.insert(0, _why(n, counts))
    return plan


def _why(n, counts):
    games = f"{n} game{'s' if n != 1 else ''}" if n < 3 else "3+ games"
    base = (f"{games} this week -> {counts['football']} football skill, {counts['gym']} gym, "
            f"{counts['recovery']} recovery session(s).")
    if n >= 3:
        base += " Games already carry the football load, so no extra football sessions; focus on gym and recovery."
    return base


if __name__ == "__main__":
    import sys
    # e.g. python weekplan.py 2:competitive 0:casual
    gs = []
    for arg in sys.argv[1:]:
        d, _, kind = arg.partition(":")
        gs.append(Game(int(d), kind or "casual"))
    wp = build_week(gs)
    print("\n".join(wp.reasons))
    print(wp.render())
