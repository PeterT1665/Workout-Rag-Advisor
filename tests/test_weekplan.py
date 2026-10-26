import unittest

from weekplan import Game, build_week, LOAD_TABLE


def kinds(plan, kind):
    return [s.day for s in plan.slots if s.kind == kind]


class WeekPlanTests(unittest.TestCase):
    def test_session_counts_follow_game_count(self):
        # Peter's rule: 1 game -> 2 football, 2 games -> 1, 3 games -> 0
        for days, expected in [([2], 2), ([2, 0], 1), ([0, 2, 4], 0)]:
            plan = build_week([Game(d) for d in days])
            self.assertEqual(len(kinds(plan, "football")), expected)

    def test_three_games_get_recovery_not_more_football(self):
        plan = build_week([Game(0), Game(2), Game(4)])
        self.assertEqual(len(kinds(plan, "football")), 0)
        self.assertEqual(len(kinds(plan, "recovery")), LOAD_TABLE[3]["recovery"])

    def test_no_hard_session_the_day_before_a_game_when_avoidable(self):
        plan = build_week([Game(2)])  # Wednesday game
        hard = kinds(plan, "football") + kinds(plan, "gym")
        self.assertNotIn(1, hard)

    def test_one_main_session_per_day_apart_from_games(self):
        plan = build_week([Game(2, "competitive"), Game(2, "casual")])
        for day, slots in plan.by_day().items():
            non_game = [s for s in slots if s.kind != "game"]
            self.assertLessEqual(len(non_game), 1, f"day {day}")

    def test_saturday_is_cheerleading(self):
        self.assertEqual(kinds(build_week([Game(2)]), "cheer"), [5])
        self.assertEqual(kinds(build_week([Game(2)], cheer=False), "cheer"), [])

    def test_recovery_follows_a_game(self):
        plan = build_week([Game(2)])
        self.assertIn(3, kinds(plan, "recovery"))

    def test_reasons_explain_the_week(self):
        plan = build_week([Game(2)])
        self.assertIn("1 game", plan.reasons[0])


if __name__ == "__main__":
    unittest.main()
