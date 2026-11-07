"""Heads-up Texas Hold'em bot with Monte Carlo equity and opponent profiling.

This file only uses the public bot API in this repository.  Opponent profiles
are ordinary dictionaries so the Model remains pickle-compatible.
"""

from __future__ import annotations

import random
from itertools import combinations
from typing import Any

from card_utils import Card, evaluate, full_deck
from player_action import DEALER, PlayerAction


# Keep evaluation bounded.  The arena's worker enforces a per-move deadline;
# these values are deliberately conservative and can be tuned after testing.
_SIMULATIONS = {3: 420, 4: 650, 5: 900}
_RIVER_EXACT_MAX_OPPONENTS = 1
_MAX_RECENT_HANDS = 60
_MAX_SHOWDOWNS = 60
_PROFILE_WARMUP_HANDS = 100
_MIN_FOLD_TO_CBET_OPPORTUNITIES = 12


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _new_position_stats() -> dict[str, int]:
    return {
        "hands": 0,
        "vpip_hands": 0,
        "pfr_hands": 0,
        "rfi_hands": 0,
        "limp_hands": 0,
    }


def _new_profile() -> dict[str, Any]:
    """Return a profile containing only pickle-safe primitive values."""
    return {
        "hands": 0,
        "vpip_hands": 0,
        "pfr_hands": 0,
        "rfi_opportunities": 0,
        "rfi_hands": 0,
        "limp_hands": 0,
        "three_bet_opportunities": 0,
        "three_bets": 0,
        "four_bet_opportunities": 0,
        "four_bets": 0,
        "fold_to_three_bet_opportunities": 0,
        "fold_to_three_bet": 0,
        "cbet_opportunities": 0,
        "cbets": 0,
        "fold_to_cbet_opportunities": 0,
        "fold_to_cbet": 0,
        "flop_bet_opportunities": 0,
        "flop_bets": 0,
        "turn_bet_opportunities": 0,
        "turn_bets": 0,
        "river_bet_opportunities": 0,
        "river_bets": 0,
        "aggressive_actions": 0,
        "raise_actions": 0,
        "all_in_raises": 0,
        "all_in_showdowns": 0,
        "weak_all_in_showdowns": 0,
        "passive_actions": 0,
        "calls": 0,
        "checks": 0,
        "folds": 0,
        "showdowns": 0,
        "showdown_wins": 0,
        "weak_showdowns": 0,
        "position": {
            "button": _new_position_stats(),
            "big_blind": _new_position_stats(),
        },
        "recent": [],
        "showdown_history": [],
        "bet_size_history": [],
    }


def _preflop_strength(hand: list[Card]) -> float:
    """A lightweight ordering heuristic, not a substitute for equity."""
    if len(hand) != 2:
        return 0.0
    a, b = sorted((card.value for card in hand), reverse=True)
    suited = hand[0].suit == hand[1].suit
    pair = a == b
    gap = a - b - 1

    if pair:
        score = 0.48 + (a - 2) * 0.034
    else:
        score = 0.13 + (a - 2) * 0.019 + (b - 2) * 0.012
        if suited:
            score += 0.055
        if gap == 0:
            score += 0.065
        elif gap == 1:
            score += 0.040
        elif gap == 2:
            score += 0.020
        elif gap >= 4:
            score -= 0.035
        if a == 14:
            score += 0.060
        if a >= 12 and b >= 10:
            score += 0.045
        if a == 14 and b >= 11:
            score += 0.035

    return _clamp(score, 0.04, 0.97)


def _rank_strength(rank: tuple[int, ...]) -> float:
    """Map the evaluator's lexicographic hand rank to a rough [0, 1] value."""
    category = rank[0]
    # Category dominates; kickers provide a small ordering within categories.
    kicker = (rank[1] / 14.0) if len(rank) > 1 else 0.0
    return _clamp(category / 8.0 + kicker * 0.055, 0.015, 0.99)


def _rate(successes: int, opportunities: int, prior: float = 0.5,
          prior_weight: float = 4.0) -> float:
    """Smoothed frequency; small samples stay close to the prior."""
    return (successes + prior * prior_weight) / (opportunities + prior_weight)


class Model:
    """A pickle-safe heads-up bot with persistent, team-keyed opponent data."""

    def __init__(self):
        self.hand: list[Card] = []
        self.player_id: int | None = None

        # Lifetime profiles survive seed() calls and therefore survive hands
        # while the arena keeps this Model instance alive.
        self.opponent_profiles: dict[str, dict[str, Any]] = {}
        self.rng = random.Random(781239)
        self.hands_seen = 0
        self._reset_hand_state()

    # ------------------------------------------------------------------
    # Public model interface
    # ------------------------------------------------------------------

    def seed(self, hand: list[Card], player_id: int):
        """Called once at the start of each hand; do not clear learned data."""
        self.hand = list(hand)
        self.player_id = player_id
        self.hands_seen += 1
        self._reset_hand_state()

    def run(self, actions: list[PlayerAction], game_state) -> PlayerAction:
        """Observe the new action batch, then make one legal decision."""
        self._register_players(game_state)
        self._observe_actions(actions, game_state)

        me = self.player_id
        if me is None:
            # Defensive fallback; normal arena flow always calls seed first.
            return PlayerAction.call(game_state.to_act)

        if game_state.street == "preflop":
            return self._decide_preflop(game_state)
        return self._decide_postflop(game_state)

    def end_game(self, actions: list[PlayerAction], game_state):
        """Process the final unseen actions and commit this hand's statistics."""
        self._register_players(game_state)
        self._observe_actions(actions, game_state)
        self._finalise_hand(game_state)

    # ------------------------------------------------------------------
    # Per-hand and persistent state
    # ------------------------------------------------------------------

    def _reset_hand_state(self):
        self._opponent_id_to_name: dict[int, str] = {}
        self._hand_stats: dict[int, dict[str, Any]] = {}
        self._hand_street = "preflop"
        self._hand_board_count = 0
        self._hand_button: int | None = None
        self._preflop_raise_count = 0
        self._preflop_aggressor: int | None = None
        self._last_preflop_raiser: int | None = None
        self._first_preflop_action: set[int] = set()
        self._opponent_preflop_action: dict[int, str] = {}
        self._opponent_last_action: dict[int, str] = {}
        self._opponent_last_action_by_street: dict[int, dict[str, str]] = {}
        self._street_has_bet = False
        self._street_first_action: set[int] = set()
        self._waiting_for_three_bet_response: set[int] = set()
        self._waiting_for_fold_to_three_bet: set[int] = set()
        self._waiting_for_fold_to_cbet: set[int] = set()
        self._opponent_revealed_cards: dict[int, list[Card]] = {}
        self._opponent_winnings: dict[int, int] = {}
        self._hand_finalised = False
        self._hand_events_seen = 0

    def _register_players(self, game_state):
        """Resolve stable team identities from the current seat assignment."""
        self._hand_button = game_state.button
        teams = list(game_state.teams)
        for player in range(game_state.num_players):
            if player == self.player_id:
                continue
            name = str(teams[player])
            self._opponent_id_to_name[player] = name
            if name not in self.opponent_profiles:
                self.opponent_profiles[name] = _new_profile()
            self._hand_stats.setdefault(player, self._new_hand_stats())

    @staticmethod
    def _new_hand_stats() -> dict[str, Any]:
        return {
            "vpip": False,
            "pfr": False,
            "rfi": False,
            "rfi_opportunity": False,
            "limp": False,
            "three_bet_opportunities": 0,
            "three_bets": 0,
            "four_bet_opportunities": 0,
            "four_bets": 0,
            "fold_to_three_bet_opportunities": 0,
            "fold_to_three_bet": 0,
            "cbet_opportunities": 0,
            "cbets": 0,
            "fold_to_cbet_opportunities": 0,
            "fold_to_cbet": 0,
            "flop_bet_opportunities": 0,
            "flop_bets": 0,
            "turn_bet_opportunities": 0,
            "turn_bets": 0,
            "river_bet_opportunities": 0,
            "river_bets": 0,
            "aggressive_actions": 0,
            "raise_actions": 0,
            "all_in_raises": 0,
            "passive_actions": 0,
            "calls": 0,
            "checks": 0,
            "folds": 0,
            "revealed_cards": [],
            "raise_sizes": [],
            "won": 0,
            "street_actions": {"preflop": 0, "flop": 0, "turn": 0, "river": 0},
        }

    def _profile(self, player_id: int) -> dict[str, Any]:
        name = self._opponent_id_to_name.get(player_id, f"seat-{player_id}")
        return self.opponent_profiles.setdefault(name, _new_profile())

    def _observe_actions(self, actions: list[PlayerAction], game_state):
        """Consume each incremental action batch once; histories do not overlap."""
        if not actions:
            return

        for action in actions:
            self._hand_events_seen += 1
            pid = action.player_id
            kind = action.kind

            if pid == DEALER:
                if kind == "reveal_card":
                    self._hand_board_count += 1
                    if self._hand_board_count >= 5:
                        self._hand_street = "river"
                    elif self._hand_board_count == 4:
                        self._hand_street = "turn"
                    elif self._hand_board_count == 3:
                        self._hand_street = "flop"
                    self._street_has_bet = False
                    self._street_first_action.clear()
                    self._waiting_for_fold_to_cbet.clear()
                continue

            if kind == "reveal_card":
                if pid != self.player_id and action.card is not None:
                    self._opponent_revealed_cards.setdefault(pid, []).append(action.card)
                    hs = self._hand_stats.setdefault(pid, self._new_hand_stats())
                    hs["revealed_cards"].append(str(action.card))
                continue

            if kind == "win":
                if pid != self.player_id:
                    self._opponent_winnings[pid] = (
                        self._opponent_winnings.get(pid, 0) + int(action.amount)
                    )
                    hs = self._hand_stats.setdefault(pid, self._new_hand_stats())
                    hs["won"] += int(action.amount)
                continue

            street = self._hand_street
            is_hero = pid == self.player_id
            if not is_hero and pid not in self._hand_stats:
                self._hand_stats[pid] = self._new_hand_stats()
                self._opponent_id_to_name.setdefault(pid, f"seat-{pid}")

            if is_hero:
                self._observe_hero_action(kind, street)
            else:
                all_in_raise = kind == "raise" and game_state.stacks[pid] == 0
                self._observe_opponent_action(
                    pid, kind, street, int(action.amount), all_in_raise
                )

    def _observe_hero_action(self, kind: str, street: str):
        if street == "preflop":
            if kind == "raise":
                # A raise after an opponent raise gives that opponent a
                # response opportunity against our 3-bet.
                if (self._preflop_raise_count == 1
                        and self._last_preflop_raiser is not None
                        and self._last_preflop_raiser != self.player_id):
                    self._waiting_for_fold_to_three_bet.add(self._last_preflop_raiser)
                # If this is our first preflop raise, the opponent can 3-bet.
                if self._preflop_raise_count == 0:
                    for opp in self._opponent_id_to_name:
                        self._waiting_for_three_bet_response.add(opp)
                self._preflop_raise_count += 1
                self._preflop_aggressor = self.player_id
                self._last_preflop_raiser = self.player_id
        else:
            if kind == "raise":
                self._street_has_bet = True
                # A flop bet by the preflop aggressor is a continuation bet.
                if street == "flop" and self._preflop_aggressor == self.player_id:
                    for opp in self._opponent_id_to_name:
                        self._waiting_for_fold_to_cbet.add(opp)

    def _observe_opponent_action(self, pid: int, kind: str, street: str,
                                 amount: int = 0, all_in_raise: bool = False):
        hs = self._hand_stats.setdefault(pid, self._new_hand_stats())
        hs["street_actions"][street] += 1
        if kind == "raise":
            hs["raise_actions"] += 1
            hs["all_in_raises"] += int(all_in_raise)
        first_action_this_street = pid not in self._street_first_action
        self._street_first_action.add(pid)

        if street == "preflop":
            if pid not in self._first_preflop_action:
                self._first_preflop_action.add(pid)
                is_button_first_in = (
                    self._hand_button is not None
                    and pid == self._hand_button
                    and self._preflop_raise_count == 0
                )
                if is_button_first_in:
                    hs["rfi_opportunity"] = True
                    if kind == "raise":
                        hs["rfi"] = True
                    elif kind == "call":
                        hs["limp"] = True
                if kind == "call" and not is_button_first_in and self._preflop_raise_count == 0:
                    self._opponent_preflop_action[pid] = "check"
                else:
                    self._opponent_preflop_action[pid] = kind
                if kind == "raise":
                    hs["vpip"] = True
                elif kind == "call":
                    # In heads-up, a button/small-blind call is a limp. A
                    # big-blind call after no raise is a check, not VPIP.
                    is_button = self._hand_button is not None and pid == self._hand_button
                    if is_button or self._preflop_raise_count > 0:
                        hs["vpip"] = True

            if kind == "fold":
                hs["folds"] += 1
            elif kind == "call":
                is_button = self._hand_button is not None and pid == self._hand_button
                if self._preflop_raise_count > 0 or is_button:
                    hs["calls"] += 1
                else:
                    hs["checks"] += 1
                hs["passive_actions"] += 1

            if pid in self._waiting_for_three_bet_response:
                hs["three_bet_opportunities"] += 1
                if kind == "raise":
                    hs["three_bets"] += 1
                self._waiting_for_three_bet_response.discard(pid)

            if kind == "raise":
                hs["vpip"] = True
                hs["pfr"] = True
                hs["raise_sizes"].append({"street": street, "total": int(amount)})
                self._opponent_preflop_action[pid] = (
                    "shove" if all_in_raise else "raise"
                )
                self._preflop_raise_count += 1
                self._preflop_aggressor = pid
                self._last_preflop_raiser = pid

            if pid in self._waiting_for_fold_to_three_bet:
                hs["fold_to_three_bet_opportunities"] += 1
                hs["four_bet_opportunities"] += 1
                if kind == "fold":
                    hs["fold_to_three_bet"] += 1
                elif kind == "raise":
                    hs["four_bets"] += 1
                self._waiting_for_fold_to_three_bet.discard(pid)

        else:
            # Record genuine opportunities to bet when no one has bet yet.
            if first_action_this_street and not self._street_has_bet:
                opportunity_keys = {
                    "flop": ("flop_bet_opportunities", "flop_bets"),
                    "turn": ("turn_bet_opportunities", "turn_bets"),
                    "river": ("river_bet_opportunities", "river_bets"),
                }.get(street)
                if opportunity_keys is not None:
                    opportunity_key, bet_key = opportunity_keys
                    hs[opportunity_key] += 1
                    if kind == "raise":
                        hs[bet_key] += 1

            # C-bet opportunity: the opponent was the preflop raiser and
            # gets to act before anyone has bet on the flop.
            if (street == "flop" and first_action_this_street
                    and self._preflop_aggressor == pid
                    and not self._street_has_bet
                    and hs["cbet_opportunities"] == 0):
                hs["cbet_opportunities"] += 1
                if kind == "raise":
                    hs["cbets"] += 1

            if pid in self._waiting_for_fold_to_cbet:
                hs["fold_to_cbet_opportunities"] += 1
                if kind == "fold":
                    hs["fold_to_cbet"] += 1
                self._waiting_for_fold_to_cbet.discard(pid)

            if kind == "raise":
                hs["aggressive_actions"] += 1
                # PlayerAction.amount is the total street bet, not the raise
                # increment. Keep it as a raw observation; do not mislabel it
                # as a pot-relative bet size without historical pot context.
                hs["raise_sizes"].append({"street": street, "total": int(amount)})
                self._street_has_bet = True
            elif kind == "fold":
                hs["folds"] += 1
            elif kind == "call":
                # The action format does not include call amounts. Use the
                # street betting context to distinguish check from call.
                if self._street_has_bet:
                    hs["calls"] += 1
                    hs["passive_actions"] += 1
                else:
                    hs["checks"] += 1
                    hs["passive_actions"] += 1

        if kind == "raise" and street == "preflop":
            hs["aggressive_actions"] += 1
        observed_action = "shove" if all_in_raise else kind
        if street != "preflop" and kind == "call":
            # In this API call() also represents a check when nothing is owed.
            observed_action = "call" if self._street_has_bet else "check"
        self._opponent_last_action[pid] = observed_action
        self._opponent_last_action_by_street.setdefault(pid, {})[street] = observed_action

    def _finalise_hand(self, game_state):
        if self._hand_finalised:
            return
        self._hand_finalised = True

        # No opponent can be profiled if the model was never given a state;
        # in normal arena play _register_players has already populated this.
        for pid, hs in self._hand_stats.items():
            profile = self._profile(pid)
            profile["hands"] += 1
            profile["vpip_hands"] += int(bool(hs["vpip"]))
            profile["pfr_hands"] += int(bool(hs["pfr"]))
            profile["rfi_opportunities"] += int(bool(hs["rfi_opportunity"]))
            profile["rfi_hands"] += int(bool(hs["rfi"]))
            profile["limp_hands"] += int(bool(hs["limp"]))

            for key in (
                "three_bet_opportunities", "three_bets",
                "four_bet_opportunities", "four_bets",
                "fold_to_three_bet_opportunities", "fold_to_three_bet",
                "cbet_opportunities", "cbets",
                "fold_to_cbet_opportunities", "fold_to_cbet",
                "flop_bet_opportunities", "flop_bets",
                "turn_bet_opportunities", "turn_bets",
                "river_bet_opportunities", "river_bets",
                "aggressive_actions", "raise_actions", "all_in_raises",
                "passive_actions", "calls", "checks", "folds",
            ):
                profile[key] += hs[key]

            position = "button" if pid == game_state.button else "big_blind"
            pos_stats = profile["position"].setdefault(position, _new_position_stats())
            pos_stats["hands"] += 1
            pos_stats["vpip_hands"] += int(bool(hs["vpip"]))
            pos_stats["pfr_hands"] += int(bool(hs["pfr"]))
            pos_stats["rfi_hands"] += int(bool(hs["rfi"]))
            pos_stats["limp_hands"] += int(bool(hs["limp"]))

            profile["bet_size_history"].extend(hs["raise_sizes"])
            profile["bet_size_history"] = profile["bet_size_history"][-60:]

            revealed = hs["revealed_cards"]
            if len(revealed) >= 2:
                profile["showdowns"] += 1
                if hs["all_in_raises"]:
                    profile["all_in_showdowns"] += 1
                won = hs["won"] > 0
                profile["showdown_wins"] += int(won)
                try:
                    shown_cards = [Card.parse(value) for value in revealed[:2]]
                    board = list(game_state.board)
                    if len(board) == 5:
                        category = evaluate(shown_cards + board)[0]
                    else:
                        category = None
                except Exception:
                    category = None
                # Store public showdown information only; cap memory growth.
                profile["showdown_history"].append({
                    "cards": list(revealed[:2]),
                    "board": [str(card) for card in game_state.board],
                    "category": category,
                    "won": bool(won),
                })
                profile["showdown_history"] = profile["showdown_history"][-_MAX_SHOWDOWNS:]
                if category is not None and category <= 1:
                    profile["weak_showdowns"] += 1
                    if hs["all_in_raises"]:
                        profile["weak_all_in_showdowns"] += 1

            recent_entry = {
                "vpip": bool(hs["vpip"]),
                "pfr": bool(hs["pfr"]),
                "rfi": bool(hs["rfi"]),
                "limp": bool(hs["limp"]),
                "raises": hs["aggressive_actions"],
                "all_in_raises": hs["all_in_raises"],
                "calls": hs["calls"],
                "checks": hs["checks"],
                "folds": hs["folds"],
                "three_bet_opportunities": hs["three_bet_opportunities"],
                "three_bets": hs["three_bets"],
                "cbet_opportunities": hs["cbet_opportunities"],
                "cbets": hs["cbets"],
                "fold_to_cbet_opportunities": hs["fold_to_cbet_opportunities"],
                "fold_to_cbet": hs["fold_to_cbet"],
                "raise_sizes": list(hs["raise_sizes"]),
                "showdown": len(revealed) >= 2,
                "won": hs["won"] > 0,
            }
            profile["recent"].append(recent_entry)
            profile["recent"] = profile["recent"][-_MAX_RECENT_HANDS:]

    # ------------------------------------------------------------------
    # Opponent statistics and range estimates
    # ------------------------------------------------------------------

    def _opponent_position(self, player_id: int, game_state) -> str:
        return "button" if player_id == game_state.button else "big_blind"

    def _opponent_rate(self, player_id: int, numerator: str,
                       denominator: str, prior: float = 0.5) -> float:
        profile = self._profile(player_id)
        return _rate(profile.get(numerator, 0), profile.get(denominator, 0), prior)

    def _opponent_is_frequent_shover(self, player_id: int) -> bool:
        """Require repeated all-in raises before treating shoves as a pattern."""
        profile = self._profile(player_id)
        raises = profile.get("raise_actions", 0)
        shoves = profile.get("all_in_raises", 0)
        return raises >= 3 and shoves / raises >= 0.75

    def _opponent_fold_to_cbet(self, player_id: int) -> float:
        profile = self._profile(player_id)
        # Ignore sparse historical reads during warm-up. Equity calculations
        # still run; this gates only the profile-based exploit estimate.
        if profile.get("hands", 0) < _PROFILE_WARMUP_HANDS:
            return 0.45
        lifetime = _rate(
            profile.get("fold_to_cbet", 0),
            profile.get("fold_to_cbet_opportunities", 0),
            prior=0.45,
        )
        recent = profile.get("recent", [])[-25:]
        recent_opportunities = sum(x.get("fold_to_cbet_opportunities", 0) for x in recent)
        recent_folds = sum(x.get("fold_to_cbet", 0) for x in recent)
        if recent_opportunities >= 4:
            recent_rate = _rate(recent_folds, recent_opportunities, prior=lifetime)
            return 0.4 * lifetime + 0.6 * recent_rate
        return lifetime

    def _candidate_range_weight(self, player_id: int, hole: list[Card],
                                board: list[Card]) -> float:
        """Importance weight for a candidate opponent hand.

        This is deliberately conservative: observed actions narrow the prior,
        but never collapse it to a single guessed hand.
        """
        profile = self._profile(player_id)
        pre_strength = _preflop_strength(hole)
        pre_action = self._opponent_preflop_action.get(player_id, "unknown")
        last_action = self._opponent_last_action_by_street.get(player_id, {}).get(
            self._hand_street, "unknown"
        )

        hands = profile.get("hands", 0)
        profile_mature = hands >= _PROFILE_WARMUP_HANDS
        # Use population priors until there are 100 completed hands against
        # this team. Actions observed in the current hand still condition range.
        vpip = 0.65
        pfr = 0.30
        if profile_mature:
            vpip = _rate(profile.get("vpip_hands", 0), hands, prior=0.65)
            pfr = _rate(profile.get("pfr_hands", 0), hands, prior=0.30)
            recent = profile.get("recent", [])[-25:]
            if len(recent) >= 5:
                recent_vpip = _rate(sum(bool(x.get("vpip")) for x in recent),
                                    len(recent), prior=vpip)
                recent_pfr = _rate(sum(bool(x.get("pfr")) for x in recent),
                                   len(recent), prior=pfr)
                vpip = 0.55 * vpip + 0.45 * recent_vpip
                pfr = 0.55 * pfr + 0.45 * recent_pfr

        if profile_mature and self._hand_button is not None:
            pos_name = "button" if player_id == self._hand_button else "big_blind"
            pos = profile.get("position", {}).get(pos_name, {})
            pos_hands = pos.get("hands", 0)
            if pos_hands >= 5:
                pos_vpip = _rate(pos.get("vpip_hands", 0), pos_hands, prior=vpip)
                pos_pfr = _rate(pos.get("pfr_hands", 0), pos_hands, prior=pfr)
                vpip = 0.65 * vpip + 0.35 * pos_vpip
                pfr = 0.65 * pfr + 0.35 * pos_pfr

        weight = 1.0
        if pre_action in ("raise", "shove"):
            exponent = 1.35 if pfr > 0.30 else 1.8
            action_weight = 0.22 + 1.65 * (pre_strength ** exponent)
            if (pre_action == "shove"
                    and self._opponent_is_frequent_shover(player_id)):
                # Repeated jams are less informative about hand strength than
                # a one-off shove. Blend the strong-range weight toward neutral.
                blend = 0.40
                shown = profile.get("all_in_showdowns", 0)
                if shown >= 3:
                    weak_rate = _rate(
                        profile.get("weak_all_in_showdowns", 0), shown,
                        prior=0.20,
                    )
                    blend = _clamp(0.35 + 0.40 * weak_rate, 0.35, 0.65)
                action_weight = (1.0 - blend) * action_weight + blend
            weight *= action_weight
        elif pre_action == "call":
            target = _clamp(0.30 + 0.28 * vpip, 0.35, 0.62)
            weight *= 0.35 + 1.15 * (1.0 - abs(pre_strength - target))
        elif pre_action == "fold":
            weight *= 0.8

        if len(board) >= 3:
            try:
                made_strength = _rank_strength(evaluate(hole + board))
            except Exception:
                made_strength = 0.35

            if last_action in ("raise", "shove"):
                weak_showdown_rate = (
                    _rate(
                        profile.get("weak_showdowns", 0),
                        profile.get("showdowns", 0),
                        prior=0.18,
                    )
                    if profile_mature else 0.18
                )
                # Weak revealed hands can be evidence of wider aggression,
                # but are not automatically classified as bluffs.
                weak_hand_floor = _clamp(0.12 + 0.18 * weak_showdown_rate, 0.12, 0.24)
                action_weight = (
                    weak_hand_floor
                    + (1.0 - weak_hand_floor) * (made_strength ** 1.35)
                )
                if (last_action == "shove"
                        and self._opponent_is_frequent_shover(player_id)):
                    blend = 0.40
                    shown = profile.get("all_in_showdowns", 0)
                    if shown >= 3:
                        weak_rate = _rate(
                            profile.get("weak_all_in_showdowns", 0), shown,
                            prior=0.20,
                        )
                        blend = _clamp(0.35 + 0.40 * weak_rate, 0.35, 0.65)
                    action_weight = (1.0 - blend) * action_weight + blend
                weight *= action_weight
            elif last_action == "call":
                weight *= 0.42 + 1.05 * made_strength
            elif last_action == "check":
                # Checks are weaker on average but retain traps and slow plays.
                weight *= 1.10 - 0.30 * made_strength

        return _clamp(weight, 0.08, 4.0)

    def _estimate_equity(self, game_state) -> float:
        """Estimate our share of the pot against all currently live opponents."""
        board = list(game_state.board)
        opponents = [p for p in game_state.players_in_hand if p != self.player_id]
        if not opponents:
            return 1.0

        known = set(self.hand + board)
        unseen = [card for card in full_deck() if card not in known]
        missing = 5 - len(board)
        needed = 2 * len(opponents) + missing
        if needed > len(unseen):
            return 0.0

        # On the river in heads-up, enumerate every opposing combination.
        if missing == 0 and len(opponents) <= _RIVER_EXACT_MAX_OPPONENTS:
            mine = evaluate(self.hand + board)
            weighted_share = 0.0
            total_weight = 0.0
            for hole_tuple in combinations(unseen, 2):
                hole = list(hole_tuple)
                theirs = evaluate(hole + board)
                weight = self._candidate_range_weight(opponents[0], hole, board)
                share = 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
                weighted_share += weight * share
                total_weight += weight
            return weighted_share / total_weight if total_weight else 0.5

        if missing == 3:
            samples = _SIMULATIONS[3]
        elif missing == 2:
            samples = _SIMULATIONS[4]
        else:
            samples = _SIMULATIONS[5]
        # Multiway equity is more expensive; reduce samples but preserve a
        # useful estimate for the repository's multi-player legality tests.
        if len(opponents) > 1:
            samples = max(100, int(samples / (1 + 0.9 * (len(opponents) - 1))))

        weighted_share = 0.0
        total_weight = 0.0
        for _ in range(samples):
            drawn = self.rng.sample(unseen, needed)
            holes = [drawn[2 * i:2 * i + 2] for i in range(len(opponents))]
            runout = board + drawn[2 * len(opponents):]
            mine = evaluate(self.hand + runout)
            opponent_ranks = [evaluate(hole + runout) for hole in holes]
            best = max([mine] + opponent_ranks)
            if mine > best:
                share = 1.0
            elif mine < best:
                share = 0.0
            else:
                tied = 1 + sum(rank == mine for rank in opponent_ranks)
                share = 1.0 / tied

            weight = 1.0
            for pid, hole in zip(opponents, holes):
                # Condition opponent ranges only on cards visible when the
                # opponent made the observed action, never on future runout.
                weight *= self._candidate_range_weight(pid, hole, board)
            weighted_share += weight * share
            total_weight += weight

        return weighted_share / total_weight if total_weight else 0.5

    # ------------------------------------------------------------------
    # Betting helpers and strategy
    # ------------------------------------------------------------------

    def _safe_raise(self, game_state, target: int) -> PlayerAction:
        if not game_state.can_raise or game_state.raise_range is None:
            return PlayerAction.call(self.player_id)
        low, high = game_state.raise_range
        total = int(_clamp(int(target), low, high))
        return PlayerAction.raise_to(self.player_id, total)

    def _raise_increment(self, game_state, pot_fraction: float) -> PlayerAction:
        """Raise by a pot-relative increment, expressed as a street total."""
        increment = max(game_state.last_raise, int(round(game_state.pot * pot_fraction)))
        target = game_state.current_bet + increment
        return self._safe_raise(game_state, target)

    def _decide_preflop(self, game_state) -> PlayerAction:
        score = _preflop_strength(self.hand)
        me = self.player_id
        is_button = me == game_state.button
        raises = self._preflop_raise_count
        facing_bet = game_state.call_amount > 0
        opponents = len(game_state.players_in_hand) - 1

        # Competition is heads-up. Keep a conservative legal fallback for
        # wider tables used by the repository's general legality test.
        if opponents > 1:
            if not facing_bet:
                if score > 0.70 and game_state.can_raise:
                    return self._raise_increment(game_state, 0.65)
                return PlayerAction.call(me)
            threshold = 0.38 + 0.12 * (opponents - 1) + 0.25 * game_state.pot_odds
            if score > threshold and game_state.can_raise and score > 0.78:
                return self._raise_increment(game_state, 0.85)
            if score > threshold:
                return PlayerAction.call(me)
            return PlayerAction.fold(me)

        if raises == 0:
            if is_button:
                # Heads-up button is the small blind and acts first preflop.
                # Use a wide opening range with a mixture of raises and limps.
                if score >= 0.285 and game_state.can_raise:
                    target = int(round(2.5 * game_state.big_blind))
                    return self._safe_raise(game_state, target)
                if score >= 0.22:
                    return PlayerAction.call(me)
                return PlayerAction.fold(me)

            # Big blind after a button limp: check most hands, raise a strong
            # value range to punish limps while retaining a wide check range.
            if not facing_bet:
                if score >= 0.73 and game_state.can_raise:
                    target = int(round(3.5 * game_state.big_blind))
                    return self._safe_raise(game_state, target)
                return PlayerAction.call(me)

        # Facing a preflop raise. Defend more widely against smaller raises;
        # tighten against 3-bets and large effective-stack commitments.
        if raises == 1:
            if score >= 0.64 and game_state.can_raise:
                return self._raise_increment(game_state, 1.05)
            call_threshold = 0.29 + 0.13 * game_state.pot_odds
            opponent_shoved = any(
                action == "shove"
                for pid, action in self._opponent_preflop_action.items()
                if pid != me
            )
            frequent_shover = any(
                self._opponent_is_frequent_shover(pid)
                for pid in self._opponent_id_to_name
            )
            if opponent_shoved and frequent_shover:
                # Widen calls modestly against a repeatedly observed jammer;
                # pot odds still matter and this is not an automatic call.
                call_threshold = max(0.18, call_threshold - 0.06)
            if score >= call_threshold:
                return PlayerAction.call(me)
            return PlayerAction.fold(me)

        # Facing a 3-bet or later raise: continue with a stronger range.
        if score >= 0.83 and game_state.can_raise:
            return self._raise_increment(game_state, 1.10)
        call_threshold = 0.53 + 0.18 * game_state.pot_odds
        if score >= call_threshold:
            return PlayerAction.call(me)
        return PlayerAction.fold(me)

    def _decide_postflop(self, game_state) -> PlayerAction:
        me = self.player_id
        if not game_state.board:
            return PlayerAction.call(me)

        equity = self._estimate_equity(game_state)
        call_amount = game_state.call_amount
        pot_odds = game_state.pot_odds
        active_opponents = [p for p in game_state.players_in_hand if p != me]
        opponent_id = active_opponents[0] if len(active_opponents) == 1 else None

        # Against an actual bet, compare equity against the price of calling.
        if call_amount > 0:
            opponent_shoved = (
                opponent_id is not None
                and self._opponent_last_action_by_street.get(opponent_id, {}).get(
                    game_state.street
                ) == "shove"
            )
            frequent_shover = (
                opponent_id is not None
                and self._opponent_is_frequent_shover(opponent_id)
            )
            margin = 0.025 if len(game_state.board) < 5 else 0.012
            if opponent_shoved and frequent_shover:
                # Don't require extra equity above the price against a known
                # frequent jammer, but never call below estimated pot odds.
                margin = 0.0
            if equity > pot_odds + margin:
                # Raise strong value hands; do not turn routine bluff-catchers
                # into raises without a specific exploitative reason.
                if game_state.can_raise and equity >= 0.79:
                    return self._raise_increment(game_state, 0.85)
                return PlayerAction.call(me)
            return PlayerAction.fold(me)

        # We can check for free. Value bet strong equity; use smaller bets for
        # continuation bets and selectively exploit opponents who overfold.
        if not game_state.can_raise:
            return PlayerAction.call(me)

        board_count = len(game_state.board)
        fold_to_cbet = (
            self._opponent_fold_to_cbet(opponent_id)
            if opponent_id is not None else 0.45
        )
        opponent_profile = self._profile(opponent_id) if opponent_id is not None else {}
        cbet_sample = opponent_profile.get("fold_to_cbet_opportunities", 0)
        mature_profile = opponent_profile.get("hands", 0) >= _PROFILE_WARMUP_HANDS
        is_preflop_aggressor = self._preflop_aggressor == me
        is_button = me == game_state.button

        if equity >= 0.78:
            return self._raise_increment(game_state, 0.72)
        if equity >= 0.62:
            return self._raise_increment(game_state, 0.55)

        # A small c-bet is useful when the preflop aggressor has initiative.
        if board_count == 3 and is_preflop_aggressor and equity >= 0.39:
            return self._raise_increment(game_state, 0.36)

        # Exploit a demonstrated tendency to fold to flop bets, but require a
        # reasonable sample and avoid bluffing every weak holding.
        if (board_count == 3 and opponent_id is not None and mature_profile
                and cbet_sample >= _MIN_FOLD_TO_CBET_OPPORTUNITIES
                and fold_to_cbet >= 0.62 and equity >= 0.24):
            return self._raise_increment(game_state, 0.30)

        # On later streets, thin value bets need stronger support.
        if board_count >= 4 and equity >= 0.55 and is_button:
            return self._raise_increment(game_state, 0.40)

        return PlayerAction.call(me)


__all__ = ["Model"]
