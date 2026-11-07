"""Self-contained adaptive poker policy. Uses only cards and public observations."""
import bisect
import itertools
import math
import random
import time

from card_utils import full_deck
from player_action import PlayerAction

# (random-opponent equity, combination-weighted percentile); generated offline.
# 4096 trials/class; 46 near-threshold equities refined to 65536 trials.
# Percentiles are intentionally frozen to isolate the equity refinement.
PREFLOP = {
    "22": (0.5040130615234375, 0.5090498),
    "32o": (0.3173828, 0.0045249),
    "32s": (0.3707275, 0.0950226),
    "33": (0.5489502, 0.688537),
    "42o": (0.3288574, 0.0135747),
    "42s": (0.3657227, 0.092006),
    "43o": (0.3485107, 0.0497738),
    "43s": (0.3771973, 0.1101056),
    "44": (0.571167, 0.7624434),
    "52o": (0.3425293, 0.0226244),
    "52s": (0.383667, 0.1402715),
    "53o": (0.3543701, 0.0588235),
    "53s": (0.3919678, 0.1553544),
    "54o": (0.3848877, 0.1463047),
    "54s": (0.4245605, 0.2669683),
    "55": (0.5993652, 0.8544495),
    "62o": (0.3426514, 0.0316742),
    "62s": (0.3737793, 0.0980392),
    "63o": (0.3560791, 0.0769231),
    "63s": (0.3905029, 0.1523379),
    "64o": (0.3830566, 0.1342383),
    "64s": (0.4146729, 0.224736),
    "65o": (0.4033203, 0.1825038),
    "65s": (0.4196777, 0.239819),
    "66": (0.6300049, 0.9193062),
    "72o": (0.3461914, 0.040724),
    "72s": (0.3974609, 0.1674208),
    "73o": (0.3552246, 0.0678733),
    "73s": (0.4095459, 0.2096531),
    "74o": (0.3796387, 0.1161388),
    "74s": (0.416626, 0.2368024),
    "75o": (0.4083252, 0.2036199),
    "75s": (0.440918, 0.3031674),
    "76o": (0.4155273, 0.2307692),
    "76s": (0.4554443, 0.3544495),
    "77": (0.6462402, 0.9449472),
    "82o": (0.3649902, 0.0859729),
    "82s": (0.4130859, 0.2217195),
    "83o": (0.3747559, 0.1040724),
    "83s": (0.4056396, 0.1975867),
    "84o": (0.3989258, 0.173454),
    "84s": (0.4306641, 0.2790347),
    "85o": (0.4207764, 0.2458522),
    "85s": (0.4493408, 0.3242836),
    "86o": (0.4311523, 0.2850679),
    "86s": (0.4544678, 0.3514329),
    "87o": (0.44927978515625, 0.4147813),
    "87s": (0.47743988037109375, 0.453997),
    "88": (0.6815186, 0.9705882),
    "92o": (0.381958, 0.1251885),
    "92s": (0.4216309, 0.2609351),
    "93o": (0.395752, 0.1613876),
    "93s": (0.4241943, 0.2639517),
    "94o": (0.4118652, 0.2156863),
    "94s": (0.4385986, 0.3001508),
    "95o": (0.4335938, 0.2941176),
    "95s": (0.4589844, 0.3755656),
    "96o": (0.4449463, 0.3182504),
    "96s": (0.4715423583984375, 0.387632),
    "97o": (0.465301513671875, 0.3936652),
    "97s": (0.49285125732421875, 0.505279),
    "98o": (0.48087310791015625, 0.4600302),
    "98s": (0.5091094970703125, 0.5520362),
    "99": (0.7290039, 0.9751131),
    "A2o": (0.5465088, 0.6726998),
    "A2s": (0.55896, 0.7315234),
    "A3o": (0.5576858520507812, 0.6425339),
    "A3s": (0.5767822, 0.7873303),
    "A4o": (0.564209, 0.7466063),
    "A4s": (0.5911865, 0.8386124),
    "A5o": (0.5773926, 0.7933635),
    "A5s": (0.5922852, 0.841629),
    "A6o": (0.5748291, 0.7782805),
    "A6s": (0.5838623, 0.8144796),
    "A7o": (0.590332, 0.8325792),
    "A7s": (0.6130371, 0.8974359),
    "A8o": (0.5952148, 0.8476621),
    "A8s": (0.6346436, 0.9230769),
    "A9o": (0.6257324, 0.9034691),
    "A9s": (0.6286621, 0.9155354),
    "AA": (0.862915, 0.9977376),
    "AJo": (0.6385498, 0.9291101),
    "AJs": (0.6495361, 0.9517345),
    "AKo": (0.6495361, 0.9577677),
    "AKs": (0.6765137, 0.9668175),
    "AQo": (0.6392822, 0.9381599),
    "AQs": (0.6634521, 0.9638009),
    "ATo": (0.6115723, 0.8914027),
    "ATs": (0.6466064, 0.9487179),
    "J2o": (0.4506836, 0.3303167),
    "J2s": (0.47283935546875, 0.3996983),
    "J3o": (0.4517822, 0.3393665),
    "J3s": (0.48487091064453125, 0.4871795),
    "J4o": (0.4630126953125, 0.4238311),
    "J4s": (0.49080657958984375, 0.4841629),
    "J5o": (0.47086334228515625, 0.3815988),
    "J5s": (0.49616241455078125, 0.4751131),
    "J6o": (0.4567871, 0.3604827),
    "J6s": (0.5062637329101562, 0.5490196),
    "J7o": (0.49629974365234375, 0.4932127),
    "J7s": (0.5247879028320312, 0.5822021),
    "J8o": (0.514190673828125, 0.5671192),
    "J8s": (0.5365142822265625, 0.6063348),
    "J9o": (0.5354156494140625, 0.6334842),
    "J9s": (0.5599365, 0.73454),
    "JJ": (0.7652588, 0.9841629),
    "JTo": (0.555542, 0.7043741),
    "JTs": (0.5827637, 0.811463),
    "K2o": (0.50714111328125, 0.5248869),
    "K2s": (0.5323715209960938, 0.627451),
    "K3o": (0.5123367309570312, 0.5761689),
    "K3s": (0.5432129, 0.6636501),
    "K4o": (0.5214614868164062, 0.6214178),
    "K4s": (0.5609131, 0.7375566),
    "K5o": (0.532318115234375, 0.612368),
    "K5s": (0.557251, 0.719457),
    "K6o": (0.546875, 0.6817496),
    "K6s": (0.5616455, 0.7405732),
    "K7o": (0.5581055, 0.7254902),
    "K7s": (0.5748291, 0.7843137),
    "K8o": (0.5554199, 0.6953243),
    "K8s": (0.5853271, 0.826546),
    "K9o": (0.5805664, 0.8024133),
    "K9s": (0.6002197, 0.8582202),
    "KJo": (0.6038818, 0.86727),
    "KJs": (0.6275635, 0.9095023),
    "KK": (0.8269043, 0.9932127),
    "KQo": (0.6079102, 0.8823529),
    "KQs": (0.6280518, 0.9125189),
    "KTo": (0.5849609, 0.8205128),
    "KTs": (0.6063232, 0.8733032),
    "Q2o": (0.4706268310546875, 0.438914),
    "Q2s": (0.501129150390625, 0.4992459),
    "Q3o": (0.479736328125, 0.4057315),
    "Q3s": (0.5098495483398438, 0.5022624),
    "Q4o": (0.48900604248046875, 0.4479638),
    "Q4s": (0.5161972045898438, 0.5882353),
    "Q5o": (0.4990997314453125, 0.5429864),
    "Q5s": (0.5277786254882812, 0.5852187),
    "Q6o": (0.5086822509765625, 0.5339367),
    "Q6s": (0.5348358154296875, 0.5912519),
    "Q7o": (0.5199508666992188, 0.6003017),
    "Q7s": (0.5410156, 0.6515837),
    "Q8o": (0.5411377, 0.6576169),
    "Q8s": (0.5444336, 0.6666667),
    "Q9o": (0.5568848, 0.7134238),
    "Q9s": (0.5805664, 0.8084465),
    "QJo": (0.5662842, 0.7556561),
    "QJs": (0.6069336, 0.8763198),
    "QQ": (0.8037109, 0.9886878),
    "QTo": (0.5717773, 0.7692308),
    "QTs": (0.6015625, 0.8612368),
    "T2o": (0.4052734, 0.1915535),
    "T2s": (0.4522705, 0.3453997),
    "T3o": (0.4211426, 0.254902),
    "T3s": (0.453125, 0.3484163),
    "T4o": (0.4265137, 0.2730015),
    "T4s": (0.465850830078125, 0.4298643),
    "T5o": (0.4417725, 0.3092006),
    "T5s": (0.47310638427734375, 0.4328808),
    "T6o": (0.4573975, 0.3695324),
    "T6s": (0.49053955078125, 0.4781297),
    "T7o": (0.4770965576171875, 0.4690799),
    "T7s": (0.5052642822265625, 0.4811463),
    "T8o": (0.4987945556640625, 0.5158371),
    "T8s": (0.52362060546875, 0.5942685),
    "T9o": (0.5121078491210938, 0.5580694),
    "T9s": (0.54266357421875, 0.6485671),
    "TT": (0.7454834, 0.979638),
}
PARAMETERS = (
    (0.30, 0.78, 0.65, 0.06),
    (0.55, 0.88, 0.75, 0.015),
    (-0.20, 0.88, 0.30, 0.01),
    (0.12, 0.62, 0.80, 0.18),
    (0.65, 0.88, 0.65, 0.02),
    (0, 0, 0, 0),
)
PRIOR = (0.40, 0.12, 0.12, 0.12, 0.12, 0.12)


def _straight_high_mask(mask):
    # Ace can play below a deuce as well as above a king.
    mask |= (mask >> 13) & 2
    consecutive = mask & (mask << 1) & (mask << 2) & (mask << 3) & (mask << 4)
    return consecutive.bit_length() - 1 if consecutive else 0


def evaluate(cards):
    """Best-five-card ranking for a legal collection of five to seven Cards."""
    if not 5 <= len(cards) <= 7:
        raise ValueError("evaluate requires five to seven cards")
    counts = [0] * 15
    suits = {}
    mask = 0
    for card in cards:
        value = card.value
        bit = 1 << value
        counts[value] += 1
        mask |= bit
        suits[card.suit] = suits.get(card.suit, 0) | bit

    flush = 0
    for suit_mask in suits.values():
        if suit_mask.bit_count() >= 5:
            high = _straight_high_mask(suit_mask)
            if high:
                return (8, high)
            flush = suit_mask
            break

    ranks = [rank for rank in range(14, 1, -1) if counts[rank]]
    quads = [rank for rank in ranks if counts[rank] == 4]
    if quads:
        quad = quads[0]
        return (7, quad, next(rank for rank in ranks if rank != quad))

    trips = [rank for rank in ranks if counts[rank] >= 3]
    pairs = [rank for rank in ranks if counts[rank] >= 2]
    if trips:
        trip = trips[0]
        pair = next((rank for rank in pairs if rank != trip), 0)
        if pair:
            return (6, trip, pair)

    if flush:
        return (5,) + tuple(rank for rank in ranks if flush & (1 << rank))[:5]
    high = _straight_high_mask(mask)
    if high:
        return (4, high)
    if trips:
        trip = trips[0]
        kickers = [rank for rank in ranks if rank != trip][:2]
        return (3, trip, *kickers)
    if len(pairs) >= 2:
        first, second = pairs[:2]
        kicker = next(rank for rank in ranks if rank != first and rank != second)
        return (2, first, second, kicker)
    if pairs:
        pair = pairs[0]
        kickers = [rank for rank in ranks if rank != pair][:3]
        return (1, pair, *kickers)
    return (0, *ranks[:5])


def _class_key(hand):
    a, b = sorted(hand, key=lambda c: c.value, reverse=True)
    return (
        a.rank + b.rank + ("" if a.rank == b.rank else "s" if a.suit == b.suit else "o")
    )


def _sigmoid(x):
    return 1 / (1 + math.exp(max(-40, min(40, -x))))


def _draw_feature(hand, board):
    if len(board) not in (3, 4):
        return 0.0
    cards = list(hand) + list(board)
    score = 0.0
    for suit in "CDHS":
        if (
            sum(c.suit == suit for c in cards) == 4
            and sum(c.suit == suit for c in board) < 4
            and any(c.suit == suit for c in hand)
        ):
            score = 0.55
    values = {c.value for c in cards}
    public = {c.value for c in board}
    if 14 in values:
        values.add(1)
    if 14 in public:
        public.add(1)
    outs = set()
    for low in range(1, 11):
        seq = set(range(low, low + 5))
        missing = seq - values
        if len(missing) == 1 and (seq & values) - public:
            outs.update(missing)
    return max(score, 0.50 if len(outs) >= 2 else 0.35 if outs else 0.0)


def _response(
    profile,
    made,
    draw,
    odds,
    facing,
    can_raise,
    pressure,
    shallow=False,
    preflop_size=0.0,
):
    """Soft behavior likelihood; not an equity estimate."""
    strength = max(made, draw)
    if profile == 5:
        fold = 0.02 if facing else 0.0
        aggressive = 0.98 if can_raise else 0.0
        sizes = (0.03, 0.07, 0.90)
    else:
        cutoff, threshold, value, bluff = PARAMETERS[profile]
        cutoff += 0.20 * odds + 0.08 * pressure
        if preflop_size:
            sensitivity = 0.25 if profile == 2 else 0.5 if profile == 3 else 1.0
            cutoff += sensitivity * 0.12 * math.log2(max(1.0, preflop_size / 2))
            cutoff = min(0.95, cutoff)
        temperature = max(0.01, 0.10 * (1 - cutoff) / 0.70) if preflop_size else 0.10
        fold = _sigmoid((cutoff - strength) / temperature) if facing else 0.0
        aggressive = (
            min(
                0.95,
                value * _sigmoid(10 * (strength - threshold))
                + bluff * (1 - made) * (1 + draw),
            )
            if can_raise
            else 0.0
        )
        sizes = (
            (0.30, 0.40, 0.30)
            if profile == 3
            else (0.70, 0.25, 0.05)
            if profile == 2
            else (0.55, 0.35, 0.10)
        )
        if shallow or strength >= 0.90:
            sizes = (sizes[0], sizes[1], sizes[2] * 3)
        denominator = sum(sizes)
        sizes = tuple(v / denominator for v in sizes)
    raising = (1 - fold) * aggressive
    q = [fold, (1 - fold) * (1 - aggressive)] + [raising * x for x in sizes]
    active = ([0] if facing else []) + [1] + ([2, 3, 4] if can_raise else [])
    for i in active:
        q[i] = 0.99 * q[i] + 0.01 / len(active)
    return q


def _heads_up_value(equity, hero_total, other_total, invested, realization=1.0):
    """Prior bets are sunk; uncontested excess is a certain refund."""
    return (
        realization * equity * 2 * min(hero_total, other_total)
        + max(0, hero_total - other_total)
        - invested
    )


def _pot_share(contributions, folded, ranks, hero):
    payout = 0.0
    previous = 0
    live = [p for p in range(len(contributions)) if not folded[p]]
    if live == [hero]:
        return float(sum(contributions))
    for level in sorted(set(contributions)):
        amount = sum(max(0, min(c, level) - previous) for c in contributions)
        previous = level
        eligible = [p for p in live if contributions[p] >= level]
        if not amount or hero not in eligible:
            continue
        best = max(ranks[p] for p in eligible)
        winners = [p for p in eligible if ranks[p] == best]
        if hero in winners:
            payout += amount / len(winners)
    return payout


class Model:
    def __init__(
        self,
        seed=72819,
        samples=128,
        adaptation=0.10,
        aggression=0,
        realization=0.92,
        integrate_shover=True,
    ):
        self.rng = random.Random(seed)
        self.samples = int(samples)
        self.adaptation = float(adaptation)
        self.aggression = int(aggression)
        self.realization = float(realization)
        self.integrate_shover = bool(integrate_shover)
        self.opponents = {}
        self.deck = full_deck()
        self.hand = []
        self.player_id = None
        self.hands_played = 0
        self.ledger_errors = 0
        self.budget_fallbacks = 0
        self.last_values = []
        self._ready = False

    def seed(self, hand, player_id):
        self.hand = list(hand)
        self.player_id = player_id
        self._ready = False
        self._feature_board = None
        self._shown = {}
        self._observations = 0
        self._done = False
        self.last_values = []

    def _start(self, state, awards=None):
        n = state.num_players
        if state.street != "preflop":
            return False
        awards = awards or [0] * n
        self._initial = [state.stacks[i] + state.bets[i] - awards[i] for i in range(n)]
        self._stacks = list(self._initial)
        self._bets = [0] * n
        self._contrib = [0] * n
        self._folded = [False] * n
        self._acted = [False] * n
        self._board = []
        self._button = state.button
        self._bb = state.big_blind
        self._last_raise = self._bb
        self._raises = 0
        sb = state.button if n == 2 else (state.button + 1) % n
        bb = (sb + 1) % n
        for p, amount in ((sb, state.small_blind), (bb, self._bb)):
            paid = min(amount, self._stacks[p])
            self._bets[p] += paid
            self._contrib[p] += paid
            self._stacks[p] -= paid
        self._names = tuple(state.teams)
        self._villain = 1 - self.player_id if n == 2 else None
        self._holes = []
        self._weights = []
        if n == 2:
            unseen = [c for c in self.deck if c not in self.hand]
            self._holes = list(itertools.combinations(unseen, 2))
            self._valid = [True] * len(self._holes)
            prior = self.opponents.get(self._names[self._villain], list(PRIOR))
            self._weights = [[p / len(self._holes)] * len(self._holes) for p in prior]
        self._ready = True
        return True

    def _features(self):
        key = tuple(self._board)
        if self._feature_board == key:
            return True
        if time.perf_counter() > self._deadline:
            return False
        if not self._board:
            self._features_by_hand = [
                (PREFLOP[_class_key(h)][1], 0.0) for h in self._holes
            ]
            self._ranks = []
        else:
            available = [c for c in self.deck if c not in self._board]
            ranks = {}
            for i, h in enumerate(itertools.combinations(available, 2)):
                if i % 64 == 0 and time.perf_counter() > self._deadline:
                    return False
                ranks[h] = evaluate(list(h) + self._board)
            ordered = sorted(ranks.values())
            percentile = {
                rank: (
                    bisect.bisect_left(ordered, rank)
                    + bisect.bisect_right(ordered, rank)
                )
                / (2 * len(ordered))
                for rank in set(ordered)
            }
            features = []
            own_ranks = []
            for i, h in enumerate(self._holes):
                if not self._valid[i]:
                    features.append((0.0, 0.0))
                    own_ranks.append(())
                else:
                    rank = ranks[h]
                    features.append((percentile[rank], _draw_feature(h, self._board)))
                    own_ranks.append(rank)
            self._features_by_hand = features
            self._ranks = own_ranks
        self._feature_board = key
        return True

    def _normalize(self):
        total = sum(sum(row) for row in self._weights)
        if total <= 0 or not math.isfinite(total):
            count = sum(self._valid)
            if count:
                self._weights = [
                    [p / count if valid else 0.0 for valid in self._valid]
                    for p in PRIOR
                ]
        else:
            for row in self._weights:
                for i in range(len(row)):
                    row[i] /= total

    def _observe(self, action):
        if not self._features():
            self.budget_fallbacks += 1
            return
        p = action.player_id
        other = 1 - p
        current = max(self._bets)
        cost = min(self._stacks[p], current - self._bets[p])
        pot = sum(self._contrib)
        contested = 2 * min(self._contrib[p] + cost, self._contrib[other])
        odds = cost / max(1, contested)
        can_raise = (
            not self._acted[p]
            and self._stacks[other] > 0
            and self._bets[p] + self._stacks[p] > current
        )
        if action.kind == "fold":
            category = 0
        elif action.kind == "call":
            category = 1
        elif action.amount == self._bets[p] + self._stacks[p]:
            category = 4
        else:
            category = 2 if (action.amount - current) / max(1, pot + cost) <= 0.5 else 3
        shallow = min(self._stacks[p] - cost, self._stacks[other]) <= pot + cost
        buckets = [category]
        if category >= 2:
            buckets = []
            maximum = self._bets[p] + self._stacks[p]
            minimum = min(maximum, current + self._last_raise)
            for bucket, fraction in enumerate((1 / 3, 1.0, None)):
                desired = (
                    maximum
                    if fraction is None
                    else int(round(current + fraction * (pot + cost)))
                )
                target = min(maximum, max(minimum, desired))
                mapped = (
                    4
                    if target == maximum
                    else 2
                    if (target - current) / max(1, pot + cost) <= 0.5
                    else 3
                )
                if mapped == category:
                    buckets.append(bucket + 2)
        likelihoods = {}
        preflop_size = cost / self._bb if not self._board else 0.0
        pressure = self._raises >= 2
        for k, row in enumerate(self._weights):
            for i, w in enumerate(row):
                if not w:
                    continue
                made, draw = self._features_by_hand[i]
                key = (k, made, draw) if k != 5 else (5, 0.0, 0.0)
                likelihood = likelihoods.get(key)
                if likelihood is None:
                    q = _response(
                        k,
                        made,
                        draw,
                        odds,
                        cost > 0,
                        can_raise,
                        pressure,
                        shallow,
                        preflop_size,
                    )
                    likelihood = max(0.001, sum(q[b] for b in buckets))
                    likelihoods[key] = likelihood
                row[i] = w * likelihood
        self._observations += 1
        self._normalize()

    def _replay(self, actions, state):
        if not self._ready and not self._start(state):
            return False
        for action in actions:
            p = action.player_id
            if action.kind == "reveal_card":
                if p == -1:
                    self._board.append(action.card)
                    if self._weights:
                        for i, h in enumerate(self._holes):
                            if self._valid[i] and action.card in h:
                                self._valid[i] = False
                                for row in self._weights:
                                    row[i] = 0.0
                        self._normalize()
                    if len(self._board) in (3, 4, 5):
                        self._bets = [0] * len(self._bets)
                        self._acted = [False] * len(self._acted)
                        self._last_raise = self._bb
                        self._raises = 0
                else:
                    self._shown.setdefault(p, []).append(action.card)
            elif action.kind == "win":
                self._stacks[p] += action.amount
            else:
                if p == self._villain:
                    self._observe(action)
                current = max(self._bets)
                if action.kind == "fold":
                    self._folded[p] = True
                else:
                    paid = (
                        min(current - self._bets[p], self._stacks[p])
                        if action.kind == "call"
                        else action.amount - self._bets[p]
                    )
                    if action.kind == "raise":
                        increment = action.amount - current
                        if increment >= self._last_raise:
                            self._last_raise = increment
                            self._acted = [False] * len(self._acted)
                        self._raises += 1
                    self._bets[p] += paid
                    self._contrib[p] += paid
                    self._stacks[p] -= paid
                self._acted[p] = True
        correct = (
            tuple(self._stacks) == tuple(state.stacks)
            and tuple(self._bets) == tuple(state.bets)
            and tuple(self._folded) == tuple(state.folded)
            and tuple(self._board) == tuple(state.board)
            and sum(self._contrib) == state.pot
        )
        if not correct:
            self.ledger_errors += 1
        return correct

    def _samples(self, count):
        self._mc_samples = 0
        if not self._features():
            return []
        if len(self._board) == 5:
            mine = evaluate(self.hand + self._board)
            return [
                (
                    w,
                    k,
                    i,
                    float(mine > self._ranks[i]) + 0.5 * float(mine == self._ranks[i]),
                )
                for k, row in enumerate(self._weights)
                for i, w in enumerate(row)
                if w
            ]
        cumulative = []
        total = 0.0
        analytic = []
        rows = self._weights
        if self.integrate_shover and not self._board:
            # This profile's likelihood is independent of private cards.
            # Integrate its uniform range using the larger offline table.
            mass = sum(self._weights[5])
            index = next(i for i, valid in enumerate(self._valid) if valid)
            analytic = [(mass, 5, index, PREFLOP[_class_key(self.hand)][0])]
            rows = rows[:5]
        for row in rows:
            for w in row:
                total += w
                cumulative.append(total)
        if total <= 1e-12:
            return analytic
        unseen = [c for c in self.deck if c not in self.hand + self._board]
        result = []
        width = len(self._holes)
        for sample in range(count):
            if sample % 32 == 0 and time.perf_counter() > self._deadline:
                break
            flat = min(
                len(cumulative) - 1,
                bisect.bisect_right(cumulative, self.rng.random() * total),
            )
            k, i = divmod(flat, width)
            h = self._holes[i]
            available = [c for c in unseen if c not in h]
            board = self._board + self.rng.sample(available, 5 - len(self._board))
            mine = evaluate(self.hand + board)
            theirs = evaluate(list(h) + board)
            result.append(
                (1.0, k, i, float(mine > theirs) + 0.5 * float(mine == theirs))
            )
        self._mc_samples = len(result)
        if not result:
            return []
        return [(w * total / len(result), k, i, e) for w, k, i, e in result] + analytic

    def _realization(self, state, hero_left, other_left):
        if len(self._board) == 5 or min(hero_left, other_left) == 0:
            return 1.0
        base = self.realization
        if len(self._board) == 4:
            base = 1 - (1 - base) * 0.5
        if len(self._board) == 0:
            base -= 0.03
        return max(
            0.70, min(1.0, base - (0 if self.player_id == state.button else 0.04))
        )

    def _score_action(self, target, samples, state):
        """Restricted response tree; future hero choices use aggregate ranges."""
        me, opp = self.player_id, self._villain
        invested = target - state.bets[me]
        hero_left = state.stacks[me] - invested
        hero_total = self._contrib[me] + invested
        cost = min(max(0, target - state.bets[opp]), state.stacks[opp])
        other_total = self._contrib[opp] + cost
        other_left = state.stacks[opp] - cost
        raised = target > state.current_bet
        equity = sum(w * e for w, k, i, e in samples)
        realization = self._realization(state, hero_left, other_left)
        if not raised and (self._acted[opp] or other_left == 0):
            return _heads_up_value(
                equity, hero_total, other_total, invested, realization
            )
        can_raise = (
            hero_left > 0
            and other_left > 0
            and (not self._acted[opp] or target - state.current_bet >= state.last_raise)
        )
        pot_response = state.pot + invested
        odds = cost / max(1, 2 * min(hero_total, other_total))
        shallow = min(hero_left, other_left) <= pot_response + cost
        raise_size = max(state.last_raise, target - state.current_bet)
        maximum = state.bets[opp] + state.stacks[opp]
        minimum = min(maximum, target + raise_size)
        response_targets = []
        for fraction in (1 / 3, 1.0, None):
            desired = (
                maximum
                if fraction is None
                else int(round(target + fraction * (pot_response + cost)))
            )
            response_targets.append(min(maximum, max(minimum, desired)))
        raise_mass = {t: [0.0, 0.0] for t in response_targets}
        raise_leaves = []
        if can_raise:
            for rtarget in response_targets:
                extra = min(hero_left, max(0, rtarget - target))
                opp_extra = rtarget - state.bets[opp]
                r = self._realization(
                    state, hero_left - extra, state.stacks[opp] - opp_extra
                )
                raise_leaves.append(
                    (
                        hero_total + extra,
                        self._contrib[opp] + opp_extra,
                        invested + extra,
                        r,
                    )
                )
        # All response contexts and raise leaf chip amounts are public and
        # constant across the sampled private hands.
        root_pot = state.pot
        pressure = self._raises + int(raised) >= 2
        preflop_size = cost / self._bb if not self._board else 0.0
        responses = {}
        value = 0.0
        for w, k, i, outcome in samples:
            made, draw = self._features_by_hand[i]
            key = (k, made, draw) if k != 5 else (5, 0.0, 0.0)
            q = responses.get(key)
            if q is None:
                q = _response(
                    k,
                    made,
                    draw,
                    odds,
                    cost > 0,
                    can_raise,
                    pressure,
                    shallow,
                    preflop_size,
                )
                responses[key] = q
            value += w * q[0] * root_pot
            value += (
                w
                * q[1]
                * _heads_up_value(
                    outcome, hero_total, other_total, invested, realization
                )
            )
            if can_raise:
                for index, rtarget in enumerate(response_targets):
                    probability = w * q[index + 2]
                    htotal, ototal, future_invested, r = raise_leaves[index]
                    call_value = _heads_up_value(
                        outcome, htotal, ototal, future_invested, r
                    )
                    raise_mass[rtarget][0] += probability
                    raise_mass[rtarget][1] += probability * call_value
        for mass, call_value in raise_mass.values():
            value += max(-invested * mass, call_value)
        return value

    def _candidates(self, state):
        me = self.player_id
        targets = {state.bets[me] + state.call_amount}
        if state.can_raise:
            low, high = state.raise_range
            fractions = (1 / 3, 2 / 3, 1.0, 1.5) if self.aggression else (1 / 3, 2 / 3)
            for f in fractions:
                target = int(
                    round(
                        state.bets[me]
                        + state.call_amount
                        + f * (state.pot + state.call_amount)
                    )
                )
                targets.add(min(high, max(low, target)))
            if state.street == "preflop" and state.current_bet <= state.big_blind:
                for multiple in (2.0, 2.5):
                    targets.add(min(high, max(low, int(multiple * state.big_blind))))
            targets.add(high)
        return sorted(targets)

    def _multiway(self, state):
        me = self.player_id
        live = [p for p in state.players_in_hand if p != me]
        available = [c for c in self.deck if c not in self.hand + list(state.board)]
        contributions = list(self._contrib)
        contributions[me] += state.call_amount
        expected = equity = 0.0
        trials = 0
        for j in range(96):
            if j % 16 == 0 and time.perf_counter() > self._deadline:
                break
            missing = 5 - len(state.board)
            cards = self.rng.sample(available, missing + 2 * len(live))
            board = list(state.board) + cards[:missing]
            ranks = {me: evaluate(self.hand + board)}
            for k, p in enumerate(live):
                ranks[p] = evaluate(
                    cards[missing + 2 * k : missing + 2 * k + 2] + board
                )
            expected += _pot_share(contributions, state.folded, ranks, me)
            best = max(ranks.values())
            equity += (
                1 / sum(r == best for r in ranks.values()) if ranks[me] == best else 0
            )
            trials += 1
        if not trials:
            return (
                PlayerAction.call(me)
                if state.call_amount == 0
                else PlayerAction.fold(me)
            )
        if state.can_raise and equity / trials > 0.76:
            low, high = state.raise_range
            target = (
                state.bets[me]
                + state.call_amount
                + int(0.67 * (state.pot + state.call_amount))
            )
            return PlayerAction.raise_to(me, min(high, max(low, target)))
        return (
            PlayerAction.call(me)
            if state.call_amount == 0 or expected / trials > state.call_amount
            else PlayerAction.fold(me)
        )

    def run(self, actions, game_state):
        self._deadline = time.perf_counter() + 1.75
        me = self.player_id
        fallback = (
            PlayerAction.call(me)
            if game_state.call_amount == 0
            else PlayerAction.fold(me)
        )
        if not self._replay(actions, game_state):
            return fallback
        if game_state.num_players != 2:
            return self._multiway(game_state)
        samples = self._samples(self.samples)
        actual_samples = self._mc_samples
        if not samples:
            self.budget_fallbacks += 1
            return fallback
        values = [(0.0, None)] if game_state.call_amount else []
        for target in self._candidates(game_state):
            if time.perf_counter() > self._deadline:
                self.budget_fallbacks += 1
                break
            values.append((self._score_action(target, samples, game_state), target))
        if not values:
            return fallback
        values.sort(key=lambda v: (-v[0], -1 if v[1] is None else v[1]))
        # Refine costly decisions with fresh common samples; do not let a
        # noisy maximum over candidate shoves dominate a small sample.
        if len(self._board) < 5 and 0 < actual_samples < 512 and values:
            best_target = values[0][1]
            commitment = (
                game_state.call_amount
                if best_target is None
                else best_target - game_state.bets[me]
            )
            effective = min(
                game_state.stacks[me],
                game_state.stacks[self._villain] + game_state.call_amount,
            )
            equity = sum(w * e for w, k, i, e in samples)
            gap = values[0][0] - values[1][0] if len(values) > 1 else float("inf")
            costly = commitment >= 0.25 * max(1, effective) and 0.15 < equity < 0.85
            close = (
                game_state.pot >= 4 * game_state.big_blind
                and gap < 0.5 * game_state.big_blind
            )
            if (costly or close) and time.perf_counter() < self._deadline - 0.35:
                fresh = self._samples(512 - actual_samples)
                if fresh:
                    n, m = actual_samples, self._mc_samples
                    combined = [
                        (w * n / (n + m), k, i, e) for w, k, i, e in samples
                    ] + [(w * m / (n + m), k, i, e) for w, k, i, e in fresh]
                    refined = [(0.0, None)] if game_state.call_amount else []
                    complete = True
                    for candidate in self._candidates(game_state):
                        if time.perf_counter() > self._deadline:
                            complete = False
                            break
                        refined.append(
                            (
                                self._score_action(candidate, combined, game_state),
                                candidate,
                            )
                        )
                    if complete:
                        samples = combined
                        actual_samples = n + m
                        values = sorted(
                            refined, key=lambda v: (-v[0], -1 if v[1] is None else v[1])
                        )
        self.last_sample_count = actual_samples
        self.last_values = values
        target = values[0][1]
        if target is None:
            return PlayerAction.fold(me)
        if target <= game_state.current_bet or not game_state.can_raise:
            return PlayerAction.call(me)
        low, high = game_state.raise_range
        action = PlayerAction.raise_to(me, int(min(high, max(low, target))))
        game_state.validate_action(action)
        return action

    def end_game(self, actions, game_state):
        if self._done:
            return
        self._deadline = time.perf_counter() + 1.75
        if not self._ready:
            if game_state.street == "preflop":
                awards = [0] * game_state.num_players
                for action in actions:
                    if action.kind == "win":
                        awards[action.player_id] += action.amount
                self._start(game_state, awards)
            else:
                self.hands_played += 1
                self._done = True
                return
        if not self._replay(actions, game_state):
            self.hands_played += 1
            self._done = True
            return
        if self._weights and self._observations:
            shown = self._shown.get(self._villain, [])
            if len(shown) == 2:
                cards = set(shown)
                for i, h in enumerate(self._holes):
                    if set(h) != cards:
                        for row in self._weights:
                            row[i] = 0.0
                self._normalize()
            posterior = [sum(row) for row in self._weights]
            prior = [
                max(0.02, (1 - self.adaptation) * p + self.adaptation * base)
                for p, base in zip(posterior, PRIOR)
            ]
            total = sum(prior)
            self.opponents[self._names[self._villain]] = [p / total for p in prior]
        self.hands_played += 1
        self._done = True
