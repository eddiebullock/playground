"""Short-form keys from published sources.

AQ-10 binary key: Autism Research Centre AQ-10 (Adult), citing Allison,
Auyeung and Baron-Cohen (2012), JAACAP 51(2):202-212. One point for
definitely or slightly agree on items 1, 7, 8 and 10; one point for
definitely or slightly disagree on items 2, 3, 4, 5, 6 and 9. Cutoff 6.

EQ-10 and SQ-R-10 point tables: Greenberg et al. (2018) PNAS SI Appendix
Tables S17 and S18 (DOI 10.1073/pnas.1811032115). Positively poled items
score 2 / 1 / 0 / 0 and negatively poled items score 0 / 0 / 1 / 2 for
strongly agree, slightly agree, slightly disagree, strongly disagree.
The agree/disagree polarity matches the binary keys used in the earlier
pass. The published point values are not binary. Both are implemented.
Wakabayashi et al. (2006) is the 22-item EQ-Short and 25-item SQ-Short,
not these 10-item forms.

SPQ-10: Greenberg et al. (2018) PNAS SI Appendix Table S19, developed from
Tavassoli, Hoekstra and Baron-Cohen (2014). Every published short-form item
is positively poled: 3 / 2 / 1 / 0. There is no reverse key in that table.
Channel 4 column order against this text is checked from instrument wording,
not assumed.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

AQ10_ITEMS = (
    "I often notice small sounds when others do not",
    "I usually concentrate more on the whole picture, rather than the small details.",
    "I find it easy to do more than one thing at once.",
    "If there is an interruption, I can switch back to what I was doing very quickly.",
    "I find it easy to 'read between the lines' when someone is talking to me.",
    "I know how to tell if someone listening to me is getting bored.",
    "When I'm reading a story I find it difficult to work out the characters' intentions.",
    "I like to collect information about categories of things (e.g. types of car, types of bird, types of train, types of plant etc).",
    "I find it easy to work out what someone is thinking or feeling just by looking at their face.",
    "I find it difficult to work out people's intentions.",
)
# 1 = agree-scored, 0 = disagree-scored. Allison et al. 2012 / ARC AQ-10.
AQ10_AGREE = (1, 0, 0, 0, 0, 0, 1, 1, 0, 1)
AQ10_CUTOFF = 6

EQ10_ITEMS = (
    "I am good at predicting how someone will feel.",
    "Other people tell me I am good at understanding how they are feeling and what they are thinking.",
    "It is hard for me to see why some things upset people so much.",
    "I can easily work out what another person might want to talk about.",
    "I can't always see why someone should have felt offended by a remark.",
    "I can tune into how someone else feels rapidly and intuitively.",
    "Other people often say that I am insensitive, though I don't always see why.",
    "In a conversation, I tend to focus on my own thoughts rather than on what my listener might be thinking.",
    "Friends usually talk to me about their problems as they say that I am very understanding.",
    "I find it hard to know what to do in a social situation.",
)
EQ10_AGREE = (1, 1, 0, 1, 0, 1, 0, 0, 1, 0)

SQR10_ITEMS = (
    "When I learn about a new category I like to go into detail to understand the small differences between different members of that category.",
    "When I'm in a plane, I do not think about the aerodynamics.",
    "I am interested in knowing the path a river takes from its source to the sea.",
    "When travelling by train, I often wonder exactly how the rail networks are coordinated.",
    "When I hear the weather forecast, I am not very interested in the meteorological patterns.",
    "I enjoy looking through catalogues of products to see the details of each product and how it compares to others.",
    "When I look at a mountain, I think about how precisely it was formed.",
    "When I look at a piece of furniture, I do not notice the details of how it was constructed.",
    "When I learn a language, I become intrigued by its grammatical rules.",
    "When I listen to a piece of music, I always notice the way it's structured.",
)
SQR10_AGREE = (1, 0, 1, 1, 0, 1, 1, 0, 1, 1)

# Greenberg et al. 2018 SI Table S19. All positively poled. Order is the
# published short-form order, which still has to be matched to columns.
SPQ10_ITEMS = (
    "I would be able to distinguish different people by their smell.",
    "I would be able to taste the difference between two brands of salty potato chips/crisps.",
    "I can hear electricity humming in the walls.",
    "I would be able to notice a tiny change (for example, 1 degree) in the temperature of the weather.",
    "I would be able to taste the difference between apparently identical pieces of candy.",
    "I would be able to tell the weight difference between two different coin sizes on the palm of my hand, if my eyes were closed.",
    "I would be able to smell the smallest gas leak from anywhere in the house.",
    "I would be the first to hear if there was a fly in the room.",
    "If I look at a pile of blue sweaters in a shop that are meant to be identical, I would be able to see differences between them.",
    "I can see dust particles in the air in most environments.",
)
SPQ10_AGREE = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
SPQ10_STATUS = "published_polarity_all_positive_column_order_unconfirmed"

# Points for response codes 1..4 under the inferred coding
# (1 strongly agree ... 4 strongly disagree).
_BINARY_AGREE = {1: 1, 2: 1, 3: 0, 4: 0}
_BINARY_DISAGREE = {1: 0, 2: 0, 3: 1, 4: 1}
_GRADED_AGREE = {1: 2, 2: 1, 3: 0, 4: 0}
_GRADED_DISAGREE = {1: 0, 2: 0, 3: 1, 4: 2}
_SPQ_POSITIVE = {1: 3, 2: 2, 3: 1, 4: 0}


def _as_frame(responses: pd.DataFrame | np.ndarray, n_items: int) -> np.ndarray:
    arr = np.asarray(responses, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != n_items:
        raise ValueError(f"expected {n_items} items, got {arr.shape[1]}")
    return arr


def score_items(
    responses: pd.DataFrame | np.ndarray,
    agree: Sequence[int],
    *,
    scheme: str,
) -> pd.Series:
    """Score rows. `scheme` is 'binary', 'graded' (0/1/2) or 'spq' (0/1/2/3).

    Missing or out-of-range responses make the total missing for that row.
    """
    if len(agree) == 0:
        raise ValueError("empty scoring key")
    arr = _as_frame(responses, len(agree))
    if scheme == "binary":
        table = [_BINARY_AGREE if flag else _BINARY_DISAGREE for flag in agree]
    elif scheme == "graded":
        table = [_GRADED_AGREE if flag else _GRADED_DISAGREE for flag in agree]
    elif scheme == "spq":
        if any(flag != 1 for flag in agree):
            raise ValueError("published SPQ-10 has no reverse-keyed items")
        table = [_SPQ_POSITIVE for _ in agree]
    else:
        raise ValueError(f"unknown scoring scheme: {scheme}")
    out = np.zeros(arr.shape[0], dtype=float)
    for j, mapping in enumerate(table):
        col = arr[:, j]
        lookup = np.full(col.shape, np.nan)
        finite = np.isfinite(col)
        codes = np.rint(col[finite]).astype(int)
        mapped = np.array(
            [mapping[int(code)] if int(code) in mapping else np.nan for code in codes],
            dtype=float,
        )
        lookup[finite] = mapped
        out = out + lookup
    return pd.Series(out, index=getattr(responses, "index", None))


def binary_total(responses, agree: Sequence[int]) -> pd.Series:
    return score_items(responses, agree, scheme="binary")


def texts_match(observed: Iterable[str], published: Sequence[str]) -> bool:
    """Case- and punctuation-insensitive item-order check."""
    def norm(text: str) -> str:
        keep = "".join(ch.lower() for ch in text if ch.isalnum() or ch.isspace())
        return " ".join(keep.split())

    obs = [norm(t) for t in observed]
    pub = [norm(t) for t in published]
    return obs == pub
