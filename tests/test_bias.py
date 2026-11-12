"""Tests for the bias maths and the preprocessing fix.

The embedding is a hand-built dict of vectors with a known gender axis, so no
corpus, gensim model or training run is needed:  pip install pytest && pytest
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import bias_detection as bd  # noqa: E402

DIM = 24


def planted_embedding(bias=0.8, seed=0):
    """Male words sit at +axis, female at -axis; career/family lean by `bias`."""
    rng = np.random.default_rng(seed)
    axis = np.zeros(DIM); axis[0] = 1.0
    topic = np.zeros(DIM); topic[1] = 1.0

    def vec(lean, base=None):
        v = lean * axis + (base if base is not None else 0) + rng.normal(0, 0.15, DIM)
        return v.astype(np.float32)

    kv = {}
    for w in bd.BIAS_SETS["male"]:
        kv[w] = vec(1.0)
    for w in bd.BIAS_SETS["female"]:
        kv[w] = vec(-1.0)
    for w in bd.BIAS_SETS["career"]:
        kv[w] = vec(bias, topic)
    for w in bd.BIAS_SETS["family"]:
        kv[w] = vec(-bias, topic)
    kv["engineer"] = vec(0.6, topic)
    kv["nurse"] = vec(-0.6, topic)
    return kv


def sets():
    s = bd.BIAS_SETS
    return s["male"], s["female"], s["career"], s["family"]


def test_weat_detects_planted_bias():
    kv = planted_embedding(bias=0.8)
    male, female, career, family = sets()
    d = bd.weat_effect_size(kv, male, female, career, family)
    p = bd.weat_p_permutation(kv, male, female, career, family, n=500)
    assert d > 1.5            # Cohen's d is bounded by 2
    assert p < 0.01


def test_weat_is_quiet_when_there_is_no_bias():
    kv = planted_embedding(bias=0.0)
    male, female, career, family = sets()
    assert abs(bd.weat_effect_size(kv, male, female, career, family)) < 1.0
    assert bd.weat_p_permutation(kv, male, female, career, family, n=500) > 0.05


def test_weat_sign_flips_with_the_bias():
    kv = planted_embedding(bias=-0.8)
    male, female, career, family = sets()
    assert bd.weat_effect_size(kv, male, female, career, family) < -1.5


def test_direction_is_male_positive_and_ranks_job_words():
    kv = planted_embedding()
    male, female, _, _ = sets()
    g = bd.combined_direction(kv, male, female)
    assert np.linalg.norm(g) == pytest.approx(1.0, abs=1e-5)
    assert np.dot(kv["he"], g) > 0 > np.dot(kv["she"], g)
    assert np.dot(kv["engineer"], g) > 0 > np.dot(kv["nurse"], g)


def test_held_out_accuracy_is_high_on_a_clean_axis_and_restores_anchor_pairs():
    kv = planted_embedding()
    male, female, _, _ = sets()
    before = list(bd.ANCHOR_PAIRS)
    acc, n = bd.held_out_lexicon_accuracy(kv, male, female)
    assert n == 2 * len(male) and acc == 1.0
    assert bd.ANCHOR_PAIRS == before


def test_held_out_accuracy_is_near_chance_without_a_gender_axis():
    rng = np.random.default_rng(1)
    male, female, _, _ = sets()
    kv = {w: rng.normal(0, 1, DIM).astype(np.float32) for w in male + female}
    in_sample_g = bd.combined_direction(kv, male, female)
    in_sample = np.mean([np.dot(kv[w], in_sample_g) > 0 for w in male] + [np.dot(kv[w], in_sample_g) < 0 for w in female])
    held_out, _ = bd.held_out_lexicon_accuracy(kv, male, female)
    # random vectors have no gender axis: the circular score still looks good,
    # the held-out one falls to about chance
    assert in_sample > 0.7
    assert held_out < 0.65
    assert in_sample > held_out + 0.1


def test_missing_words_do_not_crash():
    kv = planted_embedding()
    for w in ("career", "family", "he"):
        kv.pop(w)
    male, female, career, family = sets()
    assert np.isfinite(bd.weat_effect_size(kv, male, female, career, family))
    assert np.isnan(bd.association(kv, "career", male, female))
    assert bd.combined_direction(kv, male, female) is not None


def test_importing_the_module_creates_no_folder(tmp_path, monkeypatch):
    import importlib
    monkeypatch.chdir(tmp_path)
    importlib.reload(bd)
    assert not (tmp_path / "outputs").exists()


def test_cleaner_keeps_the_exact_words_the_analysis_looks_up():
    try:
        clean = bd.build_cleaner()
        tokens = clean("She said the children and their parents visit relatives; he works in sales and was promoted.")
    except LookupError:
        pytest.skip("NLTK data not available offline")
    for word in ("she", "he", "children", "parents", "relatives", "sales"):
        assert word in tokens, word
    assert "the" not in tokens and "wa" not in tokens   # stop-words still go
    assert "visit" in tokens and "promoted" in tokens
