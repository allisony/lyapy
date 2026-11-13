import json

import pytest

from lyapy.atomic_data import (
    ATOMIC_TRANSITIONS,
    MORTON_2003_TABLE_2,
    AtomicTransition,
    get_atomic_transition,
)


@pytest.mark.parametrize(
    ("key", "wavelength", "oscillator_strength", "damping_rate"),
    (
        ("h_i_1215", 1215.6700, 4.164e-1, 6.265e8),
        ("si_iii_1206", 1206.500, 1.63, 2.48e9),
        ("n_v_1238", 1238.821, 1.560e-1, 3.391e8),
        ("n_v_1242", 1242.804, 7.770e-2, 3.356e8),
        ("o_v_1218", 1218.344, 1.46e-6, 2.18e3),
        ("c_ii_1334", 1334.5323, 1.28e-1, 2.88e8),
        ("c_ii_1335a", 1335.6627, 1.28e-2, 2.88e8),
        ("c_ii_1335b", 1335.7077, 1.15e-1, 2.88e8),
        ("mg_ii_k", 2796.3543, 6.155e-1, 2.625e8),
        ("mg_ii_h", 2803.5315, 3.058e-1, 2.595e8),
    ),
)
def test_catalog_transcribes_morton_2003_table_2(
    key,
    wavelength,
    oscillator_strength,
    damping_rate,
):
    transition = ATOMIC_TRANSITIONS[key]

    assert transition.absorption_parameters() == (
        wavelength,
        oscillator_strength,
        damping_rate,
    )
    assert transition.reference == MORTON_2003_TABLE_2
    assert transition.has_absorption_data
    json.dumps(transition.as_dict())


def test_custom_emission_only_transition_may_omit_absorption_data():
    transition = AtomicTransition("custom", "X", "X 1000", 1000.0)

    assert not transition.has_absorption_data
    with pytest.raises(ValueError, match="has no absorption data"):
        transition.absorption_parameters()


def test_absorption_data_must_be_complete_positive_and_referenced():
    with pytest.raises(ValueError, match="provided together"):
        AtomicTransition("custom", "X", "X 1000", 1000.0, 0.5)
    with pytest.raises(ValueError, match="oscillator_strength"):
        AtomicTransition("custom", "X", "X 1000", 1000.0, 0.0, 1.0, "ref")
    with pytest.raises(ValueError, match="damping_rate"):
        AtomicTransition("custom", "X", "X 1000", 1000.0, 0.5, 0.0, "ref")
    with pytest.raises(ValueError, match="require a reference"):
        AtomicTransition("custom", "X", "X 1000", 1000.0, 0.5, 1.0)


def test_transition_lookup_accepts_keys_and_custom_values():
    assert get_atomic_transition("mg_ii_k") is ATOMIC_TRANSITIONS["mg_ii_k"]
    custom = AtomicTransition("custom", "X", "X 1000", 1000.0)
    assert get_atomic_transition(custom) is custom
    with pytest.raises(ValueError, match="unknown atomic transition"):
        get_atomic_transition("missing")
    with pytest.raises(TypeError):
        ATOMIC_TRANSITIONS["custom"] = custom
