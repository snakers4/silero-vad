"""Forced splits at max_speech_duration_s must not lose speech.

These tests drive get_speech_timestamps_from_probs with scripted
probabilities, so they depend only on the post-processing state machine.
"""
import pytest

from silero_vad import get_speech_timestamps_from_probs

WINDOW = 512  # samples per probability at 16 kHz


@pytest.mark.parametrize("use_max_poss_sil_at_max_speech", [True, False])
def test_forced_split_keeps_the_window_it_cuts_at(use_max_poss_sil_at_max_speech):
    # 700 windows of uninterrupted speech and no padding to hide a gap
    speech_probs = [0.9] * 700
    speeches = get_speech_timestamps_from_probs(
        speech_probs,
        max_speech_duration_s=2,
        min_speech_duration_ms=0,
        speech_pad_ms=0,
        use_max_poss_sil_at_max_speech=use_max_poss_sil_at_max_speech,
    )

    assert len(speeches) > 1
    assert speeches[0]['start'] == 0
    assert speeches[-1]['end'] == len(speech_probs) * WINDOW
    for previous, following in zip(speeches, speeches[1:]):
        assert following['start'] == previous['end']
    assert all(s['end'] - s['start'] <= 2 * 16000 for s in speeches)


@pytest.mark.parametrize("use_max_poss_sil_at_max_speech", [True, False])
def test_forced_split_keeps_speech_held_by_neg_threshold(use_max_poss_sil_at_max_speech):
    # After the onset the probabilities stay between neg_threshold (0.35) and
    # threshold (0.5): still speech for the state machine, so without
    # max_speech_duration_s this is a single 9.6 s chunk.
    speech_probs = [0.9] * 20 + [0.45] * 280
    audio_length_samples = len(speech_probs) * WINDOW
    assert get_speech_timestamps_from_probs(speech_probs) == [
        {'start': 0, 'end': audio_length_samples}
    ]

    speeches = get_speech_timestamps_from_probs(
        speech_probs,
        max_speech_duration_s=5,
        use_max_poss_sil_at_max_speech=use_max_poss_sil_at_max_speech,
    )

    assert len(speeches) == 2
    assert speeches[0]['start'] == 0
    assert speeches[1]['start'] == speeches[0]['end']
    assert speeches[1]['end'] == audio_length_samples
    assert all(s['end'] - s['start'] <= 5 * 16000 for s in speeches)


def test_forced_split_inside_a_pause_waits_for_speech_to_return():
    # The limit is reached during a pause that is still shorter than
    # min_silence_duration_ms; the next chunk starts when speech comes back.
    speech_probs = [0.9] * 155 + [0.05] * 10 + [0.9] * 50
    speeches = get_speech_timestamps_from_probs(
        speech_probs,
        max_speech_duration_s=5,
        min_silence_duration_ms=500,
        speech_pad_ms=0,
    )

    assert speeches == [
        {'start': 0, 'end': 156 * WINDOW},
        {'start': 165 * WINDOW, 'end': 215 * WINDOW},
    ]
