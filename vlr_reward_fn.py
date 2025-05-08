import os
from verl.utils.reward_score.math import is_equiv
from enum import IntEnum
import numpy as np


class RES(IntEnum):
    CORRECT = 1
    FORMAT_ONLY = 2
    WRONG = 3


res2str = {RES.CORRECT: "correct", RES.FORMAT_ONLY: "format_only", RES.WRONG: "wrong"}


def _find_matching_braces(text):
    start = text.rfind(r"\boxed{")
    if start == -1:
        return None

    i = start + len(r"\boxed{")
    stack = 1

    while i < len(text):
        if text[i] == '{':
            stack += 1
        elif text[i] == '}':
            stack -= 1
            if stack == 0:
                return (start, i)
        i += 1

    return None


def _get_end_token_idx(offset_mapping, pos):
    low, high = 0, len(offset_mapping) - 1
    while low <= high:
        mid = (low + high) // 2
        start, end = offset_mapping[mid]
        if start <= pos < end:
            return mid
        elif pos < start:
            high = mid - 1
        else:
            low = mid + 1
    return None


def _compute_res_verbose_len(solution_str, solution_id, offset_mapping, ground_truth):

    result = _find_matching_braces(solution_str)
    if result is None:
        return RES.WRONG, -1

    boxed_left = "\\boxed{"

    starting_slash_pos, closing_brace_pos = result

    end_token_idx = _get_end_token_idx(offset_mapping, closing_brace_pos)

    answer = solution_str[starting_slash_pos + len(boxed_left):closing_brace_pos]
    verbose_len = len(solution_id) - end_token_idx - 1

    if is_equiv(answer, ground_truth):
        return RES.CORRECT, verbose_len

    return RES.FORMAT_ONLY, verbose_len


def _relative_verbose_penalty_score(ress, lens, **kwargs):
    correct_max, correct_min, format_only, wrong = kwargs["correct_max"], kwargs["correct_min"], kwargs[
        "format_only"], kwargs["wrong"],
    ress = np.array(ress)
    lens = np.array(lens)
    scores = np.zeros_like(lens, dtype=float)

    correct_mask = (ress == RES.CORRECT)
    format_only_mask = (ress == RES.FORMAT_ONLY)
    wrong_mask = (ress == RES.WRONG)

    correct_lens = lens[correct_mask]

    if correct_lens.size > 0:
        min_len = correct_lens.min()
        max_len = correct_lens.max()
        if min_len == max_len:
            scores[correct_mask] = correct_max
        else:
            norm = (correct_lens - min_len) / (max_len - min_len)
            scores[correct_mask] = correct_max - norm * (correct_max - correct_min)

    scores[format_only_mask] = format_only
    scores[wrong_mask] = wrong

    return scores.tolist()


def _vanilla_score(ress, lens, **kwargs):
    correct_max, format_only, wrong = kwargs["correct_max"], kwargs["format_only"], kwargs["wrong"]
    ress = np.array(ress)
    lens = np.array(lens)
    scores = np.zeros_like(lens, dtype=float)

    correct_mask = (ress == RES.CORRECT)
    format_only_mask = (ress == RES.FORMAT_ONLY)
    wrong_mask = (ress == RES.WRONG)

    scores[correct_mask] = correct_max
    scores[format_only_mask] = format_only
    scores[wrong_mask] = wrong

    return scores.tolist()


def _val_score(ress, lens, **kwargs):
    ress = np.array(ress)
    lens = np.array(lens)
    scores = np.zeros_like(lens, dtype=float)

    correct_mask = (ress == RES.CORRECT)
    format_only_mask = (ress == RES.FORMAT_ONLY)
    wrong_mask = (ress == RES.WRONG)

    scores[correct_mask] = 1.0
    scores[format_only_mask] = 0.0
    scores[wrong_mask] = 0.0

    return scores.tolist()


def filter_truncated(scores, truncateds):
    return [0.0 if truncated else score for score, truncated in zip(scores, truncateds)]


def _general_score(_score,
                   solution_strs,
                   solution_ids,
                   ground_truths,
                   offset_mappings,
                   truncateds,
                   extra_info=None,
                   **kwargs):
    required_keys = ['correct_max', 'correct_min', 'format_only', 'wrong', 'filter_truncated']
    missing_keys = [k for k in required_keys if k not in kwargs]
    if missing_keys:
        raise KeyError(f"Missing required keys: {missing_keys}")

    ress, verbose_lens = zip(*[
        _compute_res_verbose_len(solution_str, solution_id, offset_mapping, ground_truth) for solution_str, solution_id,
        offset_mapping, ground_truth in zip(solution_strs, solution_ids, offset_mappings, ground_truths)
    ])
    scores = _score(ress, verbose_lens, **kwargs)
    if kwargs["filter_truncated"]:
        scores = filter_truncated(scores, truncateds)
    results = [res2str[item] for item in ress]
    return [{
        "score": s,
        "result": r,
        "verbose_len": v,
        "truncated": t
    } for s, r, v, t in zip(scores, results, verbose_lens, truncateds)]


def compute_score_rvp(solution_strs,
                      solution_ids,
                      ground_truths,
                      offset_mappings,
                      truncateds,
                      extra_infos=None,
                      **kwargs):
    return _general_score(_relative_verbose_penalty_score,
                          solution_strs,
                          solution_ids,
                          ground_truths,
                          offset_mappings,
                          truncateds,
                          extra_info=None,
                          **kwargs)


def compute_score_vanilla(solution_strs,
                          solution_ids,
                          ground_truths,
                          offset_mappings,
                          truncateds,
                          extra_infos=None,
                          **kwargs):
    return _general_score(_vanilla_score,
                          solution_strs,
                          solution_ids,
                          ground_truths,
                          offset_mappings,
                          truncateds,
                          extra_info=None,
                          **kwargs)


def val_compute_score(solution_strs,
                      solution_ids,
                      ground_truths,
                      offset_mappings,
                      truncateds,
                      extra_infos=None,
                      **kwargs):
    return _general_score(_val_score,
                          solution_strs,
                          solution_ids,
                          ground_truths,
                          offset_mappings,
                          truncateds,
                          extra_info=None,
                          **kwargs)
