# ============================================================
# plate_corrector.py
#
# Rewritten after finding the previous version was actively destructive:
# it used blind, position-unaware global replaces (text.replace('3','8')
# corrupts EVERY '3' in the string, including correct ones) and rules
# hardcoded to specific known plates from the dev's own test set
# (e.g. "if state/district == KA05, force next letter to M" - only
# true because KA05MU0712 happened to be in the test data; would
# force-corrupt any other real KA05 plate).
#
# Design here: only correct a character when the PLATE FORMAT tells us
# what type of character MUST be there (letter vs digit), and only
# flip within known OCR confusion pairs. Never touch a character that's
# already the correct type for its position. Verified this leaves
# already-correct plates untouched - see test cases at the bottom.
# ============================================================

import re

# OCR misreads a digit as one of these letters -> map back to the digit,
# ONLY when the format says a digit belongs at that position.
LETTER_TO_DIGIT = {
    'O': '0', 'I': '1', 'L': '1', 'A': '4', 'S': '5',
    'B': '8', 'G': '6', 'Z': '2', 'U': '0', 'T': '1',
}

# OCR misreads a letter as one of these digits -> map back to the letter,
# ONLY when the format says a letter belongs at that position.
DIGIT_TO_LETTER = {
    '0': 'O', '1': 'I', '4': 'A', '5': 'S',
    '8': 'B', '6': 'G', '2': 'Z',
}


def force_digit(ch):
    if ch.isdigit():
        return ch
    return LETTER_TO_DIGIT.get(ch, ch)  # leave unchanged if no known mapping


def force_letter(ch):
    if ch.isalpha():
        return ch
    return DIGIT_TO_LETTER.get(ch, ch)  # leave unchanged if no known mapping


def _correct_standard_series(text):
    """Standard format: SS DD LLL NNNN (state letters, district digits,
    series letters, number digits). Series length varies 0-3 letters,
    so total length is 9 or 10."""
    if len(text) not in (9, 10):
        return None

    state = "".join(force_letter(c) for c in text[0:2])
    district = "".join(force_digit(c) for c in text[2:4])
    series = text[4:-4]
    series_corrected = "".join(force_letter(c) for c in series)
    number = "".join(force_digit(c) for c in text[-4:])

    corrected = state + district + series_corrected + number
    valid = state.isalpha() and district.isdigit() and number.isdigit()
    return corrected, valid


def _correct_bh_series(text):
    """BH format: YY BH NNNN L(L) - e.g. 23BH1234C (9 chars, 1-letter
    suffix) or 23BH1234CD (10 chars, 2-letter suffix)."""
    if len(text) not in (9, 10):
        return None
    if text[2:4] != "BH":
        return None

    year = "".join(force_digit(c) for c in text[0:2])
    bh = "BH"
    number = "".join(force_digit(c) for c in text[4:8])
    suffix = "".join(force_letter(c) for c in text[8:])

    corrected = year + bh + number + suffix
    valid = year.isdigit() and number.isdigit() and suffix.isalpha()
    return corrected, valid


def correct_plate(text):
    """
    Position-role-aware correction: only forces a character to digit/
    letter when the KNOWN FORMAT requires that type at that position.
    Never blindly replaces a character just because it COULD be a
    misread of something else - that's what made the old version
    destroy already-correct plates.
    """
    if not text:
        return {"corrected": None, "format": "unknown", "valid": False}

    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    if not text:
        return {"corrected": None, "format": "unknown", "valid": False}

    bh_result = _correct_bh_series(text)
    if bh_result and bh_result[1]:
        corrected, valid = bh_result
        return {"corrected": corrected, "format": "bh", "valid": valid}

    std_result = _correct_standard_series(text)
    if std_result and std_result[1]:
        corrected, valid = std_result
        return {"corrected": corrected, "format": "standard", "valid": valid}

    # Neither format matched validly - return best-effort standard-format
    # guess if length is plausible, otherwise return the cleaned text
    # completely untouched rather than guessing wildly.
    if std_result:
        corrected, valid = std_result
        return {"corrected": corrected, "format": "standard", "valid": False}
    if bh_result:
        corrected, valid = bh_result
        return {"corrected": corrected, "format": "bh", "valid": False}

    return {"corrected": text, "format": "unknown", "valid": False}


if __name__ == "__main__":
    print("Regression test: already-correct plates must stay UNCHANGED")
    print("-" * 70)
    already_correct = [
        "KA41AA0033", "KA01MS7103", "AP28CC4284",
        "KA05MU0712", "KA51HB7942", "23BH1234C",
    ]
    all_ok = True
    for plate in already_correct:
        result = correct_plate(plate)
        ok = result["corrected"] == plate
        all_ok &= ok
        print(f"{plate:15} -> {str(result['corrected']):15} "
              f"{'OK' if ok else 'CORRUPTED <-- BUG'}")

    print("\nSanity test: common single-char misreads should still resolve")
    print("-" * 70)
    noisy_cases = [
        ("KA4IAA0033", "KA41AA0033"),   # 1 misread as I
        ("KA01MS7I03", "KA01MS7103"),   # 1 misread as I
        ("KAO1MS7103", "KA01MS7103"),   # 0 misread as O
    ]
    for pred, truth in noisy_cases:
        result = correct_plate(pred)
        ok = result["corrected"] == truth
        print(f"{pred:15} -> {str(result['corrected']):15} (truth: {truth}) "
              f"{'OK' if ok else 'did not resolve'}")

    print(f"\n{'ALL ALREADY-CORRECT PLATES SURVIVED - safe to use' if all_ok else 'STILL CORRUPTING GOOD PLATES - do not use'}")