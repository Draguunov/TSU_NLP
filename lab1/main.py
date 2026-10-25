"""Лабораторная № 1: поиск согласованных соседних слов."""
import argparse
from pathlib import Path
import re
from nltk import sent_tokenize, word_tokenize
import pymorphy3

WORD = re.compile(r"[А-Яа-яЁё]+(?:[-‐‑][А-Яа-яЁё]+)*")
PARTS = {"NOUN", "ADJF"}


def agrees(left, right):
    """Проверяем признаки исходных форм, а не лемм."""
    a, b = left.tag, right.tag
    if a.POS not in PARTS or b.POS not in PARTS:
        return False
    if a.number is None or a.number != b.number:
        return False
    if a.case is None or a.case != b.case:
        return False
    # Во множественном числе род прилагательного не выражен.
    if a.number == "plur":
        return True
    return a.gender is not None and a.gender == b.gender


def find_pairs(text, morph):
    """Один результат на позицию: совместимые разборы с наибольшим весом."""
    for sentence in sent_tokenize(text, language="russian"):
        tokens = word_tokenize(sentence, language="russian", preserve_line=True)
        # Пунктуацию не удаляем, чтобы не создавать новых соседств.
        for w1, w2 in zip(tokens, tokens[1:]):
            if not WORD.fullmatch(w1) or not WORD.fullmatch(w2):
                continue
            parses1 = morph.parse(w1.lower())
            parses2 = morph.parse(w2.lower())
            candidates = [(a, b) for a in parses1 for b in parses2 if agrees(a, b)]
            if candidates:
                a, b = max(candidates, key=lambda p: p[0].score * p[1].score)
                yield w1, w2, a, b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", type=Path,
                        default=Path(__file__).with_name("text.txt"))
    parser.add_argument("--explain", action="store_true",
                        help="показать исходную пару и выбранные признаки")
    args = parser.parse_args()
    try:
        text = args.file.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        parser.exit(1, f"Не удалось прочитать UTF-8 файл: {error}\n")
    morph = pymorphy3.MorphAnalyzer()
    try:
        for w1, w2, a, b in find_pairs(text, morph):
            print(a.normal_form, b.normal_form)
            if args.explain:
                print(f"  Исходная пара: {w1} {w2}\n  {a.tag}\n  {b.tag}")
    except LookupError:
        parser.exit(1, "Нет данных NLTK. Выполните: python -m nltk.downloader punkt_tab\n")


if __name__ == "__main__":
    main()
