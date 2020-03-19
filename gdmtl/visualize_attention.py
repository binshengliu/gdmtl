import argparse
import math
import multiprocessing as mp
import os
from itertools import repeat
from typing import Any, List, Tuple, Union

import imgkit
import matplotlib as mpl
import matplotlib.pyplot as plt  # noqa
import pdfkit
from pdfCropMargins import crop
from tqdm import tqdm


def colorize_mp_wrapper(zipped: Any) -> Tuple[bytes, bytes]:
    return colorize(*zipped)


def colorize(
    words: List[str],
    color_array: List[float],
    img_out: Union[bool, str] = False,
    pdf_out: Union[bool, str] = False,
    norm_max: Union[bool, float] = True,
) -> Tuple[bytes, bytes]:
    # words is a list of words
    # color_array is an array of numbers between 0 and 1 of length equal to words
    if isinstance(norm_max, bool) and norm_max:
        norm = mpl.colors.Normalize(vmin=min(color_array), vmax=max(color_array))
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    elif isinstance(norm_max, float):
        norm = mpl.colors.Normalize(vmin=min(color_array), vmax=norm_max)
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    else:
        cmap = mpl.cm.ScalarMappable(cmap=mpl.cm.Blues)

    font = "DejaVu Sans Mono"  # Linux Libertine O
    word_surround = (
        f'<span style="font-family: {font}; font-size: 12; color: black">' "{}</span>"
    )
    weight_surround = (
        f'<span style="font-family: {font}; font-size: 6; color: black">' "{}</span>"
    )

    template = '<span style="background-color: {}">{}</span>'
    word_count = 0
    strings = []
    colored_string = ""
    weight_string = ""
    for word, color in zip(words, color_array):
        color_str = f"{color:.3f}"
        word_len = max(math.ceil(len(color_str) / 2), len(word)) + 1
        if word_count + word_len > 80:
            strings.append(word_surround.format(colored_string + "<br>"))
            strings.append(weight_surround.format(weight_string + "<br>"))
            # strings.append("<br>")
            word_count = 0
            colored_string = ""
            weight_string = ""

        word_count += word_len
        color_hex = mpl.colors.rgb2hex(cmap.to_rgba(color)[:3])
        colored_string += template.format(
            color_hex, word.ljust(word_len).replace(" ", "&nbsp")
        )
        weight_string += template.format(
            color_hex, color_str.ljust(word_len * 2).replace(" ", "&nbsp")
        )
    if colored_string:
        strings.append(word_surround.format(colored_string + "<br>"))
        strings.append(weight_surround.format(weight_string))
    final_string = "\n".join(strings)
    img_figure: bytes = imgkit.from_string(
        final_string, img_out, options={"quality": 100, "quiet": 1},
    )
    pdf_figure: bytes = pdfkit.from_string(
        final_string, pdf_out, options={"quiet": 0},
    )
    return img_figure, pdf_figure


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument(
        "--hide", default=["[CLS]", "[SEP]"], type=lambda x: x.split(",")
    )

    return parser.parse_args()


def visualize_sp(zipped: Any) -> None:
    line, hide, output_dir = zipped

    step, qid, did, label, logit, attens = line.split("\t")
    splits = attens.split(" ")
    words, weights = splits[::2], [float(x) for x in splits[1::2]]
    assert len(words) == len(weights)
    for i in range(len(words)):
        if words[i] in hide:
            weights[i] = 0
    dummy = f"{output_dir}/.{step}-{qid}-{did}-{label}-{logit}.pdf"
    name = f"{output_dir}/{step}-{qid}-{did}-{label}-{logit}.pdf"
    colorize(words, weights, False, dummy, max(weights) * 1.2)
    crop(["-p", "0", "-o", name, dummy])
    os.unlink(dummy)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_arguments()

    with open(args.input, "r") as f:
        lines = f.readlines()

    os.makedirs(args.output_dir, exist_ok=True)
    with mp.Pool(mp.cpu_count() // 2) as pool:
        out = pool.imap(
            visualize_sp, zip(lines, repeat(args.hide), repeat(args.output_dir))
        )
        list(tqdm(out, total=len(lines), desc="Visualize"))
