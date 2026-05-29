from dataclasses import dataclass
from enum import Enum


class Ocr(str, Enum):
    ocr32px = "32px"
    ocr48px = "48px"
    ocr48px_ctc = "48px_ctc"
    mocr = "mocr"


@dataclass
class OcrConfig:
    use_mocr_merge: bool = False
    ocr: Ocr = Ocr.ocr48px
    min_text_length: int = 0
    ignore_bubble: int = 0
    prob: float | None = None
