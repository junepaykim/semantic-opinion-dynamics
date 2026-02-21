from enum import Enum


class Stance(str, Enum):
    pro_rto = "pro_rto"
    anti_rto = "anti_rto"
    hybrid = "hybrid"
    neutral = "neutral"


class Tone(str, Enum):
    analytical = "analytical"
    conciliatory = "conciliatory"
    aggressive = "aggressive"
    empathetic = "empathetic"
    neutral = "neutral"
