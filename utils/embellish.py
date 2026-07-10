import math


# ──────────────────────────────────────────────────────────────────────────────────────────────────


def sep(n: int) -> str:
    return n * "─" + "\n"
 

def sepComment(n: int, comment: str) -> str:
    
    result = (math.floor(n/2) * "─") + " " + comment + " " + (math.ceil(n/2) * "─") + "\n"
    return result


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# cores ANSI:


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def colorPrint(text: str, hex_color: str) -> str:
    r, g, b = hex_to_rgb(hex_color)
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def stylePrint(text: str, hex_color: str = None, bold: bool = False, italic: bool = False, underline: bool = False) -> str:
    codes = []
    if bold:      codes.append("1")
    if italic:    codes.append("3")
    if underline: codes.append("4")
    if hex_color:
        r, g, b = hex_to_rgb(hex_color)
        codes.append(f"38;2;{r};{g};{b}")
    if not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


# ──────────────────────────────────────────────────────────────────────────────────────────────────
