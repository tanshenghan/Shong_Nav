"""Readable instruction banners for navigation review videos."""

from __future__ import annotations


def wrap_instruction(text, max_width, measure):
    """Wrap by measured pixel width without truncating the instruction."""
    if not isinstance(text, str) or not text.strip() or max_width <= 0:
        raise ValueError("a nonempty instruction and positive text width are required")
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if measure(candidate) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        while measure(word) > max_width:
            end = 1
            if measure(word[:end]) > max_width:
                raise ValueError("text area is too narrow for the selected font")
            while end < len(word) and measure(word[:end+1]) <= max_width:
                end += 1
            lines.append(word[:end])
            word = word[end:]
        current = word
    if current:
        lines.append(current)
    return lines


def instruction_banner(instruction, width, font_path):
    from PIL import Image, ImageDraw, ImageFont

    font_size = 22 if width >= 800 else 18
    body = ImageFont.truetype(font_path, font_size)
    label = ImageFont.truetype(font_path, 12)
    lines = wrap_instruction(instruction, width - 32, body.getlength)
    line_height = font_size + 7
    height = 46 + line_height * len(lines)
    height += height % 2
    banner = Image.new("RGB", (width, height), (13, 20, 29))
    draw = ImageDraw.Draw(banner)
    draw.text((16, 11), "INSTRUCTIONS", font=label, fill=(93, 222, 193), anchor="lt")
    for index, line in enumerate(lines):
        draw.text((16, 33 + index * line_height), line, font=body,
                  fill=(244, 246, 249), anchor="lt")
    draw.line((0, height-1, width, height-1), fill=(55, 71, 88))
    return banner, dict(lines=lines, font_size=font_size, width=width, height=height)


def prepend_banner(image, banner):
    from PIL import Image

    if image.width != banner.width:
        raise ValueError("instruction banner and video panel widths must match")
    result = Image.new("RGB", (image.width, image.height + banner.height))
    result.paste(banner, (0, 0))
    result.paste(image, (0, banner.height))
    return result


def fit_rgb_frame(rgb, size):
    """Preserve the official field of view when panel and source aspects differ."""
    from PIL import Image, ImageOps

    image = ImageOps.contain(Image.fromarray(rgb), size, method=Image.Resampling.LANCZOS)
    result = Image.new("RGB", size, (0, 0, 0))
    result.paste(image, ((size[0]-image.width)//2, (size[1]-image.height)//2))
    return result
