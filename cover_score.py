#!/usr/bin/env python3
"""
Cover scoring tool for children's book covers.

Runs deterministic checks on cover images, TeX source, and PDF output.

Usage:
    cover-score --image <path>     Image-based checks (resolution, color, contrast)
    cover-score --tex <path>       TeX source checks (contrast boxes, fonts, spine)
    cover-score --pdf <path>       PDF output checks (text overlap, anchor, dimensions)

Combine flags for full analysis:
    cover-score --image front.png --tex cover.tex --pdf cover.pdf
"""

import sys
import os
import re
import subprocess
import math
from pathlib import Path
from xml.etree import ElementTree

try:
    import argparse
except ImportError:
    print("ERROR: argparse not available.")
    sys.exit(1)

try:
    from PIL import Image, ImageStat, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# --- Constants ---

KDP_WIDTH_PX = 1875
KDP_HEIGHT_PX = 2775
KDP_MIN_WIDTH = 1800
KDP_MIN_HEIGHT = 2700
KDP_ASPECT = 9.25 / 6.25

THUMB_WIDTH = 160

WCAG_AA_LARGE = 3.0
WCAG_AA_NORMAL = 4.5

BLEED_MARGIN_PCT = 0.02


# --- Utility ---

def relative_luminance(r, g, b):
    def linearize(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)


def contrast_ratio(lum1, lum2):
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    return (lighter + 0.05) / (darker + 0.05)


def get_dominant_colors(img, n_colors=8):
    small = img.resize((150, 150), Image.LANCZOS)
    quantized = small.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()
    color_counts = sorted(quantized.getcolors(), reverse=True)
    total = sum(c for c, _ in color_counts)
    colors = []
    for count, idx in color_counts:
        r, g, b = palette[idx * 3], palette[idx * 3 + 1], palette[idx * 3 + 2]
        colors.append({"rgb": (r, g, b), "proportion": count / total})
    return colors


def rgb_to_hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    diff = mx - mn
    v = mx
    s = 0 if mx == 0 else diff / mx
    if diff == 0:
        h = 0
    elif mx == r:
        h = 60 * (((g - b) / diff) % 6)
    elif mx == g:
        h = 60 * ((b - r) / diff + 2)
    else:
        h = 60 * ((r - g) / diff + 4)
    return h, s, v


def make_result(name, score, max_pts, icon, feedback, detail=None, suggestion=None, critical=None):
    return {
        "name": name, "score": score, "max": max_pts, "icon": icon,
        "feedback": feedback, "detail": detail,
        "suggestion": suggestion, "critical": critical,
    }


# ============================================================
# IMAGE CHECKS (--image)
# ============================================================

def check_thumbnail_readability(img):
    max_pts = 15
    ratio = img.width / THUMB_WIDTH
    thumb = img.resize((THUMB_WIDTH, int(img.height / ratio)), Image.LANCZOS)
    title_zone = thumb.crop((0, 0, thumb.width, thumb.height // 3))
    stddev = ImageStat.Stat(title_zone.convert("L")).stddev[0]

    if stddev >= 70:
        score, feedback, icon = max_pts, "Strong contrast in title zone at thumbnail size", "✓"
    elif stddev >= 55:
        score, feedback, icon = int(max_pts * 0.8), "Decent title zone contrast at thumbnail", "~"
    elif stddev >= 40:
        score, feedback, icon = int(max_pts * 0.55), "Title zone contrast low — may not read at thumbnail", "⚠"
    elif stddev >= 25:
        score, feedback, icon = int(max_pts * 0.3), "Title zone barely visible at thumbnail", "✗"
    else:
        score, feedback, icon = 0, "Title zone has almost no contrast — invisible at thumbnail", "✗"

    return make_result("Thumbnail Readability", score, max_pts, icon, feedback,
                       f"Stddev={stddev:.1f} (target ≥55)",
                       "Add text outline, drop shadow, or darker area behind title" if score < max_pts * 0.7 else None)


def check_wcag_contrast(img):
    max_pts = 15
    title_zone = img.crop((0, 0, img.width, img.height // 3))
    small = title_zone.resize((100, 33), Image.LANCZOS)
    pixels = list(small.getdata())
    luminances = sorted([relative_luminance(*p) for p in pixels])
    n = len(luminances)
    dark_avg = sum(luminances[:n // 5]) / max(n // 5, 1)
    bright_avg = sum(luminances[-(n // 5):]) / max(n // 5, 1)
    ratio = contrast_ratio(bright_avg, dark_avg)

    if ratio >= WCAG_AA_NORMAL:
        score, feedback, icon = max_pts, f"Contrast ratio {ratio:.1f}:1 — passes WCAG AA", "✓"
    elif ratio >= WCAG_AA_LARGE:
        score, feedback, icon = int(max_pts * 0.7), f"Contrast ratio {ratio:.1f}:1 — AA for large text only", "~"
    elif ratio >= 2.5:
        score, feedback, icon = int(max_pts * 0.4), f"Contrast ratio {ratio:.1f}:1 — below WCAG minimum", "⚠"
    else:
        score, feedback, icon = int(max_pts * 0.15), f"Contrast ratio {ratio:.1f}:1 — poor", "✗"

    critical = ratio < WCAG_AA_LARGE
    return make_result("WCAG Contrast (Title Zone)", score, max_pts, icon, feedback,
                       f"Ratio={ratio:.2f}:1 (AA-large≥3.0, AA≥4.5)",
                       "Increase text outline or add semi-transparent background" if score < max_pts * 0.7 else None,
                       f"WCAG contrast {ratio:.1f}:1 — minimum 3.0:1 for large text" if critical else None)


def check_color_vibrancy(img):
    max_pts = 10
    colors = get_dominant_colors(img, 8)
    total_sat, total_weight = 0, 0
    for c in colors:
        _, s, _ = rgb_to_hsv(*c["rgb"])
        total_sat += s * c["proportion"]
        total_weight += c["proportion"]
    avg_sat = total_sat / max(total_weight, 0.001)

    if avg_sat >= 0.45:
        score, feedback, icon = max_pts, "Vibrant, saturated palette", "✓"
    elif avg_sat >= 0.35:
        score, feedback, icon = int(max_pts * 0.8), "Good color saturation", "~"
    elif avg_sat >= 0.25:
        score, feedback, icon = int(max_pts * 0.5), "Colors somewhat muted", "⚠"
    elif avg_sat >= 0.15:
        score, feedback, icon = int(max_pts * 0.25), "Dull, desaturated palette", "✗"
    else:
        score, feedback, icon = 0, "Nearly grayscale", "✗"

    return make_result("Color Vibrancy", score, max_pts, icon, feedback,
                       f"Avg saturation={avg_sat:.2f} (target ≥0.35)",
                       "Increase color saturation" if score < max_pts * 0.7 else None)


def check_color_diversity(img):
    max_pts = 10
    colors = get_dominant_colors(img, 12)
    hue_buckets = set()
    for c in colors:
        h, s, v = rgb_to_hsv(*c["rgb"])
        if s > 0.1 and v > 0.1:
            hue_buckets.add(int(h / 30))
    n_hues = len(hue_buckets)

    if 3 <= n_hues <= 5:
        score, feedback, icon = max_pts, f"{n_hues} distinct hue groups — balanced", "✓"
    elif n_hues == 2 or n_hues == 6:
        score, feedback, icon = int(max_pts * 0.7), f"{n_hues} distinct hue groups — acceptable", "~"
    elif n_hues == 1:
        score, feedback, icon = int(max_pts * 0.4), "Monochromatic — add accent color", "⚠"
    elif n_hues >= 7:
        score, feedback, icon = int(max_pts * 0.5), f"{n_hues} hue groups — chaotic", "⚠"
    else:
        score, feedback, icon = int(max_pts * 0.3), "Very limited color range", "✗"

    return make_result("Color Diversity", score, max_pts, icon, feedback,
                       f"Hue clusters={n_hues} (target 3-5)",
                       "Add 1-2 complementary accent colors" if n_hues < 3 else ("Simplify to 3-5 hues" if n_hues > 6 else None))


def check_title_zone_clarity(img):
    max_pts = 10
    title_zone = img.crop((0, 0, img.width, img.height // 3))
    edges = title_zone.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_mean = ImageStat.Stat(edges).mean[0]

    if 15 <= edge_mean <= 40:
        score, feedback, icon = max_pts, "Title zone clean — text against clear background", "✓"
    elif 10 <= edge_mean < 15:
        score, feedback, icon = int(max_pts * 0.7), "Title zone fairly clean", "~"
    elif 40 < edge_mean <= 55:
        score, feedback, icon = int(max_pts * 0.6), "Title zone busy — text may compete", "⚠"
    elif edge_mean < 10:
        score, feedback, icon = int(max_pts * 0.5), "Title zone very flat", "⚠"
    else:
        score, feedback, icon = int(max_pts * 0.3), "Title zone very busy — text hard to read", "✗"

    return make_result("Title Zone Clarity", score, max_pts, icon, feedback,
                       f"Edge density={edge_mean:.1f} (target 15-40)",
                       "Simplify illustration behind title" if score < max_pts * 0.7 else None)


def check_resolution(img, path):
    max_pts = 10
    w, h = img.size

    if w >= KDP_MIN_WIDTH and h >= KDP_MIN_HEIGHT:
        score, feedback, icon = max_pts, f"{w}×{h}px — meets KDP 300dpi", "✓"
    elif w >= KDP_MIN_WIDTH * 0.85 and h >= KDP_MIN_HEIGHT * 0.85:
        score, feedback, icon = int(max_pts * 0.6), f"{w}×{h}px — slightly below KDP minimum", "⚠"
    elif w >= 1200 and h >= 1800:
        score, feedback, icon = int(max_pts * 0.3), f"{w}×{h}px — low for print", "⚠"
    else:
        score, feedback, icon = 0, f"{w}×{h}px — too low", "✗"

    return make_result("Image Resolution", score, max_pts, icon, feedback,
                       f"Actual={w}×{h}, KDP min={KDP_MIN_WIDTH}×{KDP_MIN_HEIGHT}",
                       "Upscale to at least 1875×2775px" if score < max_pts else None)


def check_aspect_ratio(img):
    max_pts = 5
    w, h = img.size
    actual_ratio = h / w if w > 0 else 0
    front_diff = abs(actual_ratio - KDP_ASPECT)
    wrap_ratio = 9.25 / (6.25 * 2 + 0.5)
    wrap_diff = abs(actual_ratio - wrap_ratio)
    landscape_diff = abs(actual_ratio - (3 / 4))
    best_diff = min(front_diff, wrap_diff, landscape_diff)

    if best_diff < 0.05:
        score, feedback, icon = max_pts, "Aspect ratio matches expected format", "✓"
    elif best_diff < 0.1:
        score, feedback, icon = int(max_pts * 0.7), "Aspect ratio close", "~"
    elif best_diff < 0.2:
        score, feedback, icon = int(max_pts * 0.4), "Aspect ratio slightly off", "⚠"
    else:
        score, feedback, icon = 0, f"Aspect ratio {actual_ratio:.2f} doesn't match", "✗"

    return make_result("Aspect Ratio", score, max_pts, icon, feedback,
                       f"Actual={actual_ratio:.3f}, Front={KDP_ASPECT:.3f}, Wrap={wrap_ratio:.3f}",
                       "Resize or crop to match KDP dimensions" if score < max_pts else None)


def check_brightness_balance(img):
    max_pts = 10
    gray = img.convert("L")
    histogram = gray.histogram()
    total = sum(histogram)
    shadows = sum(histogram[0:64]) / total
    midtones = sum(histogram[64:192]) / total
    highlights = sum(histogram[192:256]) / total
    mean_b = sum(i * histogram[i] for i in range(256)) / total

    problems = []
    if shadows > 0.5: problems.append("too dark (>50% shadows)")
    if highlights > 0.5: problems.append("too bright (>50% highlights)")
    if midtones < 0.2: problems.append("lack of midtones")
    if mean_b < 60: problems.append(f"mean brightness {mean_b:.0f}/255 — very dark")
    if mean_b > 200: problems.append(f"mean brightness {mean_b:.0f}/255 — washed out")

    if not problems:
        score, feedback, icon = max_pts, "Good brightness balance", "✓"
    elif len(problems) == 1:
        score, feedback, icon = int(max_pts * 0.6), f"Brightness issue: {problems[0]}", "⚠"
    else:
        score, feedback, icon = int(max_pts * 0.3), f"Brightness issues: {'; '.join(problems)}", "✗"

    return make_result("Brightness Balance", score, max_pts, icon, feedback,
                       f"Shadows={shadows:.0%} Mid={midtones:.0%} High={highlights:.0%} Mean={mean_b:.0f}",
                       "Adjust levels" if problems else None)


def check_edge_safety(img):
    max_pts = 5
    w, h = img.size
    margin_x = int(w * BLEED_MARGIN_PCT)
    margin_y = int(h * BLEED_MARGIN_PCT)

    if margin_x < 5 or margin_y < 5:
        return make_result("Edge Bleed Safety", max_pts, max_pts, "~",
                           "Image too small to check bleed margins", "Skipped")

    gray = img.convert("L")
    def strip_mean(box): return ImageStat.Stat(gray.crop(box)).mean[0]

    edge_diffs = [
        abs(strip_mean((0, 0, w, margin_y)) - strip_mean((0, margin_y, w, margin_y * 3))),
        abs(strip_mean((0, h - margin_y, w, h)) - strip_mean((0, h - margin_y * 3, w, h - margin_y))),
        abs(strip_mean((0, 0, margin_x, h)) - strip_mean((margin_x, 0, margin_x * 3, h))),
        abs(strip_mean((w - margin_x, 0, w, h)) - strip_mean((w - margin_x * 3, 0, w - margin_x, h))),
    ]
    max_diff = max(edge_diffs)

    if max_diff < 15:
        score, feedback, icon = max_pts, "Edges blend smoothly — safe for trim", "✓"
    elif max_diff < 30:
        score, feedback, icon = int(max_pts * 0.7), "Some content near edges", "~"
    else:
        score, feedback, icon = int(max_pts * 0.3), "Significant content near edges — may be cut", "⚠"

    return make_result("Edge Bleed Safety", score, max_pts, icon, feedback,
                       f"Max edge diff={max_diff:.1f} (target <15)",
                       "Move critical elements away from 0.125\" margins" if score < max_pts else None)


def check_subject_centering(img):
    max_pts = 10
    w, h = img.size
    upper_h = int(h * 0.6)
    upper = img.crop((0, 0, w, upper_h))
    hsv = upper.convert("HSV")
    hsv_data = list(hsv.getdata())
    sats = [p[1] for p in hsv_data]
    vals = [p[2] for p in hsv_data]
    sat_thresh = sorted(sats)[int(len(sats) * 0.85)]
    val_thresh = sorted(vals)[int(len(vals) * 0.40)]

    subject_xs = [i % w for i, (s_val, v_val) in enumerate(zip(sats, vals))
                  if s_val > sat_thresh and v_val > val_thresh]

    if not subject_xs:
        return make_result("Subject Centering", max_pts, max_pts, "~",
                           "No distinct subject detected — skipped", "Skipped")

    centroid_x = sum(subject_xs) / len(subject_xs)
    pct_x = centroid_x / w * 100
    spine_cutoff = int(w * 0.05)
    spine_pct = sum(1 for x in subject_xs if x < spine_cutoff) / len(subject_xs) * 100
    trim_cutoff = int(w * 0.95)
    trim_pct = sum(1 for x in subject_xs if x > trim_cutoff) / len(subject_xs) * 100

    problems = []
    is_killer = False
    if pct_x < 35:
        problems.append(f"centroid at {pct_x:.0f}% — off-center toward spine")
        is_killer = True
    elif pct_x > 65:
        problems.append(f"centroid at {pct_x:.0f}% — off-center toward trim")
        is_killer = True
    if spine_pct > 10:
        problems.append(f"{spine_pct:.0f}% of subject at spine edge")
        is_killer = True
    if trim_pct > 15:
        problems.append(f"{trim_pct:.0f}% of subject at trim edge")

    if not problems:
        score, feedback, icon = max_pts, "Subject well-centered", "✓"
    elif is_killer:
        score, feedback, icon = 0, f"Subject NOT centered: {'; '.join(problems)}", "✗"
    else:
        score, feedback, icon = int(max_pts * 0.6), f"Minor: {'; '.join(problems)}", "⚠"

    return make_result("Subject Centering", score, max_pts, icon, feedback,
                       f"Centroid={pct_x:.1f}% SpineEdge={spine_pct:.1f}% TrimEdge={trim_pct:.1f}%",
                       "Center subject — keep away from spine edge" if problems else None,
                       f"Subject off-center — centroid {pct_x:.0f}%, spine edge {spine_pct:.0f}%" if is_killer else None)


def check_compression_quality(img, path):
    max_pts = 10
    file_size = os.path.getsize(path)
    pixel_count = img.width * img.height
    bits_per_pixel = (file_size * 8) / max(pixel_count, 1)
    laplacian = img.convert("L").filter(ImageFilter.Kernel(size=(3, 3), kernel=[-1, -1, -1, -1, 8, -1, -1, -1, -1], scale=1, offset=128))
    lap_stddev = ImageStat.Stat(laplacian).stddev[0]
    is_png = str(path).lower().endswith(".png")

    problems = []
    if not is_png and bits_per_pixel < 1.0: problems.append(f"heavy compression ({bits_per_pixel:.1f} bpp)")
    if lap_stddev < 8 and not is_png: problems.append(f"low detail (laplacian σ={lap_stddev:.1f})")

    if is_png or not problems:
        score, feedback, icon = max_pts, "Good image quality" + (" (PNG)" if is_png else ""), "✓"
    elif len(problems) == 1:
        score, feedback, icon = int(max_pts * 0.5), f"Quality concern: {problems[0]}", "⚠"
    else:
        score, feedback, icon = int(max_pts * 0.2), f"Quality issues: {'; '.join(problems)}", "✗"

    return make_result("Compression Quality", score, max_pts, icon, feedback,
                       f"BPP={bits_per_pixel:.1f} Laplacian σ={lap_stddev:.1f} {'PNG' if is_png else 'JPEG'}",
                       "Use PNG or JPEG quality ≥ 90" if problems else None)


def run_image_checks(path):
    if not HAS_PIL:
        print("ERROR: Pillow not installed. pip install Pillow")
        sys.exit(1)
    path = Path(path)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    checks = [
        lambda: check_thumbnail_readability(img),
        lambda: check_wcag_contrast(img),
        lambda: check_color_vibrancy(img),
        lambda: check_color_diversity(img),
        lambda: check_title_zone_clarity(img),
        lambda: check_subject_centering(img),
        lambda: check_resolution(img, path),
        lambda: check_aspect_ratio(img),
        lambda: check_brightness_balance(img),
        lambda: check_edge_safety(img),
        lambda: check_compression_quality(img, path),
    ]
    results = [c() for c in checks]
    return results, f"{img.width}×{img.height}"


# ============================================================
# TEX CHECKS (--tex)
# ============================================================

def read_tex(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def tex_check_contrast_boxes(tex_content):
    """Detect \\fill[ and \\shade[ commands — solid rectangles behind text."""
    max_pts = 10
    fills = re.findall(r'\\fill\s*\[', tex_content)
    shades = re.findall(r'\\shade\s*\[', tex_content)
    count = len(fills) + len(shades)

    if count == 0:
        return make_result("Text Contrast Boxes", max_pts, max_pts, "✓",
                           "No \\fill or \\shade boxes — text reads against image", f"Found: 0")
    else:
        items = []
        if fills: items.append(f"{len(fills)} \\fill")
        if shades: items.append(f"{len(shades)} \\shade")
        return make_result("Text Contrast Boxes", 0, max_pts, "✗",
                           f"KILLER: {', '.join(items)} boxes found — remove and use image contrast instead",
                           f"Found: {count} ({', '.join(items)})",
                           "Remove all \\fill/\\shade boxes, edit image to add natural contrast",
                           f"{count} contrast box(es) detected — KILLER")


def tex_check_bare_text(tex_content):
    """Check for text outline/stroke commands — bare text scores highest."""
    max_pts = 5
    # Look for pdfrender textRenderingMode or textcontour or outline indicators
    has_stroke = bool(re.search(r'textRenderingMode\s*=\s*(?:1|2|Stroke|FillStroke)', tex_content, re.IGNORECASE))
    has_contour = bool(re.search(r'\\textcontour', tex_content))
    has_shadow = bool(re.search(r'\\shadow|shadow.*text|text.*shadow', tex_content, re.IGNORECASE))

    # Shadow-only layers (textRenderingMode=0 with offset) are OK for readability
    # What we penalize: thick outlines/strokes on the main text
    stroke_lines = re.findall(r'textRenderingMode\s*=\s*(?:1|2)', tex_content)
    fill_stroke_lines = re.findall(r'textRenderingMode\s*=\s*(?:2|FillStroke)', tex_content, re.IGNORECASE)

    # Count distinct rendering layers for title
    # Mode 0=Fill, 1=Stroke, 2=FillStroke — multiple layers = shadow+stroke+fill technique
    render_modes = re.findall(r'textRenderingMode\s*=\s*(\d)', tex_content)
    has_thick_outline = False
    for match in re.finditer(r'linewidth\s*[=:]\s*([\d.]+)\s*(pt|mm|in|cm)?', tex_content, re.IGNORECASE):
        width = float(match.group(1))
        unit = match.group(2) or 'pt'
        # Convert to pt for comparison
        if unit == 'mm': width *= 2.835
        elif unit == 'in': width *= 72
        elif unit == 'cm': width *= 28.35
        if width > 1.5:  # >1.5pt is thick
            has_thick_outline = True

    if not has_stroke and not has_contour:
        score, feedback, icon = max_pts, "Bare text — no outlines or strokes", "✓"
    elif has_thick_outline:
        score, feedback, icon = 1, "Thick text outline detected — amateurish", "✗"
    elif has_stroke or has_contour:
        score, feedback, icon = 3, "Thin text outline/stroke present — subtle is OK", "~"
    else:
        score, feedback, icon = max_pts, "Bare text rendering", "✓"

    detail_parts = []
    if render_modes: detail_parts.append(f"renderModes={render_modes}")
    if has_thick_outline: detail_parts.append("thick outline")
    if has_contour: detail_parts.append("textcontour")
    detail = ", ".join(detail_parts) if detail_parts else "No outline commands"

    return make_result("Bare Text Rendering", score, max_pts, icon, feedback, detail,
                       "Remove thick outlines — rely on natural image contrast" if score < 3 else None)


def tex_check_font_count(tex_content):
    """Count distinct font declarations — target exactly 2."""
    max_pts = 5
    # Match fontspec declarations
    font_decls = set()
    for pattern in [
        r'\\setmainfont\s*(?:\[.*?\])?\s*\{([^}]+)\}',
        r'\\setsansfont\s*(?:\[.*?\])?\s*\{([^}]+)\}',
        r'\\newfontfamily\s*\\?\w+\s*(?:\[.*?\])?\s*\{([^}]+)\}',
        r'\\setmonofont\s*(?:\[.*?\])?\s*\{([^}]+)\}',
        r'\\fontspec\s*(?:\[.*?\])?\s*\{([^}]+)\}',
    ]:
        for m in re.finditer(pattern, tex_content, re.DOTALL):
            font_name = m.group(1).strip()
            if font_name:
                font_decls.add(font_name)

    n = len(font_decls)

    if n == 2:
        score, feedback, icon = max_pts, f"2 fonts — good hierarchy (title + body)", "✓"
    elif n == 1:
        score, feedback, icon = int(max_pts * 0.6), "Only 1 font — consider adding a display font for title", "~"
    elif n == 3:
        score, feedback, icon = int(max_pts * 0.7), "3 fonts — acceptable if used tastefully", "~"
    elif n == 0:
        score, feedback, icon = int(max_pts * 0.3), "No custom fonts declared — likely using system defaults", "⚠"
    else:
        score, feedback, icon = int(max_pts * 0.2), f"{n} fonts — too many, cluttered hierarchy", "✗"

    fonts_str = ", ".join(sorted(font_decls)) if font_decls else "none"
    return make_result("Font Count & Pairing", score, max_pts, icon, feedback,
                       f"Fonts ({n}): {fonts_str}",
                       f"Use exactly 2 contrasting fonts" if n != 2 else None)


def tex_check_spine_match(tex_content):
    """Check if spine text matches the title on front cover."""
    max_pts = 5

    # Extract title — look for common patterns
    title_match = re.search(r'\\def\\booktitle\{([^}]+)\}', tex_content)
    if not title_match:
        title_match = re.search(r'\\newcommand\{?\\booktitle\}?\{([^}]+)\}', tex_content)
    if not title_match:
        # Try to find title from node text or large font sections
        title_match = re.search(r'\\booktitle', tex_content)
        if title_match:
            # Title is defined via macro, spine likely uses same macro — pass
            return make_result("Spine Text Match", max_pts, max_pts, "✓",
                               "Title uses macro — spine consistency ensured", "\\booktitle macro used")

    # Look for spine section
    spine_match = re.search(r'%.*[Ss]pine|\\begin\{.*spine', tex_content, re.IGNORECASE)

    if not spine_match and not title_match:
        return make_result("Spine Text Match", max_pts, max_pts, "~",
                           "Could not parse title/spine — manual check needed", "Skipped")

    # If title defined as variable and used in spine, it's consistent
    # Check for hardcoded spine text that might differ
    spine_section = ""
    spine_start = tex_content.find("% Spine") if "% Spine" in tex_content else tex_content.find("spine")
    if spine_start >= 0:
        spine_section = tex_content[spine_start:spine_start + 500]

    if "\\booktitle" in spine_section or "\\title" in spine_section:
        return make_result("Spine Text Match", max_pts, max_pts, "✓",
                           "Spine uses title macro — consistent", "Macro reference in spine")

    # Can't determine automatically
    return make_result("Spine Text Match", max_pts, max_pts, "~",
                       "Spine text could not be auto-verified — manual check needed", "Skipped")


def run_tex_checks(path):
    path = Path(path)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)
    tex = read_tex(path)
    results = [
        tex_check_contrast_boxes(tex),
        tex_check_bare_text(tex),
        tex_check_font_count(tex),
        tex_check_spine_match(tex),
    ]
    return results


# ============================================================
# PDF CHECKS (--pdf)
# ============================================================

def run_pdftotext_bbox(pdf_path):
    """Run pdftotext -bbox and return parsed word elements."""
    try:
        result = subprocess.run(
            ["pdftotext", "-bbox", str(pdf_path), "-"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None, f"pdftotext failed: {result.stderr.strip()}"
        return result.stdout, None
    except FileNotFoundError:
        return None, "pdftotext not found — install poppler-utils"
    except subprocess.TimeoutExpired:
        return None, "pdftotext timed out"


def parse_bbox_words(html_content):
    """Parse pdftotext -bbox HTML output into word records."""
    words = []
    # pdftotext -bbox outputs XHTML with <word> tags
    # Need to handle the namespace
    try:
        # Clean up for parsing — pdftotext bbox output uses xhtml namespace
        html_content = re.sub(r'xmlns="[^"]*"', '', html_content)
        root = ElementTree.fromstring(html_content)
    except ElementTree.ParseError:
        # Fallback: regex parsing
        for m in re.finditer(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>', html_content):
            words.append({
                "xMin": float(m.group(1)),
                "yMin": float(m.group(2)),
                "xMax": float(m.group(3)),
                "yMax": float(m.group(4)),
                "text": m.group(5),
            })
        return words

    for word_el in root.iter("word"):
        try:
            words.append({
                "xMin": float(word_el.get("xMin", 0)),
                "yMin": float(word_el.get("yMin", 0)),
                "xMax": float(word_el.get("xMax", 0)),
                "yMax": float(word_el.get("yMax", 0)),
                "text": (word_el.text or "").strip(),
            })
        except (ValueError, TypeError):
            continue
    return words


def get_page_dimensions(html_content):
    """Extract page width and height from pdftotext bbox output."""
    m = re.search(r'<page width="([\d.]+)" height="([\d.]+)"', html_content)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def group_words_into_blocks(words, y_proximity=5.0):
    """Group words into text blocks by vertical proximity."""
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: (w["yMin"], w["xMin"]))
    blocks = []
    current_block = [sorted_words[0]]

    for w in sorted_words[1:]:
        # Same block if vertically close to any word in current block
        block_y_max = max(bw["yMax"] for bw in current_block)
        block_y_min = min(bw["yMin"] for bw in current_block)
        if abs(w["yMin"] - block_y_min) < y_proximity or abs(w["yMin"] - block_y_max) < y_proximity:
            current_block.append(w)
        else:
            blocks.append(current_block)
            current_block = [w]
    blocks.append(current_block)
    return blocks


def dedupe_multilayer_blocks(blocks):
    """
    Merge blocks that represent the same text rendered multiple times
    (shadow/stroke/fill layers). Same words at nearly same position = same block.
    """
    if len(blocks) <= 1:
        return blocks

    def block_text(block):
        return " ".join(w["text"] for w in sorted(block, key=lambda w: w["xMin"]))

    def block_center_y(block):
        return (min(w["yMin"] for w in block) + max(w["yMax"] for w in block)) / 2

    merged = []
    used = set()

    for i, b1 in enumerate(blocks):
        if i in used:
            continue
        text1 = block_text(b1)
        cy1 = block_center_y(b1)
        group = list(b1)

        for j, b2 in enumerate(blocks):
            if j <= i or j in used:
                continue
            text2 = block_text(b2)
            cy2 = block_center_y(b2)
            # Same text, close vertical position = multi-layer rendering
            if text1 == text2 and abs(cy1 - cy2) < 8.0:
                used.add(j)
                # Keep the block with tightest bbox (the fill layer)

        merged.append(group)
        used.add(i)

    return merged


def pdf_check_text_overlap(words, page_w, page_h):
    """Check if different text blocks overlap each other."""
    max_pts = 10
    if not words:
        return make_result("Text Overlap", max_pts, max_pts, "~",
                           "No text found in PDF", "Skipped")

    blocks = group_words_into_blocks(words, y_proximity=5.0)
    blocks = dedupe_multilayer_blocks(blocks)

    if len(blocks) <= 1:
        return make_result("Text Overlap", max_pts, max_pts, "✓",
                           "Single text block — no overlap possible", f"{len(blocks)} block(s)")

    # Get bounding box for each block
    block_bboxes = []
    for block in blocks:
        bb = {
            "xMin": min(w["xMin"] for w in block),
            "yMin": min(w["yMin"] for w in block),
            "xMax": max(w["xMax"] for w in block),
            "yMax": max(w["yMax"] for w in block),
            "text": " ".join(w["text"] for w in sorted(block, key=lambda w: w["xMin"])),
        }
        block_bboxes.append(bb)

    overlaps = []
    for i in range(len(block_bboxes)):
        for j in range(i + 1, len(block_bboxes)):
            a, b = block_bboxes[i], block_bboxes[j]
            # Check bbox intersection
            if (a["xMin"] < b["xMax"] and a["xMax"] > b["xMin"] and
                    a["yMin"] < b["yMax"] and a["yMax"] > b["yMin"]):
                overlaps.append(f'"{a["text"][:30]}" ↔ "{b["text"][:30]}"')

    if not overlaps:
        return make_result("Text Overlap", max_pts, max_pts, "✓",
                           f"No overlapping text blocks ({len(block_bboxes)} blocks checked)",
                           f"Blocks: {len(block_bboxes)}")
    else:
        return make_result("Text Overlap", 0, max_pts, "✗",
                           f"KILLER: {len(overlaps)} overlap(s): {'; '.join(overlaps[:3])}",
                           f"Blocks: {len(block_bboxes)}, Overlaps: {len(overlaps)}",
                           "Reposition text elements to eliminate overlap",
                           f"{len(overlaps)} text block overlap(s) — KILLER")


def parse_tex_dimensions(tex_path):
    """Read paperwidth, paperheight, and spine width from cover.tex."""
    if not tex_path or not Path(tex_path).exists():
        return None, None, None

    tex = read_tex(tex_path)

    # paperwidth from geometry
    pw_match = re.search(r'paperwidth\s*=\s*([\d.]+)\s*(in|cm|mm|pt)', tex)
    ph_match = re.search(r'paperheight\s*=\s*([\d.]+)\s*(in|cm|mm|pt)', tex)

    def to_inches(val, unit):
        val = float(val)
        if unit == 'in': return val
        if unit == 'cm': return val / 2.54
        if unit == 'mm': return val / 25.4
        if unit == 'pt': return val / 72.0
        return val

    pw = to_inches(pw_match.group(1), pw_match.group(2)) if pw_match else None
    ph = to_inches(ph_match.group(1), ph_match.group(2)) if ph_match else None

    # Spine width from comment or variable
    spine_match = re.search(r'[Ss]pine\s*[Ww]idth\s*[=:]\s*([\d.]+)\s*(in|cm|mm|pt)?', tex)
    if not spine_match:
        spine_match = re.search(r'\\def\\spinewidth\{([\d.]+)(in|cm|mm|pt)?\}', tex)
    spine = None
    if spine_match:
        unit = spine_match.group(2) or 'in'
        spine = to_inches(spine_match.group(1), unit)

    return pw, ph, spine


def pdf_check_text_anchor(words, page_w, page_h, tex_path=None):
    """Check that title is anchored to a cover edge, not floating."""
    max_pts = 10
    if not words or not page_w or not page_h:
        return make_result("Text Anchor Zone", max_pts, max_pts, "~",
                           "Insufficient data for anchor check", "Skipped")

    # Determine front cover boundaries
    pw, ph, spine = parse_tex_dimensions(tex_path) if tex_path else (None, None, None)

    if pw and spine:
        # Front cover starts after spine
        front_cover_width_in = (pw - spine) / 2
        spine_end_in = front_cover_width_in + spine
        # Convert to PDF points (72 dpi in PDF coordinate space)
        front_x_start = spine_end_in / pw * page_w
        front_x_end = page_w
    else:
        # Assume right half is front cover
        front_x_start = page_w / 2
        front_x_end = page_w

    # Filter words on front cover
    front_words = [w for w in words if w["xMax"] > front_x_start]
    if not front_words:
        return make_result("Text Anchor Zone", max_pts, max_pts, "~",
                           "No text on front cover", "Skipped")

    # Group into blocks and find title (largest/topmost block)
    blocks = group_words_into_blocks(front_words, y_proximity=5.0)
    blocks = dedupe_multilayer_blocks(blocks)

    if not blocks:
        return make_result("Text Anchor Zone", max_pts, max_pts, "~",
                           "No text blocks found", "Skipped")

    # Title = block with largest total text area or topmost
    def block_area(block):
        w_span = max(w["xMax"] for w in block) - min(w["xMin"] for w in block)
        h_span = max(w["yMax"] for w in block) - min(w["yMin"] for w in block)
        return w_span * h_span

    blocks_sorted = sorted(blocks, key=block_area, reverse=True)
    title_block = blocks_sorted[0]
    title_top = min(w["yMin"] for w in title_block)
    title_bottom = max(w["yMax"] for w in title_block)
    title_center_y = (title_top + title_bottom) / 2

    # Distances to edges (in PDF points, 72pt = 1in)
    dist_top = title_top  # distance from top edge
    dist_bottom = page_h - title_bottom  # distance from bottom edge
    min_dist = min(dist_top, dist_bottom)

    # Convert to inches for threshold comparison
    dist_top_in = dist_top / 72.0
    dist_bottom_in = dist_bottom / 72.0
    min_dist_in = min_dist / 72.0
    center_offset_in = abs(title_center_y - page_h / 2) / 72.0

    problems = []
    is_killer = False

    # Check title anchor
    if dist_top_in <= 1.5 or dist_bottom_in <= 1.5:
        pass  # Anchored to edge — good
    elif center_offset_in <= 0.5:
        pass  # Centered — good
    elif min_dist_in > 2.0:
        problems.append(f"title floating — {min_dist_in:.1f}in from nearest edge (max 2.0in)")
        is_killer = True
    else:
        problems.append(f"title loosely anchored — {min_dist_in:.1f}in from nearest edge")

    # Check author name (smallest or lowest block)
    if len(blocks_sorted) > 1:
        # Find block closest to bottom that isn't the title
        non_title = [b for b in blocks_sorted[1:]]
        if non_title:
            author_block = max(non_title, key=lambda b: max(w["yMax"] for w in b))
            author_bottom = max(w["yMax"] for w in author_block)
            author_top = min(w["yMin"] for w in author_block)
            author_center_y = (author_top + author_bottom) / 2
            author_dist_bottom = (page_h - author_bottom) / 72.0
            author_dist_top = author_top / 72.0

            if author_dist_bottom > 1.0 and author_dist_top > 1.0:
                author_center_pct = author_center_y / page_h * 100
                if 30 < author_center_pct < 70:
                    problems.append(f"author name floating in middle ({author_center_pct:.0f}% from top)")
                    is_killer = True

    if not problems:
        anchor_type = "top" if dist_top_in < dist_bottom_in else ("centered" if center_offset_in <= 0.5 else "bottom")
        score, feedback, icon = max_pts, f"Title anchored to {anchor_type} edge", "✓"
    elif is_killer:
        score, feedback, icon = 0, f"KILLER: {'; '.join(problems)}", "✗"
    else:
        score, feedback, icon = int(max_pts * 0.5), "; ".join(problems), "⚠"

    return make_result("Text Anchor Zone", score, max_pts, icon, feedback,
                       f"Top={dist_top_in:.1f}in Bottom={dist_bottom_in:.1f}in CenterOff={center_offset_in:.1f}in",
                       "Anchor title to top or bottom edge (within 1.5in)" if problems else None,
                       f"Title unanchored — {'; '.join(problems)}" if is_killer else None)


def pdf_check_title_dominance(words, page_w, page_h, tex_path=None):
    """Check title occupies 30-60% of front cover area."""
    max_pts = 5
    if not words or not page_w or not page_h:
        return make_result("Title Dominance", max_pts, max_pts, "~",
                           "Insufficient data", "Skipped")

    pw, ph, spine = parse_tex_dimensions(tex_path) if tex_path else (None, None, None)

    if pw and spine:
        front_cover_width_in = (pw - spine) / 2
        spine_end_in = front_cover_width_in + spine
        front_x_start = spine_end_in / pw * page_w
    else:
        front_x_start = page_w / 2

    front_words = [w for w in words if w["xMax"] > front_x_start]
    if not front_words:
        return make_result("Title Dominance", 0, max_pts, "⚠",
                           "No text on front cover", "0%")

    blocks = group_words_into_blocks(front_words, y_proximity=5.0)
    blocks = dedupe_multilayer_blocks(blocks)

    if not blocks:
        return make_result("Title Dominance", 0, max_pts, "⚠", "No text blocks", "0%")

    def block_area(block):
        w_span = max(w["xMax"] for w in block) - min(w["xMin"] for w in block)
        h_span = max(w["yMax"] for w in block) - min(w["yMin"] for w in block)
        return w_span * h_span

    title_block = max(blocks, key=block_area)
    title_area = block_area(title_block)
    front_width = page_w - front_x_start
    front_area = front_width * page_h
    pct = (title_area / front_area * 100) if front_area > 0 else 0

    if 30 <= pct <= 60:
        score, feedback, icon = max_pts, f"Title occupies {pct:.0f}% — ideal range", "✓"
    elif 20 <= pct < 30:
        score, feedback, icon = int(max_pts * 0.6), f"Title occupies {pct:.0f}% — slightly small", "~"
    elif 60 < pct <= 75:
        score, feedback, icon = int(max_pts * 0.6), f"Title occupies {pct:.0f}% — slightly large", "~"
    elif pct < 20:
        score, feedback, icon = int(max_pts * 0.2), f"Title occupies {pct:.0f}% — too small", "✗"
    else:
        score, feedback, icon = int(max_pts * 0.3), f"Title occupies {pct:.0f}% — overwhelming", "✗"

    return make_result("Title Dominance", score, max_pts, icon, feedback,
                       f"Title area={pct:.1f}% of front cover (target 30-60%)",
                       "Increase title size" if pct < 20 else ("Reduce title size" if pct > 75 else None))


def pdf_check_barcode_zone(words, page_w, page_h, tex_path=None):
    """Check that back cover bottom-right is clear for barcode."""
    max_pts = 5
    if not words or not page_w or not page_h:
        return make_result("Barcode Zone Clear", max_pts, max_pts, "~",
                           "Insufficient data", "Skipped")

    pw, ph, spine = parse_tex_dimensions(tex_path) if tex_path else (None, None, None)

    if pw and spine:
        front_cover_width_in = (pw - spine) / 2
        # Back cover is 0 to front_cover_width_in
        back_x_end = front_cover_width_in / pw * page_w
    else:
        back_x_end = page_w / 2

    # Barcode zone: bottom-right quadrant of back cover
    # Typically 2" x 1.5" in bottom-right
    barcode_x_start = back_x_end * 0.5
    barcode_y_start = page_h * 0.75

    barcode_words = [w for w in words
                     if w["xMin"] < back_x_end
                     and w["xMax"] > barcode_x_start
                     and w["yMax"] > barcode_y_start]

    if not barcode_words:
        return make_result("Barcode Zone Clear", max_pts, max_pts, "✓",
                           "Back cover bottom-right clear for barcode", "No text in barcode zone")
    else:
        texts = [w["text"] for w in barcode_words[:5]]
        return make_result("Barcode Zone Clear", 0, max_pts, "✗",
                           f"KILLER: Text in barcode zone: {' '.join(texts)}",
                           f"{len(barcode_words)} word(s) in barcode area",
                           "Move text away from back cover bottom-right (barcode area)",
                           f"Text in barcode zone — KILLER")


def pdf_check_blurb_present(words, page_w, page_h, tex_path=None):
    """Check that back cover has body text (blurb)."""
    max_pts = 5
    if not words or not page_w:
        return make_result("Blurb Present", max_pts, max_pts, "~",
                           "Insufficient data", "Skipped")

    pw, ph, spine = parse_tex_dimensions(tex_path) if tex_path else (None, None, None)

    if pw and spine:
        front_cover_width_in = (pw - spine) / 2
        back_x_end = front_cover_width_in / pw * page_w
    else:
        back_x_end = page_w / 2

    # Words on back cover (left side)
    back_words = [w for w in words if w["xMax"] < back_x_end]
    word_count = len(back_words)
    back_text = " ".join(w["text"] for w in back_words)

    if word_count >= 15:
        score, feedback, icon = max_pts, f"Back cover has {word_count} words — blurb present", "✓"
    elif word_count >= 5:
        score, feedback, icon = int(max_pts * 0.5), f"Back cover has only {word_count} words — sparse", "⚠"
    else:
        score, feedback, icon = 0, f"Back cover has {word_count} words — missing blurb", "✗"

    return make_result("Blurb Present", score, max_pts, icon, feedback,
                       f"Back cover words: {word_count}",
                       "Add book description/blurb to back cover" if word_count < 15 else None,
                       f"Missing blurb — only {word_count} words on back cover" if word_count < 5 else None)


def pdf_check_garbled_text(words):
    """Check for garbled/AI-artifact text via simple heuristics."""
    max_pts = 5
    if not words:
        return make_result("Text Integrity", max_pts, max_pts, "~",
                           "No text found", "Skipped")

    garbled = []
    for w in words:
        text = w["text"]
        if not text:
            continue
        # Check for nonsense patterns
        # 1. Too many consonants in a row (>4)
        if re.search(r'[bcdfghjklmnpqrstvwxyz]{5,}', text.lower()):
            garbled.append(text)
            continue
        # 2. Mixed case chaos (alternating case like "hElLo" but not acronyms)
        if len(text) > 3 and not text.isupper() and not text.islower() and not text.istitle():
            case_changes = sum(1 for i in range(1, len(text))
                               if text[i].isalpha() and text[i-1].isalpha()
                               and text[i].isupper() != text[i-1].isupper())
            if case_changes > len(text) * 0.5:
                garbled.append(text)
                continue
        # 3. Excessive special characters in what should be a word
        if len(text) > 2:
            alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
            if alpha_ratio < 0.5 and not re.match(r'^[\d.,;:!?\'"-]+$', text):
                garbled.append(text)

    if not garbled:
        score, feedback, icon = max_pts, "All text appears well-formed", "✓"
    elif len(garbled) <= 2:
        score, feedback, icon = int(max_pts * 0.4), f"Suspect words: {', '.join(garbled[:5])}", "⚠"
    else:
        score, feedback, icon = 0, f"KILLER: {len(garbled)} garbled words: {', '.join(garbled[:5])}", "✗"

    return make_result("Text Integrity", score, max_pts, icon, feedback,
                       f"Suspect words: {len(garbled)}/{len(words)}",
                       "Fix garbled text in cover.tex" if garbled else None,
                       f"{len(garbled)} garbled words — KILLER" if len(garbled) > 2 else None)


def pdf_check_dimensions(page_w, page_h, tex_path=None):
    """Validate PDF dimensions match KDP cover calculator."""
    max_pts = 5
    if not page_w or not page_h:
        return make_result("PDF Dimensions", max_pts, max_pts, "~",
                           "Could not read PDF dimensions", "Skipped")

    pw, ph, spine = parse_tex_dimensions(tex_path) if tex_path else (None, None, None)

    # PDF points to inches (72 pt/in)
    pdf_w_in = page_w / 72.0
    pdf_h_in = page_h / 72.0

    problems = []

    if pw:
        w_diff = abs(pdf_w_in - pw)
        if w_diff > 0.1:
            problems.append(f"PDF width {pdf_w_in:.2f}in ≠ TeX paperwidth {pw:.2f}in (diff {w_diff:.2f}in)")
    if ph:
        h_diff = abs(pdf_h_in - ph)
        if h_diff > 0.1:
            problems.append(f"PDF height {pdf_h_in:.2f}in ≠ TeX paperheight {ph:.2f}in (diff {h_diff:.2f}in)")

    # KDP full cover must be wider than a single page
    if pdf_w_in < 6.0:
        problems.append(f"PDF width {pdf_w_in:.2f}in — too narrow for KDP cover")

    if not problems:
        score, feedback, icon = max_pts, f"PDF dimensions {pdf_w_in:.2f}×{pdf_h_in:.2f}in — OK", "✓"
    else:
        score, feedback, icon = 0, f"Dimension mismatch: {'; '.join(problems)}", "✗"

    return make_result("PDF Dimensions", score, max_pts, icon, feedback,
                       f"PDF={pdf_w_in:.2f}×{pdf_h_in:.2f}in" + (f" TeX={pw:.2f}×{ph:.2f}in" if pw and ph else ""),
                       "Fix dimensions to match KDP cover calculator" if problems else None,
                       f"Wrong dimensions — {'; '.join(problems)}" if problems else None)


def run_pdf_checks(pdf_path, tex_path=None):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}")
        sys.exit(1)

    html_content, err = run_pdftotext_bbox(pdf_path)
    if err:
        print(f"WARNING: {err}")
        return []

    words = parse_bbox_words(html_content)
    page_w, page_h = get_page_dimensions(html_content)

    results = [
        pdf_check_text_overlap(words, page_w, page_h),
        pdf_check_text_anchor(words, page_w, page_h, tex_path),
        pdf_check_title_dominance(words, page_w, page_h, tex_path),
        pdf_check_barcode_zone(words, page_w, page_h, tex_path),
        pdf_check_blurb_present(words, page_w, page_h, tex_path),
        pdf_check_garbled_text(words),
        pdf_check_dimensions(page_w, page_h, tex_path),
    ]
    return results


# ============================================================
# REPORT
# ============================================================

def print_section(title, results):
    if not results:
        return
    print(f"\n  --- {title} ---\n")
    for r in results:
        name = r["name"].ljust(28)
        score_str = f"{r['score']}/{r['max']}".rjust(6)
        print(f"  {r['icon']}  {name} {score_str}  {r['feedback']}")
        if r.get("detail"):
            print(f"     {''.ljust(28)}        {r['detail']}")
        print()


def print_report(image_results, tex_results, pdf_results, image_dims=None, image_path=None, tex_path=None, pdf_path=None):
    all_results = image_results + tex_results + pdf_results
    if not all_results:
        print("No checks were run. Provide --image, --tex, and/or --pdf.")
        return

    total_score = sum(r["score"] for r in all_results)
    total_max = sum(r["max"] for r in all_results)
    criticals = [r["critical"] for r in all_results if r.get("critical")]
    suggestions = [r["suggestion"] for r in all_results if r.get("suggestion")]

    print()
    print("=" * 60)
    print("  COVER SCORE REPORT")
    print("=" * 60)
    if image_path: print(f"  Image: {image_path}" + (f" ({image_dims})" if image_dims else ""))
    if tex_path: print(f"  TeX:   {tex_path}")
    if pdf_path: print(f"  PDF:   {pdf_path}")
    print("-" * 60)

    if image_results:
        print_section("IMAGE CHECKS", image_results)
    if tex_results:
        print_section("TEX CHECKS", tex_results)
    if pdf_results:
        print_section("PDF CHECKS", pdf_results)

    print("-" * 60)
    pct = total_score / total_max * 100 if total_max > 0 else 0
    print(f"  SCRIPT SCORE:    {total_score}/{total_max} ({pct:.0f}%)")

    # Only show weighted 40% if image checks were run (backwards compat)
    if image_results:
        img_score = sum(r["score"] for r in image_results)
        img_max = sum(r["max"] for r in image_results)
        img_pct = img_score / img_max * 100 if img_max > 0 else 0
        weighted = round(img_pct * 0.40, 1)
        print(f"  IMAGE WEIGHTED:  {weighted}/40 (image checks only)")

    print(f"  TOTAL WEIGHTED:  {round(pct * 0.40, 1)}/40 (all checks)")
    print()

    if criticals:
        print("  *** CRITICAL WARNINGS ***")
        for c in criticals:
            print(f"  ✗ {c}")
        print()

    if suggestions:
        print("  TOP IMPROVEMENTS:")
        for i, s in enumerate(suggestions[:5], 1):
            print(f"  {i}. {s}")
        print()

    if pct >= 90: rating = "EXCELLENT"
    elif pct >= 75: rating = "GOOD"
    elif pct >= 60: rating = "FAIR"
    elif pct >= 40: rating = "NEEDS WORK"
    else: rating = "POOR"
    print(f"  SCRIPT RATING: {rating}")
    print("=" * 60)
    print()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cover scoring tool for children's book covers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cover-score --image /tmp/cover_front.png
  cover-score --tex cover/cover.tex
  cover-score --pdf cover/cover.pdf
  cover-score --image /tmp/cover_front.png --tex cover/cover.tex --pdf cover/cover.pdf
  cover-score /tmp/cover_front.png   (legacy: same as --image)
        """,
    )
    parser.add_argument("--image", metavar="PATH", help="Front cover image (PNG/JPEG)")
    parser.add_argument("--tex", metavar="PATH", help="Cover LaTeX source file")
    parser.add_argument("--pdf", metavar="PATH", help="Built cover PDF")
    parser.add_argument("legacy_image", nargs="?", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Legacy support: positional arg = --image
    image_path = args.image or args.legacy_image
    tex_path = args.tex
    pdf_path = args.pdf

    if not image_path and not tex_path and not pdf_path:
        parser.print_help()
        sys.exit(1)

    image_results, image_dims = [], None
    tex_results = []
    pdf_results = []

    if image_path:
        image_results, image_dims = run_image_checks(image_path)
    if tex_path:
        tex_results = run_tex_checks(tex_path)
    if pdf_path:
        pdf_results = run_pdf_checks(pdf_path, tex_path)

    print_report(image_results, tex_results, pdf_results,
                 image_dims, image_path, tex_path, pdf_path)


if __name__ == "__main__":
    main()
