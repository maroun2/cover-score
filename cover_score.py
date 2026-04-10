#!/usr/bin/env python3
"""
Cover scoring tool for children's book covers.

Runs 10 programmatic checks on a cover image and outputs a score (0-100)
with per-check feedback.

Usage:
    cover-score <image_path>
    cover-score cover_front.png
"""

import sys
import os
from pathlib import Path

try:
    from PIL import Image, ImageStat, ImageFilter
except ImportError:
    print("ERROR: Pillow not installed.")
    print("  pip install Pillow")
    sys.exit(1)


# --- Constants ---

# KDP 6x9 cover at 300 DPI (with bleed)
KDP_WIDTH_PX = 1875   # 6.25" * 300
KDP_HEIGHT_PX = 2775  # 9.25" * 300
KDP_MIN_WIDTH = 1800
KDP_MIN_HEIGHT = 2700
KDP_ASPECT = 9.25 / 6.25  # ~1.48

# Amazon thumbnail width
THUMB_WIDTH = 160

# WCAG AA minimum contrast for large text (18pt+)
WCAG_AA_LARGE = 3.0
WCAG_AA_NORMAL = 4.5

# Bleed margin (0.125" at 300dpi = 37.5px, ~2% of dimension)
BLEED_MARGIN_PCT = 0.02


def load_image(path):
    """Load image from path, return PIL Image in RGB mode."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def relative_luminance(r, g, b):
    """WCAG relative luminance from sRGB values (0-255)."""
    def linearize(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)


def contrast_ratio(lum1, lum2):
    """WCAG contrast ratio between two luminance values."""
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    return (lighter + 0.05) / (darker + 0.05)


def get_dominant_colors(img, n_colors=8):
    """Extract dominant colors by quantizing image."""
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
    """Convert RGB (0-255) to HSV (H: 0-360, S: 0-1, V: 0-1)."""
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


# --- Checks ---

def check_thumbnail_readability(img):
    max_pts = 15
    ratio = img.width / THUMB_WIDTH
    thumb = img.resize((THUMB_WIDTH, int(img.height / ratio)), Image.LANCZOS)
    title_zone = thumb.crop((0, 0, thumb.width, thumb.height // 3))
    gray_zone = title_zone.convert("L")
    gray_stat = ImageStat.Stat(gray_zone)
    stddev = gray_stat.stddev[0]

    if stddev >= 70:
        score, feedback, icon = max_pts, "Strong contrast in title zone at thumbnail size", "✓"
    elif stddev >= 55:
        score, feedback, icon = int(max_pts * 0.8), "Decent title zone contrast at thumbnail — could be stronger", "~"
    elif stddev >= 40:
        score, feedback, icon = int(max_pts * 0.55), "Title zone contrast is low — may not read at Amazon thumbnail size", "⚠"
    elif stddev >= 25:
        score, feedback, icon = int(max_pts * 0.3), "Title zone barely visible at thumbnail size", "✗"
    else:
        score, feedback, icon = 0, "Title zone has almost no contrast — invisible at thumbnail size", "✗"

    return {
        "name": "Thumbnail Readability", "score": score, "max": max_pts, "icon": icon,
        "feedback": feedback, "detail": f"Stddev={stddev:.1f} (target ≥55)",
        "suggestion": "Add text outline, drop shadow, or darker area behind title" if score < max_pts * 0.7 else None,
    }


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
        score, feedback, icon = int(max_pts * 0.7), f"Contrast ratio {ratio:.1f}:1 — passes AA for large text only", "~"
    elif ratio >= 2.5:
        score, feedback, icon = int(max_pts * 0.4), f"Contrast ratio {ratio:.1f}:1 — below WCAG minimum", "⚠"
    else:
        score, feedback, icon = int(max_pts * 0.15), f"Contrast ratio {ratio:.1f}:1 — poor, title hard to read", "✗"

    critical = ratio < WCAG_AA_LARGE
    return {
        "name": "WCAG Contrast (Title Zone)", "score": score, "max": max_pts, "icon": icon,
        "feedback": feedback, "detail": f"Ratio={ratio:.2f}:1 (AA-large≥3.0, AA≥4.5)",
        "suggestion": "Increase text outline thickness or add semi-transparent background behind title" if score < max_pts * 0.7 else None,
        "critical": f"WCAG contrast {ratio:.1f}:1 in title zone — minimum 3.0:1 for large text" if critical else None,
    }


def check_color_vibrancy(img):
    max_pts = 10
    colors = get_dominant_colors(img, 8)
    total_sat, total_weight = 0, 0
    for c in colors:
        h, s, v = rgb_to_hsv(*c["rgb"])
        total_sat += s * c["proportion"]
        total_weight += c["proportion"]
    avg_sat = total_sat / max(total_weight, 0.001)

    if avg_sat >= 0.45:
        score, feedback, icon = max_pts, "Vibrant, saturated palette — great for children's books", "✓"
    elif avg_sat >= 0.35:
        score, feedback, icon = int(max_pts * 0.8), "Good color saturation", "~"
    elif avg_sat >= 0.25:
        score, feedback, icon = int(max_pts * 0.5), "Colors somewhat muted — children's covers benefit from vibrancy", "⚠"
    elif avg_sat >= 0.15:
        score, feedback, icon = int(max_pts * 0.25), "Dull, desaturated palette — not appealing for ages 4-6", "✗"
    else:
        score, feedback, icon = 0, "Nearly grayscale — children's book covers need color", "✗"

    return {
        "name": "Color Vibrancy", "score": score, "max": max_pts, "icon": icon,
        "feedback": feedback, "detail": f"Avg saturation={avg_sat:.2f} (target ≥0.35)",
        "suggestion": "Increase color saturation, use warmer/brighter tones" if score < max_pts * 0.7 else None,
    }


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
        score, feedback, icon = max_pts, f"{n_hues} distinct hue groups — balanced palette", "✓"
    elif n_hues == 2 or n_hues == 6:
        score, feedback, icon = int(max_pts * 0.7), f"{n_hues} distinct hue groups — acceptable", "~"
    elif n_hues == 1:
        score, feedback, icon = int(max_pts * 0.4), "Monochromatic — consider adding accent color", "⚠"
    elif n_hues >= 7:
        score, feedback, icon = int(max_pts * 0.5), f"{n_hues} hue groups — palette may feel chaotic", "⚠"
    else:
        score, feedback, icon = int(max_pts * 0.3), "Very limited color range", "✗"

    return {
        "name": "Color Diversity", "score": score, "max": max_pts, "icon": icon,
        "feedback": feedback, "detail": f"Hue clusters={n_hues} (target 3-5)",
        "suggestion": "Add 1-2 complementary accent colors" if n_hues < 3 else ("Simplify palette to 3-5 dominant hues" if n_hues > 6 else None),
    }


def check_title_zone_clarity(img):
    max_pts = 10
    title_zone = img.crop((0, 0, img.width, img.height // 3))
    gray = title_zone.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_mean = ImageStat.Stat(edges).mean[0]

    if 15 <= edge_mean <= 40:
        score, feedback, icon = max_pts, "Title zone has good clarity — text against clean background", "✓"
    elif 10 <= edge_mean < 15:
        score, feedback, icon = int(max_pts * 0.7), "Title zone fairly clean", "~"
    elif 40 < edge_mean <= 55:
        score, feedback, icon = int(max_pts * 0.6), "Title zone somewhat busy — text may compete with illustration", "⚠"
    elif edge_mean < 10:
        score, feedback, icon = int(max_pts * 0.5), "Title zone very flat — verify text is actually present", "⚠"
    else:
        score, feedback, icon = int(max_pts * 0.3), "Title zone very busy — text likely hard to read against background", "✗"

    return {
        "name": "Title Zone Clarity", "score": score, "max": max_pts, "icon": icon,
        "feedback": feedback, "detail": f"Edge density={edge_mean:.1f} (target 15-40)",
        "suggestion": "Simplify illustration behind title text or add gradient overlay" if score < max_pts * 0.7 else None,
    }


def check_resolution(img, path):
    max_pts = 10
    w, h = img.size
    meets_height = h >= KDP_MIN_HEIGHT
    meets_width = w >= KDP_MIN_WIDTH

    if meets_width and meets_height:
        score, feedback, icon = max_pts, f"{w}×{h}px — meets KDP 300dpi requirement", "✓"
    elif w >= KDP_MIN_WIDTH * 0.85 and h >= KDP_MIN_HEIGHT * 0.85:
        score, feedback, icon = int(max_pts * 0.6), f"{w}×{h}px — slightly below KDP minimum, may be acceptable", "⚠"
    elif w >= 1200 and h >= 1800:
        score, feedback, icon = int(max_pts * 0.3), f"{w}×{h}px — low for print, fine for digital only", "⚠"
    else:
        score, feedback, icon = 0, f"{w}×{h}px — too low for print or digital", "✗"

    return {
        "name": "Image Resolution", "score": score, "max": max_pts, "icon": icon,
        "feedback": feedback, "detail": f"Actual={w}×{h}, KDP min={KDP_MIN_WIDTH}×{KDP_MIN_HEIGHT}",
        "suggestion": "Upscale image to at least 1875×2775px for KDP print" if score < max_pts else None,
    }


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
        score, feedback, icon = max_pts, "Aspect ratio matches expected cover format", "✓"
    elif best_diff < 0.1:
        score, feedback, icon = int(max_pts * 0.7), "Aspect ratio close to expected", "~"
    elif best_diff < 0.2:
        score, feedback, icon = int(max_pts * 0.4), "Aspect ratio slightly off — may need cropping", "⚠"
    else:
        score, feedback, icon = 0, f"Aspect ratio {actual_ratio:.2f} doesn't match any expected format", "✗"

    return {
        "name": "Aspect Ratio", "score": score, "max": max_pts, "icon": icon,
        "feedback": feedback, "detail": f"Actual={actual_ratio:.3f}, Front={KDP_ASPECT:.3f}, Wrap={wrap_ratio:.3f}",
        "suggestion": "Resize or crop to match KDP cover dimensions" if score < max_pts else None,
    }


def check_brightness_balance(img):
    max_pts = 10
    gray = img.convert("L")
    histogram = gray.histogram()
    total = sum(histogram)
    shadows = sum(histogram[0:64]) / total
    midtones = sum(histogram[64:192]) / total
    highlights = sum(histogram[192:256]) / total
    mean_brightness = sum(i * histogram[i] for i in range(256)) / total

    problems = []
    if shadows > 0.5: problems.append("too dark (>50% shadows)")
    if highlights > 0.5: problems.append("too bright (>50% highlights)")
    if midtones < 0.2: problems.append("lack of midtones — high contrast extremes")
    if mean_brightness < 60: problems.append(f"mean brightness {mean_brightness:.0f}/255 — very dark")
    if mean_brightness > 200: problems.append(f"mean brightness {mean_brightness:.0f}/255 — washed out")

    if not problems:
        score, feedback, icon = max_pts, "Good brightness balance across the image", "✓"
    elif len(problems) == 1:
        score, feedback, icon = int(max_pts * 0.6), f"Brightness issue: {problems[0]}", "⚠"
    else:
        score, feedback, icon = int(max_pts * 0.3), f"Brightness issues: {'; '.join(problems)}", "✗"

    return {
        "name": "Brightness Balance", "score": score, "max": max_pts, "icon": icon,
        "feedback": feedback, "detail": f"Shadows={shadows:.0%} Mid={midtones:.0%} High={highlights:.0%} Mean={mean_brightness:.0f}",
        "suggestion": "Adjust levels — brighten shadows or reduce overexposure" if problems else None,
    }


def check_edge_safety(img):
    max_pts = 5
    w, h = img.size
    margin_x = int(w * BLEED_MARGIN_PCT)
    margin_y = int(h * BLEED_MARGIN_PCT)

    if margin_x < 5 or margin_y < 5:
        return {"name": "Edge Bleed Safety", "score": max_pts, "max": max_pts, "icon": "~",
                "feedback": "Image too small to meaningfully check bleed margins", "detail": "Skipped", "suggestion": None}

    gray = img.convert("L")
    def strip_mean(box): return ImageStat.Stat(gray.crop(box)).mean[0]

    edge_diffs = []
    edge_diffs.append(abs(strip_mean((0, 0, w, margin_y)) - strip_mean((0, margin_y, w, margin_y * 3))))
    edge_diffs.append(abs(strip_mean((0, h - margin_y, w, h)) - strip_mean((0, h - margin_y * 3, w, h - margin_y))))
    edge_diffs.append(abs(strip_mean((0, 0, margin_x, h)) - strip_mean((margin_x, 0, margin_x * 3, h))))
    edge_diffs.append(abs(strip_mean((w - margin_x, 0, w, h)) - strip_mean((w - margin_x * 3, 0, w - margin_x, h))))
    max_diff = max(edge_diffs)

    if max_diff < 15:
        score, feedback, icon = max_pts, "Edges blend smoothly — safe for bleed trimming", "✓"
    elif max_diff < 30:
        score, feedback, icon = int(max_pts * 0.7), "Some content near edges — verify nothing critical gets trimmed", "~"
    else:
        score, feedback, icon = int(max_pts * 0.3), "Significant content near edges — may be cut during print trimming", "⚠"

    return {
        "name": "Edge Bleed Safety", "score": score, "max": max_pts, "icon": icon,
        "feedback": feedback, "detail": f"Max edge diff={max_diff:.1f} (target <15)",
        "suggestion": "Move critical elements away from outer 0.125\" margins" if score < max_pts else None,
    }


def check_compression_quality(img, path):
    max_pts = 10
    file_size = os.path.getsize(path)
    pixel_count = img.width * img.height
    bits_per_pixel = (file_size * 8) / max(pixel_count, 1)
    gray = img.convert("L")
    laplacian = gray.filter(ImageFilter.Kernel(size=(3, 3), kernel=[-1, -1, -1, -1, 8, -1, -1, -1, -1], scale=1, offset=128))
    lap_stddev = ImageStat.Stat(laplacian).stddev[0]
    is_png = str(path).lower().endswith(".png")

    problems = []
    if not is_png and bits_per_pixel < 1.0: problems.append(f"heavy compression ({bits_per_pixel:.1f} bpp)")
    if lap_stddev < 8 and not is_png: problems.append(f"low detail — possible artifact smoothing (laplacian σ={lap_stddev:.1f})")

    if is_png or not problems:
        score, feedback, icon = max_pts, "Good image quality" + (" (PNG — lossless)" if is_png else ""), "✓"
    elif len(problems) == 1:
        score, feedback, icon = int(max_pts * 0.5), f"Quality concern: {problems[0]}", "⚠"
    else:
        score, feedback, icon = int(max_pts * 0.2), f"Quality issues: {'; '.join(problems)}", "✗"

    return {
        "name": "Compression Quality", "score": score, "max": max_pts, "icon": icon,
        "feedback": feedback, "detail": f"BPP={bits_per_pixel:.1f} Laplacian σ={lap_stddev:.1f} Format={'PNG' if is_png else 'JPEG/other'}",
        "suggestion": "Use PNG or higher-quality JPEG (quality ≥ 90)" if problems else None,
    }


# --- Main ---

ALL_CHECKS = [
    (check_thumbnail_readability, False),
    (check_wcag_contrast, False),
    (check_color_vibrancy, False),
    (check_color_diversity, False),
    (check_title_zone_clarity, False),
    (check_resolution, True),
    (check_aspect_ratio, False),
    (check_brightness_balance, False),
    (check_edge_safety, False),
    (check_compression_quality, True),
]


def score_cover(path):
    path = Path(path)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    img = load_image(path)
    results = []
    criticals = []

    for check_fn, needs_path in ALL_CHECKS:
        result = check_fn(img, path) if needs_path else check_fn(img)
        results.append(result)
        if result.get("critical"):
            criticals.append(result["critical"])

    total_score = sum(r["score"] for r in results)
    total_max = sum(r["max"] for r in results)

    return {
        "file": str(path),
        "dimensions": f"{img.width}×{img.height}",
        "results": results,
        "script_score": total_score,
        "script_max": total_max,
        "weighted_40pct": round(total_score * 0.40, 1),
        "criticals": criticals,
        "suggestions": [r["suggestion"] for r in results if r.get("suggestion")],
    }


def print_report(report):
    print()
    print("=" * 60)
    print("  COVER SCORE REPORT")
    print("=" * 60)
    print(f"  File: {report['file']}")
    print(f"  Size: {report['dimensions']}")
    print("-" * 60)
    print()

    for r in report["results"]:
        name = r["name"].ljust(28)
        score_str = f"{r['score']}/{r['max']}".rjust(6)
        print(f"  {r['icon']}  {name} {score_str}  {r['feedback']}")
        if r.get("detail"):
            print(f"     {''.ljust(28)}        {r['detail']}")
        print()

    print("-" * 60)
    score = report["script_score"]
    mx = report["script_max"]
    pct = score / mx * 100 if mx > 0 else 0
    print(f"  SCRIPT SCORE:    {score}/{mx} ({pct:.0f}%)")
    print(f"  WEIGHTED (40%):  {report['weighted_40pct']}/40")
    print()

    if report["criticals"]:
        print("  *** CRITICAL WARNINGS ***")
        for c in report["criticals"]:
            print(f"  ✗ {c}")
        print()

    if report["suggestions"]:
        print("  TOP IMPROVEMENTS:")
        for i, s in enumerate(report["suggestions"][:5], 1):
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


def main():
    if len(sys.argv) < 2:
        print("Usage: cover-score <image_path>")
        sys.exit(1)
    report = score_cover(sys.argv[1])
    print_report(report)


if __name__ == "__main__":
    main()
