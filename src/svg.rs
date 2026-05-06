//! # svg.rs
//!
//! Pure-Rust SVG renderer for Mamdani fuzzy membership functions.
//! No external dependencies — plain string generation.
//!
//! Produces self-contained `.svg` files that open in any browser.
//! Features: dark background (Catppuccin Mocha), coloured curves,
//! clipped activation areas, μ-annotation dots, and a colour legend.

use crate::variable::FuzzyVariable;

// ─── Layout ──────────────────────────────────────────────────────

const W: f64 = 660.0;
const H: f64 = 380.0; // extra height for legend row
const ML: f64 = 62.0;
const MR: f64 = 20.0;
const MT: f64 = 48.0;
const MB: f64 = 56.0;
const LEG: f64 = 36.0; // legend strip height at bottom

const PW: f64 = W - ML - MR;
const PH: f64 = H - MT - MB - LEG; // plot height shrinks to make room

const SAMPLES: usize = 500;

// ─── Palette (Catppuccin Mocha) ───────────────────────────────────

const BG: &str = "#1e1e2e";
const SURFACE: &str = "#313244";
const OVERLAY: &str = "#585b70";
const MUTED: &str = "#6c7086";
const TEXT: &str = "#cdd6f4";
const SUBTEXT: &str = "#a6adc8";
const YELLOW: &str = "#f9e2af";

const PALETTE: &[&str] = &[
    "#f38ba8", // red
    "#a6e3a1", // green
    "#89b4fa", // blue
    "#cba6f7", // mauve
    "#94e2d5", // teal
    "#fab387", // peach
    "#74c7ec", // sapphire
    "#f9e2af", // yellow
];

// ─── Coordinate helpers ───────────────────────────────────────────

#[inline]
fn px(x: f64, min: f64, max: f64) -> f64 {
    ML + (x - min) / (max - min) * PW
}
#[inline]
fn py(d: f64) -> f64 {
    MT + PH - d.clamp(0.0, 1.0) * PH
}
#[inline]
fn fv(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 && v.abs() < 10_000.0 {
        format!("{:.0}", v)
    } else {
        format!("{:.1}", v)
    }
}

// ─── SVG element helpers ──────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn rect(s: &mut String, x: f64, y: f64, w: f64, h: f64, rx: f64, fill: &str, opacity: f64) {
    s.push_str(&format!(
        r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" rx="{}" fill="{}" fill-opacity="{:.2}"/>"#,
        x, y, w, h, rx, fill, opacity
    ));
}

#[allow(clippy::too_many_arguments)]
fn line(s: &mut String, x1: f64, y1: f64, x2: f64, y2: f64, stroke: &str, sw: f64, dash: &str) {
    let da = if dash.is_empty() {
        String::new()
    } else {
        format!(r#" stroke-dasharray="{}""#, dash)
    };
    s.push_str(&format!(
        r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{}" stroke-width="{}"{}/>"#,
        x1, y1, x2, y2, stroke, sw, da
    ));
}

#[allow(clippy::too_many_arguments)]
fn text(
    s: &mut String,
    x: f64,
    y: f64,
    anchor: &str,
    fill: &str,
    size: u32,
    bold: bool,
    content: &str,
) {
    let w = if bold { r#" font-weight="bold""# } else { "" };
    s.push_str(&format!(
        r#"<text x="{:.1}" y="{:.1}" text-anchor="{}" fill="{}" font-size="{}"{} >{}</text>"#,
        x, y, anchor, fill, size, w, content
    ));
}

fn circle(s: &mut String, cx: f64, cy: f64, r: f64, fill: &str) {
    s.push_str(&format!(
        r#"<circle cx="{:.1}" cy="{:.1}" r="{}" fill="{}"/>"#,
        cx, cy, r, fill
    ));
}

// ─── Grid and axes ────────────────────────────────────────────────

fn draw_grid_axes(s: &mut String, var: &FuzzyVariable) {
    let min = var.universe.min;
    let max = var.universe.max;

    // Horizontal grid + Y labels
    for &d in &[0.25_f64, 0.5, 0.75, 1.0] {
        let y = py(d);
        let col = if (d - 1.0).abs() < 1e-9 {
            OVERLAY
        } else {
            SURFACE
        };
        line(s, ML, y, ML + PW, y, col, 1.0, "");
        text(
            s,
            ML - 6.0,
            y + 3.5,
            "end",
            MUTED,
            9,
            false,
            &format!("{:.2}", d),
        );
    }
    text(s, ML - 6.0, py(0.0) + 3.5, "end", MUTED, 9, false, "0.00");

    // Y-axis label
    let cy = MT + PH / 2.0;
    s.push_str(&format!(
        r#"<text x="11" y="{:.1}" text-anchor="middle" fill="{}" font-size="10" transform="rotate(-90 11 {:.1})">&#956;(x)</text>"#,
        cy, MUTED, cy
    ));

    // Vertical grid + X labels
    for i in 0..=4 {
        let t = i as f64 / 4.0;
        let xv = min + (max - min) * t;
        let xp = px(xv, min, max);
        line(s, xp, MT, xp, MT + PH, SURFACE, 1.0, "3,3");
        text(s, xp, MT + PH + 14.0, "middle", MUTED, 9, false, &fv(xv));
    }

    // Plot border
    s.push_str(&format!(
        r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="none" stroke="{}" stroke-width="1"/>"#,
        ML, MT, PW, PH, OVERLAY
    ));
}

// ─── Legend strip ─────────────────────────────────────────────────

/// Draws a horizontal colour legend below the X-axis labels.
fn draw_legend(s: &mut String, var: &FuzzyVariable) {
    let n = var.term_count();
    if n == 0 {
        return;
    }

    // Legend bar background
    let ly = MT + PH + MB - 4.0; // top of legend strip
    rect(s, ML, ly, PW, LEG - 6.0, 4.0, SURFACE, 0.45);

    // Title
    text(s, ML + 8.0, ly + 14.0, "start", MUTED, 8, false, "Terms:");

    // Items — evenly spaced after the title
    let title_w = 84.0;
    let avail = PW - title_w;
    let item_w = (avail / n as f64).min(110.0);
    let start_x = ML + title_w;

    for (idx, term) in var.terms().iter().enumerate() {
        let color = PALETTE[idx % PALETTE.len()];
        let ix = start_x + idx as f64 * item_w;
        let iy = ly + 14.0;

        // Coloured square
        rect(s, ix, iy - 8.0, 10.0, 10.0, 2.0, color, 0.92);
        // Stroke around the square
        s.push_str(&format!(
            r#"<rect x="{:.1}" y="{:.1}" width="10" height="10" rx="2" fill="none" stroke="{}" stroke-width="0.8"/>"#,
            ix, iy - 8.0, color
        ));
        // Label
        text(s, ix + 14.0, iy, "start", TEXT, 9, false, &term.label);
    }
}

// ─── Curve helpers ────────────────────────────────────────────────

fn sample_curve(mf: &crate::membership::MembershipFn, min: f64, max: f64) -> Vec<(f64, f64)> {
    (0..SAMPLES)
        .map(|i| {
            let x = min + (max - min) * i as f64 / (SAMPLES - 1) as f64;
            (x, mf.eval(x))
        })
        .collect()
}

fn polyline_pts(pts: &[(f64, f64)], min: f64, max: f64) -> String {
    pts.iter()
        .map(|(x, d)| format!("{:.1},{:.1}", px(*x, min, max), py(*d)))
        .collect::<Vec<_>>()
        .join(" ")
}

fn filled_poly(pts: &[(f64, f64)], min: f64, max: f64) -> String {
    let fx = px(pts.first().map(|(x, _)| *x).unwrap_or(min), min, max);
    let lx = px(pts.last().map(|(x, _)| *x).unwrap_or(max), min, max);
    let by = py(0.0);
    let mid: String = pts
        .iter()
        .map(|(x, d)| format!("{:.1},{:.1}", px(*x, min, max), py(*d)))
        .collect::<Vec<_>>()
        .join(" ");
    format!("{:.1},{:.1} {} {:.1},{:.1}", fx, by, mid, lx, by)
}

fn clipped_poly(pts: &[(f64, f64)], clip: f64, min: f64, max: f64) -> String {
    let clipped: Vec<(f64, f64)> = pts.iter().map(|(x, d)| (*x, d.min(clip))).collect();
    filled_poly(&clipped, min, max)
}

// ─── Annotation helpers ───────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn draw_intersection(
    s: &mut String,
    xv: f64,
    degree: f64,
    min: f64,
    max: f64,
    color: &str,
    label: &str,
    offset_y: f64,
) {
    let xp = px(xv, min, max);

    if degree < 1e-9 {
        circle(s, xp, py(0.0), 4.0, color);
        return;
    }

    let yp = py(degree);

    // Horizontal dashed line to Y-axis
    line(s, ML, yp, xp, yp, color, 1.0, "4,3");

    // Dot at intersection
    circle(s, xp, yp, 4.5, color);

    // Annotation label – use chars().count() to get the true character width
    let char_count = label.chars().count();
    let lw = char_count as f64 * 6.8 + 10.0;
    let lx = if xp + lw + 6.0 > ML + PW {
        xp - lw - 6.0
    } else {
        xp + 6.0
    };
    let ly = (yp + offset_y - 9.0).clamp(MT - 30.0, MT - 2.0);
    rect(s, lx, ly, lw, 14.0, 3.0, BG, 0.90);
    text(s, lx + lw / 2.0, ly + 10.0, "middle", color, 9, true, label);
}

// ─── Main render: membership function SVG ────────────────────────

/// Generates a self-contained SVG string for a `FuzzyVariable`.
///
/// When `input` is `Some(value)`:
/// - draws a vertical dashed line at the crisp input
/// - fills the clipped activation area for each term
/// - draws horizontal dashed lines at each membership degree
/// - places dots + μ annotations at the intersection points
///
/// The bottom of every SVG includes a **colour legend** mapping
/// each term label to its curve colour.
///
/// # Example
/// ```
/// use logicfuzzy_academic::{fuzzy_var};
/// let var = fuzzy_var!("temperature", 0.0, 50.0, 501,
///     "cold" => trimf [0.0,  0.0, 25.0],
///     "warm" => trimf [0.0, 25.0, 50.0],
///     "hot"  => trimf [25.0,50.0, 50.0],
/// );
/// let svg = var.to_svg();
/// assert!(svg.starts_with("<svg"));
/// ```
pub fn render_variable_svg(var: &FuzzyVariable, input: Option<f64>) -> String {
    let min = var.universe.min;
    let max = var.universe.max;

    let mut s = String::with_capacity(20_000);

    // Header + background
    s.push_str(&format!(
        r#"<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="'JetBrains Mono',monospace,ui-monospace,'Courier New'">"#
    ));
    rect(&mut s, 0.0, 0.0, W, H, 10.0, BG, 1.0);

    // Clip path for plot area
    s.push_str(&format!(
        r#"<defs><clipPath id="p"><rect x="{ML}" y="{MT}" width="{PW}" height="{PH}"/></clipPath></defs>"#
    ));

    // Title + universe subtitle
    text(&mut s, W / 2.0, 22.0, "middle", TEXT, 13, true, &var.name);
    text(
        &mut s,
        W / 2.0,
        36.0,
        "middle",
        MUTED,
        9,
        false,
        &format!("universe  [{}, {}]", fv(min), fv(max)),
    );

    // Grid and axes
    draw_grid_axes(&mut s, var);

    // ── Curves ────────────────────────────────────────────────────
    let mut intersections: Vec<(f64, &str, &str)> = Vec::new();

    for (idx, term) in var.terms().iter().enumerate() {
        let color = PALETTE[idx % PALETTE.len()];
        let pts = sample_curve(&term.mf, min, max);

        // Clipped fill (only when input is within bounds)
        if let Some(val) = input {
            if val >= min && val <= max {
                let deg = term.mf.eval(val);
                if deg > 1e-9 {
                    let cp = clipped_poly(&pts, deg, min, max);
                    s.push_str(&format!(
                        r#"<polygon points="{cp}" fill="{color}" fill-opacity="0.38" clip-path="url(#p)"/>"#
                    ));
                }
            }
        }

        // Background fill under the full curve
        let fp = filled_poly(&pts, min, max);
        s.push_str(&format!(
            r#"<polygon points="{fp}" fill="{color}" fill-opacity="0.15" clip-path="url(#p)"/>"#
        ));

        // Curve line
        let lp = polyline_pts(&pts, min, max);
        s.push_str(&format!(
            r#"<polyline points="{lp}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" clip-path="url(#p)"/>"#
        ));

        // Collect intersections
        if let Some(val) = input {
            if val >= min && val <= max {
                intersections.push((term.mf.eval(val), &term.label, color));
            }
        }
    }

    // ── Crisp input marker ────────────────────────────────────────
    if let Some(val) = input {
        if val >= min && val <= max {
            let xp = px(val, min, max);

            // Vertical dashed line
            s.push_str(&format!(
                r#"<line x1="{xp:.1}" y1="{MT}" x2="{xp:.1}" y2="{:.1}" stroke="{YELLOW}" stroke-width="1.5" stroke-dasharray="5,3" clip-path="url(#p)"/>"#,
                MT + PH
            ));

            // Triangle marker
            let ty = MT + PH;
            s.push_str(&format!(
                r#"<polygon points="{},{} {},{} {},{}" fill="{YELLOW}"/>"#,
                xp,
                ty + 3.0,
                xp - 5.0,
                ty + 11.0,
                xp + 5.0,
                ty + 11.0
            ));

            // Input value label
            text(
                &mut s,
                xp,
                ty + 26.0,
                "middle",
                YELLOW,
                9,
                true,
                &format!("x = {}", fv(val)),
            );

            // Per-term annotations
            for (offset, (deg, label, color)) in intersections.iter().enumerate() {
                let ann = format!("μ_{}({}) = {:.4}", label, fv(val), deg);
                let oy = (offset as f64 - intersections.len() as f64 / 2.0) * 14.0;
                draw_intersection(&mut s, val, *deg, min, max, color, &ann, oy);
            }
        }
    }

    // ── Colour legend ─────────────────────────────────────────────
    draw_legend(&mut s, var);

    s.push_str("</svg>");
    s
}

// ─── Aggregated output SVG ───────────────────────────────────────

/// Generates an SVG showing the aggregated consequent set and its centroid.
///
/// - **Dashed lines**: original (unclipped) membership functions
/// - **Filled areas**: each term clipped at its firing degree `α`
/// - **Grey envelope**: aggregated set (max over all clipped MFs)
/// - **Yellow dashed line**: centroid of the aggregated set
/// - **Legend**: maps each term label and firing degree to its colour
pub fn render_aggregated_svg(
    var: &FuzzyVariable,
    firing_degrees: &[(&str, f64)],
    centroid: f64,
) -> String {
    let min = var.universe.min;
    let max = var.universe.max;
    let xs: Vec<f64> = (0..SAMPLES)
        .map(|i| min + (max - min) * i as f64 / (SAMPLES - 1) as f64)
        .collect();

    let mut s = String::with_capacity(20_000);

    // Header + background
    s.push_str(&format!(
        r#"<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="'JetBrains Mono',monospace,ui-monospace,'Courier New'">"#
    ));
    rect(&mut s, 0.0, 0.0, W, H, 10.0, BG, 1.0);
    s.push_str(&format!(
        r#"<defs><clipPath id="p"><rect x="{ML}" y="{MT}" width="{PW}" height="{PH}"/></clipPath></defs>"#
    ));

    // Title
    text(
        &mut s,
        W / 2.0,
        22.0,
        "middle",
        TEXT,
        13,
        true,
        &format!("{} — aggregated output", var.name),
    );
    text(
        &mut s,
        W / 2.0,
        36.0,
        "middle",
        MUTED,
        9,
        false,
        &format!(
            "Mamdani clip (min) + aggregation (max)  |  centroid = {:.4}",
            centroid
        ),
    );

    draw_grid_axes(&mut s, var);

    // Compute aggregated MF
    let mut agg = vec![0.0_f64; SAMPLES];

    for (idx, term) in var.terms().iter().enumerate() {
        let color = PALETTE[idx % PALETTE.len()];
        let firing = firing_degrees
            .iter()
            .find(|(l, _)| *l == term.label)
            .map(|(_, d)| *d)
            .unwrap_or(0.0);
        let pts = sample_curve(&term.mf, min, max);

        // Original curve (dashed, semi-transparent)
        let lp = polyline_pts(&pts, min, max);
        s.push_str(&format!(
            r#"<polyline points="{lp}" fill="none" stroke="{color}" stroke-width="1.2" stroke-dasharray="4,3" opacity="0.60" clip-path="url(#p)"/>"#
        ));

        if firing > 1e-9 {
            // Clipped fill
            let cp = clipped_poly(&pts, firing, min, max);
            s.push_str(&format!(
                r#"<polygon points="{cp}" fill="{color}" fill-opacity="0.38" clip-path="url(#p)"/>"#
            ));
            // Clipped top line (solid)
            let clip_pts: Vec<(f64, f64)> = pts.iter().map(|(x, d)| (*x, d.min(firing))).collect();
            let cl = polyline_pts(&clip_pts, min, max);
            s.push_str(&format!(
                r#"<polyline points="{cl}" fill="none" stroke="{color}" stroke-width="2" clip-path="url(#p)"/>"#
            ));
            // α label on the horizontal clip line
            let xr = ML + PW - 8.0;
            let yr = py(firing) - 4.0;
            text(
                &mut s,
                xr,
                yr,
                "end",
                color,
                9,
                true,
                &format!("α={:.4}", firing),
            );
        }

        // Update aggregate
        for (i, (_, d)) in pts.iter().enumerate() {
            let c = d.min(firing);
            if c > agg[i] {
                agg[i] = c;
            }
        }
    }

    // Aggregated filled area
    let agg_pts: Vec<(f64, f64)> = xs.iter().zip(agg.iter()).map(|(&x, &d)| (x, d)).collect();
    let afp = filled_poly(&agg_pts, min, max);
    s.push_str(&format!(
        r#"<polygon points="{afp}" fill="{SUBTEXT}" fill-opacity="0.20" clip-path="url(#p)"/>"#
    ));

    // Aggregated envelope line
    let alp = polyline_pts(&agg_pts, min, max);
    s.push_str(&format!(
        r#"<polyline points="{alp}" fill="none" stroke="{SUBTEXT}" stroke-width="2.2" clip-path="url(#p)"/>"#
    ));

    // Centroid marker
    if centroid >= min && centroid <= max {
        let xp = px(centroid, min, max);

        s.push_str(&format!(
            r#"<line x1="{xp:.1}" y1="{MT}" x2="{xp:.1}" y2="{:.1}" stroke="{YELLOW}" stroke-width="2" stroke-dasharray="6,3" clip-path="url(#p)"/>"#,
            MT + PH
        ));

        let ty = MT + PH;
        s.push_str(&format!(
            r#"<polygon points="{},{} {},{} {},{}" fill="{YELLOW}"/>"#,
            xp,
            ty + 3.0,
            xp - 5.0,
            ty + 11.0,
            xp + 5.0,
            ty + 11.0
        ));

        text(
            &mut s,
            xp,
            ty + 26.0,
            "middle",
            YELLOW,
            9,
            true,
            &format!("centroid = {:.4}", centroid),
        );
    }

    // Legend
    draw_legend(&mut s, var);

    s.push_str("</svg>");
    s
}

// ─── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FuzzyVariable, MembershipFn, Term, Universe};

    // ─── Helpers ──────────────────────────────────────────────────
    fn make_var() -> FuzzyVariable {
        let mut v = FuzzyVariable::new("temperature", Universe::new(0.0, 50.0, 501));
        v.add_term(Term::new("cold", MembershipFn::Trimf([0.0, 0.0, 25.0])));
        v.add_term(Term::new("warm", MembershipFn::Trimf([0.0, 25.0, 50.0])));
        v.add_term(Term::new("hot", MembershipFn::Trimf([25.0, 50.0, 50.0])));
        v
    }

    fn make_simple_var() -> FuzzyVariable {
        let mut v = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        v.add_term(Term::new("tri", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        v
    }

    // ── Existing tests (preserved) ────────────────────────────────
    #[test]
    fn px_boundary_values() {
        assert!((px(0.0, 0.0, 50.0) - ML).abs() < 1e-9);
        assert!((px(50.0, 0.0, 50.0) - (ML + PW)).abs() < 1e-9);
    }

    #[test]
    fn py_boundary_values() {
        assert!((py(0.0) - (MT + PH)).abs() < 1e-9);
        assert!((py(1.0) - MT).abs() < 1e-9);
    }

    #[test]
    fn py_clamps_above_one() {
        assert!((py(1.5) - MT).abs() < 1e-9);
    }

    #[test]
    fn py_clamps_below_zero() {
        assert!((py(-0.5) - (MT + PH)).abs() < 1e-9);
    }

    // ── NEW tests for helper functions ───────────────────────────
    #[test]
    fn px_mid_value() {
        let x = px(25.0, 0.0, 50.0);
        let expected = ML + PW / 2.0;
        assert!((x - expected).abs() < 1e-9);
    }

    #[test]
    fn fv_integer() {
        assert_eq!(fv(5.0), "5");
    }
    #[test]
    fn fv_decimal() {
        assert_eq!(fv(5.1), "5.1");
    }
    #[test]
    fn fv_negative() {
        assert_eq!(fv(-3.0), "-3");
    }
    #[test]
    fn fv_large() {
        assert_eq!(fv(9999.0), "9999");
    }

    #[test]
    fn rect_generates_correct_string() {
        let mut s = String::new();
        rect(&mut s, 10.0, 20.0, 100.0, 30.0, 5.0, "#ff0000", 0.8);
        // Use r##"..."## to avoid conflict with # inside the string
        let expected = r##"<rect x="10.0" y="20.0" width="100.0" height="30.0" rx="5" fill="#ff0000" fill-opacity="0.80"/>"##;
        assert!(s.contains(expected));
    }

    #[test]
    fn line_generates_correct_string_with_dash() {
        let mut s = String::new();
        line(&mut s, 0.0, 0.0, 100.0, 100.0, "#000", 2.0, "4,2");
        assert!(s.contains(r#"stroke-dasharray="4,2""#));
    }

    #[test]
    fn line_generates_correct_string_without_dash() {
        let mut s = String::new();
        line(&mut s, 0.0, 0.0, 100.0, 100.0, "#000", 2.0, "");
        assert!(!s.contains("stroke-dasharray"));
    }

    #[test]
    fn text_with_bold_generates_font_weight_bold() {
        let mut s = String::new();
        text(&mut s, 10.0, 20.0, "start", "#fff", 12, true, "Test");
        assert!(s.contains(r#"font-weight="bold""#));
    }

    #[test]
    fn text_without_bold_omits_font_weight() {
        let mut s = String::new();
        text(&mut s, 10.0, 20.0, "start", "#fff", 12, false, "Test");
        assert!(!s.contains("font-weight"));
    }

    #[test]
    fn circle_generates_correct_string() {
        let mut s = String::new();
        circle(&mut s, 1.0, 2.0, 3.0, "#000");
        let expected = r##"<circle cx="1.0" cy="2.0" r="3" fill="#000"/>"##;
        assert!(s.contains(expected));
    }

    // ── Tests for draw_grid_axes and draw_legend ─────────────────
    #[test]
    fn variable_svg_contains_grid_labels() {
        let var = make_simple_var();
        let svg = render_variable_svg(&var, None);
        // Should contain X axis labels at 0, 2.5, 5, 7.5, 10
        assert!(svg.contains(">0<") || svg.contains(">0.0"));
        assert!(svg.contains(">10"));
        // Y axis labels 0.00, 0.25, 0.50, 0.75, 1.00
        assert!(svg.contains(">0.00<"));
        assert!(svg.contains(">1.00<"));
    }

    #[test]
    fn variable_svg_contains_legend_with_terms() {
        let var = make_simple_var();
        let svg = render_variable_svg(&var, None);
        assert!(svg.contains("Terms:"));
        assert!(svg.contains("tri"));
    }

    // ── Tests for curve helpers ──────────────────────────────────
    #[test]
    fn sample_curve_trimf_peak_mid() {
        let mf = MembershipFn::Trimf([0.0, 5.0, 10.0]);
        let pts = sample_curve(&mf, 0.0, 10.0);
        assert_eq!(pts.len(), SAMPLES);
        // os extremos devem ser ~0
        assert!((pts[0].1).abs() < 1e-9);
        assert!((pts[SAMPLES - 1].1).abs() < 1e-9);
        // o máximo deve estar próximo de 1 (pico em 5.0, mas pode não cair exatamente no grid)
        let max = pts
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        assert!(
            (max.1 - 1.0).abs() < 0.01,
            "peak should be ~1.0, got {}",
            max.1
        );
        assert!(
            (max.0 - 5.0).abs() < 0.05,
            "peak x should be ~5.0, got {}",
            max.0
        );
    }

    #[test]
    fn polyline_pts_generates_valid_points() {
        let pts = vec![(0.0, 0.0), (10.0, 1.0)];
        let s = polyline_pts(&pts, 0.0, 10.0);
        let count = s.matches(',').count();
        assert_eq!(count, 2); // two commas for two points: "x1,y1 x2,y2"
    }

    #[test]
    fn filled_poly_contains_expected_coordinates() {
        let pts = vec![(0.0, 0.0), (10.0, 0.5)];
        let s = filled_poly(&pts, 0.0, 10.0);
        assert!(s.contains(format!("{:.1},{:.1}", ML, MT + PH).as_str()));
        assert!(s.contains(format!("{:.1},{:.1}", ML + PW, py(0.5)).as_str()));
    }

    #[test]
    fn clipped_poly_applies_clip() {
        let pts = vec![(0.0, 0.0), (10.0, 1.0)];
        let s = clipped_poly(&pts, 0.5, 0.0, 10.0);
        let y_clamped = py(0.5);
        assert!(s.contains(format!("{:.1}", y_clamped).as_str()));
    }

    // ── draw_intersection ────────────────────────────────────────
    #[test]
    fn draw_intersection_nonzero_degree() {
        let mut s = String::new();
        draw_intersection(
            &mut s,
            5.0,
            0.8,
            0.0,
            10.0,
            "#ff0000",
            "μ_cold(5.0)=0.8000",
            0.0,
        );
        assert!(s.contains("<circle"));
        assert!(s.contains(r#"stroke-dasharray="4,3""#));
    }

    #[test]
    fn draw_intersection_zero_degree() {
        let mut s = String::new();
        draw_intersection(
            &mut s,
            5.0,
            0.0,
            0.0,
            10.0,
            "#ff0000",
            "μ_cold(5.0)=0.0000",
            0.0,
        );
        assert!(s.contains("<circle"));
        assert!(!s.contains("stroke-dasharray"));
    }

    // ── render_variable_svg precise checks ───────────────────────
    #[test]
    fn svg_with_input_marker_at_specific_position() {
        let var = make_simple_var();
        let svg = render_variable_svg(&var, Some(5.0));
        let xp = px(5.0, 0.0, 10.0);
        let x_str = format!("{:.1}", xp);
        // check the dashed vertical line is placed at xp
        assert!(
            svg.contains(&format!("x1=\"{}\"", x_str)),
            "missing x1 at {}",
            x_str
        );
        assert!(
            svg.contains(&format!("x2=\"{}\"", x_str)),
            "missing x2 at {}",
            x_str
        );
        // check the label with the input value
        assert!(svg.contains("x = 5"), "missing input label");
    }

    #[test]
    fn svg_contains_intersection_annotations() {
        let var = make_simple_var();
        let svg = render_variable_svg(&var, Some(2.5));
        assert!(svg.contains("μ_tri(2.5)"));
    }

    // ── render_aggregated_svg ────────────────────────────────────
    #[test]
    fn aggregated_svg_centroid_marker_at_correct_x() {
        let mut var = FuzzyVariable::new("out", Universe::new(0.0, 100.0, 1001));
        var.add_term(Term::new("low", MembershipFn::Trimf([0.0, 0.0, 50.0])));
        var.add_term(Term::new("high", MembershipFn::Trimf([50.0, 100.0, 100.0])));
        let centroid = 68.5;
        let svg = render_aggregated_svg(&var, &[("low", 0.3), ("high", 0.7)], centroid);
        let xp = px(centroid, 0.0, 100.0);
        let x_str = format!("{:.1}", xp);
        // The centroid dashed line should have x1/x2 set to xp
        assert!(
            svg.contains(&format!("x1=\"{}\"", x_str)),
            "centroid line x1 not found in SVG"
        );
        assert!(
            svg.contains(&format!("x2=\"{}\"", x_str)),
            "centroid line x2 not found in SVG"
        );
        assert!(svg.contains("centroid"), "word 'centroid' missing");
    }

    #[test]
    fn aggregated_svg_contains_alpha_labels() {
        let mut var = FuzzyVariable::new("out", Universe::new(0.0, 100.0, 1001));
        var.add_term(Term::new("mid", MembershipFn::Trimf([25.0, 50.0, 75.0])));
        let svg = render_aggregated_svg(&var, &[("mid", 0.6)], 50.0);
        assert!(svg.contains("α=0.6") || svg.contains("α=0.60"));
    }

    // ── Pre-existing tests (kept) ─────────────────────────────────
    #[test]
    fn svg_starts_and_ends_correctly() {
        let svg = render_variable_svg(&make_var(), None);
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
    }

    #[test]
    fn svg_contains_variable_name() {
        assert!(render_variable_svg(&make_var(), None).contains("temperature"));
    }

    #[test]
    fn svg_contains_all_term_labels() {
        let svg = render_variable_svg(&make_var(), None);
        assert!(svg.contains("cold"));
        assert!(svg.contains("warm"));
        assert!(svg.contains("hot"));
    }

    #[test]
    fn svg_contains_legend() {
        let svg = render_variable_svg(&make_var(), None);
        assert!(svg.contains("Terms:"));
    }

    #[test]
    fn svg_legend_contains_all_term_labels() {
        let svg = render_variable_svg(&make_var(), None);
        assert!(svg.contains("cold"));
        assert!(svg.contains("warm"));
        assert!(svg.contains("hot"));
    }

    #[test]
    fn svg_with_input_has_vertical_marker() {
        let svg = render_variable_svg(&make_var(), Some(10.0));
        assert!(svg.contains(r#"stroke-dasharray="5,3""#));
    }

    #[test]
    fn svg_with_input_has_mu_annotations() {
        let svg = render_variable_svg(&make_var(), Some(10.0));
        assert!(svg.contains("μ_cold"));
    }

    #[test]
    fn svg_without_input_no_vertical_marker() {
        let svg = render_variable_svg(&make_var(), None);
        assert!(!svg.contains(r#"stroke-dasharray="5,3""#));
    }

    #[test]
    fn svg_input_outside_universe_ignored() {
        let with_out = render_variable_svg(&make_var(), Some(999.0));
        let without = render_variable_svg(&make_var(), None);
        assert_eq!(with_out, without);
    }

    #[test]
    fn svg_input_at_lower_bound() {
        let svg = render_variable_svg(&make_var(), Some(0.0));
        assert!(svg.contains(r#"stroke-dasharray="5,3""#));
    }

    #[test]
    fn svg_input_at_upper_bound() {
        let svg = render_variable_svg(&make_var(), Some(50.0));
        assert!(svg.contains(r#"stroke-dasharray="5,3""#));
    }

    #[test]
    fn svg_trapmf_renders() {
        let mut v = FuzzyVariable::new("x", Universe::new(0.0, 100.0, 1001));
        v.add_term(Term::new(
            "mid",
            MembershipFn::Trapmf([20.0, 30.0, 70.0, 80.0]),
        ));
        let svg = render_variable_svg(&v, Some(50.0));
        assert!(svg.contains("mid"));
        assert!(svg.contains("1.0000"));
    }

    #[test]
    fn svg_gaussmf_renders_with_input() {
        let mut v = FuzzyVariable::new("x", Universe::new(0.0, 100.0, 1001));
        v.add_term(Term::new(
            "peak",
            MembershipFn::Gaussmf {
                mean: 50.0,
                sigma: 10.0,
            },
        ));
        let svg = render_variable_svg(&v, Some(50.0));
        assert!(svg.contains("μ_peak"));
    }

    #[test]
    fn svg_single_term_no_panic() {
        let mut v = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        v.add_term(Term::new("only", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        let svg = render_variable_svg(&v, Some(5.0));
        assert!(svg.contains("only"));
    }

    #[test]
    fn svg_eight_terms_all_colors() {
        let mut v = FuzzyVariable::new("x", Universe::new(0.0, 80.0, 101));
        for i in 0..8 {
            v.add_term(Term::new(
                &format!("t{}", i),
                MembershipFn::Trimf([
                    i as f64 * 10.0,
                    i as f64 * 10.0 + 5.0,
                    i as f64 * 10.0 + 10.0,
                ]),
            ));
        }
        let svg = render_variable_svg(&v, None);
        for i in 0..8 {
            assert!(svg.contains(&format!("t{}", i)));
        }
    }

    /// Check that vertical grid lines are placed at exact x coordinates.
    #[test]
    fn grid_vertical_lines_at_correct_x() {
        let mut var = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        var.add_term(Term::new("a", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        let svg = render_variable_svg(&var, None);
        for &t in &[0.0_f64, 0.25, 0.5, 0.75, 1.0] {
            let xv = 0.0 + (10.0 - 0.0) * t;
            let xp = px(xv, 0.0, 10.0);
            let expected_x = format!("x1=\"{:.1}\"", xp);
            let expected_x2 = format!("x2=\"{:.1}\"", xp);
            assert!(
                svg.contains(&expected_x) && svg.contains(&expected_x2),
                "Grid line for t={} not found at x={:.1}",
                t,
                xp
            );
        }
    }

    #[test]
    fn grid_horizontal_lines_and_labels() {
        let mut var = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        var.add_term(Term::new("a", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        let svg = render_variable_svg(&var, None);
        for &d in &[0.25_f64, 0.5, 0.75, 1.0] {
            let y = py(d);
            let expected_y1 = format!("y1=\"{:.1}\"", y);
            let expected_y2 = format!("y2=\"{:.1}\"", y);
            assert!(
                svg.contains(&expected_y1) && svg.contains(&expected_y2),
                "Horizontal grid line at d={} not found at y={:.1}",
                d,
                y
            );
        }
    }

    /// Legend items must be positioned with correct x spacing.
    #[test]
    fn legend_items_have_correct_x_offset() {
        let mut var = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        var.add_term(Term::new("one", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        var.add_term(Term::new("two", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        let svg = render_variable_svg(&var, None);
        // Legend drawing uses a layout with title at ML+8, items spaced by item_w.
        // The first coloured square after title has x = ML + title_w + 0*item_w
        let n = var.term_count() as f64;
        let title_w = 84.0;
        let avail = PW - title_w;
        let _item_w = (avail / n).min(110.0);
        let start_x = ML + title_w;
        // The label of the first term should be at x = start_x + 14.0 (square at ix, label at ix+14)
        // We'll just check that the first term label "one" is placed at a specific x coordinate.
        // The text element for "one" should contain x="..." near start_x + 14.
        let expected_x = start_x + 14.0;
        // Look for a text element containing "one" and the x coordinate
        let pat = format!("x=\"{:.1}\"", expected_x);
        // It's possible the label text is after the square, but we can search for the term name and nearby x coordinate.
        // Simpler: assert that the SVG contains the term name and the x coordinate in the same text element.
        assert!(
            svg.contains(&pat) && svg.contains(">one<"),
            "First legend item not at expected x={:.1}",
            expected_x
        );
    }

    /// Intersection dots for non-zero degree must have correct y coordinate.
    #[test]
    fn intersection_dot_at_correct_y() {
        let mut var = FuzzyVariable::new("x", Universe::new(0.0, 10.0, 101));
        var.add_term(Term::new("tri", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        let svg = render_variable_svg(&var, Some(2.5));
        // mu_tri(2.5) = 0.5
        let degree = 0.5;
        let yp = py(degree);
        // There should be a circle at that y (and at the correct x)
        let expected_circle = format!("cy=\"{:.1}\"", yp);
        assert!(
            svg.contains(&expected_circle),
            "Intersection dot at degree {} missing at y={:.1}",
            degree,
            yp
        );
    }

    #[test]
    fn aggregated_centroid_y_placement() {
        let mut var = FuzzyVariable::new("out", Universe::new(0.0, 100.0, 1001));
        var.add_term(Term::new("low", MembershipFn::Trimf([0.0, 0.0, 50.0])));
        var.add_term(Term::new("high", MembershipFn::Trimf([50.0, 100.0, 100.0])));
        let centroid = 68.5;
        let svg = render_aggregated_svg(&var, &[("low", 0.3), ("high", 0.7)], centroid);
        // The centroid line extends from y=MT to y=MT+PH
        let y_top = MT;
        let y_bot = MT + PH;
        assert!(svg.contains(format!("y1=\"{:.1}\"", y_top).as_str()));
        assert!(svg.contains(format!("y2=\"{:.1}\"", y_bot).as_str()));
    }

    /// Check that draw_grid_axes generates lines at the plot boundaries.
    /// These coordinates are mathematically fixed (ML, ML+PW, MT, MT+PH)
    /// so any mutation of arithmetic inside draw_grid_axes will shift them.
    #[test]
    fn draw_grid_axes_boundary_coordinates() {
        let var = FuzzyVariable::new("test", Universe::new(0.0, 10.0, 101));
        let mut s = String::new();
        draw_grid_axes(&mut s, &var);

        // Leftmost vertical line (t=0.0) should be at x = ML
        let left_x = format!("x1=\"{:.1}\"", ML);
        let left_x2 = format!("x2=\"{:.1}\"", ML);
        assert!(s.contains(&left_x), "Left vertical line x1 not found");
        assert!(s.contains(&left_x2), "Left vertical line x2 not found");

        // Rightmost vertical line (t=1.0) should be at x = ML + PW
        let right_x = format!("x1=\"{:.1}\"", ML + PW);
        let right_x2 = format!("x2=\"{:.1}\"", ML + PW);
        assert!(s.contains(&right_x), "Right vertical line x1 not found");
        assert!(s.contains(&right_x2), "Right vertical line x2 not found");

        // Top horizontal line (d=1.0) should be at y = MT
        let top_y = format!("y1=\"{:.1}\"", MT);
        let top_y2 = format!("y2=\"{:.1}\"", MT);
        assert!(s.contains(&top_y), "Top horizontal line y1 not found");
        assert!(s.contains(&top_y2), "Top horizontal line y2 not found");

        // The plot border is a rect with exact coordinates
        let border = format!(
            r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="none""#,
            ML, MT, PW, PH
        );
        assert!(s.contains(&border), "Plot border rect missing or incorrect");
    }

    /// Test draw_legend directly to kill arithmetic mutants inside it.
    #[test]
    fn draw_legend_puts_correct_positions() {
        let mut var = FuzzyVariable::new("test", Universe::new(0.0, 10.0, 101));
        var.add_term(Term::new("low", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        var.add_term(Term::new("high", MembershipFn::Trimf([0.0, 5.0, 10.0])));
        let mut s = String::new();
        draw_legend(&mut s, &var);

        // The first colored square is placed at (start_x, ly-8) where start_x = ML + title_w
        let title_w = 84.0;
        let start_x = ML + title_w;
        let expected_rect_x = format!("x=\"{:.1}\"", start_x);
        assert!(
            s.contains(&expected_rect_x),
            "First legend square not at expected x"
        );

        // The legend title must be visible
        assert!(s.contains("Terms:"), "Legend title missing");
    }
}
