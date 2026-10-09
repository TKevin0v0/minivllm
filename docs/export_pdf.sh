#!/usr/bin/env bash
# 将 docs 下 Markdown 导出为带图片 / 公式 / 列表 / GitHub Alerts 的 PDF。
#
# 依赖：pandoc、google-chrome（或 chromium）
# 公式：优先 docs/_vendor/katex 或系统 KaTeX，否则 MathML（离线）
# Alerts：支持 > [!NOTE|TIP|IMPORTANT|WARNING|CAUTION]（预处理为带色 callout）
#
# 用法：
#   bash docs/export_pdf.sh
#   bash docs/export_pdf.sh docs/00-MoE概述.md
#   bash docs/export_pdf.sh docs/01-路由与融合计算.md
#   bash docs/export_pdf.sh docs/00-MoE概述.md /tmp/out.pdf

set -euo pipefail

DOCS="$(cd "$(dirname "$0")" && pwd)"

SRC="${1:-$DOCS/00-MoE概述.md}"
if [[ ! -f "$SRC" ]]; then
  echo "missing markdown: $SRC" >&2
  exit 1
fi
SRC="$(cd "$(dirname "$SRC")" && pwd)/$(basename "$SRC")"
BASE="$(basename "$SRC" .md)"
OUT_PDF="${2:-$DOCS/${BASE}.pdf}"

TMPDIR="$(mktemp -d /tmp/md2pdf.XXXXXX)"
trap 'rm -rf "$TMPDIR"' EXIT
MD_PRE="$TMPDIR/pre.md"
HTML="$TMPDIR/out.html"

CHROME=""
for c in google-chrome-stable google-chrome chromium chromium-browser; do
  if command -v "$c" >/dev/null 2>&1; then
    CHROME="$c"
    break
  fi
done
if [[ -z "$CHROME" ]]; then
  echo "need google-chrome or chromium" >&2
  exit 1
fi
if ! command -v pandoc >/dev/null 2>&1; then
  echo "need pandoc" >&2
  exit 1
fi

MATH_ARGS=(--mathml)
VT=5000
KATEX_DIR=""
if [[ -f "$DOCS/_vendor/katex/katex.min.js" ]]; then
  KATEX_DIR="$DOCS/_vendor/katex"
elif [[ -f /usr/share/javascript/katex/katex.min.js ]]; then
  KATEX_DIR="/usr/share/javascript/katex"
else
  # 复用本机 VS Code / Cursor 自带的 KaTeX（无需再下载）
  KATEX_DIR="$(ls -d /home/*/.vscode-server/cli/servers/Stable-*/server/node_modules/katex/dist 2>/dev/null | head -1 || true)"
  if [[ -z "$KATEX_DIR" || ! -f "$KATEX_DIR/katex.min.js" ]]; then
    KATEX_DIR="$(ls -d /home/*/.cursor-server/cli/servers/Stable-*/server/node_modules/katex/dist 2>/dev/null | head -1 || true)"
  fi
fi
if [[ -n "$KATEX_DIR" && -f "$KATEX_DIR/katex.min.js" ]]; then
  MATH_ARGS=(--katex="$KATEX_DIR/")
  VT=20000
  echo "using katex: $KATEX_DIR"
else
  echo "using MathML (optional: put katex under docs/_vendor/katex/ for better glyphs)"
fi

# 预处理：公式兜底 + GitHub Alerts（[!NOTE] 等）→ pandoc fenced div
python3 - "$SRC" "$MD_PRE" <<'PY'
import re, sys

src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()
text = re.sub(
    r"```math\s*\n(.*?)```",
    lambda m: "\n$$\n" + m.group(1).strip() + "\n$$\n",
    text,
    flags=re.S,
)

def trim_inline(m: re.Match) -> str:
    inner = m.group(1).strip()
    if re.search(r"[\u4e00-\u9fff]", inner):
        return m.group(0)
    return f"${inner}$"

text = re.sub(
    r"(?<![A-Za-z0-9\\])\$[ \t]+([^$\n]+?)[ \t]+\$(?!\$)",
    trim_inline,
    text,
)

ALERT_START = re.compile(
    r"^>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*(.*)$",
    re.IGNORECASE,
)
ALERT_LABEL = {
    "NOTE": "NOTE",
    "TIP": "TIP",
    "IMPORTANT": "IMPORTANT",
    "WARNING": "WARNING",
    "CAUTION": "CAUTION",
}


def _is_structural(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if re.match(r"^#{1,6}\s", line):
        return True
    if ALERT_START.match(line):
        return True
    if s in ("---", "***", "___"):
        return True
    if line.startswith("```"):
        return True
    if re.match(r"^\|.+\|", s):
        return True
    if re.match(r"^!\[[^\]]*\]\(", line):
        return True
    return False


def convert_github_alerts(src_text: str) -> str:
    """Convert GFM alerts to pandoc fenced divs for PDF CSS styling.

    Strict: consecutive ``>`` lines belong to the alert.
    Lenient: after the alert header, absorb plain lines until a blank line
    (covers docs that forgot the leading ``>`` on the body).
    """
    lines = src_text.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        m = ALERT_START.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue

        kind = m.group(1).upper()
        title = m.group(2).strip()
        body: list[str] = []
        i += 1
        while i < n:
            line = lines[i]
            if line.startswith(">"):
                # 下一个 GitHub Alert：结束当前块，留给外层重新识别
                if ALERT_START.match(line):
                    break
                body.append(re.sub(r"^>\s?", "", line))
                i += 1
                continue
            if line.strip() == "":
                # Keep blank if more ``>`` body follows (but not a new alert).
                if (
                    i + 1 < n
                    and lines[i + 1].startswith(">")
                    and not ALERT_START.match(lines[i + 1])
                ):
                    body.append("")
                    i += 1
                    continue
                break
            if _is_structural(line):
                break
            # Lenient: body forgot leading ``>`` (stop at next blank)
            body.append(line)
            i += 1

        while body and body[0].strip() == "":
            body.pop(0)
        while body and body[-1].strip() == "":
            body.pop()

        label = ALERT_LABEL[kind]
        cls = f"md-alert md-alert-{kind.lower()}"
        title_md = f"**{label}**" + (f" · {title}" if title else "")
        out.append(f":::: {{.{cls.replace(' ', ' .')}}}")
        out.append("::: {.md-alert-title}")
        out.append(title_md)
        out.append(":::")
        out.append("")
        out.extend(body)
        out.append("")
        out.append("::::")
        out.append("")
    return "\n".join(out)


text = convert_github_alerts(text)

lines = text.splitlines()
out = []
for line in lines:
    if (
        re.match(r"^- ", line)
        and out
        and out[-1].strip() != ""
        and not re.match(r"^- ", out[-1])
        and not re.match(r"^\d+\. ", out[-1])
    ):
        out.append("")
    out.append(line)
text = "\n".join(out) + ("\n" if text.endswith("\n") or not text else "")
open(dst, "w", encoding="utf-8").write(text)
PY

pandoc "$MD_PRE" \
  -f markdown+tex_math_dollars+tex_math_single_backslash+fenced_divs \
  -t html5 \
  --standalone \
  "${MATH_ARGS[@]}" \
  --embed-resources \
  --resource-path="$(dirname "$SRC")" \
  --metadata title="$BASE" \
  --css "$DOCS/export_pdf.css" \
  -o "$HTML"

"$CHROME" \
  --headless=new \
  --disable-gpu \
  --no-pdf-header-footer \
  --allow-file-access-from-files \
  --virtual-time-budget="$VT" \
  --run-all-compositor-stages-before-draw \
  --print-to-pdf="$OUT_PDF" \
  "file://$HTML" \
  >/dev/null 2>&1

echo "wrote $OUT_PDF ($(du -h "$OUT_PDF" | cut -f1))"
