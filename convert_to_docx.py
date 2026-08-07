"""
将初赛方案文档.md 转换为 .docx Word 文档
- 全文字体：宋体（标题加粗，无斜体）
- 全文正文居中
- 列表去符号
- LaTeX 公式 → Unicode 可读文本
- Mermaid 图表 → PNG 嵌入
"""
import re, zlib, base64, json, io
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from pathlib import Path
import urllib.request

SRC = Path("D:/AIVC/初赛方案文档.md")
DST = Path("D:/AIVC/初赛方案文档.docx")

FONT_CN = "宋体"
FONT_EN = "Times New Roman"
FONT_MONO = "Consolas"
FONT_SIZE_NORMAL = Pt(11)
FONT_SIZE_SMALL = Pt(9.5)
FONT_SIZE_TABLE = Pt(9)
FONT_SIZE_CODE = Pt(9)
HEADING_SIZES = {1: Pt(18), 2: Pt(15), 3: Pt(13), 4: Pt(11.5)}

doc = Document()

# ── 全局默认样式 ──
style = doc.styles["Normal"]
style.font.name = FONT_EN
style.font.size = FONT_SIZE_NORMAL
style.element.rPr.rFonts.set(qn("w:eastAsia"), FONT_CN)

for section in doc.sections:
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)


# ══════════════════════════════════════════════════════════════
# LaTeX → Unicode
# ══════════════════════════════════════════════════════════════

LATEX_REPLACEMENTS = [
    # 希腊字母
    (r"\\alpha", "α"), (r"\\beta", "β"), (r"\\gamma", "γ"),
    (r"\\delta", "δ"), (r"\\Delta", "Δ"), (r"\\epsilon", "ε"),
    (r"\\mu", "μ"), (r"\\sigma", "σ"), (r"\\Sigma", "Σ"),
    # 数学符号
    (r"\\in\b", "∈"), (r"\\notin", "∉"),
    (r"\\mathbb\{R\}", "ℝ"), (r"\\mathbb\{N\}", "ℕ"),
    (r"\\mathbb\{P\}", "ℙ"),
    (r"\\times", "×"), (r"\\cdot", "·"),
    (r"\\approx", "≈"), (r"\\propto", "∝"),
    (r"\\sum", "Σ"), (r"\\prod", "Π"),
    (r"\\infty", "∞"), (r"\\to", "→"), (r"\\mapsto", "↦"),
    (r"\\pm", "±"), (r"\\leq", "≤"), (r"\\geq", "≥"),
    (r"\\neq", "≠"), (r"\\equiv", "≡"),
    (r"\\subset", "⊂"), (r"\\subseteq", "⊆"),
    # 花括号（先处理，避免被其他规则误匹配）
    (r"\\\{", "{"), (r"\\\}", "}"),
    # 上标下标
    (r"\^\{([^}]+)\}", r"^\1"),  # ^{P} → ^P
    (r"_\{([^}]+)\}", r"_\1"),   # _{treat} → _treat
    # hat
    (r"\\hat\{([^}]+)\}", r"\1̂"),  # \hat{y} → ŷ
    # text
    (r"\\text\{([^}]+)\}", r"\1"),  # \text{strain} → strain
    # mathbb (bare)
    (r"\\mathbb\{R\}", "ℝ"),
    # 剩余花括号（math grouping, 去掉）
    (r"\{", ""), (r"\}", ""),
]


def clean_latex(text: str) -> str:
    """将 LaTeX 数学公式转为可读文本"""
    # Step 1：找到所有 $...$ 段（不在代码块内的 inline math）
    result = []
    segs = re.split(r"(\$[^$]+\$)", text)
    for seg in segs:
        if seg.startswith("$") and seg.endswith("$"):
            inner = seg[1:-1]  # 去掉 $ $
            # 应用 LaTeX 替换
            for pattern, replacement in LATEX_REPLACEMENTS:
                inner = re.sub(pattern, replacement, inner)
            # 清理残留的反斜杠命令
            inner = re.sub(r"\\[a-zA-Z]+\b", "", inner)
            # 清理多余空格
            inner = inner.strip()
            result.append(inner)
        else:
            result.append(seg)
    return "".join(result)


# ══════════════════════════════════════════════════════════════
# Mermaid → PNG
# ══════════════════════════════════════════════════════════════

def encode_mermaid_pako(code: str) -> str:
    payload = json.dumps({"code": code, "mermaid": {"theme": "default"}},
                         ensure_ascii=False)
    compressed = zlib.compress(payload.encode("utf-8"), level=9)
    b64 = base64.urlsafe_b64encode(compressed).decode("utf-8").rstrip("=")
    return f"https://mermaid.ink/img/pako:{b64}?type=png"


def fetch_mermaid_image(code: str):
    url = encode_mermaid_pako(code)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return io.BytesIO(resp.read())
    except Exception as e:
        print(f"  [WARN] Mermaid 图片获取失败: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════

def set_run_font(run, font_name=FONT_CN, size=None, bold=False, color=None):
    run.font.name = font_name
    run.element.rPr.rFonts.set(qn("w:eastAsia"), FONT_CN)
    run.bold = bold
    run.italic = False
    if size:
        run.font.size = size
    if color:
        run.font.color.rgb = color


def parse_inline(text):
    """解析行内 markdown → [(text, bold, code), ...]，同时处理 LaTeX"""
    parts = []
    # 先处理行内代码 `...`
    segs = re.split(r"(`[^`]+`)", text)
    for seg in segs:
        if seg.startswith("`") and seg.endswith("`"):
            parts.append((seg[1:-1], False, True))
        else:
            # 处理 **bold**
            sub_segs = re.split(r"(\*\*[^*]+\*\*)", seg)
            for ss in sub_segs:
                if ss.startswith("**") and ss.endswith("**"):
                    # 加粗段内也做 LaTeX 清理
                    cleaned = clean_latex(ss[2:-2])
                    parts.append((cleaned, True, False))
                else:
                    # *italic* → 去斜体，转普通文本
                    ss = re.sub(r"\*([^*]+)\*", r"\1", ss)
                    # 剥离 markdown 链接
                    ss = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", ss)
                    # LaTeX $...$ → 可读文本
                    ss = clean_latex(ss)
                    if ss:
                        parts.append((ss, False, False))
    return parts


def add_rich_paragraph(inline_parts, indent=None, spacing_after=None,
                       alignment=WD_ALIGN_PARAGRAPH.CENTER):
    """添加富文本段落，默认居中"""
    p = doc.add_paragraph()
    for text, bold, code in inline_parts:
        run = p.add_run(text)
        if code:
            set_run_font(run, FONT_MONO, FONT_SIZE_CODE, bold=bold,
                         color=RGBColor(0xCC, 0x33, 0x33))
        else:
            set_run_font(run, FONT_CN, FONT_SIZE_NORMAL, bold=bold)
    if indent:
        p.paragraph_format.left_indent = indent
    if spacing_after is not None:
        p.paragraph_format.space_after = spacing_after
    if alignment is not None:
        p.alignment = alignment
    return p


def add_heading_styled(text, level):
    """标题：宋体加粗（标题不居中，保持默认左对齐）"""
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        set_run_font(run, FONT_CN, HEADING_SIZES.get(level, Pt(11)), bold=True)
    return h


def add_code_block(code_lines):
    """代码块：灰底 Consolas，不居中（代码需左对齐保持缩进）"""
    for line in code_lines:
        p = doc.add_paragraph()
        run = p.add_run(line if line else " ")
        set_run_font(run, FONT_MONO, FONT_SIZE_CODE, bold=False,
                     color=RGBColor(0x33, 0x33, 0x33))
        shading = p.paragraph_format.element.get_or_add_pPr()
        shd = shading.makeelement(qn("w:shd"), {
            qn("w:fill"): "F5F5F5", qn("w:val"): "clear",
        })
        shading.append(shd)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.left_indent = Cm(0.5)


def add_table_from_rows(rows):
    """表格：文字水平+垂直居中，表头加粗灰底"""
    ncols = max(len(r) for r in rows)
    norm_rows = []
    for r in rows:
        nr = r[:]
        while len(nr) < ncols:
            nr.append("")
        norm_rows.append(nr)

    table = doc.add_table(rows=len(norm_rows), cols=ncols, style="Table Grid")
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    for i, row_data in enumerate(norm_rows):
        row = table.rows[i]
        is_header = (i == 0)
        for j, cell_text in enumerate(row_data):
            cell = row.cells[j]
            cell.paragraphs[0].clear()
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

            parts = parse_inline(cell_text)

            if not parts:
                run = cell.paragraphs[0].add_run("")
                set_run_font(run, FONT_CN, FONT_SIZE_TABLE, bold=is_header)
            else:
                first = True
                for text, bold, code in parts:
                    p = cell.paragraphs[0] if first else cell.add_paragraph()
                    first = False
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    run = p.add_run(text)
                    if code:
                        set_run_font(run, FONT_MONO, FONT_SIZE_TABLE,
                                     bold=bold or is_header,
                                     color=RGBColor(0xCC, 0x33, 0x33))
                    else:
                        set_run_font(run, FONT_CN, FONT_SIZE_TABLE,
                                     bold=bold or is_header)
                    p.paragraph_format.space_before = Pt(1)
                    p.paragraph_format.space_after = Pt(1)
                    p.paragraph_format.line_spacing = Pt(12)

            if is_header:
                p = cell.paragraphs[0]
                shading = p.paragraph_format.element.get_or_add_pPr()
                shd = shading.makeelement(qn("w:shd"), {
                    qn("w:fill"): "E8E8E8", qn("w:val"): "clear",
                })
                shading.append(shd)

    doc.add_paragraph()
    return table


# ══════════════════════════════════════════════════════════════
# 主解析
# ══════════════════════════════════════════════════════════════

lines = SRC.read_text(encoding="utf-8").split("\n")

i = 0
in_code_block = False
code_buffer = []
in_table = False
table_buffer = []
in_mermaid = False
mermaid_buffer = []
blockquote_buffer = []
mermaid_count = 0

while i < len(lines):
    line = lines[i]

    # ── Mermaid 代码块 → PNG ──
    if line.strip().startswith("```mermaid"):
        in_mermaid = True
        mermaid_buffer = []
        i += 1
        continue
    if in_mermaid:
        if line.strip().startswith("```"):
            in_mermaid = False
            mermaid_count += 1
            mermaid_code = "\n".join(mermaid_buffer)
            print(f"  导出 Mermaid 图 {mermaid_count}...")
            img_data = fetch_mermaid_image(mermaid_code)
            if img_data:
                try:
                    doc.add_picture(img_data, width=Inches(5.5))
                    last_p = doc.paragraphs[-1]
                    last_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    last_p.paragraph_format.space_after = Pt(6)
                    print(f"    ✓ 已嵌入")
                except Exception as e:
                    print(f"    ✗ 嵌入失败: {e}")
                    add_rich_paragraph([("[ 图表导出失败，请参考 Markdown 原文 ]", False, False)])
            else:
                add_rich_paragraph([("[ 图表导出失败，请参考 Markdown 原文 ]", False, False)])
            i += 1
            continue
        mermaid_buffer.append(line)
        i += 1
        continue

    # ── 普通代码块 ──
    if line.strip().startswith("```") and not in_code_block:
        in_code_block = True
        code_buffer = []
        i += 1
        continue
    if in_code_block:
        if line.strip() == "```":
            in_code_block = False
            add_code_block(code_buffer)
            doc.add_paragraph()
            i += 1
            continue
        code_buffer.append(line)
        i += 1
        continue

    # ── 表格 ──
    if "|" in line and line.strip().startswith("|"):
        if not in_table:
            in_table = True
            table_buffer = []
        table_buffer.append(line)
        i += 1
        if i < len(lines) and re.match(r"^\|[\s\-:|]+\|$", lines[i].strip()):
            table_buffer.append(lines[i])
            i += 1
        continue
    elif in_table:
        in_table = False
        rows = []
        for tl in table_buffer:
            if re.match(r"^\|[\s\-:|]+\|$", tl.strip()):
                continue
            cells = [c.strip() for c in tl.strip().strip("|").split("|")]
            rows.append(cells)
        if rows:
            add_table_from_rows(rows)
        table_buffer = []

    # ── 引用块 ──
    if line.strip().startswith(">"):
        clean = re.sub(r"^>\s?", "", line)
        blockquote_buffer.append(clean)
        i += 1
        if i < len(lines) and lines[i].strip().startswith(">"):
            continue
        else:
            text = " ".join(blockquote_buffer)
            parts = parse_inline(text)
            p = add_rich_paragraph(parts, indent=Cm(1.0), spacing_after=Pt(4))
            for run in p.runs:
                run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
                run.font.size = FONT_SIZE_SMALL
            blockquote_buffer = []
            continue

    # ── 水平线 ──
    if line.strip() in ("---", "***", "___"):
        p = doc.add_paragraph("─" * 60)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        i += 1
        continue

    # ── 空行 ──
    if line.strip() == "":
        i += 1
        continue

    # ── 标题（不居中，保持默认）──
    if line.startswith("# "):
        add_heading_styled(line[2:].strip(), 1)
        i += 1
        continue
    if line.startswith("## "):
        add_heading_styled(line[3:].strip(), 2)
        i += 1
        continue
    if line.startswith("### "):
        add_heading_styled(line[4:].strip(), 3)
        i += 1
        continue
    if line.startswith("#### "):
        add_heading_styled(line[5:].strip(), 4)
        i += 1
        continue

    # ── 无序列表 → 去符号，缩进段落，居中 ──
    if re.match(r"^[\-\*]\s+", line):
        text = re.sub(r"^[\-\*]\s+", "", line.strip())
        parts = parse_inline(text)
        add_rich_paragraph(parts, indent=Cm(0.75), spacing_after=Pt(2))
        i += 1
        continue

    # ── 有序列表 → 保留序号，缩进段落，居中 ──
    if re.match(r"^\d+\.\s+", line):
        text = re.sub(r"^\d+\.\s+", "", line.strip())
        parts = parse_inline(text)
        add_rich_paragraph(parts, indent=Cm(0.75), spacing_after=Pt(2))
        i += 1
        continue

    # ── 普通段落（居中）──
    parts = parse_inline(line.strip())
    if parts:
        add_rich_paragraph(parts)
    i += 1

# ── 清理残留 ──
if in_code_block and code_buffer:
    add_code_block(code_buffer)
if in_table and table_buffer:
    rows = []
    for tl in table_buffer:
        if re.match(r"^\|[\s\-:|]+\|$", tl.strip()):
            continue
        cells = [c.strip() for c in tl.strip().strip("|").split("|")]
        rows.append(cells)
    if rows:
        add_table_from_rows(rows)

# ── 保存 ──
doc.save(str(DST))
print(f"已保存到: {DST}")
print(f"文件大小: {DST.stat().st_size / 1024:.1f} KB")
print(f"Mermaid 图表: {mermaid_count} 个")
