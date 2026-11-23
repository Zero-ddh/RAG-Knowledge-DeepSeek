"""文档解析：txt / md / pdf / docx -> 纯文本段落（保留页码信息）。"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Section:
    text: str
    page: int | None = None


class DocumentParseError(Exception):
    """文档无法解析时抛出，消息可直接展示给用户。"""


def load_sections(path: Path) -> list[Section]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        sections = [Section(text=_read_text(path))]
    elif suffix == ".pdf":
        sections = _load_pdf(path)
    elif suffix == ".docx":
        sections = _load_docx(path)
    else:
        raise DocumentParseError(f"暂不支持的文件类型：{suffix}")

    cleaned = [Section(text=_normalize(s.text), page=s.page) for s in sections]
    cleaned = [s for s in cleaned if s.text]
    if not cleaned:
        raise DocumentParseError("未能从文件中提取到有效文本，请确认文件不是扫描件或空文件")
    return cleaned


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentParseError("无法识别文本编码，请将文件另存为 UTF-8 后重试")


def _load_pdf(path: Path) -> list[Section]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 依赖缺失时给出明确提示
        raise DocumentParseError("缺少 pypdf 依赖，请先执行 pip install -r requirements.txt") from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise DocumentParseError(f"PDF 文件读取失败：{exc}") from exc

    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception as exc:
            raise DocumentParseError("PDF 已加密，无法解析") from exc

    sections: list[Section] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            sections.append(Section(text=text, page=page_number))
    return sections


def _load_docx(path: Path) -> list[Section]:
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:  # pragma: no cover
        raise DocumentParseError("缺少 python-docx 依赖，请先执行 pip install -r requirements.txt") from exc

    try:
        document = DocxDocument(str(path))
    except Exception as exc:
        raise DocumentParseError(f"DOCX 文件读取失败：{exc}") from exc

    blocks: list[str] = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))
    return [Section(text="\n".join(blocks))]


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [line.strip() for line in text.split("\n")]
    result: list[str] = []
    for line in lines:
        if not line and result and not result[-1]:
            continue
        result.append(line)
    return "\n".join(result).strip()
