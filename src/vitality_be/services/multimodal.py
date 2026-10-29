"""Servizio di validazione e sanitizzazione di allegati multimodali in RAM.

Responsabilità:
1. Validazione rigorosa: whitelist MIME, verifica Magic Bytes (file signature) e limiti di dimensione.
2. Sanitizzazione privacy & sicurezza: rimozione metadati EXIF, GPS e camera dalle immagini in RAM.
3. Prevenzione minacce documentali: blocco di file contraffatti e script malevoli (/JavaScript) nei PDF.
4. Generazione metadati strutturati per la traccia di contesto (`chat_message.rag_context`).
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import io
import re
from typing import Literal

from loguru import logger

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "application/pdf",
}

AttachmentKind = Literal[
    "meal_photo",
    "glucometer_photo",
    "medical_report",
    "medication_photo",
    "generic_image",
    "pdf_document",
]


class MultimodalValidationError(ValueError):
    """Sollevata quando un file allegato non rispetta i vincoli di sicurezza o integrità."""


@dataclass(frozen=True)
class ProcessedAttachment:
    clean_bytes: bytes
    mime_type: str
    kind: AttachmentKind
    size_bytes: int
    exif_stripped: bool
    sha256_prefix: str

    def to_rag_context_dict(self) -> dict:
        return {
            "has_attachment": True,
            "kind": self.kind,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "exif_stripped": self.exif_stripped,
            "sha256_prefix": self.sha256_prefix,
            "vision_context_hint": f"File allegato dall'utente ({self.kind}, {self.mime_type})",
        }


def _verify_magic_bytes(data: bytes, mime_type: str) -> None:
    """Verifica che la firma binaria reale corrisponda al tipo MIME dichiarato."""
    if len(data) < 12:
        raise MultimodalValidationError("File troppo piccolo o corrotto.")

    if mime_type == "image/jpeg":
        if not data.startswith(b"\xff\xd8\xff"):
            raise MultimodalValidationError("Firma JPEG non valida (magic bytes errati).")
    elif mime_type == "image/png":
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise MultimodalValidationError("Firma PNG non valida (magic bytes errati).")
    elif mime_type == "image/webp":
        if not (data.startswith(b"RIFF") and data[8:12] == b"WEBP"):
            raise MultimodalValidationError("Firma WebP non valida (magic bytes errati).")
    elif mime_type == "application/pdf":
        if not data.startswith(b"%PDF-"):
            raise MultimodalValidationError("Firma PDF non valida (magic bytes errati).")
    else:
        raise MultimodalValidationError(f"Tipo MIME non supportato: {mime_type}")


def _strip_jpeg_exif(data: bytes) -> bytes:
    """Rimuove i segmenti EXIF, XMP e commenti da un'immagine JPEG in RAM.

    Scansiona i marker standard JPEG (da APP1 0xFFE1 ad APP15 0xFFEF e COM 0xFFFE)
    senza richiedere librerie native C, preservando la tabella quantizzazione, Huffman e i dati immagine.
    """
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return data

    out = bytearray(b"\xff\xd8")
    idx = 2
    data_len = len(data)

    while idx < data_len:
        if data[idx] != 0xFF:
            # Dati non allineati o SOS stream, conserviamo il resto
            out.extend(data[idx:])
            break

        # Salta eventuali 0xFF di padding
        while idx < data_len and data[idx] == 0xFF:
            idx += 1

        if idx >= data_len:
            break

        marker = data[idx]
        idx += 1

        # SOS (Start of Scan) o EOI (End of Image): da qui in poi sono dati compressi
        if marker == 0xDA or marker == 0xD9:
            out.append(0xFF)
            out.append(marker)
            out.extend(data[idx:])
            break

        # Marcatori senza payload di lunghezza
        if marker in (0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0x01):
            out.append(0xFF)
            out.append(marker)
            continue

        if idx + 2 > data_len:
            break

        seg_len = (data[idx] << 8) | data[idx + 1]
        if seg_len < 2 or idx + seg_len > data_len:
            break

        segment = data[idx - 2 : idx + seg_len]
        # Segnaposto EXIF/XMP/IPTC/Commenti: APP1..APP15 (0xE1..0xEF) e COM (0xFE)
        is_metadata = (0xE1 <= marker <= 0xEF) or (marker == 0xFE)

        if not is_metadata:
            out.extend(segment)

        idx += seg_len

    return bytes(out) if len(out) > 4 else data


def _sanitize_image_metadata(data: bytes, mime_type: str) -> tuple[bytes, bool]:
    """Sanitizza i metadati privacy da un'immagine in memoria."""
    if mime_type == "image/jpeg":
        clean = _strip_jpeg_exif(data)
        return clean, True

    # Per PNG e WebP proviamo a usare PIL se disponibile, altrimenti validiamo la struttura
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        # Crea una nuova immagine pulita senza EXIF/metadati
        clean_buf = io.BytesIO()
        format_name = "PNG" if mime_type == "image/png" else "WEBP"
        img.save(clean_buf, format=format_name)
        return clean_buf.getvalue(), True
    except Exception:
        # Fallback sicuro: se PIL non è presente, i dati grezzi PNG/WebP vengono accettati
        return data, False


def _check_pdf_safety(data: bytes) -> None:
    """Verifica che un PDF non contenga macro o script JavaScript malevoli."""
    # Controllo rapido su script attivi
    lower = data.lower()
    if b"/javascript" in lower or b"/js" in lower:
        # Verifica se è un tag reale di azione PDF
        if re.search(rb"/(?:javascript|js)\b", lower):
            raise MultimodalValidationError("Il PDF contiene script attivi non ammessi per ragioni di sicurezza.")


def process_and_sanitize_attachment(
    *,
    data_base64: str,
    mime_type: str,
    kind: AttachmentKind = "generic_image",
    max_bytes: int = 5 * 1024 * 1024,
) -> ProcessedAttachment:
    """Decodifica, valida e ripulisce in RAM un allegato multimodale.

    Solleva `MultimodalValidationError` se il file è invalido, corrotto o pericoloso.
    """
    if mime_type not in ALLOWED_MIME_TYPES:
        raise MultimodalValidationError(
            f"Tipo MIME '{mime_type}' non consentito. Formati ammessi: {', '.join(sorted(ALLOWED_MIME_TYPES))}"
        )

    try:
        raw_bytes = base64.b64decode(data_base64, validate=True)
    except Exception:
        raise MultimodalValidationError("Dati base64 dell'allegato non validi.") from None

    if len(raw_bytes) > max_bytes:
        raise MultimodalValidationError(
            f"Allegato troppo grande ({len(raw_bytes)} byte). Il limite massimo è {max_bytes} byte."
        )

    _verify_magic_bytes(raw_bytes, mime_type)

    exif_stripped = False
    if mime_type in ("image/jpeg", "image/png", "image/webp"):
        clean_bytes, exif_stripped = _sanitize_image_metadata(raw_bytes, mime_type)
    elif mime_type == "application/pdf":
        _check_pdf_safety(raw_bytes)
        clean_bytes = raw_bytes
    else:
        clean_bytes = raw_bytes

    sha256_prefix = hashlib.sha256(clean_bytes).hexdigest()[:12]

    logger.info(
        "multimodal processed kind={} mime={} bytes_in={} bytes_out={} exif_stripped={}",
        kind,
        mime_type,
        len(raw_bytes),
        len(clean_bytes),
        exif_stripped,
    )

    return ProcessedAttachment(
        clean_bytes=clean_bytes,
        mime_type=mime_type,
        kind=kind,
        size_bytes=len(clean_bytes),
        exif_stripped=exif_stripped,
        sha256_prefix=sha256_prefix,
    )
