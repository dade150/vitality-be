import base64
import pytest

from vitality_be.services.multimodal import (
    MultimodalValidationError,
    process_and_sanitize_attachment,
)

# Minimal valid JPEG: SOI (\xff\xd8) + APP0 marker (\xff\xe0\x00\x10JFIF...) + DQT + SOF + SOS + EOI (\xff\xd9)
MINIMAL_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00\x43\x00" + b"\x01" * 64 +
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00"
    b"\xff\xd9"
)

# JPEG con segmento EXIF APP1 contenente metadati GPS fittizi
JPEG_WITH_EXIF = (
    b"\xff\xd8"
    b"\xff\xe1\x00\x1aExif\x00\x00GPS_COORDINATES_45_1234"
    b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00\x43\x00" + b"\x01" * 64 +
    b"\xff\xd9"
)

# Minimal valid PNG
MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)

# Minimal valid PDF
MINIMAL_PDF = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF"

# Malicious PDF with JavaScript tag
MALICIOUS_PDF = b"%PDF-1.4\n1 0 obj\n<< /Type /Action /S /JavaScript /JS (app.alert('evil')) >>\nendobj\n%%EOF"


def test_process_valid_jpeg():
    b64 = base64.b64encode(MINIMAL_JPEG).decode()
    res = process_and_sanitize_attachment(
        data_base64=b64,
        mime_type="image/jpeg",
        kind="meal_photo",
    )
    assert res.mime_type == "image/jpeg"
    assert res.kind == "meal_photo"
    assert res.size_bytes > 0
    assert len(res.sha256_prefix) == 12
    rag_dict = res.to_rag_context_dict()
    assert rag_dict["has_attachment"] is True
    assert rag_dict["kind"] == "meal_photo"


def test_strip_exif_gps_from_jpeg():
    b64 = base64.b64encode(JPEG_WITH_EXIF).decode()
    res = process_and_sanitize_attachment(
        data_base64=b64,
        mime_type="image/jpeg",
        kind="meal_photo",
    )
    assert res.exif_stripped is True
    # I dati GPS/EXIF non devono essere presenti nei byte puliti
    assert b"GPS_COORDINATES" not in res.clean_bytes
    assert b"Exif" not in res.clean_bytes


def test_process_valid_png_and_pdf():
    png_b64 = base64.b64encode(MINIMAL_PNG).decode()
    png_res = process_and_sanitize_attachment(
        data_base64=png_b64,
        mime_type="image/png",
        kind="glucometer_photo",
    )
    assert png_res.mime_type == "image/png"

    pdf_b64 = base64.b64encode(MINIMAL_PDF).decode()
    pdf_res = process_and_sanitize_attachment(
        data_base64=pdf_b64,
        mime_type="application/pdf",
        kind="medical_report",
    )
    assert pdf_res.mime_type == "application/pdf"


def test_reject_magic_bytes_mismatch():
    # File di testo mascherato da JPEG
    fake_jpeg = b"Ciao sono un file di testo normale"
    b64 = base64.b64encode(fake_jpeg).decode()
    with pytest.raises(MultimodalValidationError, match="magic bytes errati"):
        process_and_sanitize_attachment(
            data_base64=b64,
            mime_type="image/jpeg",
        )


def test_reject_unsupported_mime():
    with pytest.raises(MultimodalValidationError, match="non consentito"):
        process_and_sanitize_attachment(
            data_base64="dGVzdA==",
            mime_type="application/zip",
        )


def test_reject_malicious_pdf():
    b64 = base64.b64encode(MALICIOUS_PDF).decode()
    with pytest.raises(MultimodalValidationError, match="script attivi"):
        process_and_sanitize_attachment(
            data_base64=b64,
            mime_type="application/pdf",
            kind="medical_report",
        )


def test_reject_oversized_attachment():
    big_data = b"\xff\xd8\xff\xe0" + b"\x00" * 2000
    b64 = base64.b64encode(big_data).decode()
    with pytest.raises(MultimodalValidationError, match="troppo grande"):
        process_and_sanitize_attachment(
            data_base64=b64,
            mime_type="image/jpeg",
            max_bytes=1000,
        )
