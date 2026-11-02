"""Test unitari per la generazione del Report Medico PDF (PyMuPDF) e relativo endpoint API."""
from datetime import datetime, timezone
import types as pytypes

import pymupdf as fitz
import pytest

from vitality_be.api.routes.reports import export_summary_pdf
from vitality_be.services.pdf_summary import generate_patient_summary_pdf


def _run_coro_sync(coro):
    """Esegue una coroutine puramente in-memory senza aprire socket di loopback su Windows sandbox."""
    try:
        while True:
            coro.send(None)
    except StopIteration as exc:
        return exc.value


class MockQueryBuilder:
    def __init__(self, data: list[dict]):
        self._data = data

    def select(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    def order(self, *_a, **_k):
        return self

    def eq(self, *_a, **_k):
        return self

    def in_(self, *_a, **_k):
        return self

    def not_(self, *_a, **_k):
        return self

    def gte(self, *_a, **_k):
        return self

    def execute(self):
        return pytypes.SimpleNamespace(data=self._data)


class MockUserDb:
    def __init__(self, tables: dict[str, list[dict]]):
        self._tables = tables

    def table(self, name: str):
        data = self._tables.get(name, [])
        return MockQueryBuilder(data)


@pytest.fixture
def sample_complete_data():
    now_iso = datetime.now(timezone.utc).isoformat()
    return {
        "user": [{"name": "Mario", "surname": "Rossi", "birth": "1965-04-12"}],
        "profile": [{"peso": 78.5, "altezza": 175, "sesso": "maschio", "note_anamnestiche": "Diabete T2 da 5 anni"}],
        "diabetes_data": [
            {"metric_type": "GLICEMIA", "metric_value": "125", "measured_at": now_iso},
            {"metric_type": "GLICEMIA", "metric_value": "195", "measured_at": now_iso},
            {"metric_type": "GLICEMIA", "metric_value": "65", "measured_at": now_iso},
            {"metric_type": "PRESSIONE", "metric_value": "135/85", "measured_at": now_iso},
            {"metric_type": "PRESSIONE", "metric_value": "150/95", "measured_at": now_iso},
            {"metric_type": "HBA1C", "metric_value": "6.9", "measured_at": "2026-09-15T08:00:00Z"},
            {"metric_type": "HBA1C", "metric_value": "7.2", "measured_at": "2026-05-10T08:00:00Z"},
            {"metric_type": "UACR", "metric_value": "18.5", "measured_at": "2026-09-15T08:00:00Z"},
            {"metric_type": "EGFR", "metric_value": "88", "measured_at": "2026-09-15T08:00:00Z"},
            {"metric_type": "COLESTEROLO_TOTALE", "metric_value": "185", "measured_at": "2026-09-15T08:00:00Z"},
            {"metric_type": "HDL", "metric_value": "48", "measured_at": "2026-09-15T08:00:00Z"},
            {"metric_type": "LDL", "metric_value": "105", "measured_at": "2026-09-15T08:00:00Z"},
            {"metric_type": "TRIGLICERIDI", "metric_value": "140", "measured_at": "2026-09-15T08:00:00Z"},
        ],
        "therapies": [
            {"id": 1, "therapy_type": "COMPRESSE_ORALI", "drug_name": "Metformina", "dosage": "1000 mg", "frequency": "2 volte al dì", "scheduled_time": "08:00:00", "is_active": True}
        ],
        "therapy_logs": [
            {"therapy_id": 1, "taken_at": now_iso, "status": "taken"}
        ],
        "meals": [
            {"meal_type": "PRANZO", "description": "Riso integrale e salmone", "calories": 650, "carbs_grams": 75, "is_estimated": False, "meal_at": now_iso},
            {"meal_type": "CENA", "description": "Pizza e birra", "calories": 2600, "carbs_grams": 320, "is_estimated": True, "meal_at": now_iso},
        ],
        "physical_activities": [
            {"activity_type": "CAMMINATA", "duration_minutes": 45, "intensity": "MODERATA", "steps": 5200, "notes": "Passeggiata al parco", "activity_at": now_iso}
        ],
        "patient_reports": [
            {"id": 101, "source_name": "Visita Diabetologica Ospedale", "report_type": "VISITA_DIABETOLOGICA", "clean_summary": "Paziente con compenso glicometabolico stabile. Proseguire terapia.", "content": "Visita diabetologica", "extracted_params": {}, "report_at": now_iso, "created_at": now_iso}
        ],
    }


def test_generate_patient_summary_pdf_complete(sample_complete_data):
    db = MockUserDb(sample_complete_data)
    pdf_bytes = _run_coro_sync(generate_patient_summary_pdf(user_db=db, months=3))

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    assert len(doc) >= 1

    first_page_text = doc[0].get_text()
    assert "Mario Rossi" in first_page_text
    assert "GLICEMIA" in first_page_text
    assert "NOTE" in first_page_text

    # Verifica presenza delle sezioni
    full_text = "".join(page.get_text() for page in doc)
    assert "PASTI REGISTRATI" in full_text
    assert "ATTIVITÀ REGISTRATA" in full_text
    assert "PROFILO GLICEMICO" in full_text
    assert "PROFILO RENALE" in full_text
    assert "PROFILO LIPIDICO" in full_text
    assert "ULTIMI REFERTI CLINICI" in full_text

    # Verifica rimozione tag obsoleti e sovrapposizioni come richiesto dall'utente
    assert "[Ultimo valore]" not in full_text
    assert "[Penultimo valore]" not in full_text
    assert "[Terzultimo valore]" not in full_text
    assert "Minima settimanale" not in full_text
    assert "Ultimi 3 mesi (90 giorni)" not in first_page_text
    doc.close()


def test_generate_patient_summary_pdf_omits_empty_sections():
    now_iso = datetime.now(timezone.utc).isoformat()
    sparse_data = {
        "user": [{"name": "Luigi", "surname": "Bianchi", "birth": "1980-01-01"}],
        "profile": [],
        "diabetes_data": [
            {"metric_type": "GLICEMIA", "metric_value": "110", "measured_at": now_iso}
        ],
        "therapies": [],
        "therapy_logs": [],
        "meals": [],
        "physical_activities": [],
        "patient_reports": [],
    }
    db = MockUserDb(sparse_data)
    pdf_bytes = _run_coro_sync(generate_patient_summary_pdf(user_db=db, months=2))

    assert pdf_bytes.startswith(b"%PDF-")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = "".join(page.get_text() for page in doc)

    assert "Luigi Bianchi" in full_text
    assert "GLICEMIA" in full_text
    # Sezioni vuote omesse
    assert "PASTI REGISTRATI" not in full_text
    assert "ATTIVITÀ REGISTRATA" not in full_text
    assert "ULTIMI REFERTI CLINICI" not in full_text
    doc.close()


def test_generate_patient_summary_pdf_invalid_months():
    db = MockUserDb({})
    with pytest.raises(ValueError, match="months deve essere 2, 3 o 4"):
        _run_coro_sync(generate_patient_summary_pdf(user_db=db, months=5))


def test_api_export_summary_pdf_direct(sample_complete_data):
    fake_user = pytypes.SimpleNamespace(id="user-123")
    fake_db = MockUserDb(sample_complete_data)

    response = _run_coro_sync(export_summary_pdf(user=fake_user, user_db=fake_db, months=3))

    assert response.status_code == 200
    assert response.media_type == "application/pdf"
    assert "Report_Clinico_3M_" in response.headers["Content-Disposition"]
    assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate"
    assert response.body.startswith(b"%PDF-")
