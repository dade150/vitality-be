"""Modulo per la generazione del Report Clinico PDF multi-mensile per il Medico.

Caratteristiche:
- 100% deterministico e in RAM (zero chiamate a LLM/Vertex AI, massima privacy GDPR Art. 5 e 25).
- Query eseguite su Supabase tramite UserDbDep (SECURITY INVOKER, rispetto RLS).
- Omissione automatica delle sezioni per cui il paziente non ha registrato dati nel periodo.
- Grafici vettoriali multi-settimanali per Glicemia e Pressione con note fuori target affiancate.
- Segnalazione in rosso delle giornate critiche nei pasti.
- Storico ultimi 3 esami per i profili Glicemico, Renale e Lipidico.
- Sintesi ultimi referti clinici differenziati per tipologia di visita.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import math
import re
from typing import Any, Literal

import pymupdf as fitz
from loguru import logger
from supabase import Client

from vitality_be.services.patient_context import (
    calculate_age_years,
    calculate_bmi,
    normalize_metric_type,
)

# Costanti tipografiche e layout A4 (595.28 x 841.89 pt)
PAGE_WIDTH = 595.28
PAGE_HEIGHT = 841.89
MARGIN_LEFT = 36.0
MARGIN_RIGHT = 36.0
CONTENT_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT  # ~523.28 pt
MAX_Y = 800.0

# Colori RGB normalizzati [0.0 .. 1.0]
COLOR_TEXT_MAIN = (0.12, 0.14, 0.17)
COLOR_TEXT_MUTED = (0.42, 0.45, 0.50)
COLOR_TEXT_ALERT = (0.80, 0.12, 0.12)  # Rosso evidenziazione situazioni gravi
COLOR_PRIMARY = (0.90, 0.32, 0.00)     # Arancio Vitality (#E65100)
COLOR_SECONDARY = (0.10, 0.45, 0.85)   # Blu / Azzurro per curve secondarie
COLOR_BORDER = (0.82, 0.85, 0.88)
COLOR_BG_CARD = (0.98, 0.98, 0.99)
COLOR_TARGET_LINE = (0.70, 0.73, 0.76)

REPORT_TYPE_LABELS: dict[str, str] = {
    "VISITA_DIABETOLOGICA": "Visita Diabetologica",
    "VISITA_CARDIOLOGICA": "Visita Cardiologica",
    "RETINOPATIA": "Screening Retinopatia (Fondo Oculare)",
    "NEFROPATIA": "Valutazione Nefrologica",
    "ESAME_PODOLOGICO": "Esame Podologico (Piede Diabetico)",
    "SALUTE_ORALE": "Salute Orale / Visita Odontoiatrica",
    "DISFUNZIONE_ERETTILE": "Valutazione Andrologica",
    "ANALISI_SANGUE": "Analisi Ematochimiche / Laboratorio",
    "ALTRO": "Altri Referti Specialistici",
}


def _safe_str(val: Any) -> str:
    if val is None:
        return ""
    return str(val).strip()


def _extract_number(val: Any) -> float | None:
    if val is None:
        return None
    cleaned = str(val).strip().replace(",", ".")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _format_date(iso_str: Any, fmt: str = "%d/%m/%Y") -> str:
    if not iso_str:
        return "—"
    raw = str(iso_str).strip()
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        return dt.strftime(fmt)
    except Exception:
        return raw[:10]


_IT_MONTHS = {
    1: "Gennaio", 2: "Febbraio", 3: "Marzo", 4: "Aprile",
    5: "Maggio", 6: "Giugno", 7: "Luglio", 8: "Agosto",
    9: "Settembre", 10: "Ottobre", 11: "Novembre", 12: "Dicembre",
}


def _format_date_italian(iso_str: Any) -> str:
    if not iso_str:
        return "—"
    raw = str(iso_str).strip()
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        m_name = _IT_MONTHS.get(dt.month, str(dt.month))
        return f"{dt.day} {m_name}"
    except Exception:
        return raw[:10]


class PdfReportBuilder:
    """Costruttore vettoriale multipagina del report per il medico."""

    def __init__(self, patient_info: dict[str, Any], months: int, period_start: datetime, period_end: datetime):
        self.doc = fitz.open()
        self.patient_info = patient_info
        self.months = months
        self.period_start = period_start
        self.period_end = period_end
        self.current_page = self._new_page()
        self.y = 36.0

    def _new_page(self) -> fitz.Page:
        page = self.doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        self.y = 36.0
        return page

    def ensure_space(self, needed_pt: float) -> None:
        """Controlla se c'è spazio sufficiente nella pagina corrente, altrimenti crea nuova pagina con mini header."""
        if self.y + needed_pt > MAX_Y:
            self.current_page = self._new_page()
            self._draw_running_header()

    def _draw_running_header(self) -> None:
        name = self.patient_info.get("full_name", "Paziente")
        page = self.current_page
        shape = page.new_shape()
        shape.draw_line(fitz.Point(MARGIN_LEFT, 28), fitz.Point(PAGE_WIDTH - MARGIN_RIGHT, 28))
        shape.finish(color=COLOR_BORDER, width=0.5)
        shape.commit()
        page.insert_text(
            fitz.Point(MARGIN_LEFT, 22),
            f"Vitality Assist — Report Medico ({self.months} Mesi) | Paziente: {name}",
            fontsize=8.5,
            fontname="helv",
            color=COLOR_TEXT_MUTED,
        )
        self.y = 44.0

    def finalize(self) -> bytes:
        """Aggiunge numerazione pagine nel footer e produce il PDF in byte compresso."""
        total = len(self.doc)
        for idx in range(total):
            page = self.doc[idx]
            footer_text = f"Pagina {idx + 1} di {total}  •  Generato con Vitality Assist  •  Documento ad uso clinico"
            shape = page.new_shape()
            shape.draw_line(
                fitz.Point(MARGIN_LEFT, PAGE_HEIGHT - 28),
                fitz.Point(PAGE_WIDTH - MARGIN_RIGHT, PAGE_HEIGHT - 28),
            )
            shape.finish(color=COLOR_BORDER, width=0.5)
            shape.commit()
            page.insert_text(
                fitz.Point(MARGIN_LEFT, PAGE_HEIGHT - 16),
                footer_text,
                fontsize=8,
                fontname="helv",
                color=COLOR_TEXT_MUTED,
            )
        return self.doc.tobytes(garbage=4, deflate=True)

    # -----------------------------------------------------------------------
    # 1. Header Pagina 1
    # -----------------------------------------------------------------------
    def draw_header(self, profile: dict[str, Any], therapies: list[dict[str, Any]], therapy_adherence_pct: int | None) -> None:
        page = self.current_page
        full_name = self.patient_info.get("full_name") or "Paziente"
        age = self.patient_info.get("age")
        age_str = f"{age} anni" if age is not None else "Età n.d."
        sesso = _safe_str(profile.get("sesso") or self.patient_info.get("sesso"))
        sesso_str = f" · Sesso: {sesso}" if sesso else ""

        # Colonna Sinistra: Anagrafica e Antropometria
        page.insert_text(fitz.Point(MARGIN_LEFT, self.y + 16), full_name, fontsize=17, fontname="hebo", color=COLOR_TEXT_MAIN)
        page.insert_text(fitz.Point(MARGIN_LEFT, self.y + 30), f"Età: {age_str}{sesso_str}", fontsize=10.5, fontname="helv", color=COLOR_TEXT_MUTED)

        peso = profile.get("peso")
        altezza = profile.get("altezza")
        bmi_info = calculate_bmi(peso, altezza) if (peso and altezza) else None
        if bmi_info:
            bmi_val, bmi_cat = bmi_info
            antropo_str = f"Antropometria: Peso {peso} kg · Altezza {altezza} cm · BMI {bmi_val} kg/m² ({bmi_cat})"
        elif peso:
            antropo_str = f"Antropometria: Peso {peso} kg"
        else:
            antropo_str = ""

        if antropo_str:
            page.insert_text(fitz.Point(MARGIN_LEFT, self.y + 43), antropo_str, fontsize=9.5, fontname="helv", color=COLOR_TEXT_MAIN)

        # Colonna Destra: Periodo validità
        date_start_str = self.period_start.strftime("%d/%m/%Y")
        date_end_str = self.period_end.strftime("%d/%m/%Y")
        right_x = 345.0
        page.insert_text(fitz.Point(right_x, self.y + 14), "Periodo validità dati:", fontsize=10, fontname="hebo", color=COLOR_PRIMARY)
        page.insert_text(fitz.Point(right_x, self.y + 28), f"Dal {date_start_str} al {date_end_str}", fontsize=10.5, fontname="hebo", color=COLOR_TEXT_MAIN)

        self.y += 50.0

        # Separatore o Fascetta Terapia Attiva
        if therapies:
            active_drugs = []
            for t in therapies:
                name = _safe_str(t.get("drug_name"))
                dosage = _safe_str(t.get("dosage"))
                sched = _safe_str(t.get("scheduled_time"))[:5]
                item_str = name
                if dosage and dosage != "-":
                    item_str += f" ({dosage})"
                if sched:
                    item_str += f" ore {sched}"
                active_drugs.append(item_str)

            ther_text = "Terapia in atto: " + "; ".join(active_drugs)
            if therapy_adherence_pct is not None:
                ther_text += f"  |  Aderenza periodo: {therapy_adherence_pct}%"

            ther_rect = fitz.Rect(MARGIN_LEFT, self.y, PAGE_WIDTH - MARGIN_RIGHT, self.y + 24)
            shape = page.new_shape()
            shape.draw_rect(ther_rect)
            shape.finish(color=COLOR_BORDER, fill=COLOR_BG_CARD, width=0.8)
            shape.commit()

            page.insert_text(fitz.Point(MARGIN_LEFT + 8, self.y + 16), ther_text[:115], fontsize=8.5, fontname="hebo", color=COLOR_TEXT_MAIN)
            self.y += 32.0
        else:
            shape = page.new_shape()
            shape.draw_line(fitz.Point(MARGIN_LEFT, self.y), fitz.Point(PAGE_WIDTH - MARGIN_RIGHT, self.y))
            shape.finish(color=COLOR_BORDER, width=1.0)
            shape.commit()
            self.y += 12.0

    # -----------------------------------------------------------------------
    # 2. Sezione GLICEMIA (Grafico multi-settimanale + Colonna NOTE)
    # -----------------------------------------------------------------------
    def draw_glucose_section(self, glucose_readings: list[dict[str, Any]]) -> None:
        if not glucose_readings:
            return

        # Raccogli e ordina tutte le letture valide
        parsed_readings: list[tuple[datetime, float]] = []
        for r in glucose_readings:
            val = _extract_number(r.get("metric_value"))
            raw_dt = r.get("measured_at")
            if val is None or not raw_dt:
                continue
            try:
                dt_str = str(raw_dt).replace("Z", "+00:00")
                dt = datetime.fromisoformat(dt_str)
            except Exception:
                continue
            parsed_readings.append((dt, val))

        if not parsed_readings:
            return

        parsed_readings.sort(key=lambda t: t[0])
        all_vals = [t[1] for t in parsed_readings]

        self.ensure_space(185.0)
        page = self.current_page
        start_y = self.y

        # Dimensioni e posizionamento grafico e note (senza sovrapposizione!)
        chart_left = MARGIN_LEFT + 22.0  # ~58.0
        chart_w = 260.0
        chart_right = chart_left + chart_w  # 318.0
        notes_x = 345.0  # 27 pt di distacco da chart_right per evitare collisioni
        chart_h = 130.0
        chart_top = start_y + 22.0
        chart_bottom = chart_top + chart_h

        # Titoli sezioni
        page.insert_text(fitz.Point(MARGIN_LEFT, start_y + 12), "GLICEMIA (mg/dL)", fontsize=12, fontname="hebo", color=COLOR_TEXT_MAIN)
        page.insert_text(fitz.Point(notes_x, start_y + 12), "NOTE (Fuori target)", fontsize=12, fontname="hebo", color=COLOR_TEXT_MAIN)

        # Determina periodo effettivo con dati: il grafico parte dalla prima misurazione reale (no mesi nulli)
        first_dt = parsed_readings[0][0]
        effective_start = first_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        span_days = max(1, (self.period_end.date() - effective_start.date()).days)

        if span_days <= 14:
            num_buckets = max(2, span_days + 1)
            bucket_vals: list[list[float]] = [[] for _ in range(num_buckets)]
            for dt, val in parsed_readings:
                b_idx = (dt.date() - effective_start.date()).days
                if 0 <= b_idx < num_buckets:
                    bucket_vals[b_idx].append(val)
        else:
            num_buckets = max(2, math.ceil(span_days / 7.0))
            bucket_vals = [[] for _ in range(num_buckets)]
            for dt, val in parsed_readings:
                w_idx = int((dt - effective_start).total_seconds() // (7 * 86400))
                if 0 <= w_idx < num_buckets:
                    bucket_vals[w_idx].append(val)

        # Note solo per valori alti (> 180 mg/dL): eliminata la soglia minima ipoglicemica come richiesto
        hyper_readings = [(dt, val) for dt, val in parsed_readings if val > 180.0]

        # Statistiche sintetiche
        avg_val = round(sum(all_vals) / len(all_vals)) if all_vals else 0
        hyper_cnt = len(hyper_readings)

        kpi_str = f"Media: {avg_val} mg/dL  ·  Rilevazioni: {len(all_vals)}  ·  Sopra target (>180): {hyper_cnt}"
        page.insert_text(fitz.Point(MARGIN_LEFT, chart_top - 4), kpi_str, fontsize=8, fontname="helv", color=COLOR_TEXT_MUTED)

        # Assi del grafico
        min_y_val = 50.0
        max_y_val = max(220.0, max(all_vals) + 20) if all_vals else 220.0

        def to_y(v: float) -> float:
            ratio = (v - min_y_val) / (max_y_val - min_y_val)
            return chart_bottom - ratio * chart_h

        shape = page.new_shape()
        # Asse Y
        shape.draw_line(fitz.Point(chart_left, chart_top), fitz.Point(chart_left, chart_bottom))
        # Asse X
        shape.draw_line(fitz.Point(chart_left, chart_bottom), fitz.Point(chart_right, chart_bottom))

        # Linea guida SOLO soglia alta 180 tratteggiata (eliminata soglia minima come da richiesta)
        y180 = to_y(180.0)
        shape.draw_line(fitz.Point(chart_left, y180), fitz.Point(chart_right, y180))
        shape.finish(color=COLOR_BORDER, width=0.6, dashes="[2 2] 0")
        shape.commit()

        # Etichetta asse Y
        page.insert_text(fitz.Point(MARGIN_LEFT, y180 + 3), "180", fontsize=7.5, fontname="helv", color=COLOR_TEXT_MUTED)

        # Curva media rilevazioni
        pts_avg: list[fitz.Point] = []
        step_x = chart_w / max(1, num_buckets - 1)

        for i in range(num_buckets):
            vals = bucket_vals[i]
            x_pos = chart_left + i * step_x
            if vals:
                b_avg = sum(vals) / len(vals)
                pts_avg.append(fitz.Point(x_pos, to_y(b_avg)))

        # Traccia curva media (nera/scura)
        if len(pts_avg) >= 2:
            s_avg = page.new_shape()
            for idx in range(len(pts_avg) - 1):
                s_avg.draw_line(pts_avg[idx], pts_avg[idx + 1])
            s_avg.finish(color=COLOR_TEXT_MAIN, width=2.0)
            s_avg.commit()

        # Disegna pallini
        s_dots = page.new_shape()
        for p in pts_avg:
            s_dots.draw_circle(p, 3.0)
        s_dots.finish(color=COLOR_TEXT_MAIN, fill=COLOR_TEXT_MAIN, width=1.0)
        s_dots.commit()

        # Legenda grafico sotto asse X
        page.insert_text(fitz.Point(chart_left, chart_bottom + 12), "• Linea nera: Media rilevazioni", fontsize=7.5, fontname="hebo", color=COLOR_TEXT_MAIN)
        page.insert_text(fitz.Point(chart_left + 140, chart_bottom + 12), "- - - Target: 180 mg/dL", fontsize=7.5, fontname="helv", color=COLOR_TEXT_MUTED)

        # Colonna NOTE a destra (nessuna sovrapposizione con chart_right=318)
        note_y = start_y + 30.0
        if not hyper_readings:
            page.insert_text(
                fitz.Point(notes_x, note_y),
                "• Tutti i valori registrati nel periodo",
                fontsize=8.5,
                fontname="helv",
                color=COLOR_TEXT_MUTED,
            )
            page.insert_text(
                fitz.Point(notes_x + 8, note_y + 11),
                "rientrano nel target (<= 180 mg/dL).",
                fontsize=8.5,
                fontname="helv",
                color=COLOR_TEXT_MUTED,
            )
        else:
            hyper_readings.sort(key=lambda t: t[0], reverse=True)
            for idx, (dt, val) in enumerate(hyper_readings[:10]):
                dt_lbl = dt.strftime("%d/%m")
                val_int = round(val)
                line = f"• Giorno {dt_lbl} valore glicemia {val_int} (alto)"
                is_crit = val > 220
                c = COLOR_TEXT_ALERT if is_crit else COLOR_TEXT_MAIN
                page.insert_text(fitz.Point(notes_x, note_y + idx * 11.5), line, fontsize=8, fontname="helv", color=c)
            if len(hyper_readings) > 10:
                more = len(hyper_readings) - 10
                page.insert_text(
                    fitz.Point(notes_x, note_y + 10 * 11.5 + 4),
                    f"...e altri {more} episodi sopra soglia",
                    fontsize=7.5,
                    fontname="helv",
                    color=COLOR_TEXT_MUTED,
                )

        self.y = max(chart_bottom + 22.0, note_y + min(len(hyper_readings), 10) * 11.5 + 16.0)

    # -----------------------------------------------------------------------
    # 3. Sezione PRESSIONE (Grafico Sistolica/Diastolica + Colonna NOTE)
    # -----------------------------------------------------------------------
    def draw_pressure_section(self, pressure_readings: list[dict[str, Any]]) -> None:
        if not pressure_readings:
            return

        parsed_readings: list[tuple[datetime, float, float, str]] = []
        for r in pressure_readings:
            raw_val = _safe_str(r.get("metric_value"))
            raw_dt = r.get("measured_at")
            if not raw_val or "/" not in raw_val or not raw_dt:
                continue
            parts = raw_val.split("/")
            try:
                s_val = float(parts[0].strip())
                d_val = float(parts[1].strip())
            except ValueError:
                continue
            try:
                dt_str = str(raw_dt).replace("Z", "+00:00")
                dt = datetime.fromisoformat(dt_str)
            except Exception:
                continue
            parsed_readings.append((dt, s_val, d_val, raw_val))

        if not parsed_readings:
            return

        parsed_readings.sort(key=lambda t: t[0])

        self.ensure_space(185.0)
        page = self.current_page
        start_y = self.y

        # Dimensioni e posizionamento grafico e note (senza sovrapposizione!)
        chart_left = MARGIN_LEFT + 22.0  # ~58.0
        chart_w = 260.0
        chart_right = chart_left + chart_w  # 318.0
        notes_x = 345.0  # 27 pt di distacco da chart_right per evitare collisioni
        chart_h = 130.0
        chart_top = start_y + 22.0
        chart_bottom = chart_top + chart_h

        page.insert_text(fitz.Point(MARGIN_LEFT, start_y + 12), "PRESSIONE ARTERIOSA (mmHg)", fontsize=12, fontname="hebo", color=COLOR_TEXT_MAIN)
        page.insert_text(fitz.Point(notes_x, start_y + 12), "NOTE (Soglie pressorie)", fontsize=12, fontname="hebo", color=COLOR_TEXT_MAIN)

        # Determina periodo effettivo con dati (evita mesi vuoti iniziali)
        first_dt = parsed_readings[0][0]
        effective_start = first_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        span_days = max(1, (self.period_end.date() - effective_start.date()).days)

        if span_days <= 14:
            num_buckets = max(2, span_days + 1)
            bucket_sys: list[list[float]] = [[] for _ in range(num_buckets)]
            bucket_dia: list[list[float]] = [[] for _ in range(num_buckets)]
            for dt, s_val, d_val, _ in parsed_readings:
                b_idx = (dt.date() - effective_start.date()).days
                if 0 <= b_idx < num_buckets:
                    bucket_sys[b_idx].append(s_val)
                    bucket_dia[b_idx].append(d_val)
        else:
            num_buckets = max(2, math.ceil(span_days / 7.0))
            bucket_sys = [[] for _ in range(num_buckets)]
            bucket_dia = [[] for _ in range(num_buckets)]
            for dt, s_val, d_val, _ in parsed_readings:
                w_idx = int((dt - effective_start).total_seconds() // (7 * 86400))
                if 0 <= w_idx < num_buckets:
                    bucket_sys[w_idx].append(s_val)
                    bucket_dia[w_idx].append(d_val)

        all_sys = [t[1] for t in parsed_readings]
        all_dia = [t[2] for t in parsed_readings]
        out_of_range_notes: list[tuple[datetime, str]] = []

        for dt, s_val, d_val, raw_val in parsed_readings:
            d_str = dt.strftime("%d/%m")
            if s_val >= 140:
                out_of_range_notes.append((dt, f"• Giorno {d_str} sistolica {int(s_val)} (alta)"))
            elif s_val < 90:
                out_of_range_notes.append((dt, f"• Giorno {d_str} sistolica {int(s_val)} (bassa)"))
            elif d_val >= 90:
                out_of_range_notes.append((dt, f"• Giorno {d_str} diastolica {int(d_val)} (alta)"))
            elif d_val < 60:
                out_of_range_notes.append((dt, f"• Giorno {d_str} diastolica {int(d_val)} (bassa)"))

        avg_s = round(sum(all_sys) / len(all_sys)) if all_sys else 0
        avg_d = round(sum(all_dia) / len(all_dia)) if all_dia else 0
        kpi_str = f"Media periodo: {avg_s}/{avg_d} mmHg  ·  Totale misurazioni: {len(all_sys)}"
        page.insert_text(fitz.Point(MARGIN_LEFT, chart_top - 4), kpi_str, fontsize=8, fontname="helv", color=COLOR_TEXT_MUTED)

        min_y_val = 50.0
        max_y_val = 190.0

        def to_y(v: float) -> float:
            ratio = (v - min_y_val) / (max_y_val - min_y_val)
            return chart_bottom - ratio * chart_h

        shape = page.new_shape()
        shape.draw_line(fitz.Point(chart_left, chart_top), fitz.Point(chart_left, chart_bottom))
        shape.draw_line(fitz.Point(chart_left, chart_bottom), fitz.Point(chart_right, chart_bottom))

        # Linee guida 140 (alta) e 90 (limite)
        y140 = to_y(140.0)
        y90 = to_y(90.0)
        shape.draw_line(fitz.Point(chart_left, y140), fitz.Point(chart_right, y140))
        shape.draw_line(fitz.Point(chart_left, y90), fitz.Point(chart_right, y90))
        shape.finish(color=COLOR_BORDER, width=0.6, dashes="[2 2] 0")
        shape.commit()

        page.insert_text(fitz.Point(MARGIN_LEFT, y140 + 3), "140", fontsize=7.5, fontname="helv", color=COLOR_TEXT_MUTED)
        page.insert_text(fitz.Point(MARGIN_LEFT, y90 + 3), "90", fontsize=7.5, fontname="helv", color=COLOR_TEXT_MUTED)

        # Disegno curve
        pts_s: list[fitz.Point] = []
        pts_d: list[fitz.Point] = []
        step_x = chart_w / max(1, num_buckets - 1)

        for i in range(num_buckets):
            s_list = bucket_sys[i]
            d_list = bucket_dia[i]
            x_pos = chart_left + i * step_x
            if s_list and d_list:
                pts_s.append(fitz.Point(x_pos, to_y(sum(s_list) / len(s_list))))
                pts_d.append(fitz.Point(x_pos, to_y(sum(d_list) / len(d_list))))

        if len(pts_d) >= 2:
            s_d_line = page.new_shape()
            for idx in range(len(pts_d) - 1):
                s_d_line.draw_line(pts_d[idx], pts_d[idx + 1])
            s_d_line.finish(color=COLOR_SECONDARY, width=1.5)
            s_d_line.commit()

        if len(pts_s) >= 2:
            s_s_line = page.new_shape()
            for idx in range(len(pts_s) - 1):
                s_s_line.draw_line(pts_s[idx], pts_s[idx + 1])
            s_s_line.finish(color=COLOR_TEXT_MAIN, width=2.0)
            s_s_line.commit()

        s_dots = page.new_shape()
        for p in pts_d:
            s_dots.draw_circle(p, 2.5)
        s_dots.finish(color=COLOR_SECONDARY, fill=(1, 1, 1), width=1.2)
        s_dots.commit()

        s_dots2 = page.new_shape()
        for p in pts_s:
            s_dots2.draw_circle(p, 3.0)
        s_dots2.finish(color=COLOR_TEXT_MAIN, fill=COLOR_TEXT_MAIN, width=1.0)
        s_dots2.commit()

        page.insert_text(fitz.Point(chart_left, chart_bottom + 12), "• Linea nera: Sistolica media", fontsize=7.5, fontname="hebo", color=COLOR_TEXT_MAIN)
        page.insert_text(fitz.Point(chart_left + 125, chart_bottom + 12), "• Linea blu: Diastolica media", fontsize=7.5, fontname="hebo", color=COLOR_SECONDARY)

        # Colonna NOTE a destra (nessuna sovrapposizione con chart_right=318)
        note_y = start_y + 30.0
        if not out_of_range_notes:
            page.insert_text(fitz.Point(notes_x, note_y), "• Tutte le misurazioni rientrano nei", fontsize=8.5, fontname="helv", color=COLOR_TEXT_MUTED)
            page.insert_text(fitz.Point(notes_x + 8, note_y + 11), "limiti target abituali (<140/90 mmHg).", fontsize=8.5, fontname="helv", color=COLOR_TEXT_MUTED)
        else:
            out_of_range_notes.sort(key=lambda t: t[0], reverse=True)
            for idx, (_, n_str) in enumerate(out_of_range_notes[:10]):
                page.insert_text(fitz.Point(notes_x, note_y + idx * 11.5), n_str[:46], fontsize=8, fontname="helv", color=COLOR_TEXT_MAIN)
            if len(out_of_range_notes) > 10:
                more = len(out_of_range_notes) - 10
                page.insert_text(fitz.Point(notes_x, note_y + 10 * 11.5 + 4), f"...e altri {more} valori sopra soglia", fontsize=7.5, fontname="helv", color=COLOR_TEXT_MUTED)

        self.y = max(chart_bottom + 22.0, note_y + min(len(out_of_range_notes), 10) * 11.5 + 16.0)

    # -----------------------------------------------------------------------
    # 4. Sezione PASTI REGISTRATI (Pagina 2 Prototipo)
    # -----------------------------------------------------------------------
    def draw_meals_section(self, meals: list[dict[str, Any]]) -> None:
        if not meals:
            return  # Ometti se vuota

        self.ensure_space(60.0)
        page = self.current_page

        # Raggruppa per giorno (YYYY-MM-DD)
        daily_kcal: dict[str, int] = {}
        daily_carbs: dict[str, int] = {}

        for m in meals:
            raw_dt = _safe_str(m.get("meal_at"))
            day_key = raw_dt[:10] if len(raw_dt) >= 10 else "Sconosciuto"
            cals = int(m.get("calories") or 0)
            carbs = int(m.get("carbs_grams") or 0)
            daily_kcal[day_key] = daily_kcal.get(day_key, 0) + cals
            daily_carbs[day_key] = daily_carbs.get(day_key, 0) + carbs

        # Ordine cronologico: dalla prima data all'ultima (ascending)
        days_sorted = sorted(daily_kcal.keys())
        if not days_sorted:
            return

        avg_daily_kcal = round(sum(daily_kcal.values()) / len(daily_kcal))

        page.insert_text(fitz.Point(MARGIN_LEFT, self.y + 12), "PASTI REGISTRATI - Kcal contate per giorno", fontsize=11.5, fontname="hebo", color=COLOR_TEXT_MAIN)
        sub_str = f"Giorni monitorati: {len(days_sorted)}  ·  Media giornaliera: {avg_daily_kcal} kcal  ·  (In rosso: apporto calorico/glucidico elevato)"
        page.insert_text(fitz.Point(MARGIN_LEFT, self.y + 24), sub_str, fontsize=8, fontname="helv", color=COLOR_TEXT_MUTED)
        self.y += 32.0

        # Elenco verticale singolo (una sotto l'altra, senza bordo nero)
        for day_key in days_sorted:
            self.ensure_space(14.0)
            page = self.current_page

            kcal = daily_kcal[day_key]
            carbs = daily_carbs[day_key]
            d_formatted = _format_date_italian(day_key)

            # Soglia di allerta in rosso (approvata dall'utente)
            is_critical = kcal > 2500 or carbs > 300 or (kcal > avg_daily_kcal * 1.30 and kcal > 2000)
            text_color = COLOR_TEXT_ALERT if is_critical else COLOR_TEXT_MAIN
            alert_tag = " (elevato)" if is_critical else ""
            carb_suffix = f" · {carbs}g carb" if carbs > 0 else ""

            line = f"• Giorno {d_formatted} - {kcal} kcal{carb_suffix}{alert_tag}"
            font_to_use = "hebo" if is_critical else "helv"
            page.insert_text(fitz.Point(MARGIN_LEFT + 6.0, self.y), line[:75], fontsize=8.5, fontname=font_to_use, color=text_color)
            self.y += 13.5

        self.y += 10.0

    # -----------------------------------------------------------------------
    # 5. Sezione ATTIVITÀ REGISTRATA (Pagina 3 Prototipo)
    # -----------------------------------------------------------------------
    def draw_activity_section(self, activities: list[dict[str, Any]]) -> None:
        if not activities:
            return  # Ometti se vuota

        self.ensure_space(60.0)
        page = self.current_page

        total_mins = sum(int(a.get("duration_minutes") or 0) for a in activities)
        total_steps = sum(int(a.get("steps") or 0) for a in activities)
        weeks = max(1.0, (self.period_end - self.period_start).days / 7.0)
        avg_mins_week = round(total_mins / weeks)

        page.insert_text(fitz.Point(MARGIN_LEFT, self.y + 12), "ATTIVITÀ REGISTRATA", fontsize=11.5, fontname="hebo", color=COLOR_TEXT_MAIN)
        sub_str = f"Totale minuti: {total_mins} min (Media: {avg_mins_week} min/settimana)  ·  Passi registrati: {total_steps:,}".replace(",", ".")
        page.insert_text(fitz.Point(MARGIN_LEFT, self.y + 24), sub_str, fontsize=8, fontname="helv", color=COLOR_TEXT_MUTED)
        self.y += 32.0

        # Ordine cronologico: dalla prima data all'ultima (ascending)
        acts_sorted = sorted(activities, key=lambda a: _safe_str(a.get("activity_at")))

        # Elenco verticale singolo (una sotto l'altra, senza bordo nero)
        for act in acts_sorted:
            self.ensure_space(14.0)
            page = self.current_page

            d_str = _format_date_italian(act.get("activity_at"))
            a_type = _safe_str(act.get("activity_type") or "Attività").capitalize()
            dur = int(act.get("duration_minutes") or 0)
            steps = int(act.get("steps") or 0)

            steps_str = f" ({steps:,} passi)".replace(",", ".") if steps > 0 else ""
            line = f"• Giorno {d_str} - {a_type} {dur} min{steps_str}"
            page.insert_text(fitz.Point(MARGIN_LEFT + 6.0, self.y), line[:75], fontsize=8.5, fontname="helv", color=COLOR_TEXT_MAIN)
            self.y += 13.5

        self.y += 10.0

    # -----------------------------------------------------------------------
    # 6. Tabelle Esami Ematochimici: PROFILO GLICEMICO, RENALE, LIPIDICO
    # -----------------------------------------------------------------------
    def draw_lab_tables(self, lab_history: dict[str, list[tuple[str, str]]]) -> None:
        """Disegna la griglia a 3 valori storici per parametro come da prototipo (Pagine 3 e 4)."""
        profiles = [
            ("PROFILO GLICEMICO", [("HBA1C", "HbA1c", "%")]),
            ("PROFILO RENALE", [("UACR", "UACR", "mg/g"), ("EGFR", "eGFR", "mL/min")]),
            ("PROFILO LIPIDICO", [
                ("COLESTEROLO_TOTALE", "Colesterolo Totale", "mg/dL"),
                ("HDL", "Colesterolo HDL", "mg/dL"),
                ("LDL", "Colesterolo LDL", "mg/dL"),
                ("TRIGLICERIDI", "Trigliceridi", "mg/dL"),
            ]),
        ]

        # Verifica se c'è almeno un dato da mostrare
        has_any = any(code in lab_history and lab_history[code] for _, params in profiles for code, _, _ in params)
        if not has_any:
            return

        self.ensure_space(150.0)

        for prof_title, params in profiles:
            # Mostra il profilo solo se c'è almeno un esame presente
            prof_has_data = any(code in lab_history and lab_history[code] for code, _, _ in params)
            if not prof_has_data:
                continue

            self.ensure_space(32.0 + len(params) * 26.0)
            page = self.current_page

            page.insert_text(fitz.Point(MARGIN_LEFT, self.y + 12), prof_title, fontsize=11, fontname="hebo", color=COLOR_PRIMARY)
            self.y += 18.0

            # Posizioni colonne: nome parametro a sinistra e 3 rilevazioni storiche
            col_param_x = MARGIN_LEFT
            col1_x = MARGIN_LEFT + 155.0
            col2_x = col1_x + 115.0
            col3_x = col2_x + 115.0

            # Nota: eliminati i tag "[Ultimo valore]", "[Penultimo valore]", "[Terzultimo valore]" come richiesto

            for code, label, default_unit in params:
                records = lab_history.get(code, [])
                row_y = self.y + 10.0

                # Nome parametro
                page.insert_text(fitz.Point(col_param_x, row_y), f"{label} ({default_unit})", fontsize=9, fontname="hebo", color=COLOR_TEXT_MAIN)

                # 3 rilevazioni: valore sopra e data posta direttamente sotto al valore
                for col_idx, target_x in enumerate([col1_x, col2_x, col3_x]):
                    if col_idx < len(records):
                        val_str, dt_str = records[col_idx]
                        date_fmt = _format_date(dt_str)
                        page.insert_text(fitz.Point(target_x, row_y), f"{val_str} {default_unit}", fontsize=9, fontname="hebo", color=COLOR_TEXT_MAIN)
                        page.insert_text(fitz.Point(target_x, row_y + 11), date_fmt, fontsize=7.5, fontname="helv", color=COLOR_TEXT_MUTED)
                    else:
                        page.insert_text(fitz.Point(target_x, row_y), "—", fontsize=9, fontname="helv", color=COLOR_TEXT_MUTED)

                self.y += 25.0

            self.y += 8.0

    # -----------------------------------------------------------------------
    # 7. Sezione Finale: ULTIMI REFERTI CLINICI PER TIPOLOGIA DI VISITA
    # -----------------------------------------------------------------------
    def draw_recent_reports_section(self, reports: list[dict[str, Any]]) -> None:
        if not reports:
            return  # Nessun referto caricato

        self.ensure_space(120.0)
        page = self.current_page

        page.insert_text(fitz.Point(MARGIN_LEFT, self.y + 12), "ULTIMI REFERTI CLINICI PER TIPOLOGIA DI VISITA", fontsize=11.5, fontname="hebo", color=COLOR_TEXT_MAIN)
        page.insert_text(fitz.Point(MARGIN_LEFT, self.y + 24), "Sintesi dei documenti clinici e visite specialistiche archiviate", fontsize=8, fontname="helv", color=COLOR_TEXT_MUTED)
        self.y += 30.0

        # Raggruppa per report_type (ordine clinico)
        ordered_types = [
            "VISITA_DIABETOLOGICA",
            "VISITA_CARDIOLOGICA",
            "RETINOPATIA",
            "NEFROPATIA",
            "ESAME_PODOLOGICO",
            "SALUTE_ORALE",
            "DISFUNZIONE_ERETTILE",
            "ANALISI_SANGUE",
            "ALTRO",
        ]

        reports_by_type: dict[str, list[dict[str, Any]]] = {}
        for r in reports:
            rtype = _safe_str(r.get("report_type") or "ALTRO").upper()
            reports_by_type.setdefault(rtype, []).append(r)

        for rtype in ordered_types:
            r_list = reports_by_type.get(rtype, [])
            if not r_list:
                continue

            type_label = REPORT_TYPE_LABELS.get(rtype, rtype.replace("_", " ").title())
            latest_report = r_list[0]  # Il più recente per questa visita

            title = _safe_str(latest_report.get("source_name") or "Referto medico")
            date_str = _format_date(latest_report.get("report_at") or latest_report.get("created_at"))
            summary = _safe_str(latest_report.get("clean_summary") or latest_report.get("content"))

            # Pulisci summary
            summary = re.sub(r"\s+", " ", summary).strip()
            if len(summary) > 320:
                summary = summary[:315] + "..."

            # Calcola altezza box
            box_h = 48.0 if len(summary) > 120 else 38.0
            self.ensure_space(box_h + 8.0)
            page = self.current_page

            card_rect = fitz.Rect(MARGIN_LEFT, self.y, PAGE_WIDTH - MARGIN_RIGHT, self.y + box_h)
            shape = page.new_shape()
            shape.draw_rect(card_rect)
            shape.finish(color=COLOR_BORDER, fill=COLOR_BG_CARD, width=0.8)
            shape.commit()

            header_line = f"• {type_label}  —  Data: {date_str}  [{title}]"
            page.insert_text(fitz.Point(MARGIN_LEFT + 8, self.y + 13), header_line[:85], fontsize=8.5, fontname="hebo", color=COLOR_PRIMARY)

            if summary:
                # Textbox per consentire a capo elegante
                tb_rect = fitz.Rect(MARGIN_LEFT + 8, self.y + 17, PAGE_WIDTH - MARGIN_RIGHT - 8, self.y + box_h - 4)
                page.insert_textbox(tb_rect, summary, fontsize=7.5, fontname="helv", color=COLOR_TEXT_MAIN)

            self.y += box_h + 6.0


def _safe_query_sync(label: str, fn) -> list[dict[str, Any]]:
    try:
        resp = fn()
        data = getattr(resp, "data", None)
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        return []
    except Exception as exc:
        logger.warning("pdf_summary query fallita table={} exc={}", label, type(exc).__name__)
        return []


def _fetch_all_summary_data_sync(user_db: Client, months: int) -> dict[str, Any]:
    """Esegue tutte le query per la generazione del report su Supabase rispettando RLS."""
    now = datetime.now(timezone.utc)
    since_dt = now - timedelta(days=months * 30)
    since_iso = since_dt.isoformat()

    user_rows = _safe_query_sync(
        "user",
        lambda: user_db.table("user").select("name, surname, birth").limit(1).execute(),
    )
    profile_rows = _safe_query_sync(
        "profile",
        lambda: user_db.table("profile").select("peso, altezza, sesso, note_anamnestiche, created_at").order("created_at", desc=True).limit(1).execute(),
    )
    vitals_rows = _safe_query_sync(
        "diabetes_data_period",
        lambda: user_db.table("diabetes_data").select("metric_type, metric_value, measured_at").gte("measured_at", since_iso).order("measured_at", desc=False).limit(3000).execute(),
    )
    all_labs_rows = _safe_query_sync(
        "diabetes_data_labs_history",
        lambda: user_db.table("diabetes_data").select("metric_type, metric_value, measured_at").order("measured_at", desc=True).limit(300).execute(),
    )
    therapies_rows = _safe_query_sync(
        "therapies",
        lambda: user_db.table("therapies").select("id, therapy_type, drug_name, dosage, frequency, scheduled_time, is_active").eq("is_active", True).order("scheduled_time", desc=False).execute(),
    )
    logs_rows = _safe_query_sync(
        "therapy_logs",
        lambda: user_db.table("therapy_logs").select("therapy_id, taken_at, status").gte("taken_at", since_iso).order("taken_at", desc=False).limit(3000).execute(),
    )
    meals_rows = _safe_query_sync(
        "meals",
        lambda: user_db.table("meals").select("meal_type, description, calories, carbs_grams, is_estimated, meal_at").gte("meal_at", since_iso).order("meal_at", desc=False).limit(1500).execute(),
    )
    activities_rows = _safe_query_sync(
        "physical_activities",
        lambda: user_db.table("physical_activities").select("activity_type, duration_minutes, intensity, steps, notes, activity_at").gte("activity_at", since_iso).order("activity_at", desc=False).limit(1500).execute(),
    )
    reports_rows = _safe_query_sync(
        "patient_reports",
        lambda: user_db.table("patient_reports").select("id, source_name, report_type, clean_summary, content, extracted_params, report_at, created_at").order("report_at", desc=True).limit(40).execute(),
    )

    # 1. Anagrafica
    full_name = "Paziente"
    age = None
    if user_rows:
        u = user_rows[0]
        fn = _safe_str(u.get("name"))
        ln = _safe_str(u.get("surname"))
        if fn or ln:
            full_name = f"{fn} {ln}".strip()
        age = calculate_age_years(u.get("birth"), today=now.date())

    patient_info = {
        "full_name": full_name,
        "age": age,
        "sesso": _safe_str(profile_rows[0].get("sesso")) if profile_rows else "",
    }
    profile_info = profile_rows[0] if profile_rows else {}

    # 2. Vitals separati
    glucose_readings = [r for r in vitals_rows if normalize_metric_type(r.get("metric_type")) == "GLICEMIA"]
    pressure_readings = [r for r in vitals_rows if normalize_metric_type(r.get("metric_type")) == "PRESSIONE"]

    # 3. Lab exams storici (ultimi 3 per tipo tra quelli non glicemia/pressione)
    lab_history: dict[str, list[tuple[str, str]]] = {}
    for r in all_labs_rows:
        m_code = normalize_metric_type(r.get("metric_type"))
        if m_code in ("GLICEMIA", "PRESSIONE"):
            continue
        val = _safe_str(r.get("metric_value"))
        dt = _safe_str(r.get("measured_at"))
        if not m_code or not val:
            continue
        curr_list = lab_history.setdefault(m_code, [])
        if len(curr_list) < 3:
            curr_list.append((val, dt))

    # 4. Aderenza terapia nel periodo
    adherence_pct = None
    if therapies_rows and logs_rows:
        total_expected = len(therapies_rows) * (months * 30)
        taken_count = len(logs_rows)
        adherence_pct = min(100, round((taken_count / max(1, total_expected)) * 100))

    return {
        "patient_info": patient_info,
        "profile": profile_info,
        "glucose_readings": glucose_readings,
        "pressure_readings": pressure_readings,
        "lab_history": lab_history,
        "therapies": therapies_rows,
        "therapy_adherence_pct": adherence_pct,
        "meals": meals_rows,
        "activities": activities_rows,
        "reports": reports_rows,
        "period_start": since_dt,
        "period_end": now,
    }


def _build_summary_pdf_sync(user_db: Client, months: int) -> bytes:
    """Esegue il fetch sincrono dei dati e assembla il documento PDF vettoriale con PyMuPDF."""
    data = _fetch_all_summary_data_sync(user_db, months)

    builder = PdfReportBuilder(
        patient_info=data["patient_info"],
        months=months,
        period_start=data["period_start"],
        period_end=data["period_end"],
    )

    # 1. Header e anagrafica
    builder.draw_header(
        profile=data["profile"],
        therapies=data["therapies"],
        therapy_adherence_pct=data["therapy_adherence_pct"],
    )

    # 2. Glicemia
    builder.draw_glucose_section(data["glucose_readings"])

    # 3. Pressione
    builder.draw_pressure_section(data["pressure_readings"])

    # 4. Pasti (omessi se vuoti)
    builder.draw_meals_section(data["meals"])

    # 5. Attività (omesse se vuote)
    builder.draw_activity_section(data["activities"])

    # 6. Esami ematochimici (ultimi 3 per tipo)
    builder.draw_lab_tables(data["lab_history"])

    # 7. Ultimi referti clinici differenziati per visita (omessi se vuoti)
    builder.draw_recent_reports_section(data["reports"])

    return builder.finalize()


async def generate_patient_summary_pdf(user_db: Client, months: int = 3) -> bytes:
    """Entrypoint asincrono per la generazione del PDF riepilogativo per il medico."""
    if months not in (2, 3, 4):
        raise ValueError("Il parametro months deve essere 2, 3 o 4.")

    return _build_summary_pdf_sync(user_db, months)
