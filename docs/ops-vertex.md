# Checklist operativa Vertex AI — Vitality Assist

Legenda: ✅ fatto nel codice · ⬜ da fare da te · ⛔ blocca l'uso di dati reali

**Con dati reali NON si va avanti finché non sono chiusi tutti i punti ⛔.**

## 1. Progetto e accesso
- ⬜ Progetto GCP dedicato con fatturazione attiva.
- ⬜ Abilitare l'API: `gcloud services enable aiplatform.googleapis.com`
- ⬜ In locale: `gcloud auth application-default login` (nessuna chiave nel repo).
- ⬜ Service account per Render con **solo** `roles/aiplatform.user`; la chiave JSON come *secret file* di Render
  (`GOOGLE_APPLICATION_CREDENTIALS=/etc/secrets/<file>`), mai nel repo.
- ✅ `.env` in `.gitignore`, `.env.example` senza valori.

## 2. Costi (denial of wallet)
- ⬜ **Budget alert** sul progetto (soglie 50/90/100%).
- ⬜ **Quote massime** per le richieste Vertex del progetto.
- ✅ Rate limit per utente, semaforo globale, tetto token in uscita, limite caratteri in ingresso (in memoria, per processo).

## 3. Dati e residenza
- ⬜ Verificare con `scripts/smoke_vertex.py` che il modello scelto sia disponibile su `GCP_LOCATION=eu`.
  Se non lo è, l'alternativa è un endpoint regionale UE, che per la documentazione **non garantisce da solo** la residenza dell'elaborazione ML.
- ⬜ Leggere la pagina "Data residency" della documentazione Google Cloud.
- ⬜ Disattivare la cache in memoria di Gemini a livello di progetto (comando REST/PowerShell dalla documentazione; serve `roles/aiplatform.admin`).
- ⬜ Valutare l'eccezione per abuse monitoring / zero data retention.
- ✅ Nessun grounding, nessun logging di richieste/risposte attivato, nessun identificativo del paziente inviato al modello.

## 4. GDPR ⛔
- ⛔ DPA con Google Cloud firmato/accettato.
- ⛔ Informativa privacy che citi il trattamento da parte dell'LLM.
- ⛔ Consenso esplicito (valutare un controllo di `consenso_privacy` lato BE prima di chiamare il modello).
- ⛔ DPIA (dati sanitari di persone fragili).
- ⛔ Revisione del system prompt da parte di un professionista sanitario.
