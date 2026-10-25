# Contesto di Progetto: Back-End Coaching Conversazionale Diabete Tipo 2 (T2D)

## 1. REGOLE DI SICUREZZA ASSOLUTE (CRITICO)
- **DIVIETO DI ACCESSO A CREDENZIALI:** Non aprire, leggere, stampare in terminale o modificare MAI il file `.env` o file di credenziali Google Cloud (`*.json`).
- Per conoscere le variabili d'ambiente disponibili, controlla solo come vengono richiamate nel codice (`os.getenv`) o chiedi all'utente.
- Prima di modificare qualsiasi file esistente, spiega brevemente la modifica all'utente e attendi conferma.

## 2. ARCHITETTURA ATTUALE (BACK-END PYTHON)
- **Modello LLM:** Gemini 2.5 tramite Vertex AI (in esecuzione locale per ambiente di sviluppo).
- **Database & Vector Store:** Supabase (PostgreSQL + `pgvector`). Gli embedding del corpus clinico sono già stati caricati a DB.
- **Ruolo del BE:** Esporre API per il Front-End Flutter, gestire la pipeline Dual-RAG (corpus clinico statico + dati dinamici del paziente), applicare i Guardrail di sicurezza e gestire l'estrazione strutturata di dati clinici dalla chat.

## 3. REQUISITI DI DOMINIO (TESI MAGISTRALE)
1. **Pipeline RAG Ibrida (Statico + Dinamico):**
   - Recuperare i chunk pertinenti dagli embedding già presenti su Supabase (Linee guida ADA 2026, SID/AMD 2025, framework ADCES7).
   - Recuperare i dati specifici dell'utente autenticato da Supabase (profilo, diario glicemico SMBG, pasti, attività fisica, terapia, pressione, parametri HbA1c, eGFR, UACR, lipidi).
2. **Guardrails e Sicurezza Clinica:**
   - Il sistema fornisce SOLO supporto educativo, motivazionale e di autogestione.
   - **VIETATO:** Fornire diagnosi mediche, prescrivere nuovi farmaci o consigliare modifiche ai dosaggi terapeutici (es. dosi di insulina o ipoglicemizzanti).
   - **Emergenze:** Intercettare valori fuori range (ipoglicemia/iperglicemia severa) e attivare avvisi di sicurezza immediati.
   - Prevenire attacchi di *indirect prompt injection* e allucinazioni estrinseche/intrinseche.
3. **Inserimento Valori da Chat:**
   - L'assistente deve riconoscere nel testo della conversazione eventuali dati clinici comunicati dall'utente (es. glicemia, pasto, peso, pressione) e preparare un payload strutturato che il FE salverà solo dopo il consenso esplicito dell'utente.