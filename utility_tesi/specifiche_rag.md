# Appunti Operativi per la Tesi — Architettura Dual-RAG

Documento sintetico di riferimento sulle scelte progettuali e implementative del modulo **Dual-RAG** nel back-end (`vitality-be`).

---

## 1. Visione d'Insieme: Perché un "Dual-RAG"
A differenza di un RAG tradizionale (che recupera solo estratti da documenti statici), l'assistente conversazionale per il Diabete di Tipo 2 (T2D) combina a runtime **due flussi di recupero paralleli (`asyncio.gather`)** iniettati come blocchi dati distinti nell'ultimo turno utente:

1. **RAG Statico Ibrido (`<contesto_clinico>`):** recupera le evidenze mediche validate dal corpus documentale (Linee guida ADA 2026, SID/AMD 2025, framework ADCES7, LARN/SINU, ISS, Ministero della Salute).
2. **RAG Dinamico Paziente (`<contesto_paziente>`):** recupera lo stato clinico e comportamentale aggiornato dell'utente autenticato (profilo, diario glicemico SMBG, esami ematochimici, terapie, pasti, attività fisica, visite), applicando **pseudonimizzazione e minimizzazione dei dati (GDPR Art. 5 e 25)** prima dell'invio a Vertex AI.

---

## 2. Parte 1 — RAG Statico Documentale (Ricerca Ibrida + RRF)

### 2.1 Corpus e Indicizzazione (`public.document_chunks`)
- **Modello di Embedding:** `gemini-embedding-2` su endpoint regionale europeo Vertex AI (`https://aiplatform.eu.rep.googleapis.com`).
- **Dimensionalità:** `1536` dimensioni (scelta per bilanciare precisione semantica, occupazione memoria e supporto nativo dell'indice HNSW su `pgvector`).
- **Formattazione asimmetrica query/documento:**
  - In fase di indicizzazione: documento formattato con titolo e testo.
  - A runtime sulla domanda utente: prefisso `task: search result | query: <domanda_utente>`.
- **Doppio Indice su PostgreSQL (Supabase):**
  1. **Indice Vettoriale HNSW (`vector_cosine_ops`):** ricerca semantica densa (cattura sinonimi, parafrasi e concetti clinici anche tra italiano e inglese).
  2. **Indice Full-Text Search GIN (`tsvector` bilingue `italian` + `english`):** ricerca lessicale esatta con pesatura dei campi (`A` per `source_name`/`heading_path`, `B` per `summary`, `C` per `content`). Fondamentale per sigle mediche, farmaci, codici esenzione (`013`) e soglie numeriche (`70 mg/dL`, `HbA1c`).

### 2.2 Algoritmo di Retrieval (`public.hybrid_search_chunks`)
Implementato tramite RPC SQL su Supabase invocata da `src/vitality_be/services/retriever.py`:
1. Estrae i **Top-10** chunk per similarità coseno vettoriale (`semantic_top_k = 10`).
2. Estrae i **Top-10** chunk per ranking lessicale FTS (`fts_top_k = 10` con `websearch_to_tsquery`).
3. **Fusione Reciprocal Rank Fusion (RRF, $k = 60$):**
   $$\text{RRF\_score}(d) = \frac{1}{k + \text{rank}_{\text{sem}}(d)} + \frac{1}{k + \text{rank}_{\text{fts}}(d)}$$
4. Restituisce i **Top-8** chunk finali (`match_count = 8`), formattati nel blocco `<contesto_clinico>` con indicazione esplicita di `[Fonte N: <source_name> | Sezione: <heading_path>]`.

---

## 3. Parte 2 — RAG Dinamico (Contesto Paziente e Privacy-by-Design)

### 3.1 Recupero Dati in Singolo Round-Trip (`src/vitality_be/services/patient_context.py`)
- L'accesso avviene esclusivamente tramite il client Supabase autenticato con il JWT dell'utente (`get_user_db`), soggetto alle policy **Row-Level Security (RLS)** (`auth.uid() = user_id`).
- Per azzerare l'overhead di rete (evitando 9 chiamate HTTP distinte verso Supabase), la RPC `public.get_patient_context_bundle()` (`SECURITY INVOKER`) aggrega in un'unica query JSONB le 9 tabelle utente (`user`, `profile`, `diabetes_data`, `therapies`, `therapy_logs`, `meals`, `physical_activities`, `appointments`, `patient_reports`), mantenendo un fallback automatico sulle singole tabelle.

### 3.2 Elaborazione e Pre-calcolo Lato Back-End (Riduzione Allucinazioni Matematiche)
Prima di costruire il blocco `<contesto_paziente>`, il back-end esegue calcoli deterministici in Python per evitare che l'LLM debba fare aritmetica sui dati grezzi:
- **Antropometria:** calcolo automatico dell'**IMC (BMI)** ($\text{peso} / \text{altezza}^2$) e classificazione clinica (normopeso, sovrappeso, obesità grado I/II-III).
- **Diario Glicemico (SMBG):** sulle ultime 10 misurazioni calcola **media, minimo, massimo**, numero di valori **in target (`70-180 mg/dL`)**, episodi di **ipoglicemia (`< 70 mg/dL`)** e **iperglicemia (`> 180 mg/dL`)**.
- **Esami Ematochimici e Pressione:** normalizza le etichette (`HbA1c`, `eGFR`, `UACR`, `Colesterolo Totale`, `LDL`, `HDL`, `Trigliceridi`) ed estrae l'ultimo valore refertato con relativa data.
- **Attività Fisica:** somma i minuti di esercizio registrati negli ultimi 7 giorni confrontandoli con il target delle linee guida ADA/SID (`150 min/settimana`).
- **Terapie e Pasti:** elenca i farmaci attivi con posologia/orari, le ultime assunzioni (`therapy_logs`) e gli ultimi 7 pasti con grammi di carboidrati e calorie.

### 3.3 Privacy, Generalizzazione e Tokenizzazione Reversibile
Nessun identificativo diretto (PII) lascia mai il back-end verso Google Cloud / Vertex AI:
1. **Rimozione totale:** cognome, email, numero di telefono, codice fiscale e `user_id` non vengono mai inclusi nel prompt.
2. **Generalizzazione dei quasi-identificatori:** la data di nascita (`birth`) non viene inviata; viene convertita nell'**età in anni compiuti** (dato clinicamente necessario per contestualizzare i target glicemici senza esporre la data esatta).
3. **Scrubbing Regex sui testi liberi (`scrub_free_text`):** note anamnestiche, descrizioni dei pasti, referti e messaggi della chat vengono filtrati con espressioni regolari per oscurare email (`[EMAIL_RIMOSSA]`), numeri di telefono (`[TELEFONO_RIMOSSO]`), codici fiscali (`[CF_RIMOSSO]`) ed eventuali occorrenze del nome/cognome dell'utente.
4. **Token Reversibili (`{{NOME}}` e `{{MEDICO}}`):**
   - Il nome di battesimo dell'utente e il nome del medico curante vengono salvati solo in RAM (`PatientMaskingMap`) per la durata della singola richiesta HTTP.
   - Al modello LLM vengono passati solo i segnaposto `{{NOME}}` e `{{MEDICO}}`.
   - Al ritorno della risposta da Vertex AI, `rehydrate_assistant_reply(...)` sostituisce i segnaposto con i nomi reali prima di inviare il JSON al Front-End Flutter.

---

## 4. Iniezione Sicura nel Prompt (`src/vitality_be/services/chat.py`)

Per difendere il sistema da **Indirect Prompt Injection** (es. istruzioni malevole nascoste dentro una nota alimentare o un referto):
- Il **System Prompt** (`src/vitality_be/llm/prompts/chat_system.md`) viaggia isolato nel canale `system_instruction` di Vertex AI e **non contiene mai dati dinamici o chunk recuperati**.
- I contesti vengono allegati esclusivamente nell'ultimo messaggio `user` con delimitatori XML strutturati:
  ```xml
  <contesto_paziente>
  ...scheda clinica pseudonimizzata...
  </contesto_paziente>

  <contesto_clinico>
  ...estratti linee guida ADA/SID/ADCES7...
  </contesto_clinico>

  <messaggio_utente>
  ...domanda dell'utente de-identificata...
  </messaggio_utente>
  ```
- **Degradazione Graziosa (Fault Tolerance):** se l'embedder, la ricerca vettoriale o una qualsiasi delle tabelle utente non risponde, il sistema registra solo il tipo di eccezione nei log (senza mai loggare il contenuto clinico) e prosegue la conversazione con i blocchi disponibili.
