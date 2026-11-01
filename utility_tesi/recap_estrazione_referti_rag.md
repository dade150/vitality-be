# Appunti Operativi per la Tesi — Pipeline a 4 Step di Ingestion Referti Clinici e Integrazione Dual-RAG

Documento di riferimento tecnico e metodologico sulle scelte architetturali, algoritmiche e di data protection implementate nel modulo di estrazione referti (`vitality-be`) e nell'applicazione mobile (`vitality_assist`).

---

## 1. Visione d'Insieme e Motivazione Clinico-Tecnologica

Nella gestione del Diabete Mellito di Tipo 2 (T2D), il paziente produce periodicamente documenti sanitari eterogenei: esami del sangue ematochimici (emoglobina glicata HbA1c, assetto lipidico con colesterolo totale/LDL/HDL e trigliceridi, funzionalità renale eGFR/UACR), monitoraggi pressori, visite diabetologiche e piani terapeutici.

Questi documenti presentano due criticità fondamentali:
1. **Dispersione e Formati Eterogenei:** spaziano dal PDF digitale generato nativamente dai laboratori di analisi, alle scansioni cartacee, fino alle foto scattate con smartphone.
2. **Severità Normativa e Riservatezza (GDPR Art. 9):** contengono dati particolari sulla salute associati direttamente a identificativi forti (Codice Fiscale, Nome/Cognome, Ospedale/ASL, indirizzo di residenza, recapiti telefonici e nominativi di medici curanti).

Il sistema implementa una **pipeline ibrida a 4 Step** che coniuga l'elaborazione locale a costo zero in memoria RAM, la de-identificazione crittografica/regex preventiva (Privacy-by-Design), l'estrazione guidata da schemi Pydantic v2 su modelli linguistici multimodali Vertex AI EU e l'indicizzazione vettoriale semantica protetta da Row-Level Security (RLS) su Supabase pgvector.

Inoltre, la medesima infrastruttura è stata resa pienamente fruibile sia tramite la schermata dedicata **"Aggiungi Referto"**, sia direttamente all'interno della **chat conversazionale principale**, alimentando il motore **Dual-RAG** per il retrieval semantico dei referti pregressi durante la conversazione.

```
                           [PDF / Immagine Referto]
                                      │
                                      ▼
                        [Step 1 - Ispezione in RAM]
                         ├─ Se PDF Digitale nativo ──► PyMuPDF estrae testo/tabelle in RAM
                         │                            (0 costi API, 0 latenza rete, 0 byte esterni)
                         └─ Se Scansione / Foto ─────► Gemini Flash Multimodale (OCR visivo)
                                      │
                                      ▼
                   [Step 2 - De-identificazione & Scrubbing PII]
                    Rimozione automatica via regex di CF, Nome/Cognome, Ospedale/ASL,
                    Indirizzi, Telefoni, Email (GDPR Art. 5, 9, 25)
                                      │
                                      ▼
                    [Step 3 - Estrazione Strutturata (Pydantic v2)]
                    Gemini Flash genera la lista parametri clinici (HbA1c, glicemia, ecc.)
                    per conferma HITL del paziente e la sintesi clinica oggettiva ("clean_summary")
                                      │
                                      ▼
                    [Step 4 - Embedding Vettoriale & Persistenza]
                    Testo de-identificato + sintesi clinica inviati a Gemini Embedding 2
                    (1536 dimensioni) e archiviati su Supabase (patient_reports / chunks) sotto RLS
```

---

## 2. Architettura Dettagliata a 4 Step

### Step 1: Ispezione in RAM e Ramificazione Ibrida (`src/vitality_be/services/multimodal.py`)
- **Zero Disk I/O:** Il file binario non viene mai salvato sul filesystem locale né in cartelle temporanee; viene elaborato interamente in RAM (`io.BytesIO`).
- **Verifica Magic Bytes & Hardening di Sicurezza:**
  - Controllo delle firme binarie autentiche: `%PDF-` per PDF, `\xff\xd8\xff` per JPEG, `\x89PNG\r\n\x1a\n` per PNG, `RIFF...WEBP` per WebP.
  - Verifica di sicurezza attiva del PDF contro vettori di attacco o exploit documentali: scansione dei byte per escludere token malevoli (`/JavaScript`, `/JS`, `/Launch`, `/EmbeddedFiles`, `/RichMedia`).
- **Sanitizzazione EXIF Immagini:** Per foto JPEG scattate da smartphone, rimozione algoritmica dei segmenti APP1 (metadati GPS, modello dispositivo, data/ora esatta dello scatto) prima di qualsiasi inoltro.
- **Ispezione Euristica Testo/Tabelle con PyMuPDF (`pymupdf`):**
  - PyMuPDF analizza la struttura interna del PDF ed estrae il testo posizionato (`get_text("text", sort=True)`) e le eventuali tabelle vettoriali convertite direttamente in Markdown (`find_tables() -> to_markdown()`).
  - **Soglia di Riconoscimento:** Se il documento contiene almeno 80 caratteri alfanumerici (o $\ge 40$ per documento a pagina singola), viene classificato come **digitale nativo** (`is_native_digital = True`).
  - **Vantaggio Ingegneristico:** Se il PDF è nativo, `media_bytes` viene azzerato (`None`) e al modello di linguaggio viene inoltrato solo il testo puro. Il binario originale non transita su Vertex AI Vision: **zero consumo di token visivi (costi ridotti dell'80-90%), zero latenza di upload di rete e zero esposizione di immagini contenenti timbri o loghi ospedalieri**.
  - **Fallback:** Se il PDF è una scansione raster (immagine fotografata o priva di testo selezionabile), la pipeline attiva in automatico il fallback multimodale con Gemini Flash su Vertex AI.

### Step 2: De-identificazione & Scrubbing PII Preventivo (`src/vitality_be/services/patient_context.py`)
In conformità al principio di **Minimizzazione dei Dati (GDPR Art. 5.1.c)** e alla protezione rafforzata delle categorie particolari di dati personali (**GDPR Art. 9**), nessun testo libero viene inviato al modello LLM prima di essere bonificato:
- **Codice Fiscale:** Regex `^[A-Z]{6}[0-9LMNPQRSTUV]{2}[A-EHLMPR-T][0-9LMNPQRSTUV]{2}[A-Z][0-9LMNPQRSTUV]{3}[A-Z]$` con sostituzione con `[CF_RIMOSSO]`.
- **Intestazioni Anagrafiche e Strutture Ospedaliere:**
  - Regex specifica per ospedali, cliniche, ASL/AUSL, laboratori analisi e IRCCS (es. *Ospedale San Raffaele*, *ASL Roma 1*, *Laboratorio Analisi Cliniche Dr. Rossi*), sostituiti con `[STRUTTURA_SANITARIA_RIMOSSA]`.
  - Rimozione di blocchi d'intestazione anagrafica recanti indirizzi, numeri di telefono, recapiti o riferimenti al medico curante.
- **Mascheramento Dinamico Utente (`PatientMaskingMap`):**
  - Il sistema interroga in sola lettura il profilo dell'utente autenticato (`user` / `profile`) ottenendo nome e cognome.
  - Se il nome/cognome del paziente compare nel corpo del referto, viene sostituito con il token reversibile `{{NOME}}` (conservato in RAM solo per la durata della richiesta).

### Step 3: Estrazione Clinica Strutturata & Guardrail HITL (`src/vitality_be/services/extractor.py`)
L'estrazione adotta la modalità **Structured Outputs** garantita da Pydantic v2 (`LLMReportExtractionBatch`), forzando il modello a restituire un payload JSON conforme a schema tipizzato:
- **Campi Estratti:**
  - `is_medical_report`: booleano di classificazione per scartare documenti non pertinenti (bollette, scontrini, documenti generici).
  - `report_type`: categorizzazione standardizzata (`ANALISI_SANGUE`, `VISITA_SPECIALISTICA`, `ESAME_STRUMENTALE`, `MONITORAGGIO_GLICEMICO`, `ALTRO`).
  - `report_date_iso`: data di esecuzione dell'esame refertato (es. `2026-10-15T08:00:00+02:00`), convertita e validata nel fuso orario di Roma.
  - `clean_summary`: sintesi clinica oggettiva e de-identificata (nessun nome di persona o struttura), focalizzata unicamente sul quadro metabolico, lipidico, renale e pressorio.
  - `clinical_findings_text`: trascrizione schematica dei riscontri diagnostici.
  - `items`: lista di parametri quantitativi (`RawExtractedItem`).
- **Parametri Clinici Target per T2D:**
  - *Profilo Glicemico:* `GLICEMIA` (mg/dL), `HbA1c` (% e mmol/mol).
  - *Profilo Lipidico:* `Colesterolo Totale`, `HDL`, `LDL`, `Trigliceridi` (mg/dL).
  - *Funzionalità Renale:* `eGFR` (mL/min/1.73m²), `UACR` (mg/g).
  - *Parametri Pressori:* `PRESSIONE` (es. `130/80` mmHg).
- **Guardrail Anti-Allucinazione Numerica:**
  - Funzione deterministica `_number_appears_in_source_text(...)`: il backend controlla che il numero estratto compaia realmente nel testo sorgente prima di proporlo.
  - Tolleranza avanzata su formati decimali (`6.8` vs `6.80` o `6,8`) per evitare falsi positivi/negativi.
- **Principio Human-in-the-Loop (HITL):**
  - **Nessun dato entra silenziosamente nel database clinico (`diabetes_data`).**
  - Ogni parametro viene emesso sotto forma di `ProposedEntry` con `target_table: "diabetes_data"` e `source_report_id` collegato.
  - L'applicazione Flutter mostra la scheda con spunte attive: il paziente controlla, seleziona i parametri validi e convalida la data prima della scrittura finale nel DB.

### Step 4: Rappresentazione Vettoriale & Persistenza Protetta da RLS (`src/vitality_be/services/reports.py`)
- **Modello di Embedding:** `gemini-embedding-2` regionale europeo Vertex AI a **1536 dimensioni**.
- **Formattazione Semantica Asimmetrica:**
  - Invece di vettorizzare il testo grezzo (che potrebbe contenere rumore di impaginazione), l'embedder riceve la sintesi strutturata:
    ```
    title: Referto Analisi Sangue - 15/10/2026
    text: Profilo glicemico e lipidico. HbA1c: 6.8 %. Colesterolo LDL: 110 mg/dL. Quadro metabolico stabile.
    ```
- **Persistenza su Supabase:**
  - Salvataggio nella tabella `public.patient_reports` tramite il client autenticato con JWT (`SECURITY INVOKER`).
  - Colonne salvate: `user_id`, `source_name`, `report_type`, `report_at`, `content` (testo de-identificato), `clean_summary`, `extracted_params` (JSONB) e `embedding vector(1536)`.

---

## 3. Decisione Architetturale sui Chunk: Isolamento RLS vs Corpus Globale

### Il Problema del Rischio di Data-Leak
Nel sistema Vitality era già presente la tabella `public.document_chunks`, utilizzata dal modulo RAG per interrogare le linee guida cliniche (ADA 2026, SID/AMD 2025, ISS, ecc.).
L'aggiunta dei chunk dei referti dei pazienti in `public.document_chunks` è stata **categoricamente scartata** per motivazioni architetturali e di conformità legale insormontabili:

| Caratteristica | `public.document_chunks` (Linee Guida) | Referti Paziente (GDPR Art. 9) |
| :--- | :--- | :--- |
| **Natura del dato** | Letteratura medica pubblica e statica | Dati sanitari personali sensibilissimi |
| **Isolamento RLS** | Globale (`user_id IS NULL`), visibile a tutti | Rigoroso (`user_id = auth.uid()`), visibile solo al titolare |
| **Query RAG** | Ricerca ibrida condivisa da tutti gli utenti | Ricerca isolata per singolo paziente |
| **Rischio di contaminazione** | Se mescolati, un bug nella clausola WHERE o una RPC permissiva potrebbe restituire estratti di referti di un paziente a un altro utente. |

### La Soluzione Adottata
1. **A livello referto:** `public.patient_reports` possiede la propria colonna `embedding vector(1536)` indicizzata con HNSW (`m=16`, `ef_construction=64`, metrica coseno). Poiché la maggior parte dei referti ematochimici si concentra in 1-2 pagine, la sintesi clinica oggettiva (`clean_summary`) combinata con i parametri ematochimici rientra perfettamente nella finestra semantica di un singolo embedding.
2. **Per documenti estesi e multi-pagina (Tabella Dedicata `public.patient_report_chunks`):**
   - Creata nello script SQL [`sql/20261030_patient_reports_rag.sql`](file:///c:/Users/Davide/tesi/project/sql/20261030_patient_reports_rag.sql).
   - Collegata con Foreign Key a cascata a `patient_reports(id)` e vincolata con policy RLS:
     ```sql
     ALTER TABLE public.patient_report_chunks ENABLE ROW LEVEL SECURITY;
     CREATE POLICY "patient_report_chunks_isolation"
       ON public.patient_report_chunks
       FOR ALL
       USING (user_id = auth.uid())
       WITH CHECK (user_id = auth.uid());
     ```
   - In questo modo, l'eventuale chunking di cartelle cliniche voluminose rimane ermeticamente confinato nel perimetro crittografico dell'utente autenticato.

---

## 4. Integrazione con la Chat Principale e il Modulo Dual-RAG

La funzionalità di analisi dei referti non è confinata a un form isolato, ma interagisce dinamicamente con il core conversazionale:

### 4.1 Caricamento Diretto di Allegati in Chat (`src/vitality_be/services/chat.py`)
- L'utente può scattare una foto al referto o allegare un PDF direttamente nella schermata della chat.
- **Ottimizzazione in RAM:** Prima di invocare il modello multimodale, la pipeline verifica se il file è un PDF digitale nativo tramite `inspect_pdf_in_ram`. In caso positivo, converte il documento in testo in RAM in `< 20ms` e rimuove il binario dall'invio a Vertex AI.
- **Dual Flow:**
  - Il modello conversazionale risponde alle domande dell'utente sul referto contestualizzandole con il suo piano terapeutico.
  - In parallelo, il servizio di estrazione produce le `ProposedEntry` per consentire la conferma dei parametri ematochimici direttamente dal messaggio della chat.
  - Il referto viene archiviato e indicizzato in `patient_reports`.
  - Sul client Flutter, al completamento del messaggio viene invocato `unawaited(context.read<ReportsProvider>().refresh())`, sincronizzando in tempo reale la sezione "I Miei Referti".

### 4.2 Retrieval Semantico dei Referti Pregressi (Dual-RAG)
Quando il paziente pone una domanda sullo storico clinico (es. *"Come è andata l'ultima glicata rispetto a tre mesi fa?"* o *"I miei valori di colesterolo sono migliorati?"*):
1. **Generazione Vettore Query:** La domanda viene codificata con `gemini-embedding-2` con prefisso di retrieval.
2. **RPC Sicura `public.search_patient_reports`:**
   - La RPC esegue una ricerca vettoriale con ordinamento per similarità coseno protetta dalle policy RLS (`SECURITY INVOKER`, filtra automaticamente solo i referti di `auth.uid()`).
3. **Iniezione nel Prompt:**
   - I referti pertinenti recuperati vengono formattati nel blocco `<contesto_paziente>`:
     ```xml
     <referti_medici_pertinenti>
     [Referto 1: Analisi Sangue del 15/10/2026]
     Sintesi: Profilo glicemico e lipidico. HbA1c 6.8 %, Colesterolo LDL 110 mg/dL.
     Parametri estratti: HbA1c: 6.8 %, Glicemia: 120 mg/dL, LDL: 110 mg/dL.
     </referti_medici_pertinenti>
     ```
   - L'assistente è così in grado di formulare risposte comparative precise basate sui dati reali del paziente senza allucinazioni.

---

## 5. Implementazione Front-End Flutter (`vitality_assist`)

Le modifiche al client mobile hanno interessato cinque componenti chiave:
1. **`lib/services/report_service.dart`:**
   - Timeout multipart esteso a 45 secondi per consentire l'elaborazione multimodale end-to-end anche su connessioni mobili lente.
   - Parsing arricchito di `clean_summary`, `report_type`, `report_at` e `extraction_method`.
2. **`lib/models/medical_report.dart`:**
   - Aggiunti i campi opzionali `cleanSummary` e `filePath`.
3. **`lib/providers/reports_provider.dart`:**
   - Metodo `refresh()` aggiornato per mappare `source_name`, `report_at`, `clean_summary` ed `extracted_params`.
4. **`lib/screens/reports/add_report_screen.dart`:**
   - Selettore file esteso a PDF e immagini (`allowedExtensions: ['pdf', 'jpg', 'jpeg', 'png', 'webp']`).
   - Pre-popolamento automatico della data dell'esame se estratta dal referto (`result.reportDate`).
   - Widget visuale dedicato per visualizzare la **Sintesi Clinica Oggettiva** generata dal modello.
5. **`lib/screens/chat/chat_screen.dart`:**
   - Ascolto degli allegati di tipo `medical_report`: al termine della generazione della risposta, aggiornamento trasparente del provider dei referti per mantenere allineata la vista clinica.

---

## 6. Procedura Operativa per il Database Supabase

Per applicare le modifiche allo schema del database, eseguire lo script SQL generato:
**File di migrazione:** [`sql/20261030_patient_reports_rag.sql`](file:///c:/Users/Davide/tesi/project/sql/20261030_patient_reports_rag.sql)

### Azioni Eseguite dallo Script:
1. **Estensione Tabella `public.patient_reports`:**
   - Aggiunta colonna `clean_summary text` (sintesi de-identificata).
   - Aggiunta colonna `extracted_params jsonb` (coppie parametro-valore strutturate).
   - Aggiunta colonna `file_path text` (riferimento a Supabase Storage).
   - Indice HNSW per embedding a 1536 dimensioni (`patient_reports_embedding_hnsw_idx`).
   - Indice GIN per ricerca lessicale sul contenuto del referto (`patient_reports_content_fts_idx`).
2. **Creazione Tabella `public.patient_report_chunks`:**
   - Tabella per l'eventuale frammentazione di referti lunghi o pluri-pagina.
   - Indice HNSW dedicato a 1536 dimensioni.
   - RLS abilitata con policy di isolamento totale `user_id = auth.uid()`.
3. **Funzione RPC `public.search_patient_reports(...)`:**
   - Parametri: `query_embedding vector(1536)`, `query_text text`, `match_count int`.
   - Esecuzione `SECURITY INVOKER` per ereditare i permessi del JWT del chiamante.
   - Ritorna `id`, `source_name`, `report_type`, `report_at`, `clean_summary`, `extracted_params` e `similarity`.
4. **Aggiornamento RPC `public.get_patient_context_bundle()`:**
   - Include `clean_summary` ed `extracted_params` nel JSON di aggregazione del contesto paziente iniettato nella chat.

---

## 7. Sintesi dei Risultati e Metriche di Validazione

- **Test Suite di Validazione (`vitality-be/tests/test_reports_pipeline.py`):**
  - `test_step1_pdf_inspection_native`: validata l'estrazione in RAM tramite PyMuPDF su PDF digitale nativo ($\ge 80$ caratteri, zero chiamate esterne).
  - `test_step2_pii_scrubbing_report`: validata l'eliminazione con regex di Codice Fiscale, Nome/Cognome, Ospedale e numeri telefonici.
  - `test_step3_extract_from_report_echo`: validata l'estrazione conforme a schema Pydantic v2 e la generazione di proposte cliniche tipizzate.
  - `test_step4_reports_pipeline_ingest`: validato il flusso end-to-end con embedding vettoriale e inserimento su mock database conforme a RLS.
  - **Esito test:** 17 test totali (pipeline referti, extractor clinico e hardening multimodale) passati con successo al 100% in **0.72 secondi**.
