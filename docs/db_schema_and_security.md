# Riferimento Schema Database Supabase & Checklist Sicurezza RLS

Questo documento fotografa lo schema del database PostgreSQL (`public`) su Supabase per il progetto **Vitality Assist** e raccoglie i miglioramenti di sicurezza RLS da applicare in una fase successiva.

> **Nota di sicurezza:** Questo file contiene esclusivamente metadati strutturali (nomi di tabelle, colonne, tipi `ENUM` e definizioni logiche delle policy RLS). Non contiene e non deve mai contenere credenziali, URL di produzione, chiavi API o dati reali dei pazienti.

---

## 1. Tipi `ENUM` definiti nello schema `public`

| Nome Enum | Valori ammessi | Note applicative |
| :--- | :--- | :--- |
| `activity_enum` | `CAMMINATA`, `CORSA`, `BICICLETTA`, `PALESTRA`, `NUOTO`, `ALTRO` | Usato in `physical_activities.activity_type` |
| `intensity_enum` | `LEGGERA`, `MODERATA`, `INTENSA` | Default `'MODERATA'` in `physical_activities.intensity` |
| `meal_enum` | `COLAZIONE`, `SPUNTINO_MATTINA`, `PRANZO`, `SPUNTINO_POMERIGGIO`, `CENA`, `FUORI_PASTO`, `SPUNTINO` | Usato in `meals.meal_type` |
| `metric_enum` | `GLICEMIA`, `PRESSIONE`, `HbA1c`, `eGFR`, `UACR`, `Colesterolo Totale`, `HDL`, `LDL`, `Trigliceridi`, `HBA1C`, `EGFR`, `COLESTEROLO_TOTALE`, `TRIGLICERIDI` | **Attenzione:** contiene varianti miste e maiuscole. Il back-end Python normalizza sempre in uppercase con underscore. |
| `report_enum` | `ANALISI_SANGUE`, `VISITA_DIABETOLOGICA`, `VISITA_CARDIOLOGICA`, `RETINOPATIA`, `NEFROPATIA`, `ESAME_PODOLOGICO`, `SALUTE_ORALE`, `DISFUNZIONE_ERETTILE`, `ALTRO` | Usato in `patient_reports.report_type` |
| `sesso_biologico` | `maschio`, `femmina`, `altro` | Usato in `profile.sesso` |
| `therapy_type_enum` | `COMPRESSE_ORALI`, `INSULINA_BASALE`, `INSULINA_RAPIDA`, `GLP1_INIETTIVO`, `ALTRO` | Usato in `therapies.therapy_type` |
| `font_size_enum` | `PICCOLO`, `MEDIO`, `GRANDE` | Usato in `user_settings.font_size` |
| `user_role` | `admin`, `paziente`, `paziente_premium` | Usato in `user.role` |

---

## 2. Mappa delle Tabelle (`public`)

### 2.1 Anagrafica, Profilo Clinico e Impostazioni
- **`public.user`** *(Identificativi diretti - MAI inviare a Vertex AI)*
  - `id` (`uuid`, PK, combacia con `auth.uid()`)
  - `name` (`varchar`, NOT NULL) — Nome di battesimo (usato solo per il token reversibile `{{NOME}}`)
  - `surname` (`varchar`, NOT NULL) — Cognome (oscuro all'LLM)
  - `birth` (`date`, NOT NULL) — Data di nascita (convertita in **età in anni compiuti** prima dell'invio all'LLM)
  - `email` (`varchar`, NOT NULL) — Oscurata all'LLM
  - `phone` (`varchar`, NULL) — Oscurato all'LLM
  - `consenso_privacy` (`bool`, NOT NULL), `consenso_privacy_at` (`timestamptz`, NULL)
  - `role` (`user_role`, NOT NULL)
  - `created_at` (`timestamptz`, NOT NULL)

- **`public.profile`** *(Profilo antropometrico e anamnestico)*
  - `id` (`int8`, PK), `user_id` (`uuid`, NOT NULL)
  - `peso` (`numeric`, NOT NULL, in kg)
  - `altezza` (`int8`, NOT NULL, in cm)
  - `sesso` (`sesso_biologico`, NOT NULL)
  - `note_anamnestiche` (`text`, NULL)
  - `created_at` (`timestamptz`, NOT NULL)

- **`public.user_settings`** *(Preferenze UI Front-End)*
  - `id` (`int8`, PK), `user_id` (`uuid`, NOT NULL)
  - `show_meals`, `show_activities`, `show_reports`, `show_appointments` (`bool`, NOT NULL)
  - `font_size` (`font_size_enum`, NOT NULL), `dark_mode` (`bool`, NOT NULL)
  - `notify_therapy`, `notify_appointments` (`bool`, NOT NULL)
  - `created_at`, `updated_at` (`timestamptz`, NOT NULL)

### 2.2 Monitoraggio Clinico, Terapie e Stile di Vita
- **`public.diabetes_data`** *(Glicemie SMBG, Pressione Arteriosa, Esami Ematochimici)*
  - `id` (`int8`, PK), `user_id` (`uuid`, NOT NULL, default `auth.uid()`)
  - `metric_type` (`metric_enum`, NOT NULL)
  - `metric_value` (`text`, NOT NULL — es. `"118"`, `"130/80"`, `"6.8"`)
  - `measured_at` (`timestamptz`, NOT NULL)
  - `source_report_id` (`int8`, NULL)
  - `created_at` (`timestamptz`, NOT NULL)

- **`public.therapies`** *(Piano terapeutico attivo)*
  - `id` (`int8`, PK), `user_id` (`uuid`, NOT NULL)
  - `therapy_type` (`therapy_type_enum`, NOT NULL)
  - `drug_name` (`text`, NOT NULL)
  - `dosage` (`text`, NOT NULL)
  - `frequency` (`text`, NOT NULL)
  - `scheduled_time` (`time`, NULL)
  - `notes` (`text`, NULL)
  - `is_active` (`bool`, NOT NULL)
  - `created_at` (`timestamptz`, NOT NULL)

- **`public.therapy_logs`** *(Registro assunzioni farmaci)*
  - `id` (`int8`, PK), `therapy_id` (`int8`, FK -> `therapies.id`), `user_id` (`uuid`, NOT NULL)
  - `taken_at` (`timestamptz`, NOT NULL)
  - `status` (`text`, NOT NULL)
  - `created_at` (`timestamptz`, NOT NULL)

- **`public.meals`** *(Diario alimentare)*
  - `id` (`int8`, PK), `user_id` (`uuid`, NOT NULL)
  - `meal_type` (`meal_enum`, NOT NULL)
  - `description` (`text`, NOT NULL)
  - `carbs_grams` (`int4`, NULL), `calories` (`int4`, NULL)
  - `is_estimated` (`bool`, NOT NULL, default `false`)
  - `meal_at` (`timestamptz`, NOT NULL), `created_at` (`timestamptz`, NOT NULL)

- **`public.physical_activities`** *(Diario attività motoria)*
  - `id` (`int8`, PK), `user_id` (`uuid`, NOT NULL)
  - `activity_type` (`activity_enum`, NOT NULL)
  - `duration_minutes` (`int4`, NOT NULL)
  - `intensity` (`intensity_enum`, NOT NULL, default `'MODERATA'`)
  - `steps` (`int4`, NULL)
  - `notes` (`text`, NULL)
  - `activity_at` (`timestamptz`, NOT NULL), `created_at` (`timestamptz`, NOT NULL)

- **`public.appointments`** *(Visite mediche e appuntamenti)*
  - `id` (`int8`, PK), `user_id` (`uuid`, NOT NULL)
  - `title` (`text`, NOT NULL)
  - `doctor_name` (`text`, NULL — sostituito con token reversibile `{{MEDICO}}` verso l'LLM)
  - `appointment_at` (`timestamptz`, NOT NULL)
  - `notes` (`text`, NULL)
  - `created_at` (`timestamptz`, NOT NULL)

- **`public.patient_reports`** *(Referti clinici caricati dall'utente)*
  - `id` (`int8`, PK), `user_id` (`uuid`, NOT NULL)
  - `source_name` (`text`, NOT NULL)
  - `report_type` (`report_enum`, NOT NULL)
  - `content` (`text`, NOT NULL)
  - `embedding` (`vector`, NULL)
  - `report_at` (`timestamptz`, NULL), `created_at` (`timestamptz`, NOT NULL)

### 2.3 Chat & Knowledge Base Statica
- **`public.chat_session`**: `id`, `user_id`, `title`, `created_at`
- **`public.chat_message`**: `id`, `session_id`, `role`, `content`, `rag_context` (`jsonb`), `tokens_used`, `created_at`
- **`public.document_chunks`**: Corpus clinico per RAG Statico Ibrido (`embedding` vector 1536 dim + `fts` tsvector)
- **`public.alimenti` & `public.ricette`**: Tabelle nutrizionali di consultazione pubblica (`SELECT` abilitata per `anon` e `authenticated`)

---

## 3. RPC SQL per il RAG Dinamico (`public.get_patient_context_bundle`)

Questa funzione PostgreSQL (`SECURITY INVOKER`, quindi **rispetta al 100% le policy RLS dell'utente autenticato**) permette al back-end Python di recuperare tutte le 9 tabelle del contesto paziente in **1 sola chiamata HTTP** anziché 9 chiamate separate.

> **Nota:** Il servizio [`patient_context.py`](file:///c:/Users/Davide/tesi/project/vitality-be/src/vitality_be/services/patient_context.py) prova automaticamente a invocare questa RPC; se non è ancora stata creata su Supabase, ripiega in automatico sulle query `.table(...)` senza alcun errore.

```sql
CREATE OR REPLACE FUNCTION public.get_patient_context_bundle()
RETURNS jsonb
LANGUAGE sql
STABLE
SECURITY INVOKER
SET search_path = public
AS $$
    SELECT jsonb_build_object(
        'user', COALESCE(
            (SELECT jsonb_agg(row_to_json(u))
             FROM (
                 SELECT name, surname, birth
                 FROM public."user"
                 WHERE id = auth.uid()
                 LIMIT 1
             ) u),
            '[]'::jsonb
        ),
        'profile', COALESCE(
            (SELECT jsonb_agg(row_to_json(p))
             FROM (
                 SELECT peso, altezza, sesso, note_anamnestiche, created_at
                 FROM public.profile
                 WHERE user_id = auth.uid()
                 ORDER BY created_at DESC
                 LIMIT 1
             ) p),
            '[]'::jsonb
        ),
        'diabetes_data', COALESCE(
            (SELECT jsonb_agg(row_to_json(d))
             FROM (
                 SELECT metric_type, metric_value, measured_at, created_at
                 FROM public.diabetes_data
                 WHERE user_id = auth.uid()
                 ORDER BY measured_at DESC
                 LIMIT 60
             ) d),
            '[]'::jsonb
        ),
        'therapies', COALESCE(
            (SELECT jsonb_agg(row_to_json(t))
             FROM (
                 SELECT id, therapy_type, drug_name, dosage, frequency, scheduled_time, notes, is_active
                 FROM public.therapies
                 WHERE user_id = auth.uid() AND is_active = true
                 ORDER BY created_at DESC
                 LIMIT 15
             ) t),
            '[]'::jsonb
        ),
        'therapy_logs', COALESCE(
            (SELECT jsonb_agg(row_to_json(tl))
             FROM (
                 SELECT therapy_id, taken_at, status
                 FROM public.therapy_logs
                 WHERE user_id = auth.uid()
                 ORDER BY taken_at DESC
                 LIMIT 10
             ) tl),
            '[]'::jsonb
        ),
        'meals', COALESCE(
            (SELECT jsonb_agg(row_to_json(m))
             FROM (
                 SELECT meal_type, description, carbs_grams, calories, meal_at, is_estimated, created_at
                 FROM public.meals
                 WHERE user_id = auth.uid()
                 ORDER BY meal_at DESC
                 LIMIT 7
             ) m),
            '[]'::jsonb
        ),
        'physical_activities', COALESCE(
            (SELECT jsonb_agg(row_to_json(pa))
             FROM (
                 SELECT activity_type, duration_minutes, intensity, steps, notes, activity_at, created_at
                 FROM public.physical_activities
                 WHERE user_id = auth.uid()
                 ORDER BY activity_at DESC
                 LIMIT 7
             ) pa),
            '[]'::jsonb
        ),
        'appointments', COALESCE(
            (SELECT jsonb_agg(row_to_json(ap))
             FROM (
                 SELECT title, doctor_name, appointment_at, notes
                 FROM public.appointments
                 WHERE user_id = auth.uid()
                 ORDER BY appointment_at DESC
                 LIMIT 5
             ) ap),
            '[]'::jsonb
        ),
        'patient_reports', COALESCE(
            (SELECT jsonb_agg(row_to_json(pr))
             FROM (
                 SELECT report_type, source_name, content, report_at, created_at
                 FROM public.patient_reports
                 WHERE user_id = auth.uid()
                 ORDER BY report_at DESC NULLS LAST
                 LIMIT 3
             ) pr),
            '[]'::jsonb
        )
    );
$$;

REVOKE ALL ON FUNCTION public.get_patient_context_bundle() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.get_patient_context_bundle() TO authenticated;
```

---

## 4. TODO Sicurezza Database (Hardening RLS da applicare più avanti)

Questi due interventi sono **trasparenti al 100% per il Front-End Flutter** (non richiedono alcuna modifica al codice Dart/Flutter) e blindano il database contro accessi non autorizzati diretti tramite le API PostgREST di Supabase.

### TODO 1 (Priorità Alta) — Prevenire Privilege Escalation su `public.user.role`
**Problema:** Tutte le tabelle cliniche consentono la lettura se `current_user_role() = 'admin'::user_role`. Sulla tabella `public."user"`, la policy `update_own` permette all'utente di aggiornare la propria riga (`auth.uid() = id`) senza impedire la modifica della colonna `role`. Un utente malintenzionato potrebbe invocare `PATCH /rest/v1/user` con il proprio JWT e impostare `{"role": "admin"}`.

**Soluzione SQL (Trigger `BEFORE UPDATE` trasparente per il FE):**
```sql
CREATE OR REPLACE FUNCTION public.prevent_user_role_escalation()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    -- Se la colonna role viene modificata e chi esegue la query non è già admin o service_role
    IF NEW.role IS DISTINCT FROM OLD.role THEN
        IF current_setting('request.jwt.claim.role', true) IS DISTINCT FROM 'service_role'
           AND public.current_user_role() IS DISTINCT FROM 'admin'::public.user_role THEN
            RAISE EXCEPTION 'Modifica del ruolo non consentita.'
                USING ERRCODE = '42501';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_prevent_user_role_escalation ON public."user";
CREATE TRIGGER trg_prevent_user_role_escalation
BEFORE UPDATE ON public."user"
FOR EACH ROW
EXECUTE FUNCTION public.prevent_user_role_escalation();
```

### TODO 2 (Priorità Media) — Restringere le policy delle tabelle cliniche da `TO public` a `TO authenticated`
**Problema:** Le policy sulle tabelle personali (`appointments`, `chat_message`, `chat_session`, `diabetes_data`, `meals`, `patient_reports`, `physical_activities`, `profile`, `therapies`, `user`, `user_settings`) hanno `roles = {public}` anziché `{authenticated}`. Anche se `auth.uid()` vale `NULL` per `anon`, PostgreSQL valuta comunque l'espressione `OR (current_user_role() = 'admin'::user_role)` per le richieste anonime.

**Soluzione SQL:**
```sql
-- Esempio di hardening (ripetibile per ciascuna tabella personale):
ALTER POLICY "select_own_or_admin" ON public.diabetes_data TO authenticated;
ALTER POLICY "insert_own"          ON public.diabetes_data TO authenticated;
ALTER POLICY "update_own"          ON public.diabetes_data TO authenticated;
ALTER POLICY "delete_own"          ON public.diabetes_data TO authenticated;

ALTER POLICY "select_own_or_admin" ON public.profile TO authenticated;
ALTER POLICY "insert_own"          ON public.profile TO authenticated;
ALTER POLICY "update_own"          ON public.profile TO authenticated;
ALTER POLICY "delete_own"          ON public.profile TO authenticated;

ALTER POLICY "select_own_or_admin" ON public.meals TO authenticated;
ALTER POLICY "insert_own"          ON public.meals TO authenticated;
ALTER POLICY "update_own"          ON public.meals TO authenticated;
ALTER POLICY "delete_own"          ON public.meals TO authenticated;

ALTER POLICY "select_own_or_admin" ON public.physical_activities TO authenticated;
ALTER POLICY "insert_own"          ON public.physical_activities TO authenticated;
ALTER POLICY "update_own"          ON public.physical_activities TO authenticated;
ALTER POLICY "delete_own"          ON public.physical_activities TO authenticated;

ALTER POLICY "select_own_or_admin" ON public.therapies TO authenticated;
ALTER POLICY "insert_own"          ON public.therapies TO authenticated;
ALTER POLICY "update_own"          ON public.therapies TO authenticated;
ALTER POLICY "delete_own"          ON public.therapies TO authenticated;

ALTER POLICY "select_own_or_admin" ON public.therapy_logs TO authenticated;
ALTER POLICY "delete_own"          ON public.therapy_logs TO authenticated;

ALTER POLICY "select_own_or_admin" ON public.appointments TO authenticated;
ALTER POLICY "insert_own"          ON public.appointments TO authenticated;
ALTER POLICY "update_own"          ON public.appointments TO authenticated;
ALTER POLICY "delete_own"          ON public.appointments TO authenticated;

ALTER POLICY "select_own_or_admin" ON public.patient_reports TO authenticated;
ALTER POLICY "insert_own"          ON public.patient_reports TO authenticated;
ALTER POLICY "update_own"          ON public.patient_reports TO authenticated;
ALTER POLICY "delete_own"          ON public.patient_reports TO authenticated;

ALTER POLICY "select_own_or_admin" ON public.chat_session TO authenticated;
ALTER POLICY "insert_own"          ON public.chat_session TO authenticated;
ALTER POLICY "delete_own"          ON public.chat_session TO authenticated;

ALTER POLICY "select_own_or_admin" ON public.chat_message TO authenticated;
ALTER POLICY "insert_own"          ON public.chat_message TO authenticated;

ALTER POLICY "select_own_or_admin" ON public."user" TO authenticated;
ALTER POLICY "update_own"          ON public."user" TO authenticated;

ALTER POLICY "select_own_or_admin" ON public.user_settings TO authenticated;
ALTER POLICY "insert_own"          ON public.user_settings TO authenticated;
ALTER POLICY "update_own"          ON public.user_settings TO authenticated;
ALTER POLICY "delete_own"          ON public.user_settings TO authenticated;
```
