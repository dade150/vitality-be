-- =============================================================================
-- 07_align_rpc_to_thesis_table2.sql
-- Allinea:
-- 1. public.get_patient_context_bundle() alle finestre temporali e limiti di Tabella 2 (sez. 3.4 tesi)
-- 2. public.search_patient_reports() alla ricerca ibrida HNSW + GIN FTS con fusione RRF (k=60)
-- =============================================================================

BEGIN;

CREATE OR REPLACE FUNCTION public.get_patient_context_bundle()
RETURNS jsonb
LANGUAGE sql
STABLE
SECURITY INVOKER
SET search_path TO 'public'
AS $function$
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
                 (
                     SELECT metric_type, metric_value, measured_at, created_at
                     FROM public.diabetes_data
                     WHERE user_id = auth.uid()
                       AND upper(replace(metric_type::text, ' ', '_')) = 'GLICEMIA'
                       AND COALESCE(measured_at, created_at) >= (now() - interval '7 days')
                     ORDER BY COALESCE(measured_at, created_at) DESC
                     LIMIT 20
                 )
                 UNION ALL
                 (
                     SELECT metric_type, metric_value, measured_at, created_at
                     FROM public.diabetes_data
                     WHERE user_id = auth.uid()
                       AND upper(replace(metric_type::text, ' ', '_')) = 'PRESSIONE'
                       AND COALESCE(measured_at, created_at) >= (now() - interval '14 days')
                     ORDER BY COALESCE(measured_at, created_at) DESC
                     LIMIT 10
                 )
                 UNION ALL
                 (
                     SELECT metric_type, metric_value, measured_at, created_at
                     FROM (
                         SELECT DISTINCT ON (upper(replace(metric_type::text, ' ', '_')))
                             metric_type, metric_value, measured_at, created_at
                         FROM public.diabetes_data
                         WHERE user_id = auth.uid()
                           AND upper(replace(metric_type::text, ' ', '_')) IN (
                               'HBA1C', 'EGFR', 'UACR', 'COLESTEROLO_TOTALE', 'HDL', 'LDL', 'TRIGLICERIDI'
                           )
                         ORDER BY upper(replace(metric_type::text, ' ', '_')),
                                  COALESCE(measured_at, created_at) DESC
                     ) latest_labs
                 )
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
                   AND taken_at >= (now() - interval '5 days')
                 ORDER BY taken_at DESC
                 LIMIT 25
             ) tl),
            '[]'::jsonb
        ),
        'meals', COALESCE(
            (SELECT jsonb_agg(row_to_json(m))
             FROM (
                 SELECT meal_type, description, carbs_grams, calories, meal_at, is_estimated, created_at
                 FROM public.meals
                 WHERE user_id = auth.uid()
                   AND COALESCE(meal_at, created_at) >= (now() - interval '3 days')
                 ORDER BY COALESCE(meal_at, created_at) DESC
                 LIMIT 15
             ) m),
            '[]'::jsonb
        ),
        'physical_activities', COALESCE(
            (SELECT jsonb_agg(row_to_json(pa))
             FROM (
                 SELECT activity_type, duration_minutes, intensity, steps, notes, activity_at, created_at
                 FROM public.physical_activities
                 WHERE user_id = auth.uid()
                   AND COALESCE(activity_at, created_at) >= (now() - interval '7 days')
                 ORDER BY COALESCE(activity_at, created_at) DESC
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
                   AND appointment_at >= (now() - interval '1 day')
                 ORDER BY appointment_at ASC
                 LIMIT 5
             ) ap),
            '[]'::jsonb
        ),
        'patient_reports', COALESCE(
            (SELECT jsonb_agg(row_to_json(pr))
             FROM (
                 SELECT id, report_type, source_name, content, clean_summary, extracted_params, report_at, created_at
                 FROM public.patient_reports
                 WHERE user_id = auth.uid()
                 ORDER BY report_at DESC NULLS LAST, created_at DESC
                 LIMIT 5
             ) pr),
            '[]'::jsonb
        )
    );
$function$;

CREATE OR REPLACE FUNCTION public.search_patient_reports(
    query_embedding vector,
    query_text text DEFAULT ''::text,
    match_count integer DEFAULT 4
)
RETURNS TABLE(
    id bigint,
    report_type report_enum,
    source_name text,
    report_at timestamp with time zone,
    clean_summary text,
    content text,
    extracted_params jsonb,
    similarity double precision
)
LANGUAGE sql
STABLE
SECURITY INVOKER
SET search_path TO 'public'
AS $function$
WITH fts_query AS (
    SELECT
        NULLIF(
            replace(plainto_tsquery('italian'::regconfig, COALESCE(query_text, ''))::text, ' & ', ' | '),
            ''
        )::tsquery AS tsq
),
semantic AS (
    SELECT
        r.id,
        (1.0 - (r.embedding <=> query_embedding))::double precision AS sim,
        ROW_NUMBER() OVER (ORDER BY r.embedding <=> query_embedding ASC) AS r_sem
    FROM public.patient_reports r
    WHERE r.user_id = auth.uid()
      AND r.embedding IS NOT NULL
    ORDER BY r.embedding <=> query_embedding ASC
    LIMIT 20
),
lexical AS (
    SELECT
        r.id,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank_cd(
                to_tsvector('italian'::regconfig,
                    (((COALESCE(r.source_name, ''::text) || ' '::text) || COALESCE(r.clean_summary, ''::text)) || ' '::text) || COALESCE(r.content, ''::text)
                ),
                fq.tsq
            ) DESC
        ) AS r_fts
    FROM public.patient_reports r
    CROSS JOIN fts_query fq
    WHERE r.user_id = auth.uid()
      AND fq.tsq IS NOT NULL
      AND to_tsvector('italian'::regconfig,
            (((COALESCE(r.source_name, ''::text) || ' '::text) || COALESCE(r.clean_summary, ''::text)) || ' '::text) || COALESCE(r.content, ''::text)
          ) @@ fq.tsq
    LIMIT 20
)
SELECT
    r.id,
    r.report_type,
    r.source_name,
    r.report_at,
    COALESCE(r.clean_summary, r.content) AS clean_summary,
    r.content,
    r.extracted_params,
    GREATEST(
        COALESCE(s.sim, 0.0),
        CASE WHEN l.r_fts IS NOT NULL THEN 0.45 ELSE 0.0 END
    )::double precision AS similarity
FROM semantic s
FULL OUTER JOIN lexical l ON s.id = l.id
JOIN public.patient_reports r ON r.id = COALESCE(s.id, l.id)
ORDER BY
    (COALESCE(1.0 / (60 + s.r_sem), 0.0) + COALESCE(1.0 / (60 + l.r_fts), 0.0)) DESC,
    COALESCE(s.sim, 0.0) DESC
LIMIT LEAST(GREATEST(match_count, 1), 10);
$function$;

COMMIT;
