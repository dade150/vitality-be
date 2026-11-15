-- =============================================================================
-- 06_hardening_privileges_and_policies.sql
-- Hardening dei privilegi su tabelle di catalogo/log e funzioni RPC/Trigger
-- =============================================================================

BEGIN;

-- 1. Rimuove privilegi pericolosi (TRUNCATE, TRIGGER, REFERENCES, INSERT, UPDATE, DELETE)
--    su foods e ricette per anon e authenticated (in PostgreSQL TRUNCATE bypassa la RLS).
REVOKE TRUNCATE, TRIGGER, REFERENCES, INSERT, UPDATE, DELETE ON TABLE public.foods FROM PUBLIC, anon, authenticated;
REVOKE TRUNCATE, TRIGGER, REFERENCES, INSERT, UPDATE, DELETE ON TABLE public.ricette FROM PUBLIC, anon, authenticated;
GRANT SELECT ON TABLE public.foods TO anon, authenticated;
GRANT SELECT ON TABLE public.ricette TO anon, authenticated;

-- 2. Revoca UPDATE/DELETE su app_logs e UPDATE su chat_message per authenticated
--    (allineamento dei grant tabellari con le policy RLS di append-only / immutabilità).
REVOKE UPDATE, DELETE ON TABLE public.app_logs FROM PUBLIC, anon, authenticated;
REVOKE UPDATE ON TABLE public.chat_message FROM PUBLIC, anon, authenticated;

-- 3. Restringe permessi EXECUTE sulle funzioni SECURITY DEFINER e sulle RPC RAG/cliniche
--    (principio del minimo privilegio: rimuove EXECUTE per PUBLIC e anon).
REVOKE ALL ON FUNCTION public.handle_new_user() FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.handle_new_user() TO postgres, service_role;

REVOKE ALL ON FUNCTION public.current_user_role() FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.current_user_role() TO authenticated, service_role;

REVOKE ALL ON FUNCTION public.get_patient_context_bundle() FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.get_patient_context_bundle() TO authenticated, service_role;

REVOKE ALL ON FUNCTION public.search_patient_reports(vector, text, integer) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.search_patient_reports(vector, text, integer) TO authenticated, service_role;

REVOKE ALL ON FUNCTION public.hybrid_search_chunks(text, vector, integer, integer, integer, integer) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.hybrid_search_chunks(text, vector, integer, integer, integer, integer) TO authenticated, service_role;

COMMIT;
