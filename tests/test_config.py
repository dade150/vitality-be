import pytest
from pydantic import ValidationError

from vitality_be.core.config import Settings

BASE = dict(supabase_url="x", supabase_anon_key="y")


def test_echo_senza_gcp():
    assert Settings(_env_file=None, **BASE).llm_provider == "echo"


@pytest.mark.parametrize("kw", [
    dict(llm_provider="vertex"),  # mancano progetto e modello
    dict(llm_provider="vertex", gcp_project="p", llm_model="m", gcp_location="global"),
    dict(llm_provider="vertex", gcp_project="p", llm_model="m", gcp_location="us-central1"),
    dict(embedding_location="us-central1"),
    dict(embedding_api_endpoint="https://us-central1-aiplatform.googleapis.com"),
    dict(environment="prod", llm_model="gemini-x-preview", allowed_hosts="a.it"),
    dict(environment="prod"),  # allowed_hosts di default non ammessi in prod
    dict(environment="prod", allowed_hosts="*"),
    dict(chat_request_timeout_s=60),
    dict(llm_timeout_s=55, chat_request_timeout_s=50),
    dict(llm_max_output_tokens=10),
])
def test_config_non_valida(kw):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **BASE, **kw)


def test_vertex_valida():
    s = Settings(_env_file=None, **BASE, llm_provider="vertex", gcp_project="p", llm_model="gemini-3.6-flash")
    assert s.gcp_location == "eu"
    assert s.rag_top_k == 8
    s2 = Settings(_env_file=None, **BASE, llm_provider="vertex", gcp_project="p", llm_model="gemini-2.5-flash", gcp_location="europe-west1")
    assert s2.llm_model == "gemini-2.5-flash"
