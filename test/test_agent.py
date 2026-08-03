import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage, AIMessage

import Agent as ra


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_triage(answerable=False, clarification=False, escalation=False, out_of_scope=False):
    """Shape matches UserQuery.model_dump()."""
    return {
        "answerable": answerable,
        "Requires_clarification": clarification,
        "Requires_escalation": escalation,
        "Out_of_scope": out_of_scope,
    }


def fake_triage_output(**kwargs):
    """Mimic what Query_chain.invoke(...) returns: a UserQuery-like object
    with .model_dump()."""
    obj = MagicMock()
    obj.model_dump.return_value = make_triage(**kwargs)
    return obj


def gen_json(answer="some answer", sources=None, confidence=0.9):
    sources = sources if sources is not None else [{"document": "a.md", "passage": "chunk_0"}]
    return AIMessage(content=json.dumps({
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
    }))


def initial_state(query_text):
    return {
        "Query": [HumanMessage(content=query_text)],
        "Response": [],
        "logs": "",
        "Answerable": {},
        "retrieved_passages": [],
        "Retries": 0,
        "Confidence": 0.0,
        "require_human": False,
        "reason": "",
    }


@pytest.fixture
def graph():
    return ra.build_graph()


# ---------------------------------------------------------------------------
# 1. Directly answerable question
# ---------------------------------------------------------------------------
def test_directly_answerable(monkeypatch, graph):
    fake_query = MagicMock()
    fake_query.invoke.return_value = fake_triage_output(answerable=True)
    monkeypatch.setattr(ra, "Query_chain", fake_query)
    
    monkeypatch.setattr(ra, "retrieve", lambda q, top_k=3: [
        {"document": "setup.md", "passage_id": "chunk_0",
         "text": "Install with pip install foo.", "score": 0.95},
    ])
    
    fake_generation = MagicMock()
    fake_generation.invoke.return_value = gen_json(
        answer="Run pip install foo.",
        sources=[{"document": "setup.md", "passage": "chunk_0"}],
        confidence=0.9,
    )
    monkeypatch.setattr(ra, "generation_chain", fake_generation)

    result = graph.invoke(initial_state("How do I install the package?"))
    parsed = json.loads(result["Response"][-1].content)

    assert result["Answerable"]["answerable"] is True
    assert parsed["sources"]
    assert result.get("require_human", False) is False
    assert "[Retrieval]" in result["logs"]
    assert "[Generation]" in result["logs"]
    assert "[Verification] passed=True" in result["logs"]


# ---------------------------------------------------------------------------
# 2. Question requiring information from two documents
# ---------------------------------------------------------------------------
def test_multi_document_question(monkeypatch, graph):
    fake_query = MagicMock()
    fake_query.invoke.return_value = fake_triage_output(answerable=True)
    monkeypatch.setattr(ra, "Query_chain", fake_query)
    
    monkeypatch.setattr(ra, "retrieve", lambda q, top_k=3: [
        {"document": "billing.md", "passage_id": "chunk_1",
         "text": "Refunds take 5 business days.", "score": 0.88},
        {"document": "account.md", "passage_id": "chunk_2",
         "text": "Close your account from Settings.", "score": 0.81},
    ])
    
    fake_generation = MagicMock()
    fake_generation.invoke.return_value = gen_json(
        answer="Refunds take 5 days; you can close your account from Settings.",
        sources=[
            {"document": "billing.md", "passage": "chunk_1"},
            {"document": "account.md", "passage": "chunk_2"},
        ],
        confidence=0.85,
    ) 
    monkeypatch.setattr(ra, "generation_chain", fake_generation)

    result = graph.invoke(initial_state("If I close my account, how long until I get a refund?"))
    parsed = json.loads(result["Response"][-1].content)
    docs_cited = {s["document"] for s in parsed["sources"]}

    assert len(result["retrieved_passages"]) == 2
    assert docs_cited == {"billing.md", "account.md"}
    assert "[Verification] passed=True" in result["logs"]


# ---------------------------------------------------------------------------
# 3. Ambiguous question requiring clarification
# ---------------------------------------------------------------------------
def test_requires_clarification(monkeypatch, graph):
    fake_query = MagicMock()
    fake_query.invoke.return_value = fake_triage_output(clarification=True)
    monkeypatch.setattr(ra, "Query_chain", fake_query)

    gen_spy = MagicMock()
    monkeypatch.setattr(ra, "generation_chain", gen_spy)

    result = graph.invoke(initial_state("It doesn't work."))
    parsed = json.loads(result["Response"][-1].content)

    assert result["Answerable"]["Requires_clarification"] is True
    assert parsed["reason"] == "requires clarification"
    assert parsed["requires_human"] is False
    assert result.get("require_human", False) is False
    gen_spy.invoke.assert_not_called()  # Graph short-circuits to END, never generates


# ---------------------------------------------------------------------------
# 4. Out-of-scope request
# ---------------------------------------------------------------------------
def test_out_of_scope(monkeypatch, graph):
    fake_query = MagicMock()
    fake_query.invoke.return_value = fake_triage_output(answerable=False, out_of_scope=True)
    monkeypatch.setattr(ra, "Query_chain", fake_query)

    retrieve_spy = MagicMock()
    monkeypatch.setattr(ra, "retrieve", retrieve_spy)

    result = graph.invoke(initial_state("Can you book me a flight to Paris?"))
    parsed = json.loads(result["Response"][-1].content)

    assert parsed["reason"] == "out of scope"
    assert parsed["requires_human"] is False
    retrieve_spy.assert_not_called()  # Retrieval node must never run


# ---------------------------------------------------------------------------
# 5. Initial answer fails verification, then is fixed on revise
# ---------------------------------------------------------------------------
def test_fails_verification_then_recovers_on_revise(monkeypatch, graph):
    fake_query = MagicMock()
    fake_query.invoke.return_value = fake_triage_output(answerable=True)
    monkeypatch.setattr(ra, "Query_chain", fake_query)

    monkeypatch.setattr(ra, "retrieve", lambda q, top_k=3: [
        {"document": "policy.md", "passage_id": "chunk_3",
         "text": "Cancellations within 30 days are free.", "score": 0.9},
    ])

    bad = gen_json(answer="You can cancel anytime.", sources=[], confidence=0.4)  # no sources -> fails
    good = gen_json(
        answer="Cancellations are free within 30 days.",
        sources=[{"document": "policy.md", "passage": "chunk_3"}],
        confidence=0.8,
    )

    fake_gen = MagicMock()
    fake_gen.invoke.return_value = bad
    monkeypatch.setattr(ra, "generation_chain", fake_gen)

    fake_revise = MagicMock()
    fake_revise.invoke.return_value = good
    monkeypatch.setattr(ra, "revise_chain", fake_revise)

    result = graph.invoke(initial_state("What's your cancellation policy?"))
    parsed = json.loads(result["Response"][-1].content)

    assert "[Verification] passed=False reason=no sources cited" in result["logs"]
    assert "[Revise] retrying" in result["logs"]
    assert result["Retries"] == 1
    assert parsed["sources"]
    assert result.get("require_human", False) is False


def test_fails_verification_twice_then_safe_fails(monkeypatch, graph):
    """Confirms the retry cap routes to SafeFail rather than looping forever."""
    fake_query = MagicMock()
    fake_query.invoke.return_value = fake_triage_output(answerable=True)
    monkeypatch.setattr(ra, "Query_chain", fake_query)

    monkeypatch.setattr(ra, "retrieve", lambda q, top_k=3: [
        {"document": "policy.md", "passage_id": "chunk_3",
         "text": "Cancellations within 30 days are free.", "score": 0.9},
    ])

    bad = gen_json(answer="You can cancel anytime.", sources=[], confidence=0.4)
    
    fake_gen = MagicMock()
    fake_gen.invoke.return_value = bad
    monkeypatch.setattr(ra, "generation_chain", fake_gen)

    fake_revise = MagicMock()
    fake_revise.invoke.return_value = bad  # continues failing after revise
    monkeypatch.setattr(ra, "revise_chain", fake_revise)

    result = graph.invoke(initial_state("What's your cancellation policy?"))
    parsed = json.loads(result["Response"][-1].content)

    assert result["require_human"] is True
    assert parsed["requires_human"] is True
    assert "[SafeFail] returning safe fallback" in result["logs"]


# ---------------------------------------------------------------------------
# 6. Routing tests — pure functions, no model/graph
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("triage,expected", [
    (make_triage(answerable=True), "answerable"),
    (make_triage(clarification=True), "Requires_clarification"),
    (make_triage(escalation=True), "Requires_escalation"),
    (make_triage(out_of_scope=True), "out_of_scope"),
    (make_triage(), "out_of_scope"),  # fallback when no flags set
])
def test_routing_triage_classification(triage, expected):
    assert ra.check_triage_classification({"Answerable": triage}) == expected


@pytest.mark.parametrize("state,expected", [
    ({"reason": "", "Retries": 0}, "end"),
    ({"reason": "bad JSON format", "Retries": 0}, "revise"),
    ({"reason": "bad JSON format", "Retries": 1}, "safe_fail"),
    ({"reason": "no sources cited", "Retries": 2}, "safe_fail"),
])
def test_routing_verification_check(state, expected):
    assert ra.check_verification(state) == expected