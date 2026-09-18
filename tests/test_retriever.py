from backend.context.retriever import get_context


def test_get_context():
    context = get_context()

    assert "brd" in context
    assert "repo_rules" in context

    assert isinstance(context["brd"], str)
    assert isinstance(context["repo_rules"], str)

    assert len(context["brd"]) > 0
    assert len(context["repo_rules"]) > 0
