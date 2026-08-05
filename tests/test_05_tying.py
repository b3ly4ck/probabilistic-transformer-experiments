"""Check 5 -- input and output word matrices are the same tensor object."""

import torch


def test_S_is_one_parameter_object(model, tokens):
    """Not a copy, not an equal tensor: the same object (§16(b), tying is forced)."""
    input_matrix = model.S
    tokens_embedded = model.S[tokens]
    assert input_matrix is model.S
    assert tokens_embedded.shape[-1] == model.S.shape[1]

    # mutating the single parameter must move both roles at once
    with torch.no_grad():
        before_in = model.S[tokens].clone()
        before_out = model.exact_logits(model.content_stream(tokens)).clone()
        model.S += 1.0
        assert not torch.allclose(before_in, model.S[tokens])
        assert not torch.allclose(
            before_out, model.exact_logits(model.content_stream(tokens))
        )


def test_no_separate_output_projection(model):
    """A d x |V| output matrix is exactly the rejected design (§15)."""
    shapes = {tuple(p.shape) for p in model.parameters()}
    d, V = model.cfg.d, model.cfg.vocab_size
    assert (d, V) not in shapes
    assert sum(1 for s in shapes if s == (V, d)) == 1
