"""Training-only tool-choice weighting; inference keeps the existing interface."""
import json
import re


def choice_span(tokenizer, tokens, offset, length):
    """Return absolute token indices covering the target JSON function name."""
    target = tokens[offset:length]
    text = tokenizer.decode(target)
    calls = list(re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.S))
    if len(calls) != 1:
        raise ValueError('Choice weighting requires one complete target tool call')
    body = calls[0].group(1)
    action = json.loads(body)
    if set(action) != {'name', 'arguments'} or not isinstance(action['name'], str):
        raise ValueError('Invalid native target envelope')
    match = re.search(r'"name"\s*:\s*"([^"\\]+)"', body)
    if match is None or match.group(1) != action['name']:
        raise ValueError('Missing unambiguous target function name')
    char_start = calls[0].start(1) + match.start(1)
    char_end = calls[0].start(1) + match.end(1)
    selected = []
    previous = 0
    for index in range(len(target)):
        boundary = len(tokenizer.decode(target[:index+1]))
        if previous < char_end and boundary > char_start:
            selected.append(offset+index)
        previous = boundary
        if boundary >= char_end:
            break
    if not selected:
        raise ValueError('Function name has no supervised tokens')
    return selected[0], selected[-1]+1


def balanced_loss(model, batch, lengths):
    """Half mean function-name CE, half mean remaining supervised-token CE."""
    import mlx.core as mx
    import mlx.nn as nn
    logits = model(batch[:, :-1])
    ce = nn.losses.cross_entropy(logits, batch[:, 1:]).astype(mx.float32)
    steps = mx.arange(1, batch.shape[1])
    # Match installed MLX-LM's supervised range, including its EOS/padding convention.
    supervised = (steps >= lengths[:, 0:1]) & (steps <= lengths[:, 1:2])
    choice = supervised & (steps >= lengths[:, 2:3]) & (steps < lengths[:, 3:4])
    remaining = supervised & ~choice
    choice_ce = (ce * choice).sum(axis=1) / choice.sum(axis=1)
    remaining_ce = (ce * remaining).sum(axis=1) / remaining.sum(axis=1)
    return (0.5 * choice_ce + 0.5 * remaining_ce).mean(), supervised.sum()


def make_choice_batches(tokenizer, original_iterator):
    def iterate(*args, **kwargs):
        import mlx.core as mx
        for batch, lengths in original_iterator(*args, **kwargs):
            extended = []
            for tokens, (offset, length) in zip(batch.tolist(), lengths.tolist()):
                start, end = choice_span(tokenizer, tokens, offset, length)
                if not (offset <= start < end < length):
                    raise ValueError('Choice and remaining target tokens must both exist')
                extended.append([offset, length, start, end])
            yield batch, mx.array(extended)
    return iterate
