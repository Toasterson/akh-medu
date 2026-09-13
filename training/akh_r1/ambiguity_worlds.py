"""Two distinct finite mechanisms, not independent random population samples."""
def generate(split):
    if split=='development':
        return dict(id='dev-causal-inquiry',mechanism='instrument_diagnosis_of_hidden_fault',question='diagnose',answers=['capacity','permission'],actions=['expand','authorize'])
    return dict(id='val-participant-inquiry',mechanism='participant_selection_between_valid_tradeoffs',question='ask_priority',answers=['continuity','latency'],actions=['drain','restart'])
