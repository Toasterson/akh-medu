"""Optional domain-independent feedback about identical request/result repeats."""
import hashlib
import json


class RepetitionFeedback:
    def __init__(self):
        self.counts = {}

    def observe(self, request, response):
        key = hashlib.sha256(json.dumps([request, response], sort_keys=True).encode()).hexdigest()
        self.counts[key] = self.counts.get(key, 0) + 1
        count = self.counts[key]
        if count < 2:
            return None
        return dict(same_request_and_result_count=count,
                    message='This request has returned the same result before. Review the remaining goal '
                            'and choose an action that advances it, or report the unresolved obstacle.')
