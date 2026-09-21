"""Classify provider errors, never assistant prose or task outcomes."""
import json
import re


def classify_provider_error(error):
    if not error:
        return None
    text = (error if isinstance(error, str) else json.dumps(error)).lower()
    if any(s in text for s in ('refresh token was revoked', 'token could not be refreshed',
                              'invalid_api_key', 'authentication_error', 'unauthorized',
                              'http 401', 'http 403')):
        return 'authentication'
    if any(s in text for s in ('usage limit', 'usage_limit', 'insufficient_quota',
                              'quota exceeded', 'quota exhausted', 'out of usage',
                              'credit balance', 'billing_hard_limit')):
        return 'quota'
    if re.search(r'\b429\b', text) or any(s in text for s in ('rate_limit', 'rate limit', 'too many requests')):
        return 'rate_limit'
    return None
