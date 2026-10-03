"""Teacher-akh curriculum generators: yellow variation, green sparring, black guidance and lineage.

Everything a generator asks of a model goes through `acp_worker.Teacher` (the sealed `dojo-teacher` akh over
ACP). There is no batch or direct model-API path in this package. Generated data is private and never lives
in the repo; only these scripts do.
"""
