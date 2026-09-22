# Coordination-to-learning integration

The coordinator can now admit successful assignments into the existing verified
lesson store by opting in with `--lesson-db`. This uses the same registered
contract, source snapshot and independent verifier as manual admission.

Work status and learning status remain distinct. Lost acknowledgements can be
replayed without duplicate lessons. Storage errors propagate so reconciliation
can resume. Semantic admission failures are recorded as rejected. A persistent
store UUID binds each learning receipt to its store; switching stores fails
explicitly. Reconciliation refreshes a previously admitted lesson's revocation
state. Neither replay nor completed work reactivates revoked evidence.

[Combined test evidence](unit-tests.log) covers the local Lilith suite, adapter
preflight and legacy training safety regressions. Integration tests use a fake
session API with the actual lesson store and artifact verifier. The additional
historical-assignment replay uses a real earlier source/receipt, but makes no new
model call. This is recovery evidence, not a new live coding result.

The companion [reference curriculum](../curriculum-20260921/README.md),
[evaluation protocol](../../../benchmarks/lilith/apprenticeship-v2/README.md) and
[adapter readiness audit](../adapter-readiness-20260921/README.md) extend training
preparation. No optimizer or adapter activation occurs in this integration.
