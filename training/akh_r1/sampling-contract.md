# R0 structural sampling contract

Count mechanisms, sampled environments and correlated controls separately.
A deterministic pseudorandom substream keyed by master seed, split and index
makes a draw reproducible; it does not prove statistical independence or broad
real-world representativeness. Shared latent world state, translations, identifier
changes and ablations belong to the same correlation group.

Hold out a mechanism across development/validation when claiming mechanism
transfer. New sampled topologies from the same mechanism test within-distribution
coverage. Neither a parameter grid nor every sample from one generator should be
automatically equated with an independent mechanism.

Freeze generator revision, sampling distribution, seed schedule, draw count,
selection rules, outcome strata, intervention policy and budget before an
execution batch. Keep rejected, empty, ambiguous and unchanged cases visible;
never silently resample until convenient outcomes appear. A deliberate balanced
stratified design is allowed if declared before drawing; report stratum weights
and do not present it as natural incident prevalence.

Preserve public observations and complete private initial/later outcomes. Use a
separate oracle for scoring. Structural signatures must exclude entity spellings;
the current layered adjacency signature is construction-order based and is not a
general graph-isomorphism test. Qualification requires native complete episodes,
source provenance, controls and audit; signature uniqueness alone earns no credit.

Initial pilot: branching reachability versus intersecting separately generated
routes, seed 1309, eight draws per split. This seed also appears in implementation
tests, so this is a development pilot, not a sealed evaluation. No rejection or
resampling occurred. The plan was saved before writing the batch; reference
solver/oracle implementation tests had already inspected the same samples.
