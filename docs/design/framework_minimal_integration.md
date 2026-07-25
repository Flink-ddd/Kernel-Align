# Minimal RL Framework Integration Plan

This plan describes the smallest sustainable way for an RL framework to consume
the cross-configuration alignment work from PR 230. vime is the first adapter,
but the ownership boundary is intended to apply to any RL framework.

## Goal

An RL framework should integrate RL-Kernel by importing RL-Kernel-owned
contracts wherever possible. The framework should not maintain a parallel copy
of alignment profiles, score schemas, comparison rules, reference operators, or
strict-admission rules.

The framework remains responsible for runtime facts that only it can observe:
model handles, batches, masks, tokenizer policy, distributed topology, fallback
routing, logging, and user-facing configuration.

## Ownership Boundary

RL-Kernel owns:

- cross-configuration score-artifact schema, including `ScoreArtifact`,
  `SemanticIdentitySpec`, `ScorerSpec`, `ScoreSide`, and `RuntimeProvenance`;
- selected-logprob comparison, tolerance resolution, mismatch attribution, and
  tolerance fingerprints;
- the alignment profile standard consumed by downstream adapters;
- generic single-card operator comparison/reference/admission helpers;
- stable metadata that identifies which RL-Kernel contract was used.

The RL framework owns:

- converting framework-native batches and model state into RL-Kernel schema
  inputs;
- selecting and invoking framework-native rollout/training callables;
- querying optional RL-Kernel runtime/backend capability providers;
- deciding fallback behavior for local policy knobs;
- emitting framework-native logs, telemetry, and test reports;
- preserving compatibility when RL-Kernel is absent, when the feature is
  optional.

## Adapter Shape

A minimal adapter should be a narrow import-and-normalize layer:

1. Import RL-Kernel public APIs lazily at the adapter boundary.
2. Convert framework records into RL-Kernel dataclasses.
3. Pass framework-owned tensors/callables/capabilities into RL-Kernel helpers.
4. Return RL-Kernel result objects or thin framework report wrappers.
5. Fail closed with a clear unavailable error if the requested strict feature
   depends on an RL-Kernel API that is missing.

Adapters should not:

- define their own A0-A5 matrix;
- fork RL-Kernel score schemas or comparator semantics;
- keep local reference implementations for generic operators;
- locally decide strict-fast admission when RL-Kernel exposes that rule;
- import RL-Kernel throughout unrelated framework modules.

## Recommended Imports

Framework adapters should import these RL-Kernel surfaces when available:

- `rl_engine.alignment.cross_config.get_alignment_standard`
- `rl_engine.alignment.cross_config.schema`
- `rl_engine.alignment.cross_config.compare_score_artifacts`
- `rl_engine.alignment.cross_config.operator_comparison`
- `rl_engine.kernels.gtest.tolerance`

The adapter can still provide framework names or wrappers, but those wrappers
should delegate to RL-Kernel and avoid duplicating the implementation.

## vime Example

For vime, the minimal integration shape is:

- `vime.backends.rl_kernel_utils.standard` imports the RL-Kernel alignment
  standard provider and converts vime score records into RL-Kernel
  `ScoreArtifact` objects;
- `vime.backends.rl_kernel_utils.operator_comparison` re-exports
  RL-Kernel-owned operator comparison helpers instead of carrying reference
  implementations in vime;
- Megatron/vLLM integration code keeps tensor extraction, config wiring,
  fallback routing, and telemetry because those are vime runtime concerns;
- production modules should continue to avoid direct `rl_engine` imports outside
  the RL-Kernel adapter package.

This keeps vime-specific changes small at the architectural level: vime owns
the bridge to its runtime, while RL-Kernel owns the shared alignment standard.

## PR Review Rule

When reviewing an RL framework integration PR, ask whether the changed code is
describing framework runtime facts or defining reusable alignment semantics.

If it is a runtime fact, keep it in the framework. If it is a reusable alignment
semantic, move it to RL-Kernel and let the framework import it.
