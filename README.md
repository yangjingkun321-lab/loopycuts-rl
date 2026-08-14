# LoopyCuts RL

Research code for learning a state-dependent Stage-2 loop execution
policy for the LoopyCuts volumetric cutting pipeline.

## Research scope

The project keeps the original LoopyCuts Stage-1 loop generation
pipeline and learns how to select the next legal loop during Stage-2
volumetric cutting.

The C++ LoopyCuts RL server remains authoritative for action legality
and geometry transitions.

## Current state

Current development phase:

    Phase 2E-B — baseline and train resource feasibility audit

Completed and frozen components include:

- persistent LoopyCuts C++ RL server
- RL V1 Stage-2 episode semantics
- Observation V1
- FINALIZE_EVAL and finalization outcome taxonomy
- Reward V2
- Dataset Split V2
- Original / Random baseline infrastructure
- passive process resource monitoring

Formal SAC training has not started yet.

## Important experiment policy

The held-out test split remains sealed until the model architecture,
training protocol, checkpoint selection, and evaluation protocol are
frozen.

Mesh and loop paths must be read from the frozen dataset manifest and
must not be reconstructed from model names.

## Project state

See:

    docs/project_state/LoopyCuts_RL_Project_Handoff_2026-08-14.md

for the complete project handoff, frozen semantics, experiment results,
and the exact next development step.
