# Dated analysis plan

The Research Transparency Statement refers to a dated internal analysis plan. This folder holds it. The documents are written in
Chinese, as they were used during the project; they are copied byte for byte.

| File | Content | Date and time (EDT = UTC-4) | SHA-256 |
|---|---|---|---|
| `PAPER_OUTLINE_v1_2026-10-19.md` | Study design and planned tests: constructs, writers and pairs, the reading of the internal states, Studies 1 to 4 and their primary tests (section 6), the analysis of training | created and last modified 19 Oct 2026, 14:48:49 | `6670576bab02e4f0...` |
| `PRERUN_ADJUSTMENTS_v6.md` | Exclusion rules for the pairs of narratives and other decisions taken after the materials were written and before the formal runs | last modified 19 Oct 2026, 17:14:52; its SHA-256 is recorded in `materials_v6/MANIFEST_v6.json` (Zenodo archive), frozen 19 Oct 2026, 21:14 UTC | `0c415c5172cccc80...` |
| `GPU_PLAN_v6_as_of_2026-10-19_1923EDT.md` | The plan of the GPU steps (what each step computes, the model settings) as it stood before the formal runs | saved 19 Oct 2026, 19:23:15 | `66ce52233aaff870...` |
| `GPU_PLAN_v6.md` | The same plan in its final form | last modified 20 Oct 2026, 13:55:26 | `0474f3f78e5755bc...` |

The first formal GPU run (run 1: Qwen3.6 27B, Gemma 4 31B, OLMo 3.1 32B) began on 20 Oct 2026 at 00:10 UTC (19 Oct, 20:10 EDT).
The final `GPU_PLAN_v6.md` differs from the version of 19:23 EDT only by one added section at the end of part 3, which records
the actual times of run 1 and the follow-up analyses agreed on 20 Oct (inductions at later layers, removal of larger subspaces,
the model's own manipulation check).

**What the plan fixed before the formal runs.**
- *Pairs of situations.* Automatic checks of every composed pair (banned words, length difference, numbers in both versions or
  neither; outline section 4) and the exclusion rules R1 (the pair's high version is not rated higher than its low version on the
  pair's own construct) and R2 (near-duplicate texts of one writer and construct; the earlier text is kept)
  (`PRERUN_ADJUSTMENTS_v6.md`, section 1); the resulting set of 8,311 pairs is frozen in `materials_v6/MANIFEST_v6.json`.
- *Reading of the internal states.* The mean over the narrative's word fragments, at the layers 30% to 50% of the way through
  each LLM, with dimensions standardized (outline section 3.4; band 1 in the GPU plan).
- *Held-out writers.* Each construct's representation is estimated from five writers and tested on the pairs of the sixth
  (outline section 3.5; GPU plan, step V1).
- *Planned tests of Studies 1 to 4.* Outline section 6: convergent minus discriminant correlations, identification beyond the
  sentence encoder, read-out words identified above chance, the criterion correlation (Study 1); structure across LLMs and
  agreement with the theories' processes (Study 2); activation of the mediator's representation more than of the other
  constructs' representations (Study 3); effects of inducing the representations on judgments and responses and their
  moderation (Study 4); effect sizes d_z with 95% confidence intervals; responses whose digit probabilities sum to less than .5
  are excluded and reported.
- *Settings of the inductions.* Dose in units of the construct's natural difference, induction only at the narrative's tokens,
  the induction layer chosen by a fixed rule (the first layer that reaches 95% of the LLM's best held-out-writer
  identification), directions not estimated from the materials of the text's own writer, random directions drawn from the space
  of real differences (outline section 6; GPU plan, part 1).

The analyses at other positions in the text and at other layers, the profiles in shared dimensions, the inductions at later
layers, and the analyses over the course of training were specified afterward.

**Where the times come from.** The project folder was not under version control for these files (its git history ends on
18 Oct 2026), so the times are file-system times on the first author's computer and, for the version of 19:23 EDT, the local edit
history of the editing tool (Claude Code), which kept a copy of the file each time it was saved (versions of 18:08:47 and
19:23:15 EDT on 19 Oct and of 13:55:26 EDT on 20 Oct).
