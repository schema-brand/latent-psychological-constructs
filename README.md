# Code for "Not your average Clever Hans: Latent representation of psychological constructs in large language models"

Code for the article *Not your average Clever Hans: Latent representation of psychological constructs in large language models* (Jack Chen, Xiao Li, Zelin Zhou, Weiheng Xiao, and Chunlin An; submitted to *Psychological Science*). The materials, all results that the analyses
read, and a snapshot of this repository are archived at Zenodo (https://doi.org/10.5281/zenodo.23771694). An earlier version of the article is available as a
preprint on SSRN (https://doi.org/10.2139/ssrn.7652158).

The study measures 66 psychological constructs inside large language models (LLMs). Six LLMs from different developers (the
writers) composed pairs of narratives that differ only in one construct; the internal states of the examined LLMs while they read
these situations give each construct's representation. Study 1 tests the construct validity of the representations, Study 2
their structure across LLMs, Study 3 their activation by the manipulations of 60 published experiments (silicon samples), Study 4
the effects of inducing and removing them, and Study 5 the representations of individual characteristics; a final analysis
follows the representations over the training of one LLM.

License: the code is released under the MIT License (`LICENSE`); the materials and data in the Zenodo archive under the Creative
Commons Attribution 4.0 International License (CC BY 4.0). Running the GPU steps downloads openly available language models from
Hugging Face under their own licenses; re-creating the materials sends requests to third-party language-model services
(OpenRouter) with the user's own API key.

## Two kinds of steps

| Kind | What it does | Where it runs | Folder |
|---|---|---|---|
| GPU steps | Run the examined LLMs on the materials, record their internal states, induce or remove representations, and save summaries and compressed exports of the states | Google Colab GPU (96 GB), 19-21 Oct 2026 | `code_v6/`, `jobs6/` |
| Laptop steps | Every number, table and figure of the article, computed from the saved GPU results and the materials | any computer with Python; no GPU, no internet, no API calls | `analysis_v6/`, `paper/` |
| Materials | Compose, check, rate and code the pairs of narratives, the rewordings of the experiments and the individual characteristics (LLM calls through OpenRouter, which cost money) | laptop or Colab CPU | top-level `make_*_v6.py`, `or_runner/` |

To check the reported results, only the laptop steps are needed. The internal states were recorded once on the GPU; the archive
contains their exports, so nothing has to be recomputed on a GPU.

## Set up

1. Download this repository and the Zenodo archive. Unzip every Zenodo archive (`materials_v6.zip`, `results_v6.zip`,
   `analysis_v6.zip`, `or_runner.zip`, `question_coding.zip`, `external.zip`, `materials_v4.zip`, `paper.zip`) **into the
   repository folder**, so that `results_v6/`, `materials_v6/` and the other folders sit next to `analysis_v6/` and `code_v6/`.
   The data folders are listed in `.gitignore`.
2. Python 3.13 (3.13.14 was used, on Windows 11): `pip install -r requirements.txt` (see the file for the CPU build of torch).
3. On Windows, set `PYTHONIOENCODING=utf-8` before running scripts that print non-ASCII text. The figures use the Arial font.
4. Run every command from the repository folder.

## Reproduce every reported number (laptop)

Each script reads `results_v6/`, `materials_v6/` and earlier outputs in `analysis_v6/`, and writes its output (`.json`, `.csv`,
`.npz`) next to itself (`ceiling_v6.py` and `moderation_v6.py` write into `results_v6/<job>/`), overwriting the copy from the
Zenodo archive. To check that a run reproduces the
archive, copy `MANIFEST.sha256` from the Zenodo record into the repository folder and run `sha256sum -c MANIFEST.sha256`
afterwards (it lists every archived file; a re-run output that differs is reported as FAILED).
The whole sequence takes about 75 minutes on a laptop (`ngram_floor_v6.py` alone about 35 minutes, `beyond_words_v6.py` and
`criterion_v6.py` about 15 each; all others under 5 minutes).

```
# Study 1: construct validity
python analysis_v6/per_construct_ident_v6.py          # identification of each construct (main LLMs)
python analysis_v6/discriminant_v6.py                 # discriminant validity, rival constructs
python analysis_v6/bow_floor_v6.py                    # baseline: the words of the narratives
python analysis_v6/ngram_floor_v6.py                  # baseline: word sequences
python analysis_v6/encoder_floor_v6.py                # baseline: sentence encoder
python analysis_v6/beyond_words_v6.py                 # representations beyond the words
python analysis_v6/readout_judge_v6.py score          # content validity: raters' choices for the read-out words (main LLMs)
R_TAG=R2 python analysis_v6/readout_judge_v6.py score #   ... other LLMs and training stages (Windows: set R_TAG=R2 first)
python analysis_v6/readout_echo_v6.py                 # content validity beyond word matching
python analysis_v6/criterion_v6.py                    # criterion validity: the LLM's own judgments
# Study 2: shared structure
python analysis_v6/shape_v6.py                        # construct profiles in dimensions shared by the main LLMs
python analysis_v6/structure_v6.py                    # similarity structure across LLMs and processes
# Study 3: activation in the published experiments
python analysis_v6/ceiling_v6.py V610_qwen36 V611_gemma4_31b V612_olmo32   # responses at the end of the scale (exclusion rule)
python analysis_v6/v3_all_models.py                   # effects on the responses, activation by replication outcome
python analysis_v6/cross_paper_v6.py                  # rank of the mediator's representation among the 66 constructs
python analysis_v6/depth_s3_v6.py                     # other positions and depths
python analysis_v6/depth_s3_spec_v6.py                # specificity by position
python analysis_v6/example_s3_v6.py                   # per-experiment activation at each position
python analysis_v6/reliability_s3_v6.py               # reliability across wordings
python analysis_v6/exact_tests_v6.py                  # exact tests (replication benchmark)
python analysis_v6/repl_activation_v6.py              # activation in replicated vs failed experiments
python analysis_v6/far_positions_bf_v6.py             # Bayes factors and equivalence tests at positions far from the response
# Study 4: inducing and removing the mediator's representation
python analysis_v6/run1_summary.py                    # summary of the three main LLMs
python analysis_v6/reliability_v6.py V610_qwen36 V611_gemma4_31b V612_olmo32
python analysis_v6/removal_v6.py V610_qwen36 V611_gemma4_31b V612_olmo32
python analysis_v6/removal_ci_v6.py                   # share of the effect left after removal, with 95% CIs (Fig. 7c)
python analysis_v6/subspace_share_v6.py V610_qwen36 V611_gemma4_31b V612_olmo32
python analysis_v6/moderation_v6.py V610_qwen36 V611_gemma4_31b V612_olmo32    # situational moderation (Table S7)
python analysis_v6/study4_specificity_bf_v6.py        # Bayes factors for the specificity of the induced mediator's effect
# Study 5: individual characteristics
python analysis_v6/mediate_summary_v6.py              # removing a characteristic's representation
# Training of OLMo 3 7B
python analysis_v6/v3_ckpt.py                         # Study 3 over the training stages
# Statistics required by Psychological Science (CIs, equivalence tests, Bayes factors, family-wise corrections, exclusions)
python analysis_v6/psychsci_stats_v6.py 1000          # bootstrap with 1,000 resamples, as reported
python analysis_v6/qcoding_v6.py                      # whether the experiment's question concerns the mediator (Supplemental Material)
# Supplemental tables and figures
python paper/make_si_tables.py                        # Tables S1-S9 -> paper/SUPPLEMENTARY_TABLES_v1.md (same numbers as the archived
                                                      #   SUPPLEMENTARY_TABLES_v9.md, whose captions, headers and notes were edited afterwards
                                                      #   and whose p values are printed in APA format)
python paper/psychsci/make_table_s10.py               # Table S10 (estimates, CIs, equivalence tests, Bayes factors) -> paper/psychsci/TABLE_S10.md
python paper/figs/make_figs_ps.py                     # Figures 2-8 and Figure S1 -> paper/figs/figs_ps/fig1-fig7, figS1 + plotted_values.json
                                                      #   (file figN = Figure N+1 of the article; fig1, the measurement schematic of
                                                      #   Figure 2, is drawn by paper/figs/fig_measure_ps.py)
python paper/figs/fig_cleverhans.py                   # Figure 1 (illustration) -> paper/figs/figs_ps/fig_cleverhans.png
python paper/figs/make_shape_fig.py                   # stand-alone figure of the construct profiles
python paper/figs/make_figs_v3.py                     # the same figures in the earlier (Nature Communications) layout
```

`paper/figs/figs_ps/plotted_values.json` lists every number drawn in a figure together with the file it comes from.

**Check of this package (22 Oct 2026).** In a fresh copy of this repository with the Zenodo files unzipped into it, every command
above ran without error on a Windows 11 laptop, and 1,386 of the 1,392 archived files were byte-identical afterwards. The
exceptions: `shape_v6.py` uses a randomized principal-component
decomposition (`torch.pca_lowrank`) without a fixed seed, so its outputs and the values computed from them (the Study 2 profiles in
`psychsci_stats_v6.json` and in the plotted values of Fig. 4a) vary in the third decimal from run to run (identification from the
profiles 93.3% in the archive, 93.4% in the check). `cross_paper_v6.json` was written before the Study 3 results of Gemma 4
26B-A4B were added; a re-run adds that LLM and changes the correlations averaged over all LLMs in the third decimal.
In Table S5 (`paper/SUPPLEMENTARY_TABLES_v9.md`), one correlation that lies on a rounding boundary (0.0948, stored as 0.095;
Qwen3.6 27B, scenario, layers 60-75%) is printed as .09, as in Table S10, whereas `make_si_tables.py` prints .10.
`paper/psychsci/make_table_s10.py` stops, after writing the table, at a final cross-check against the manuscript text, which is
not part of the repository; the archived `TABLE_S10.md` has three rows and three Bayes factors (Study 4) that this version of the
script does not yet write.

## Where each result comes from

Job names (folders of `results_v6/`): `V610_qwen36` = Qwen3.6 27B, `V611_gemma4_31b` = Gemma 4 31B, `V612_olmo32` = OLMo 3.1 32B
Instruct (the three main LLMs, run 1, every step); `V600`-`V608`, `V6c1`-`V6c4`, `V701` = smoke tests and code checks; `V613` = an
extra run of step V4a; `V710`-`V712` = follow-ups of the main LLMs;
`V720` = Gemma 4 26B-A4B (the LLM of MVCA's cognitive agents); `V721`, `V722` = two LLMs before instruction tuning; `V730`-`V742` =
the other LLMs; `V7500`-`V7512`, `V760`-`V762` = reading at the training stages of OLMo 3 7B; `V770_embed` = the sentence
encoder; `V810`-`V813` = inductions at other layers; `V820`-`V825` = judgments at the training stages; `V830`-`V839` = Study 3 in
the other LLMs; `V840`-`V843` = Study 5, removal; `V844`-`V84c` = Study 3 at the training stages; `V812a`-`V813b`, `V850`-`V859`
= Study 5, individual characteristics; `V84d`, `V84e` = item tables exported for the laptop analyses.

| Article | GPU step(s) (`code_v6/`) | Laptop analysis | Display |
|---|---|---|---|
| Study 1: multitrait-multimethod correlations; identification of the held-out writer's construct | `V1_read` | `per_construct_ident_v6`, `discriminant_v6`, `psychsci_stats_v6` (CIs) | Fig. 3a-b; Table 2; Table S3 |
| Study 1: validity beyond the words; sentence encoder; LLM before training | `V0_embed`, `V1_read` | `bow_floor_v6`, `ngram_floor_v6`, `encoder_floor_v6`, `beyond_words_v6` | Fig. 3b; Table S3 |
| Study 1: content validity (read-out words) | `V1_read` (read-out), `V2_jlens` | `readout_judge_v6`, `readout_echo_v6` | Fig. 3c; Table S4 |
| Study 1: criterion validity (the LLM's judgments) | `V1_read`, `V4a_causal` | `criterion_v6` | Fig. 3d; Table S3 |
| Study 2: construct profiles in shared dimensions | `V1_read` (export `export_dz_v6.pt`) | `shape_v6` | Fig. 4a |
| Study 2: similarity structure across LLMs and appraisal processes | `V1_read`, `V0_embed` | `structure_v6`, `encoder_floor_v6` | Fig. 4b-c; Table S3 |
| Study 3: activation of the mediator's representation; relation to the effect on the response; positions and depths; replication outcomes | `V3_exp`, `V5b_sweep` | `v3_all_models`, `cross_paper_v6`, `depth_s3_v6`, `depth_s3_spec_v6`, `example_s3_v6`, `reliability_s3_v6`, `ceiling_v6`, `exact_tests_v6`, `repl_activation_v6`, `far_positions_bf_v6`, `psychsci_stats_v6`, `qcoding_v6` | Figs. 5, 6a; Table S5 |
| Study 4: inducing representations (judgments, responses) | `V4a_causal`, `V4b_steer`, `V5a_incontext`, `V6a_dlda`, `V6b_steer2` | `run1_summary`, `reliability_v6` | Fig. 7a-b; Table S6 |
| Study 4: removing the mediator's representation; situational moderation | `V4c_remove`, `V5c_remove2`, `V4d_moderation`, `V6d_moderation2` | `removal_v6`, `removal_ci_v6`, `subspace_share_v6`, `moderation_v6` | Fig. 7c; Tables S6-S7 |
| Study 5: individual characteristics and their representations | `V7i_indiv`, `V7m_mediate` | `mediate_summary_v6` | Fig. 8; Table S8 |
| Representations over training (OLMo 3 7B stages) | `V1_read`, `V4a_causal`, `V3_exp` on the stages | `v3_ckpt`; `olmo7_training_v1_read.csv`, `olmo7_training_v4a.csv` | Fig. S1; Table S2 |
| Relations of the mediator's nomological network (Studies 3 to 5) | the above | the outputs above | Fig. 6b; Table 3 |
| Measurement of the representations (schematic) and illustration | none | `paper/figs/fig_measure_ps.py`, `paper/figs/fig_cleverhans.py` | Figs. 1-2 |
| Exclusions, sensitivity, CIs and tests for claims of absence | — | `psychsci_stats_v6`, `far_positions_bf_v6`, `repl_activation_v6`, `study4_specificity_bf_v6`, `paper/psychsci/make_table_s10.py` | Method, Results; Table S10 |

Supplemental tables: S1 constructs; S2 LLMs; S3 Studies 1-2; S4 content validity; S5 Study 3; S6 Study 4 inducing and removing;
S7 situational moderation; S8 Study 5; S9 example pairs and pair counts; S10 estimates, confidence intervals, equivalence tests
and Bayes factors of the main results.

**Results of the LLMs other than the three main ones.** For the run-2 and run-3 jobs (the 13 other LLMs, the training stages,
several follow-up analyses), only the files that the laptop analyses read were downloaded from the GPU runner: the read-out word
lists, the Study 3 items, the Study 5 tables and the step digests. Their headline numbers (for example the identification at
layers 30% to 50% of the depth for each LLM) were copied from the step digests into `analysis_v6/run23_digests_2026-10-21.md`, and
`paper/make_si_tables.py` reads them from there; a few values that exist only in the digests are entered in
`paper/make_si_tables.py` and `paper/figs/make_figs_ps.py` with their source in a comment. `analysis_v6/fixed_depth_ident_v6.py`
(identification at the layer nearest 50% of the depth for all 16 LLMs) needs the full V1 digests of these LLMs, which are not in
the archive (`RAW_STATES_NOT_INCLUDED.md` in the Zenodo archive); it is kept for reference.

## GPU steps (Colab)

| Step | Study | What it computes |
|---|---|---|
| `V0_embed` | 1-2 | Sentence-encoder embedding (Qwen3-Embedding-8B) of every narrative (text baseline) |
| `V1_read` | 1-2, training | Reads every pair of narratives (each text alone, no agent profile) and records, at each layer, the mean over the text's word fragments and the state at its last fragment; held-out-writer identification among the 66 constructs at each layer; multitrait-multimethod correlations; the induction layer L*; the construct directions; read-out words (logit lens); compressed exports for the laptop analyses |
| `V2_jlens` | 1 | Read-out words of each construct with the J-lens (Qwen3.6 27B and Gemma 4 31B, the LLMs with a fitted lens) |
| `V3_exp` | 3 | The 60 experiments (MVCA's scenario and the rewordings) x conditions x agent profiles, without intervention: the response (expected value over the digits 1-9) and the internal states at the manipulation, the scenario, the question and the response position, projected on the 66 representations |
| `V4a_causal` | 1 (criterion), 4 | Induction of each construct's representation at the narrative's tokens and its effect on the LLM's first-person judgments of every construct |
| `V4b_steer` | 4 | Induction of the mediator's representation in the 52 experiments of MVCA's Study 2 and its effect on the response |
| `V4c_remove` | 4 | Removal of the mediator's representation (a subspace of 10 dimensions; zeroing and mean replacement) and of comparison subspaces |
| `V4d_moderation` | 4 | Induction of the mediator's representation at both levels of a situational moderator (7 experiments) |
| `V5a_incontext`, `V5b_sweep`, `V5c_remove2` | 4 | Follow-ups (20 Oct): the LLM's own manipulation check; inductions at other depths and token ranges; removal of larger subspaces |
| `V6a_dlda`, `V6b_steer2`, `V6d_moderation2` | 4 | Run 3: another estimate of the directions; inductions at about 50% and 70% of the depth |
| `V7i_indiv`, `V7m_mediate` | 5 | Representations of the seven individual characteristics; their effect when described in the agent profile, induced, or removed |
| `V8_verify` | — | Check of the batched forward passes (run 2 on) against run 1's one-at-a-time passes |
| `V9_pack` | — | Packs each job's results for download |

- `code_v6/lib_v6.py` (shared code) is executed after `runner_lib.py` (or `code_v6/runner_lib_b.py`, a second runner), `latent_lib.py`,
  `test_lib3.py` and `test_lib4.py`. The runner notebooks `runner_v1.ipynb` (built by `build_runner.py`) and `code_v6/runner_v1b.ipynb`
  mount Google Drive, log in to Hugging Face with the Colab secret `HF_TOKEN`, and execute the job files placed in
  `MyDrive/MVCA_latent/runner/jobs_pending/`. The notebooks' instructions are written in Chinese.
- Job files: `jobs6/<run>/<job>.py`, written by `make_v6_jobs.py`. Each names the Hugging Face model id and revision (`_MODEL`,
  `_REV`; `_REV = None` means the main branch as of 19-21 Oct 2026), the steps and the settings; most also list the expected
  SHA-256 prefix of every code file they run. The materials bundle that the GPU reads (`materials_v6/upload_v6/`, placed on Drive as
  `runner/materials/v6/`) is written by `make_v6_bundle.py` (and `make_indiv_v6.py bundle`).
- Results are written to `runner/results/<job>/<step>/` on Drive (`summary.json`, `digest.md`, CSV tables, `.pt` exports);
  `fetch_v6.py` reassembles the downloaded packs into `results_v6/`.
- `code_v6/` holds the final version of every step. The forward passes were batched from run 2 on; the check `V701`/`V8_verify`
  reproduced run 1 (item-level correlations .9986 to .9997; the reported per-experiment effects differed by at most .019), so the
  final code reproduces run 1's results up to this rounding.
- `mock/mock_v6.py` runs every step on a CPU with each LLM's real tokenizer and chat template and a tiny random network in place of
  the weights (a dry run of the code; it needs `transformers` and downloads the tokenizers).
- Hardware: NVIDIA RTX PRO 6000 (96 GB) on Colab; the 27B-32B LLMs need about 55-65 GB of GPU memory in bf16.

## Materials (re-creating them costs money; the frozen materials are in the Zenodo archive)

Writing and rating ran through OpenRouter. Set the environment variable `OPENROUTER_API_KEY` to your own key (it is read at call
time and never printed); `or_runner/local_run.py` runs a job file locally, `or_runner/or_runner.ipynb` on a Colab CPU.

1. `make_or_jobs_v6.py spec`, `make_or_jobs_v6.py jobs 1` and `jobs 2`: construct specifications and writing jobs (six writers).
2. Run the jobs; `collect_v6.py pairs` and `collect_v6.py exp`: collect and check the texts.
3. `make_or_rate_v6.py pairs|exp <tag>` and `collect_ratings_v6.py`: blind ratings by two raters.
4. `finalize_v6.py`: apply the exclusion rules; `redo_uniqueness_v6.py`: the one construct rewritten before the formal runs;
   `qc_report_v6.py`: quality report; `make_manifest_v6.py`: freeze the materials (`materials_v6/MANIFEST_v6.json`).
5. `make_exp_v6.py jobs|v00`: the 60 experiments and their rewordings (reads MVCA's frozen experiment materials from
   `external/agent-psychonomicus/materials/`, which comes from the repository of the MVCA article cited in the article; only the files that the analyses read are in the
   Zenodo archive).
6. `make_indiv_v6.py write|collect|rate|check|bundle`: the individual characteristics (Study 5).
7. `make_or_code_v6.py`: blind coding of the constructs (state vs fact; similarity of the definitions).

## Changes from the scripts as they ran

The scripts are the versions that produced the article's results, with these path changes (each marked `# REPOSITORY:`):
- The MVCA data are read from `external/agent-psychonomicus/` inside the repository instead of a sibling folder
  (`analysis_v6/cross_paper_v6.py`, `paper/make_si_tables.py`, `paper/figs/make_figs_ps.py`, `paper/figs/make_figs_v3.py`,
  `make_indiv_v6.py`, `make_exp_v6.py`, `make_v6_bundle.py`).
- The folder of Drive downloads is the environment variable `TOOL_RESULTS` (default `tool-results/`) instead of a local folder
  (`collect_v6.py`, `fetch_v6.py`, `analysis_v6/mediate_summary_v6.py`; with the archived results nothing needs to be downloaded).
- The question codes are read from `question_coding/` (`analysis_v6/qcoding_v6.py`, `paper/psychsci/make_table_s10.py`).
- `or_runner/local_run.py` reads the OpenRouter key from the environment variable `OPENROUTER_API_KEY` first.
- The figure and table scripts were moved from the manuscript folder (`draft_2026-10-21/`, `draft_2026-10-21/psychsci/`) to `paper/`
  and `paper/psychsci/`; their relative paths are unchanged. Docstrings still name the original folders.

Paths that remain specific to the original setup: the Colab notebooks and job files use the Drive folder
`/content/drive/MyDrive/MVCA_latent/`; `or_runner/local_run.py` falls back to a key file `../MVCA_X_app_v4/.env` outside the
repository.

## Analysis plan

`analysis_plan/` holds the dated internal analysis plan (in Chinese) and the record of what it fixed before the formal runs
(`analysis_plan/README.md`).

## Citation

Chen, J., Li, X., Zhou, Z., Xiao, W., & An, C. (2026). *Not your average Clever Hans: Latent representation of psychological constructs in large language models* [Manuscript submitted for publication]. Code: https://github.com/schema-brand/latent-psychological-constructs. Materials, data, and code archive: https://doi.org/10.5281/zenodo.23771694.
