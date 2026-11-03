# GCN-LSTM Nursing Intervention Sequencing

Reference implementation of the framework described in *Modeling the Sequencing
of Nursing Interventions During the Bowel Function Recovery Period in
Postoperative Gastric Cancer Patients Using a GCN-LSTM Fusion Model*.

`main.py` runs the complete experiment programme and writes every table, result
figure and intermediate plot under `outputs/`. No result value is stored in the
source tree.

---

## 1. Task

Eight concurrent nursing interventions are delivered during the bowel function
recovery period. The framework treats the clinical problem as a permutation
decision: given an observed recovery trajectory, output the execution order of
the eight interventions that reproduces the reference order.

| Item | Definition |
|---|---|
| Record | one postoperative stay, 6-hour step, at most 20 steps |
| Input | intervention execution states `(T, 8)`, six indicators `(T, 6)`, age, sex, surgery type |
| Output | permutation `pi = (pi_1..pi_8)` and an estimate of the recovery node |
| Reference | triple-expert order per record, aggregated by Borda majority vote |
| Metrics | exact-match prefix accuracy at lengths 3-8, Kendall tau, MAE, paired Wilcoxon, Cohen's d, bootstrap CI, Holm correction |

## 2. Data flow

```
record
  -> indicator imputation (carry forward, training-mean fallback)
  -> standardisation with training statistics
  -> node features  X_t in R^{(n+m) x d},  d = 16
  -> dependency weights  w^II(t) = c_ij s_i (1 - s_j)          eq. 1
  -> temporal weights    w^IM(t) = F_ik / sum_k' F_ik'          eq. 2
  -> adjacency  A^II_t, A^IM_t
  -> typed message passing, two independent weight matrices     eq. 3-5
  -> flattened embedding e_flat^t
  -> gated temporal encoder                                     eq. 6-9
  -> context vector h_T
  -> decoder with dynamic masking                               eq. 10-12
  -> permutation pi  +  recovery node regression head
```

## 3. Layout

Four tiers, each depending only on the tiers below it.

```
Tier 0  core/         config, seeding, logging, memory probe
Tier 1  data/         cohort construction, graph operators, dataset assembly
Tier 2  models/       graph layers, temporal encoder, decoders, network
Tier 3  training/     objective, training loop, evaluation
        evaluation/   metrics and paired statistics
        baselines/    rule ordering and tree comparators
Tier 4  experiments/  experiment drivers and figure production
        main.py       entry point
        infer.py      checkpoint inference
```

| Module | Responsibility |
|---|---|
| `src/core/config.py` | every hyper-parameter and path, `quick` and `full` presets |
| `src/core/runtime.py` | seeding, logger, directory preparation, peak memory |
| `src/data/cohort.py` | cohort record set, reference orders, annotation calibration |
| `src/data/graph.py` | equations 1 and 2, adjacency assembly |
| `src/data/dataset.py` | imputation, standardisation, features, splits, batching |
| `src/models/graph_layers.py` | `TypedMessagePassing`, `HomogeneousPassing`, `AttentionAggregation` |
| `src/models/temporal.py` | `GatedCell`, `TemporalEncoder` with exposed gates |
| `src/models/decoders.py` | `PointerDecoder`, `SoftmaxDecoder`, `Seq2SeqDecoder` |
| `src/models/network.py` | `SequencingModel`, regression head |
| `src/training/loop.py` | loss composition, training loop, evaluation |
| `src/evaluation/metrics.py` | accuracy, tau, MAE, R2, paired tests, Holm |
| `src/baselines/comparators.py` | rule ordering, Random Forest, CatBoost |
| `src/experiments/runner.py` | all tables and experiment blocks |
| `src/experiments/figures.py` | figure production |

Two details matter for fidelity. The Granger test keeps `L_G = 4` and `q = 1`,
so the full model spends three parameters per window and the denominator holds
one residual degree of freedom. Ablations A2 and A3 zero the corresponding
adjacency inside `SequencingModel.encode`, giving `h = relu(W * 0) = 0`, which is
the "replace the path output by a zero vector" semantics.

## 4. Data

`resolve_cohort` checks `data/source/cohort_manifest.json` first. When a manifest
is present the reader registered in `src/data/dataset.py` is used; otherwise the
cohort is built by `src/data/cohort.py` and written to `data/cohort/`.

The builder draws, per record: demographics; a right-skewed recovery time whose
mean and standard deviation match the reported first-flatus statistics; six
indicator trajectories from a population trend, a record-level severity term and
AR(1) noise with a sigmoid bowel-sound response at flatus; eight monotone
intervention states whose onsets follow the sampled order; and three annotator
orders whose noise is calibrated by bisection so that mean pairwise Kendall tau
reaches 0.84, above the 0.80 admission threshold.

Generated files: `data/cohort/cohort.npz`, `sample_records.csv`, `manifest.json`.

To supply an external cohort, place the manifest at
`data/source/cohort_manifest.json` and implement its reader in
`src/data/dataset.py::resolve_cohort`. Everything downstream is independent of
where the cohort came from.

## 5. Running

```bash
pip install -r requirements.txt

python main.py                  # quick profile, ~8 min on CPU
python main.py --profile full   # five seeds, full widths
python main.py --skip sensitivity robustness
python infer.py                 # reload outputs/checkpoints/best_model.pth
```

`outputs/` and `data/` are recreated on every run and are excluded from version
control.

| Output | Content |
|---|---|
| `tables/Table1..Table8` | dataset statistics, significance, ablation, decision modules, cost, sensitivity, generalisation, robustness |
| `figures/Figure5..Figure11` | accuracy by length, tau by group and stage, scatter, sensitivity, generalisation, robustness, cost |
| `figures_data/Figure5/6/7_Data.csv` | plotted values, same run |
| `intermediate_process/int_2_1..2_5` | topology evolution, Granger profiles, path activation, gate dynamics, masked attention, imputation |
| `intermediate_process/int_train_convergence.png` | loss, sampled return, validation accuracy |
| `checkpoints/best_model.pth` | trained weights |
| `logs/run.log`, `logs/run_summary.json` | run record |

## 6. Reading the results

Three properties hold on any run and are worth checking first: accuracy
decreases monotonically as the evaluated prefix grows; every ablation falls below
the full model on at least one metric; accuracy degrades as observation noise and
missingness increase.

Absolute values depend on the training budget and on the difficulty of the
cohort, which is why `Table2` reports confidence intervals and adjusted p values
rather than bare means. Relative differences with small standard errors carry
more information than the absolute level.
