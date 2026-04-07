# Embryo Ranking Classifier — Design Document

## 1. Aim

Build a model that **mimics expert embryologist selection style** when choosing
the best embryo from a small bundle (n = 2–5). The model is trained on
**flow-match generated synthetic blastocysts** ranked by 5 expert
embryologists, and is intended to:

- Reproduce the *consensus* selection of experts on unseen bundles.
- Preserve and surface *inter-expert disagreement* (ambiguity is signal).
- Generalize from synthetic training images to real microscope images at
  evaluation time.
- Serve as a decision-support tool, not a decision-making one.

We explicitly **do not** model clinical outcomes (live birth, implantation).
The label is "which embryo would an expert pick", not "which embryo will
implant."

## 2. Scope and non-goals

In scope:
- Static images (or, optionally, a small set of canonical-timepoint keyframes
  per embryo).
- Bundles of 2–5 embryos, all from the same culture day.
- 5 named experts, each providing a full ranking per bundle.

Out of scope (for v1):
- Full timelapse video encoders.
- Outcome prediction.
- Mixed-day bundles.
- Patient-level features (age, BMI, protocol, etc.).

## 3. Data pipeline

### 3.1 Image generation
- Flow-match generator produces synthetic blastocysts.
- **Generator should be conditioned on culture day** (d3, d5, d6, d7). If not
  possible immediately, day is assigned as metadata, but day-conditioning the
  generator is the highest-leverage upgrade on the data side.
- Optional: generate 2–3 keyframes per embryo at canonical hpi values
  (tB, tEB, tHB) sharing a `parent_embryo_id`. Captures some morphokinetic
  signal without needing a video model.

### 3.2 Bundle construction (curriculum)
Bundles are sampled in three difficulty buckets, mixed roughly 40/40/20:

| Difficulty | Method | Mean pairwise cosine (DINOv2 feat space) |
|---|---|---|
| Easy | uniform random | low |
| Medium | k-NN of a seed at moderate cosine | 0.5–0.75 |
| Hard | k-NN of a seed at high cosine | 0.75–0.90 |

Notes:
- Similarity is computed in **DINOv2 feature space**, the same encoder the
  ranker uses — "hard for the model" ≠ "hard in pixel space."
- Cap upper cosine ~0.90; near-duplicates make experts disagree randomly.
- All embryos in a bundle share the same culture day (day-homogeneous).
- After v1 of the ranker exists, mine new bundles by **model uncertainty**
  (high-entropy softmax) → active-learning loop.
- Log `difficulty` (mean pairwise cosine) and `sampling_mode` per bundle.

### 3.3 Expert annotation (Supabase + Vercel webapp)

Annotation UX requirements:
- Each expert sees each bundle exactly once, in randomized embryo display
  order. Display order is **logged** so we can audit position bias.
- Expert produces a **full ranking** of the n embryos.
- Optional "tied with previous" checkbox → populates `tie_group`.
- Capture `time_spent_ms` per bundle; flag rushed annotations.
- Capture `session_id` for fatigue analysis.
- ~5% gold/repeat bundles per expert for intra-rater consistency.
- Experts never see each other's labels or any model output during
  annotation.
- Append-only audit log: rank changes are versioned, never overwritten.

### 3.4 Database schema (Supabase / Postgres)

```sql
-- Raw image registry
images (
  image_id        uuid primary key,
  storage_path    text not null,
  generator_seed  bigint not null,
  generation_params jsonb,
  day             int not null,           -- 3,5,6,7
  hpi             int,                    -- optional
  frame_index     int,                    -- if multi-frame per embryo
  parent_embryo_id uuid,                  -- groups frames of one embryo
  created_at      timestamptz default now()
)

-- Bundle registry
bundles (
  bundle_id       uuid primary key,
  n               int not null check (n between 2 and 5),
  day             int not null,
  difficulty      float,                  -- mean pairwise cosine
  sampling_mode   text check (sampling_mode in ('easy','medium','hard','active')),
  generator_seed  bigint,                 -- for split hashing
  status          text default 'active',  -- 'active'|'qa_failed'|'withdrawn'
  created_at      timestamptz default now()
)

-- Bundle membership (canonical order set at creation)
bundle_images (
  bundle_id       uuid references bundles,
  image_id        uuid references images,
  position        int not null,           -- canonical index 0..n-1
  primary key (bundle_id, image_id)
)

-- Expert registry
experts (
  expert_id       int primary key,
  expert_name     text unique,
  years_experience int,
  clinic          text
)

-- Rankings (append-only)
rankings (
  ranking_id      bigserial primary key,
  bundle_id       uuid references bundles,
  expert_id       int  references experts,
  image_id        uuid references images,
  rank            int  not null,          -- 1 = best
  tie_group       int,                    -- nullable
  display_position int,                   -- as shown in UI
  time_spent_ms   int,
  session_id      uuid,
  version         int default 1,
  created_at      timestamptz default now(),
  unique (bundle_id, expert_id, image_id, version)
)
```

Pre-export sanity SQL:
- Every (bundle, expert) pair has exactly `n` ranks.
- Every bundle has all 5 experts.
- No rank collisions within (bundle, expert) unless `tie_group` is set.
- `display_position` distribution is uniform per expert (no UI bug).

### 3.5 Splits

- Split **by `generator_seed`**, not by bundle and not randomly. Hash the seed
  into train/val/test (80/10/10) deterministically:
  ```python
  def split_of(seed):
      h = int(hashlib.md5(str(seed).encode()).hexdigest(), 16)
      p = (h % 1000) / 1000
      return "train" if p < 0.8 else "val" if p < 0.9 else "test"
  ```
- Prevents near-duplicate synthetic embryos leaking across splits.
- Stable as new data arrives.

### 3.6 Training-ready export

One Parquet file with one row per bundle:

```python
{
  "bundle_id": uuid,
  "n": int,
  "day": int,
  "difficulty": float,
  "image_ids": [uuid, ...],          # canonical order
  "rankings": {
      expert_id: [perm_indices_into_image_ids],   # length n
      ...
  },
  "generator_seed": int,
  "split": "train"|"val"|"test",
}
```

Plus a `image_features.parquet` cache:
```
image_id | feat (float16[d])         # precomputed DINOv2 features
```

Precomputing features avoids JPEG decode in the dataloader and keeps training
fast. Re-encode only if you change augmentations or unfreeze the encoder.

## 4. Model

### 4.1 Architecture

```
images of bundle (n ≤ 5)
   │
   ▼
DINOv2 encoder (frozen initially)
   │  per-embryo embedding   [n, d_dino]
   ▼
concat day embedding         [n, d_dino + d_day]
   │  d_day ~ 16, nn.Embedding(num_days, d_day)
   ▼
Set-Transformer (2 self-attention layers, no positional encoding)
   │  permutation-equivariant, context-aware over the *current* pool
   ▼
Linear head → logits          [n, E]   (E = 5 expert heads)
   │
   ▼
masked softmax over the pool, per expert head
```

Key properties:
- **Permutation invariance** of the *bundle* (test by shuffling and asserting
  identical scores up to permutation).
- **Context-aware**: each embryo's score depends on the set it sits in.
  This is what makes iterative removal at inference meaningful.
- **5 expert heads**: structurally preserves disagreement. Averaging across
  heads gives a calibrated ambiguity distribution.

Day enters as a small learned embedding concatenated to each embryo token.
Bundles are day-homogeneous, but the model still needs day so it can transfer
between days at inference and so the encoder doesn't have to infer stage from
pixels alone.

### 4.2 Loss — Plackett–Luce per expert head

For each bundle and each expert e with ranking permutation π, unroll into
n−1 cross-entropy steps with **teacher forcing on the expert's true prefix**:

```
remaining = {0..n-1}
L_e = 0
for k in 0..n-2:
    logits = model(embryos, mask=remaining)[:, e]    # [n], -inf on removed
    L_e += cross_entropy(softmax(logits), target=π[k])
    remaining.remove(π[k])
L_bundle = (1 / (n-1)) * mean_e L_e
```

- **Teacher forcing**, never student forcing → no loss collapse, no exposure
  bias issues at this scale.
- **Per-expert heads** mean each head learns one annotator's style; averaging
  the 5 head distributions reproduces ambiguity (e.g. 3/2 splits become bimodal
  predictions, not flat ones).
- **`1/(n-1)` weighting** ensures bundles of different sizes contribute
  equally.

A bundle of 5 ranked by 5 experts contributes `5 × 4 = 20` cross-entropy
terms. With n_max = 5, we can either loop the 4 steps (simple) or batch them
along a step axis `[B, n_max, n_max]` (faster). Start with the loop.

### 4.3 Batching & padding

- `n_max = 5`. Pad bundles with zeros, carry a boolean mask `[B, n_max]`.
- Set padded logits to `-inf` before softmax → 0 prob, 0 gradient.
- A custom `collate_fn` stacks `(feats, step_masks, targets)` of fixed shape.

### 4.4 Dataset (per-step unrolling happens in `__getitem__`)

```python
class BundleDataset(Dataset):
    def __getitem__(self, i):
        b = self.bundles[i]
        n, n_max = b["n"], 5
        feats = torch.stack([self.feats[x] for x in b["image_ids"]])  # [n,d]
        feats = F.pad(feats, (0,0,0,n_max-n))                          # [n_max,d]
        base_mask = torch.tensor([1]*n + [0]*(n_max-n), dtype=torch.bool)

        E = len(b["rankings"])
        step_masks = torch.zeros(E, n-1, n_max, dtype=torch.bool)
        targets    = torch.full((E, n-1), -100, dtype=torch.long)
        for e_id, perm in b["rankings"].items():
            removed = set()
            for k in range(n-1):
                m = base_mask.clone()
                for r in removed: m[r] = False
                step_masks[e_id, k] = m
                targets[e_id, k] = perm[k]
                removed.add(perm[k])

        return {"feats": feats, "step_masks": step_masks, "targets": targets,
                "n": n, "day": b["day"], "bundle_id": b["bundle_id"]}
```

`bundle_id` is metadata only — never an input feature.

### 4.5 Inference

For a full ranking, decode iteratively (Plackett–Luce):
```
remaining = pool
ranking = []
while len(remaining) > 1:
    logits = model(embryos, mask=remaining)            # context recomputed
    p = softmax(logits.mean(dim=-1))                   # avg over expert heads
    winner = argmax(p)
    ranking.append(winner)
    remaining.remove(winner)
ranking.append(remaining[0])
```

Outputs per embryo:
- `score` (head-averaged logit)
- `p_top1` (head-averaged softmax at step 1)
- `per_expert_p_top1` (5 numbers — expose for ambiguity UX)
- `rank` from iterative decoding

## 5. Training plan

1. **Linear-probe phase.** Freeze DINOv2. Train set-transformer + heads only.
   This prevents the encoder from overfitting to ~1000 bundles of labels.
2. **Optional fine-tune.** Once linear probe plateaus on val NDCG, unfreeze
   the last DINO blocks with a 10× lower LR.
3. **Curriculum order.** Start with easy + medium bundles; phase in hard
   bundles after a few epochs.
4. **Active mining.** After v1, generate new hard bundles by model
   uncertainty, send to experts, retrain.
5. **Regularization.** Dropout in attention layers, weight decay, early stop
   on val Kendall τ.
6. **Augmentation.** Rotations and flips OK (no canonical orientation),
   conservative color jitter (color is signal).

## 6. Evaluation

**Inter-rater agreement is the ceiling.** Compute pairwise Kendall τ and
Spearman ρ between experts up front; report model metrics relative to that.

Metrics, all stratified by `day` and `difficulty`:
- Top-1 accuracy vs each expert and vs consensus
- Top-1 in expert's top-2 (more forgiving, more clinical)
- NDCG@n
- Kendall τ on full ranking
- Calibration of per-expert head distributions (predicted vs empirical
  expert agreement on held-out)

**Hold out a real-image test set.** Even ~50–100 real bundles ranked
blinded by the same experts. Without this, you cannot tell if synthetic→real
transfer works. This is the single most important eval artifact.

Diagnostics:
- Real-vs-synthetic discriminator on encoder features. If > 60% accuracy,
  the ranker is partially scoring "syntheticness."
- Permutation invariance test (shuffle bundle order, scores must be
  equivariant).
- Position-bias check: does `rank ~ display_position` have signal in expert
  data? If yes, account for it.

## 7. Pitfalls & mitigations

| Risk | Mitigation |
|---|---|
| Synthetic→real distribution shift | Hold-out real test set; SSL pretraining on real unlabeled images; conservative augmentation; discriminator diagnostic |
| Generator artifacts as shortcuts | Same diagnostic; ablate on multiple generator checkpoints |
| Saturation on easy bundles | Curriculum with cosine-mined hard bundles |
| Near-duplicate hard bundles → random expert labels | Cap upper cosine ~0.90; check κ per difficulty bin |
| Annotator fatigue / drift | Session caps, gold repeats, periodic re-annotation, intra-rater κ tracking |
| Position bias in UI | Randomize and log display order |
| Label leakage across splits | Hash by `generator_seed`, not bundle |
| Overfitting (~1000 bundles) | Linear-probe first, dropout, weight decay, early stop |
| Loss collapse from student forcing | Teacher-force on expert's true prefix only |
| Iterative decoding ≡ argsort (no benefit) | Set-transformer over the *current* pool so context actually changes |
| Bundle-size imbalance | Weight loss by `1/(n-1)` |
| Day signal absent from static images | Day as explicit conditioning input; day-homogeneous bundles; day-condition the generator if possible |
| Missing morphokinetic dynamics | Optional multi-frame keyframes via `parent_embryo_id` |
| Annotators biased by model output | Never show predictions during annotation |
| Untracked retrains | Version dataset hash, code commit, encoder ckpt, split seed per run |

## 8. Deployment / UX considerations (later)

- Show the 5-head spread, not just the argmax — surface ambiguity.
- Treat top-2 within ε as tied in the UI.
- Log every (bundle, model_version, prediction, user_choice) — future training
  data and audit trail.
- Decision support only. No patient-facing claims.

## 9. Open questions

- Does the flow-match generator support day conditioning today? If not,
  prioritize that work above any ranker improvement.
- How many bundles can the 5 experts realistically annotate? Sets the
  scale of everything else.
- Multi-frame keyframes vs. single static image — when do we add them?
- Is there an existing real-image bundle set we can blind and re-rank for
  the held-out test set?

## 10. Milestones

1. **M1 — Annotation infra.** Supabase schema live, Vercel UI, 5 experts
   onboarded with calibration set, gold/repeats wired in.
2. **M2 — Bundle generation.** Cosine-mined easy/medium/hard bundles, day
   metadata correct, difficulty logged.
3. **M3 — First labels.** ~500 bundles fully ranked. Compute inter-rater κ.
4. **M4 — Linear-probe model.** Set-transformer + 5 heads + PL loss on
   frozen DINOv2. Eval against expert consensus.
5. **M5 — Real-image hold-out test.** Blinded rerank by same experts.
6. **M6 — Active mining loop.** Hard bundles via model uncertainty.
7. **M7 — Optional encoder fine-tune / multi-frame extension.**
