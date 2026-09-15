# S2Looking — temporal direction and label semantics

Investigation only. No labels were regenerated, no images swapped, no model
trained, no production code touched. Machine-readable record:
`data/s2looking/metadata/semantics_check.json`
(reproduce with `python scripts/check_s2looking_semantics.py`).

## Question

1. Which of `Image1` / `Image2` is the **earlier** acquisition?
2. What exactly do `label1` and `label2` mean?

## 1. Sources checked

| Source | Checked | Outcome |
|---|---|---|
| Paper, Remote Sensing 13(24):5094 (full text) | yes | **Fixes label meanings. Silent on image order.** |
| Official GitHub README | yes | Download links only |
| Official repo file tree (API, all folders) | yes | Only nested empty dirs + `.DS_Store`. No README, data dictionary or licence |
| GitHub issues #1–#4, incl. all comments | yes | **Issue #2 has an OWNER reply** (below) |
| Embedded file metadata (PNG text chunks / EXIF) | yes | Empty for all four raster types |
| Zip entry timestamps | yes | Packaging times (2021-04-09), not acquisition |
| MDPI supplementary / data availability | yes | Availability statement only ("datasets can be found here") |
| Open-CD dataset config | yes | Downstream convention, not author evidence |
| torchgeo | yes | No S2Looking module exists |
| ar5iv / MDPI HTML mirrors | attempted | 403 / connection reset; full text obtained locally from the PDF instead |

## 2. Evidence

### 2.1 Documentary — authoritative

Paper, Figure 2 caption (verbatim):

> "Images 1 and 2 in Figure 2 are bitemporal remote-sensing images, while
> Labels 1 and 2 are the corresponding annotation maps. **Labels 1 and 2
> indicate pixel-precise newly built and demolished areas of buildings,
> respectively.**"

Paper, Annotation and Quality Control (verbatim):

> "All newly built and demolished building regions in the dataset were annotated
> at the pixel level in separate auxiliary graphs."

The full text contains **no** statement binding `Image1`/`Image2` to a temporal
order — no "first image", "earlier", "later", "pre-change" or "post-change" used
of the pair. The string `Images 1 and 2` occurs exactly once, in the caption above.

Author reply, GitHub issue #2 (`author_association: OWNER`, 2021-10-08):

> 我们的数据集一共有三种标签，红蓝两种分别代表图A相对图B的变化和图B相对图A的变化，
> 二值的黑白图表示图A和图B之间的变化区域

> "Our dataset has three kinds of labels in total. The red and blue ones
> respectively represent the change of image A relative to image B, and the
> change of image B relative to image A. The binary black-and-white image
> represents the changed area between image A and image B."

This confirms the two colour maps are **genuinely directional** and that the
red/blue channel encoding is intended. It does **not** bind "image A" to
`Image1`, nor state which is earlier.

### 2.2 Measurement — supporting only, and only after passing a control

A building-presence proxy is worthless unless it can recover a **known** answer
first. LEVIR-CD is the control: it ships `A/` (earlier) and `B/` (later) with
documented order, and its change is predominantly building growth, so a valid
feature must separate A from B there.

| Feature | LEVIR A→B, frac(B>A) | Verdict |
|---|---|---|
| gradient-orientation concentration | **0.931** (0.0928 → 0.1625) | **passes** |
| edge energy | 0.443 (109.4 → 101.0) | **fails — discarded** |

Edge energy fails and moves *against* intuition: on LEVIR a new building
*lowers* it, because "before" regions are textured construction sites while new
roofs are smooth. **An earlier iteration of this investigation leaned on edge
energy; that evidence is withdrawn.** Orientation concentration is retained
because it is physically principled (buildings impose a few dominant straight
edge directions) and it passed the control.

Applying the surviving feature to S2Looking pure regions (n = 780 regions):

| Split | label1: frac(Image2 > Image1) | label2: frac(Image2 > Image1) |
|---|---|---|
| train | 0.127 (n=165) | 0.907 (n=86) |
| val | 0.112 (n=205) | 0.918 (n=61) |
| test | 0.204 (n=181) | 0.890 (n=82) |

Consistent across all three splits:

- **label1 regions contain the building in `Image1`**
- **label2 regions contain the building in `Image2`**

Direct visual inspection of 7 rendered examples agrees: `train/280`,
`train/4599`, `test/746` (label1) show buildings in `Image1` and bare ground in
`Image2`; `train/765`, `test/4100` (label2) show bare ground in `Image1` and
buildings in `Image2`.

## 3. Conclusion

**Partly resolvable.**

- **Label meanings — RESOLVED (documentary).** `label1` = newly built,
  `label2` = demolished. Stated explicitly in the paper and corroborated by the
  authors' own issue reply that the two maps are directional.
- **Image temporal order — NOT RESOLVED by any official source.** No primary
  document states which of `Image1`/`Image2` is earlier.

Combining the two strands: if `label1` really marks *newly built* areas, and
`label1` regions demonstrably contain the building in `Image1`, then **`Image1`
is the later acquisition and `Image2` the earlier one** — the reverse of what
the folder names suggest.

## 4. Confidence

| Claim | Confidence |
|---|---|
| `label1`/`label2` are directional, mutually exclusive in intent | **HIGH** — paper + author statement |
| `label1` regions hold the building in `Image1`; `label2` in `Image2` | **HIGH** — control-validated feature, consistent across 3 splits, plus visual inspection |
| Therefore `Image1` is the later image (paper's naming taken as correct) | **MEDIUM-HIGH** — inference, not documentation |
| Which *name* is right ("Image1 is later" vs "labels are swapped") | **LOW** — not determinable from available sources |

## 5. Remaining ambiguity

Two readings fit every piece of evidence equally well:

- **(A)** `Image1` is later, `Image2` is earlier; label names as printed in the paper.
- **(B)** `Image1` is earlier, `Image2` is later; the label1/label2 meanings are
  the reverse of the paper's wording.

They are **observationally equivalent**: both yield the identical
`(before, after, target)` pairing. They differ only in which artefact carries the
error. No available source discriminates between them, so this document does not
pretend to. The practical consequence is nil for training and total for naming.

## 6. Recommendation

```
before image      = Image2
after  image      = Image1
construction      = label1
demolition        = label2
```

Equivalently, keep `Image1` as "before" and swap the label meanings — identical
training pairs. What must **not** happen is the current combination in
`region_labels.csv`, which pairs `construction → label1` while treating
`Image1` as "before"; that is internally inconsistent and would train inverted
semantics while every metric still looked healthy.

`region_labels.csv` is deliberately left as generated (40,221 rows: 24,472
construction / 13,889 demolition / 1,860 mixed) pending this decision.

## 7. Author query (prepared, not sent)

> Subject: S2Looking — which of Image1/Image2 is the earlier acquisition?
>
> Dear S2Looking authors,
>
> We are using S2Looking to study construction-vs-demolition directionality and
> want to be certain we interpret the data as you intended.
>
> The paper states that Labels 1 and 2 indicate "newly built and demolished
> areas of buildings, respectively", and your reply on GitHub issue #2 explains
> that the red and blue maps represent the change of image A relative to image B
> and vice versa. What neither states is the temporal order of the image folders.
>
> Could you confirm, for the released archive
> (`S2Looking/{train,val,test}/{Image1,Image2,label1,label2}`):
>
> 1. Which of `Image1` and `Image2` is the **earlier** acquisition?
> 2. Does `label1` mark buildings that are **present in the later image and
>    absent in the earlier one** (newly built), and `label2` the reverse
>    (demolished)?
> 3. Which folder corresponds to "image A" and which to "image B" in your issue
>    #2 reply?
>
> Our measurements on the released files indicate that label1 regions contain a
> building in `Image1` and label2 regions contain a building in `Image2`, which
> would imply `Image1` is the later image. We would like to confirm this rather
> than rely on our inference.
>
> Thank you for releasing the dataset.

Contact: `shenli@bjirs.org.cn` (given in the official README).

## 8. Ratified project decision

Interpretation **(A)** has been ratified as an explicit project decision:

```
BEFORE = Image2        AFTER = Image1
CONSTRUCTION = label1  DEMOLITION = label2
```

**What this is, and what it is not.** This is a project decision taken on the
balance of the evidence in sections 2–4. It is **not** a claim that the official
S2Looking documentation states `Image2` is the earlier acquisition — section 3
establishes that no primary source states the image order at all. The paper
fixes only that `label1` = newly built and `label2` = demolished. Anyone citing
this project must describe the ordering as an assumption, not as a documented
property of the dataset.

Interpretation (B) remains equally compatible with every source; it yields
identical `(before, after, target)` pairings and differs only in which artefact
is deemed mislabelled. Ratifying (A) therefore costs nothing in training terms
and buys a single consistent convention.

**Where the decision lives.** In four constants in
`src/domains/built_environment/data/s2looking.py`:

```python
BEFORE_IMAGE_DIR = "Image2"
AFTER_IMAGE_DIR  = "Image1"
CONSTRUCTION_LABEL_DIR = "label1"
DEMOLITION_LABEL_DIR   = "label2"
```

Everything that reads the dataset goes through them rather than naming a folder
directly, so the convention cannot drift between the preprocessing script, the
visual verification and any future loader.

**Guards.** `tests/test_stage3b.py` pins all four constants, asserts the
`TEMPORAL_ORDERING` metadata is self-describing, and asserts
`documented_by_authors is False` so the decision can never be quietly upgraded
into a claim about the authors' documentation. Flipping any constant inverts
every target while all statistics still look healthy — which is precisely why it
is pinned rather than trusted.

**Provenance in artifacts.** `TEMPORAL_ORDERING` — including its `basis` string
and `documented_by_authors: false` — is written verbatim into
`dataset_stats.json` and `quality_checks.json`, so any regenerated artifact
carries the reasoning with it.

The author query in section 7 remains worth sending. A reply would upgrade this
from a ratified assumption to a documented fact, or tell us to flip it.

### 8.1 Post-ratification verification

Everything was regenerated under the ratified mapping and re-checked.

**The region targets did not change — and that is the expected result.** The
regenerated `region_labels.csv` is byte-identical to the pre-ratification file
(sha256 `48d2e499f8a3fbf1de8acca177d72b2edb99c227e1cac4024a61f8f65f8039b5`,
2,759,073 bytes, 40,221 rows). Ratification corrected the *image-ordering
convention*, which the CSV never encoded; the label→target mapping
(`construction ← label1`) was already what the file contained. What changed is
everything that consumes the ordering: the visual verification panels, the
provenance stamped into the artifacts, and the code path that reads the images.

**Semantic consistency** (`scripts/verify_target_consistency.py`, 500 regions
≥1000 px, using only the LEVIR-control-validated orientation feature):

| Target | orient. BEFORE | orient. AFTER | frac(after > before) | Expected building in | Result |
|---|---|---|---|---|---|
| construction | 0.1217 | 0.2102 | 0.872 | AFTER | **consistent** |
| demolition | 0.2526 | 0.1311 | 0.136 | BEFORE | **consistent** |

**Visual re-verification.** The four scenes that previously read inverted now
read correctly: `train/280` and `test/746` (construction) show cleared ground in
BEFORE and buildings in AFTER; `train/765` and `test/3245` (demolition) show
buildings in BEFORE and cleared ground in AFTER. Every figure caption carries
`BEFORE = Image2, AFTER = Image1 (ratified project decision, not
author-documented)`.

**Unchanged statistics** (as expected, since targets are identical): 5,000
scenes, 40,221 regions — 24,472 construction (60.844%), 13,889 demolition
(34.532%), 1,860 mixed (4.624%); all eight quality checks unchanged; split
integrity ok; region ids deterministic.

## 9. Status

Ratified and applied to data preparation only. Still **not** done: no classifier
training, no `Region.change_type`, no `ChangeResult` schema change, no UI change,
and no construction/demolition prediction in production. The LEVIR detector,
its checkpoint and its published metrics remain untouched.
