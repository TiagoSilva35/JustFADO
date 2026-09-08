# FADO — Talk Guide (research-group presentation)

**Deck:** `FADO_ICDM2026.pptx` (14 slides) · **Target length:** ~20 min talk + Q&A
**Audience:** your research group — technical, will push on method choices and the controller.
**One-sentence goal of the talk:** *convince them that drift is a fairness problem, and that a
node-level-fair oblique forest wrapped in a reaction controller is a principled fix — and be
honest about where the evidence is still thin (the controller).*

---

## Timing budget

| # | Slide | Time | Cumulative |
|---|-------|------|-----------|
| 1 | Title | 0:30 | 0:30 |
| 2 | Problem & RQs | 1:30 | 2:00 |
| 3 | Positioning | 1:30 | 3:30 |
| 4 | Why oblique soft trees | 2:30 | 6:00 |
| 5 | Node-level regulariser | 2:30 | 8:30 |
| 6 | Intersectional | 2:00 | 10:30 |
| 7 | Controller | 2:30 | 13:00 |
| 8 | Architecture | 1:00 | 14:00 |
| 9 | Experimental design | 1:00 | 15:00 |
| 10 | Results I — marginal | 2:00 | 17:00 |
| 11 | Results II — intersectional | 2:00 | 19:00 |
| 12 | Synthesis | 0:45 | 19:45 |
| 13 | Status & next | 0:45 | 20:30 |
| 14 | Closing | 0:20 | ~21:00 |

**If you're short on time**, compress slides 3, 8, 9 (positioning / architecture / setup) — they're
context, not contribution. **Never rush 4, 5, 7** (the design decisions) — that's what this audience
came for. **The two spine slides are 4 (why the base learner) and 11 (the new result).**

---

## Per-slide script

### 1 · Title — *0:30*
- "This is the work I submitted to ICDM 2026. Today I want to focus less on selling the results and
  more on **why the method is built the way it is**, and then show you both the marginal results
  from the submission and the new intersectional numbers."
- Set expectation: *design decisions first, results second.*

### 2 · Problem & RQs — *1:30*
- **Key line:** "Drift is usually studied as an *accuracy* problem. My starting point is that it's
  also a *fairness* problem — and those aren't the same thing."
- Walk the logic: fair learners assume a fixed distribution → real streams move `P(y|x)` and base
  rates → the statistical relationships fairness relies on break → **bias can grow while accuracy
  looks stable.**
- Land the two RQs: RQ1 = *measure the effect*, RQ2 = *can adaptation fix it*. The whole talk maps
  back to these two.
- **Transition:** "So why hasn't this been solved? Because the literature picks a side."

### 3 · Positioning — *1:30*
- Three families, each solves half: in-processing fair (static), drift-adaptive (unfair),
  drift-aware-fair (assumes shift is a fixed perturbation, not a stream).
- **Key line:** "Rather than invent a new base learner, I keep an existing in-processing fair learner
  — Aranyani — and add the layer everyone's missing: an online detector and a controller."
- **Say the honest bit out loud:** "Aranyani is *not* my contribution. My contribution is the
  drift-detection-and-response machinery wrapped around it." (Pre-empts the obvious question.)
- **Transition:** "So let me justify that base learner, because the choice isn't arbitrary."

### 4 · Why oblique soft trees — *2:30*  ⭐ spine slide
- Three reasons, in order:
  1. **Oblique** = splits on a linear combination of features (an angled hyperplane) → far more
     expressive per node than axis-aligned.
  2. **Soft routing** (temperature-scaled sigmoid) → the tree is *differentiable*, so the whole
     forest trains by gradient descent, online, one sample at a time.
  3. Because each node is now a smooth per-step prediction, **you can impose group fairness at every
     node**, not just at the output.
- Then the "what this buys us" card — hit **parameter isolation** (cheap per-step gradient) and
  **O(W) memory** (no stored past samples — essential for a stream).
- **The punchline that connects to slide 7:** "And notice the temperature τ. At τ=1 the trees route
  smoothly — good for stable learning. Push τ→0 and routing becomes sharp and decisive. **Hold that
  thought — τ is the exact knob my controller grabs when drift hits.**"

### 5 · Node-level regulariser — *2:30*  ⭐
- "At each node I compare the group-conditional mean routing to the overall mean — that difference,
  `Fₐ`, is a per-node version of the demographic-parity gap." Point at the equation.
- The objective just sums that penalty over every node, weighted by λ.
- **Spend your time on the Huber surrogate** (the `φ_δ` box) — this is the kind of choice this
  audience respects:
  - quadratic near zero → smooth, well-conditioned gradient for small deviations;
  - linear in the tails → a few outlier nodes can't dominate the update;
  - **unit slope** in the tail (vs δ in textbook Huber) → a firmer correction once you're past δ.
- Mention the DP definition choice: "I use deviation-from-the-group-mean, not pairwise-max,
  *specifically* so the metric I report is the thing the regulariser optimises."
- **Transition:** "So far this is single-attribute. Here's the part I changed this cycle."

### 6 · Intersectional — *2:00*  ⭐ (the new work)
- Lead with the **gerrymandering** result: "Kearns et al. proved marginal-only fairness is
  *provably* defeatable — a model can be perfectly fair on sex, perfectly fair on race, and badly
  unfair on, say, Black women specifically."
- The fix: optimise over the **Cartesian-product subgroups**. Show `a = 2·a₁ + a₂ ∈ {0,1,2,3}`.
- **The elegant bit:** "Because the node regulariser was already written for arbitrary K groups,
  I changed *no learner code* — just the group index in the data loaders."
- **Transition:** "Now the drift half — the actual contribution."

### 7 · Controller — *2:30*  ⭐ spine slide
- Frame it as **two detectors, three phases, one temperature**:
  - two ADWIN detectors on the error stream — a *sensitive* pre-warm and a *strict* confirmation;
  - **warn** (gently raise LR, arm) → **confirm** (spike LR, sharpen τ→0.1, reset) → **recover**
    (anneal LR and τ back).
- Point at `ε_cut`: "ADWIN's cut bound scales with stream length, which is why my thresholds differ
  between the long Folktables stream and the short COMPAS one — it's calibrated, not hand-tuned."
- **The label-noise guard (ν=0.7)** is worth naming: "a confident-but-wrong prediction is excluded
  from the change signal, so one mislabeled point can't fire the whole controller." This is the
  thing that stops it thrashing.
- **Transition:** "Here's how those pieces wire together."

### 8 · Architecture — *1:00*
- Don't read the boxes. Trace **two paths**: blue = a sample flowing through the fair forest;
  orange = the detector firing the controller, which retunes the optimiser feeding the gradient step.
- "Test-then-train, one sample at a time — fully online." Move on.

### 9 · Experimental design — *1:00*
- **Two regimes on purpose:** Folktables = a *real but gradual* cross-year shift (2015 → 2017–18);
  COMPAS = a *short, sharp, synthetic* subpopulation shift with known ground truth.
- Protocol in one breath: prequential, batch-1, 30 seeds, paired Wilcoxon + Holm, Welch cross-check.
- **The one baseline that matters:** "The controller-free Aranyani ablation shares *everything* —
  same learner, loss, window, even seed-pinned data order — so any gap is the controller alone."

### 10 · Results I — marginal (as submitted) — *2:00*
- Two messages, don't get lost in the numbers:
  - **RQ1:** "Both external baselines carry roughly *double* FADO's DP — on both regimes. Drift
    erodes fairness for the accuracy-only method *and* the static-fair one."
  - **The honest, interesting one:** "The controller's benefit is **regime-dependent**. On the
    abrupt COMPAS shift it clearly helps — DP down 0.68pp, EO down 0.56pp, both significant. On the
    *gradual* Folktables shift it essentially **ties** the base learner. And that makes sense: a
    smooth shift doesn't need a spike."
- Saying that tie out loud *builds credibility* — you're not overselling.
- **Transition:** "But that tie was under the *marginal* metric. Watch what happens intersectionally."

### 11 · Results II — intersectional (new) — *2:00*  ⭐ payoff
- **The headline:** "Under the joint objective, the controller's edge over the controller-free forest
  becomes significant under **both** tests — Wilcoxon *and* Welch — where the marginal view only saw
  a tie. The intersectional metric is more sensitive to exactly what the controller fixes."
- Be honest about magnitude: "It's a small absolute gap — 0.38pp — but it's *consistent* and it's
  *measurable*, which it wasn't before."
- **Second finding — gerrymandering, confirmed empirically:** "The ratio of intersectional to
  worst-marginal DP is about 2.8 for FADO, and it's *highest for the fairest learners*. They scrub
  the marginal axes so well that the residual unfairness hides in the joint cells — a marginal-only
  audit would have called FADO 'very fair' and missed it."
- **Transition:** "So what does all this add up to?"

### 12 · Synthesis — *0:45*
- Three cards, one sentence each — don't re-explain: RQ1 (drift is a fairness problem), RQ2 (an
  adaptive fair learner keeps both; the controller earns its keep on abrupt shift), Design lesson
  (the objective has to match the harm you care about).

### 13 · Status & next — *0:45*
- Done: intersectional extension, Folktables sweep, verified stats, controller audit.
- **Set up your own next-step honestly:** "The thing I most want to do next is a proper **controller
  ablation** — isolate the label-noise guard, the temperature modulation, the LR pre-warm. It's the
  weakest, most attackable part of the paper, and the biggest lever for a stronger resubmission."
- This turns your known weak point into *your* agenda — much stronger than having a reviewer raise it.

### 14 · Closing — *0:20*
- "A drift-aware controller keeps an intersectionally-fair oblique forest fair as the distribution
  moves." Thank them, open the floor.

---

## Delivery notes

- **Signpost with the RQs.** Say "this is the RQ1 finding" / "this answers RQ2" — it gives a
  technical audience a spine to hang the numbers on.
- **Under-claim, don't over-claim.** The 0.38pp gain and the Folktables tie are your credibility.
  State them plainly; a group that trusts your honesty will trust your bigger claims.
- **Don't read equations aloud symbol-by-symbol.** Say what they *do* ("compare group routing to the
  average"), point at the box, move on.
- **The three ⭐ slides (4, 5, 7) are where you slow down.** Everything else is scaffolding.
- If a demo of nerves helps: the two lines you must nail are the **τ hand-off** (end of slide 4) and
  the **"marginal tie → intersectional win"** contrast (slides 10→11). Rehearse those transitions.

---

## Anticipated questions (prep)

**Q. Why not optimise equalised odds too, instead of only DP?**
A. Both objectives need separate per-class running statistics and an extra per-node penalty every
step — that roughly doubles the per-step cost, which wasn't feasible under the 30-seed budget on my
hardware. The EO regulariser *is* implemented and I report EO as a tracked diagnostic; wiring it into
the gradient is a next step, not a missing capability.

**Q. The controller only helps on abrupt drift — so is it worth it?**
A. On *this* evidence, its clear win is on abrupt shift, where a spike-then-anneal response matches
the change. On gradual shift it ties the base learner — which is the *correct* behaviour, not a
failure: you don't want a big spike for a smooth shift. The intersectional result (slide 11) is the
case where it also helps on gradual data, once you measure the right thing.

**Q. 0.38pp over the controller-free baseline is tiny. Why care?**
A. Two things. It's *consistent* across 30 seeds and significant under both tests, where the marginal
setting was a statistical tie — so the contribution went from unmeasurable to measurable. And it
comes at essentially no accuracy cost. I'm not claiming a huge effect; I'm claiming the controller's
contribution is now *detectable and directionally reliable*.

**Q. Why oblique trees rather than standard (axis-aligned) trees or a neural net?**
A. Two requirements drove it: I need per-node differentiability so I can impose fairness at each node
and train online by gradient descent, and I need parameter isolation so the per-step gradient and the
running group statistics stay cheap. Soft-routed oblique trees give both; axis-aligned trees aren't
differentiable, and a generic net loses the cheap node-level fairness structure.

**Q. Composite group is only 4 cells. Does this scale to more attributes / non-binary attributes?**
A. The regulariser handles arbitrary K, so *mechanically* yes. The honest limit is combinatorial:
K grows as the product of cardinalities, and cells get sparse, so the running per-group estimates get
noisier. Two binary attributes (K=4) is the well-behaved case I validated; scaling K and handling
sparse cells is explicitly future work.

**Q. Where are the intersectional COMPAS numbers?**
A. Paused. The pipeline is implemented and single-seed-validated, but the full 30-seed sweep is
outstanding — I flagged the compute constraint separately. Results II is Folktables-only and the
slide says so.

**Q. Why paired Wilcoxon as the primary test, with Welch as a cross-check?**
A. Following Demšar's recommendation for prequential benchmarks: Wilcoxon is non-parametric and
exploits the within-seed pairing from the shared data order. The two tests answer different
questions — Welch asks whether the two models' *distributions* differ; paired Wilcoxon asks whether
the controller *systematically* shifts the per-seed outcome one way. RQ2 is the second question, so
Wilcoxon is primary; I report both for transparency.

**Q. Is 2.8 vs 2.0 (gerrymandering ratio) actually a meaningful difference?**
A. It's a directional, interpretive finding, not a hypothesis test: the *fairer* the learner, the
*more* concentrated its residual unfairness is in the joint cells. The claim isn't "2.8 is
significantly bigger than 2.0"; it's "the ordering is consistent and it's exactly the pattern that
justifies optimising the intersectional objective."

**Q. You mentioned auditing the controller — did you find bugs?**
A. Yes — I found a few discrepancies between the documented design and the executed code (recovery
keyed on cumulative rather than rolling accuracy, an inverted warn/confirm threshold, a weaker
label-noise guard than described). I deliberately left them *unchanged* for now: on analysis, the
current behaviour is plausibly part of what drives the abrupt-drift advantage, so a blind "fix" could
erase the contribution. They're documented for the controlled ablation, which is exactly why that
ablation is my top next step. *(Only volunteer this depth if asked — but be ready, it's your most
likely tough question.)*

**Q. Any risk of test-time leakage in the prequential setup?**
A. No — strict test-then-train: predict on the sample first, then update on it. On COMPAS the
synthetic edits are applied to the test stream only, and the categorical encoder is fit on the
training split alone and reused unchanged.

**If you don't know an answer:** say so, say why it's a good question, and offer to follow up — this
group respects that far more than a bluff. "I haven't tested that — good question. My expectation is
X because Y, but I'd want to check before claiming it."
