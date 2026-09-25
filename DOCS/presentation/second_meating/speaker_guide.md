# Meeting 2 — speaker guide

What to say on each slide, in plain words. The λ controller (slides 06–07) is
where the audience is most likely to get lost, so it has a step-by-step script
and a list of likely questions.

Slides 08–09 show a three-seed check on COMPAS. Say the seed count every time
you show a number: it's a direction, not a claim.

---

## 01 · Outline

> "Three parts: what went wrong and what we fixed, the decisions we took, and
> the new piece — a controller that adjusts the fairness pressure by itself."

## 02 · Weekly reviews — KANs

Keep it short; this is context, not the main thread.

## 03 · The review pass

**Message:** some earlier results cannot be trusted, and we found out before a
reviewer did.

- Say plainly: *every COMPAS number we had was computed on 10 test samples*
  because of a debugging line left in the loader. They will be re-run.
- The other two columns are the same kind of problem: experiments that
  *looked* like they varied something but did not (the sweep), and arms that
  were not measured the same way (different windows).
- End on the footer: one thing we feared (the variable-dispatch collision)
  turned out not to affect any stored result.

Avoid dwelling on each bug. The point is "the evaluation is now trustworthy",
not the list.

## 04 · Correction to Meeting 1

**Say it first, before anyone asks.**

> "Last time I showed 37 → 13 minutes on Adult. That benchmark timed a copy of
> the updater, not the code the pipeline actually ran. The complexity argument
> still holds; the wall-clock number does not until I re-measure it."

Correcting your own number builds trust in everything else you show.

## 05 · Decisions taken

Read the left column; add one sentence only where it helps:

- **Trade-off curves:** "Instead of picking a λ for each method, we show every
  λ. Nothing to cherry-pick."
- **Tuning budget:** "If a setting affects every method, every method gets
  tuned on it. The controller's own knobs are only tuned in controller
  ablations."
- **Metrics:** "We report the whole stream *and* just the part after the
  drift, because that is where FADO is supposed to differ."

---

## 06–07 · The λ controller — how to explain it

### The one-sentence version

> "Instead of fixing how much the model cares about fairness, we let it
> *charge itself a price* for unfairness: the price goes up while it is
> unfair, and comes back down when it is fair."

If you only get one sentence in, it is that one.

### The 3-minute version, in four beats

**Beat 1 — What we actually want (the problem with a fixed λ).**
> "Aranyani adds a fairness penalty times a number, λ, that you choose by hand.
> But nobody really wants 'λ = 0.3'. What you want is a rule like 'the
> difference between groups must stay below 5%'. The λ that achieves that
> depends on the data — and under drift the data changes, so a fixed λ is
> right at one moment and wrong the next."

Analogy if needed: *a fixed λ is like setting a fixed thermostat power instead
of a target temperature. It works until the weather changes.*

**Beat 2 — Turn the rule into a price.**
> "Optimisation has a classic trick for rules like this: attach a *price* to
> breaking the rule. If you break it, you pay; if you respect it, you pay
> nothing. At the best solution, the price is exactly what it takes to keep
> the rule. So the right λ is not something to tune — it is a price the
> algorithm can find."

(This is the Lagrangian and complementary slackness. You don't need to name
them unless someone asks.)

**Beat 3 — How the price moves (the update rule on the slide).**
> "Every new sample, we check: is the gap above the target? If yes, raise the
> price a little, in proportion to how far above. If no, lower it a little.
> It never goes below the λ the baseline uses, and never above a cap."

Analogy: *a fine that grows every day you keep speeding, and is only slowly
forgiven once you slow down.* Because it keeps adding up while the violation
lasts, it doesn't stop until the rule is met. That's why it beats a rule that
just maps "current unfairness → λ".

**Beat 4 — How we measure unfairness from one sample (slide 07).**
> "We only see one person at a time. We need an estimate of the gap from each
> single sample that is correct *on average*. Weighting each prediction by how
> common that person's group is gives exactly that. The obvious alternative —
> a rolling fairness score over the last few hundred samples — is almost the
> same number twice in a row, which fools drift detectors: on a stream with
> no drift at all, it raised 24 false alarms; ours raised none."

Point at the table on slide 07 while saying the last sentence.

**Close with the footer of slide 06 (forgetting).**
> "One more piece: when a drift is confirmed, the penalty forgets its old
> statistics. Otherwise we would raise the price correctly but push the model
> in the direction that was right *before* the drift."

### What to be careful not to claim

- **Not "guaranteed" or "provably".** The textbook guarantees for this kind of
  update assume a convex model; decision forests are not convex. Say
  *"it comes from constrained optimisation, which gives the motivation; we
  have no guarantee for trees yet."*
- **Not "unbiased" without the qualifier.** The group shares are estimated,
  so the estimate is only approximately unbiased.
- **Not "FADO is fairer" as a general result** until the sweeps have run.

### Likely questions and short answers

| question | answer |
|---|---|
| "Isn't this just tuning λ online?" | "It replaces tuning λ with choosing a target ε, which is a requirement you can state and defend. λ then follows from the data." |
| "How do you choose ε?" | "Open question (slide 11): either per dataset on the tuning seeds, or as a stated requirement. It's in the sensitivity sweep." |
| "Why not just set λ very high?" | "Accuracy pays for it. The price only rises while the rule is broken and falls back when it holds, so it pays only as much as needed." |
| "Does λ come back down after the drift?" | "Yes, but slowly: at most ε per sample, times the step size. On a short stream it can end still raised. That's the price of not overreacting to noise." |
| "Is it stable? Can it oscillate?" | "The integrator removes steady-state error, and the cap bounds it. A formal stability argument for trees is still open (slide 11)." |
| "Why reset the statistics?" | "They were averages over the whole stream. After a drift they describe the old data, so the penalty would push in the wrong direction." |
| "What does the baseline do?" | "The same penalty with λ fixed at the value where FADO's price starts. FADO can only add pressure, never less." |

---

## 08 · First results

**Message:** turning the price on roughly halves the unfairness after the
drift, and accuracy doesn't move.

> "Same model, same data, same seeds. The only difference between the first
> and third rows is whether λ is fixed or priced. DP goes from 0.094 to 0.058,
> and post-drift from 0.117 to 0.066. Accuracy is identical."

Then the three points on the right, in order:
1. *Direction:* "It goes the same way on all three seeds: a 54%, 18% and 40%
   drop."
2. *Cost:* "About one prediction in eight changes, and half of those become
   right while half become wrong. The controller moves the borderline cases,
   and on COMPAS, at 61% accuracy, there are plenty of them."
3. *Sanity:* "It's not cheating by predicting 'no' for everyone: the positive
   rate only goes from 0.40 to 0.37."

Point at row two (**no reset**): "Without the reset, the price climbs even
higher but helps less. That's the forgetting doing its job: without it, a
higher price pushes in the old, pre-drift direction."

Close with the footer, in your own words:
> "Two honest caveats. Even a perfectly fair classifier would show about 0.026
> on this windowed measure, just from noise, so FADO is closer to the floor
> than the raw numbers suggest. And this is three seeds on one scenario."

## 09 · How it happened

Walk the plot left to right; this is the slide that makes the idea click.

1. *Left of the drift line:* "Before the drift, all three curves overlap. The
   price creeps up a little, but nothing changes."
2. *At the drift line:* "The drift opens a gap. With λ fixed, both arms jump
   to 0.15–0.20 and stay there." Point at the green and orange lines.
3. *Bottom panel:* "With the controller, the price starts climbing exactly
   when DP goes above the dashed target line, and it keeps climbing until DP
   comes back down. It doubles, to about 2." Point at the dotted reset lines:
   "Those are the resets after each detected drift."
4. *Right side, recovery:* "In recovery the price drains slowly and DP drifts
   back up at the very end. The controller reacts with a lag; that's one of
   the things the sweeps will tune."

If asked why the price drains in a straight line: "Most predictions are
'no', and those contribute a fixed small decrease each step. In this version
the two groups' constraints are also counted twice, which doubles the step.
I'll fix that before the sweeps."

## 10 · Ablation plan

> "Everything is ready to run. The two ablations that matter most for the
> novelty claim are FADO *without* the λ controller — that's what we had
> before — and the λ controller *alone*."

## 11 · Open questions

Ask for input here; these are real decisions, not rhetorical questions. The
one to spend time on is **the claim**: with a fixed target, the natural claim
is "after a drift, FADO spends fewer samples above the target than the
baseline."

## 12 · Next steps

Read the list. If asked for timing: the COMPAS re-run and the sweeps come
first, because every other result depends on them.
