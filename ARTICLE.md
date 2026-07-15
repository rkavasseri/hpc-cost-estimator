# Estimating GPU Compute Costs Shouldn't Require Guesswork: A Small Tool for a Common Research Problem

Every ML research group eventually hits the same wall: someone needs to write down, in dollars, what a training pipeline is going to cost. A PI budgeting a grant or preparing a submission to a conference. Or, a student requesting a compute allocation. A department deciding whether to buy time on a commercial cluster, or run it inhouse, or run on desktop GPUs. The number has to go on a form, and it has to be defensible.

What I found, trying to do this myself (for a machine learning paper) is that the tooling for this specific problem barely exists. There are excellent total-cost-of-ownership calculators for procurement teams deciding whether to build a data center. There are generic GPU-hours calculators built around commercial cloud billing. What's missing is something built around how academic cluster (high performance computing - HPC) compute actually works with teh concept of a Service Unit (SU) rate, a multi-stage training pipeline, and a grant budget that needs a contingency line a reviewer can believe.

So, here's my first attempt in building one. This post walks through what it does, the assumptions baked into it, and some ideas for further development.

---

## Picking One Front End, Not Two

My first instinct was to build this twice: a CLI for anyone comfortable in a terminal, and a spreadsheet with the same logic in live formulas for anyone who isn't. I actually tried both, and cross-validated them against each other until they produced identical numbers on the same input, down to the cent, against a costing document I'd built by hand earlier.

Then I cut the spreadsheet. The people who run ML workflows aren't the spreadsheet types anyways. And, woudn't it be nice to integrate the cost estimator while setting up teh repo itself? Maybe that can be used as a metric in developing leaner and more efficient runs, I thought.

The people who actually use this, machine learning engineers, graduate students and postdocs scoping their own pipelines are already comfortable at a command line. The spreadsheet was solving a problem I didn't have: it added a second codebase to keep in sync, a second place for the math to drift, and an entire formula-validation workflow, for an audience that wouldn't understand hot to use it. A single well-documented CLI, with plain JSON config files anyone can read or diff, was the better fit for who's actually going to use this. Building the spreadsheet wasn't wasted effort, though — it's how I found the bug below, which is worth describing regardless of which interface it happened in.

---

## How a Phase Gets Costed

The tool thinks of a research pipeline as a sequence of **phases** — pretraining, fine-tuning, ablations, evaluation, whatever your project actually has, and each phase can be costed three different ways, tried in this priority order:

**1. You already know the GPU-hours.** Maybe from a prior run, maybe from a paper. Just enter it directly. This is always the most trustworthy number when you have it.

**2. You know the wall-clock time and how many GPUs it'll use.** Wall-clock hours × GPU count = GPU-hours. Simple, and it's how most people actually plan — "this'll run for about five days on an 8-GPU node."

**3. You only know the model size and dataset size.** This is where it gets more interesting, and more uncertain. This needs more work as I thought about the researchers working on grant applications and research projects where there is a lot of experimentation.

## The FLOPs-Based Estimator

For the third case, the tool falls back to the standard compute approximation used throughout the scaling-laws literature: training FLOPs ≈ 6 × parameters × training tokens. This comes from counting the forward pass (roughly 2 FLOPs per parameter per token) plus the backward pass (roughly double the forward cost).

Given a target GPU's peak throughput and an assumed **Model FLOPs Utilization (MFU)** which is the fraction of theoretical peak compute you actually achieve in practice, which is always well under 100% due to data loading, communication overhead, and imperfect kernels, you can convert total FLOPs into GPU-hours.

The honest part of this: MFU is the single biggest source of error in any estimate built this way, and it varies a lot. Industry benchmarks for well-optimized large-scale training runs often land somewhere in the 35–45% range. Smaller-scale or less-optimized research code frequently does worse. The tool defaults to 30% as a reasonably conservative middle ground, but it's a per-phase override, not a fixed constant and it should be treated as the first thing you calibrate against your own team's real throughput once you have some.

To sanity-check the formula itself (not just pick a plausible-sounding default), I calibrated it against a published reference point: training a 1.3B-parameter model on 100B tokens is widely cited as costing roughly 1,000 H100 GPU-hours. Running that through the tool's formula at 22% MFU reproduces 995.8 hours which is close enough to trust the underlying math, while also showing that real-world MFU for that kind of run sits well below the optimistic end of the range.

## Grant-Budget Contingency, Modeled Explicitly

A raw compute estimate is not what goes on a grant application. Research compute budgets get eaten by things that are entirely predictable in aggregate even if you can't predict which specific thing will happen: training runs that crash and need restarting, exploratory runs that don't make the final paper, and, if you're aiming at a competitive venue, be prepared for additional experiments a reviewer asks for during revision.

Rather than fold this into a single vague "add 30% for safety" multiplier, the tool breaks it into three named, independently adjustable layers:

| Contingency layer | Default | Rationale |
|---|---|---|
| Training restarts & instability | 10% | Standard allowance for interrupted or diverging runs |
| Exploratory runs & debugging | 15% | Typical overhead in early-stage research |
| Peer review / revision experiments | 10% | Reviewer-requested ablations before camera-ready |

Naming each layer explicitly, rather than hiding it in one multiplier, makes the final number easier to defend to a committee — and easier to trim if a reviewer pushes back on the total.

## What It Costs Elsewhere

Institutional SU rates are heavily subsidized relative to the commercial market, and that gap is worth making visible rather than assuming everyone already knows it. The tool includes a comparison against two rough commercial tiers: specialized GPU clouds (the RunPod/Lambda/CoreWeave category) and hyperscaler on-demand pricing (AWS/GCP/Azure), so a PI can see, concretely, what the institutional allocation is actually worth. In practice this tends to land somewhere around 3x for specialized clouds and 5-6x for hyperscalers, though this drifts monthly and shouldn't be trusted as a live quote.



## What the Tool Deliberately Doesn't Do

It's worth being upfront about the limits, because a costing tool that hides its assumptions is more dangerous than no tool at all.

- **The vision-model FLOPs estimate is a rough heuristic**, not a validated formula the way the language-model one is. Prefer direct GPU-hours or wall-clock times for vision phases whenever you have them.
- **Commercial cloud rates are ballpark figures from a fast-moving market.** They're useful for showing the *shape* of the institutional-vs-commercial gap, not for a real vendor quote.
- **It doesn't model multi-node communication overhead, checkpoint I/O, or data loading bottlenecks** beyond what's folded into the MFU assumption. A pipeline with heavy inter-node communication will run slower, in practice, than the naive GPU-hours number suggests.
- **It's a planning tool, not a monitoring tool.** It estimates before you run; it doesn't track actual spend once you're underway. That's a different, harder problem.

## Where This Leaves Things

None of this is groundbreaking research, it's a few hundred lines of code encoding a handful of standard formulas and some honest bookkeeping about where those formulas break down. But I think that's exactly the kind of thing worth sharing: the gap between "commercial GPU pricing" and "grant budget line item" is small and unglamorous enough that nobody's built a good general tool for it, and every lab ends up reinventing a worse version privately.

If you're in a similar position, writing a compute budget, staring down a training pipeline, trying to translate parameter counts into a number a grants office will accept, feel free to reach out. Happy to share the tool, walk through the assumptions, or hear what's wrong with them.
