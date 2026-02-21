<!-- CANONICAL_HTML_MIRROR -->
# Markdown Mirror

This file is a mirror. Canonical visual artifact: `/Users/akashc/my-trankcopy/ml model/docs/diagrams/model_architecture_dashboard.html`

Summary: Generative model DAG mirrored to canonical HTML architecture dashboard.

Last mirror refresh: 2026-02-20 20:55:19


# Markdown Mirror

This file is a mirror. Canonical visual artifact: `/Users/akashc/my-trankcopy/ml model/docs/diagrams/model_architecture_dashboard.html`

Summary: Generative model DAG mirrored to canonical HTML architecture dashboard.

Last mirror refresh: 2026-02-20 20:52:26


# Markdown Mirror

This file is a mirror. Canonical visual artifact: `/Users/akashc/my-trankcopy/ml model/docs/diagrams/model_architecture_dashboard.html`

Summary: Generative model DAG mirrored to canonical HTML architecture dashboard.

Last mirror refresh: 2026-02-20 19:18:17


# Markdown Mirror

This file is a mirror. Canonical visual artifact: `/Users/akashc/my-trankcopy/ml model/docs/diagrams/model_architecture_dashboard.html`

Summary: Generative model DAG mirrored to canonical HTML architecture dashboard.

Last mirror refresh: 2026-02-20 11:52:26


# Markdown Mirror

This file is a mirror. Canonical visual artifact: `/Users/akashc/my-trankcopy/ml model/docs/diagrams/model_architecture_dashboard.html`

Summary: Generative model DAG mirrored to canonical HTML architecture dashboard.

Last mirror refresh: 2026-02-20 11:51:36


# Markdown Mirror

This file is a mirror. Canonical visual artifact: `/Users/akashc/my-trankcopy/ml model/docs/diagrams/model_architecture_dashboard.html`

Summary: Generative model DAG mirrored to canonical HTML architecture dashboard.

Last mirror refresh: 2026-02-20 11:50:14


# Markdown Mirror

This file is a mirror. Canonical visual artifact: `/Users/akashc/my-trankcopy/ml model/docs/diagrams/model_architecture_dashboard.html`

Summary: Generative model DAG mirrored to canonical HTML architecture dashboard.

Last mirror refresh: 2026-02-20 11:18:49


# Generative Prospect Model: Visual Guide

**For explaining to anyone**

---

## The Big Picture (One Sentence)

> We learn **hidden skills** from college stats, then figure out how those skills **combine** to produce NBA impact — and we can explain exactly which skills contributed how much.

---

## Simple Analogy

Think of it like a **cooking recipe**:

| Cooking | Our Model |
|---------|-----------|
| Ingredients (flour, eggs, sugar) | College stats (3PT%, rebounds, assists) |
| Flavor profiles (sweet, savory, umami) | **Latent traits** (rim pressure, passing vision, shooting gravity) |
| How flavors combine in a dish | **Trait interactions** (rim pressure + passing = lob threat) |
| Final dish rating | **NBA RAPM** (how good the player actually is) |

The model learns: "This player has lots of 'rim pressure' and 'passing' traits, and those two together create extra value beyond what each does alone."

---

## The DAG (Directed Acyclic Graph)

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                           COLLEGE DATA (What We See)                           ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          ║
║   │  Shooting   │  │  Passing    │  │  Rebounding │  │  Defense    │   ...    ║
║   │  Stats      │  │  Stats      │  │  Stats      │  │  Stats      │          ║
║   │             │  │             │  │             │  │             │          ║
║   │ • 3PT%      │  │ • AST%      │  │ • ORB%      │  │ • STL%      │          ║
║   │ • FT%       │  │ • AST/TO    │  │ • DRB%      │  │ • BLK%      │          ║
║   │ • TS%       │  │ • Touches   │  │ • Box-outs  │  │ • DBPM      │          ║
║   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          ║
║          │                │                │                │                  ║
╚══════════╪════════════════╪════════════════╪════════════════╪══════════════════╝
           │                │                │                │
           ▼                ▼                ▼                ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                     ENCODER: Stats → Hidden Traits                             ║
║                                                                                ║
║   The model learns to compress 60+ stats into ~8-15 meaningful "traits"        ║
║   (We start with 32 possible traits, but most shrink to zero = not needed)     ║
║                                                                                ║
║                         ┌─────────────────────────┐                            ║
║                         │    ARD SHRINKAGE        │                            ║
║                         │                         │                            ║
║                         │  "Which traits matter?" │                            ║
║                         │                         │                            ║
║                         │  Trait 1: ████████ 0.9  │  ← Important!              ║
║                         │  Trait 2: ██ 0.1        │  ← Shrinks away            ║
║                         │  Trait 3: ███████ 0.8   │  ← Important!              ║
║                         │  Trait 4: █ 0.05        │  ← Shrinks away            ║
║                         │  ...                    │                            ║
║                         └─────────────────────────┘                            ║
║                                                                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    LATENT TRAIT VECTOR z (The Hidden Skills)                   ║
║                                                                                ║
║   Each player becomes a point in "trait space":                                ║
║                                                                                ║
║   ┌────────────────────────────────────────────────────────────────────┐       ║
║   │  z = [ 1.4,  -0.2,  0.9,  0.1,  1.1,  -0.5,  0.8,  ... ]          │       ║
║   │        ───    ───   ───   ───   ───    ───   ───                  │       ║
║   │        Rim   Shot   Pass  Off-  Def.  Perim  Motor                │       ║
║   │        Press Create Visn  Ball  Disr  Cont                        │       ║
║   └────────────────────────────────────────────────────────────────────┘       ║
║                                                                                ║
║   Example: Cooper Flagg might be z = [+1.8 rim, +1.2 defense, +0.9 passing]    ║
║                                                                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                      │
                 ┌────────────────────┴────────────────────┐
                 │                                         │
                 ▼                                         ▼
╔════════════════════════════════╗    ╔════════════════════════════════════════╗
║     AUX HEAD: p(NBA stats | z) ║    ║      IMPACT HEAD: p(RAPM | z, h)       ║
║                                ║    ║                                        ║
║  "If you have these traits,    ║    ║  "How do traits COMBINE to create      ║
║   what NBA stats should you    ║    ║   actual winning impact?"              ║
║   produce?"                    ║    ║                                        ║
║                                ║    ║  ┌──────────────────────────────────┐  ║
║  Used for VALIDATION:          ║    ║  │  MAIN EFFECTS (each trait alone) │  ║
║  • High rim trait → high       ║    ║  │                                  │  ║
║    FG% at rim in NBA           ║    ║  │  RAPM = β₁·z₁ + β₂·z₂ + β₃·z₃   │  ║
║  • High passing trait → high   ║    ║  │         ─────   ─────   ─────    │  ║
║    AST% in NBA                 ║    ║  │         +0.5    +0.3    +0.4     │  ║
║                                ║    ║  └──────────────────────────────────┘  ║
║  If this doesn't hold, our     ║    ║                    +                    ║
║  traits are meaningless!       ║    ║  ┌──────────────────────────────────┐  ║
╚════════════════════════════════╝    ║  │  INTERACTIONS (traits together)  │  ║
                                      ║  │                                  │  ║
                                      ║  │  + ρ₁₂·z₁·z₂ + ρ₁₃·z₁·z₃ + ...  │  ║
                                      ║  │    ─────────   ─────────         │  ║
                                      ║  │      +0.2        +0.1            │  ║
                                      ║  │                                  │  ║
                                      ║  │  HORSESHOE PRIOR: Most ρ → 0     │  ║
                                      ║  │  Only REAL synergies survive!    │  ║
                                      ║  └──────────────────────────────────┘  ║
                                      ║                    +                    ║
                                      ║  ┌──────────────────────────────────┐  ║
                                      ║  │  RESIDUAL h (unmeasured value)   │  ║
                                      ║  │                                  │  ║
                                      ║  │  IQ, communication, leadership   │  ║
                                      ║  │  (Strongly shrunk toward 0)      │  ║
                                      ║  └──────────────────────────────────┘  ║
                                      ╚════════════════════════════════════════╝
                                                         │
                                                         ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         OUTPUT: Trait Decomposition                            ║
║                                                                                ║
║   ┌─────────────────────────────────────────────────────────────────────────┐  ║
║   │  COOPER FLAGG PROJECTION                                                │  ║
║   │  ═══════════════════════════════════════════════════════════════════════│  ║
║   │                                                                         │  ║
║   │  EFFECTIVE TRAITS (top 80% of impact):                                  │  ║
║   │    • Rim Pressure:      +1.8σ  ────────────────────▶  +0.72 RAPM       │  ║
║   │    • Defensive Disr:    +1.2σ  ──────────────▶        +0.41 RAPM       │  ║
║   │    • Passing Vision:    +0.9σ  ──────────▶            +0.28 RAPM       │  ║
║   │                                                                         │  ║
║   │  ACTIVE INTERACTIONS:                                                   │  ║
║   │    • Rim × Passing:     ρ=0.34 ────────▶              +0.19 RAPM       │  ║
║   │      (Lob threat / roll gravity)                                        │  ║
║   │    • Rim × Defense:     ρ=0.21 ──────▶                +0.11 RAPM       │  ║
║   │      (Two-way anchor)                                                   │  ║
║   │                                                                         │  ║
║   │  RESIDUAL (unmeasured):                               +0.05 RAPM       │  ║
║   │                                                       ───────────       │  ║
║   │  TOTAL PROJECTED PEAK RAPM:                           +1.76 ± 0.62     │  ║
║   └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## Key Concepts Explained Simply

### 1. Latent Traits (z)

**What**: Hidden skills we can't directly measure, but we can infer from stats.

**Analogy**: You can't directly measure "basketball IQ" but you can see it in assist/turnover ratio, shot selection, defensive positioning.

**How we find them**: The model learns which combinations of stats tend to go together and represent a coherent "skill."

### 2. ARD Shrinkage (Automatic Relevance Determination)

**What**: We start with 32 possible trait dimensions, but the model automatically figures out which ones actually matter.

**Analogy**: Like starting with 32 possible spices for a recipe, then realizing only 8-12 actually contribute to the flavor.

**How it works**: Each trait has a "scale" parameter. If the scale shrinks to near-zero, that trait doesn't matter.

### 3. Horseshoe Prior (For Interactions)

**What**: Most trait combinations DON'T have special synergy. The horseshoe prior assumes most interaction effects are zero, but allows a few to be large.

**Analogy**: Most ingredient pairs don't have special chemistry (flour + salt = just flour and salt). But some do (chocolate + peanut butter = magic).

**Why it matters**: With 32 traits, there are 496 possible pairs. Without the horseshoe, we'd overfit to noise. The horseshoe lets the data tell us which pairs actually matter.

### 4. The Forked Structure (Aux + Impact)

**Why two heads?**

- **Aux Head**: Sanity check. "If we say this player has high 'rim pressure' trait, they should actually score well at the rim in the NBA." If not, our traits are meaningless.

- **Impact Head**: The actual prediction. "Given these traits, how much does this player help their team win?"

**Why separate?** Because stats ≠ impact. A player might have great stats but negative impact (empty calories), or mediocre stats but positive impact (glue guy). The residual h captures the gap.

---

## What Makes This Different From Basic ML?

| Basic ML (XGBoost) | This Generative Model |
|--------------------|----------------------|
| "3PT% and AST% predict RAPM" | "There's a 'spacing gravity' trait that manifests as 3PT%, and it combines with 'playmaking' to create extra value" |
| Feature importance list | **Trait decomposition**: exactly how much each skill contributes |
| Black box | **Fully interpretable**: main effects + specific interactions |
| Point prediction | **Uncertainty quantified**: ±0.62 RAPM |
| No interaction discovery | **Learns which trait pairs have synergy** |

---

## Flow Summary

```
COLLEGE STATS (60+)
       │
       ▼
   ENCODER ──────────────────────┐
       │                         │
       ▼                         ▼
   TRAITS z (8-15 effective)   ARD tells us which matter
       │
       ├─────────────────────────────────────┐
       ▼                                     ▼
   AUX HEAD                             IMPACT HEAD
   (validates traits                    (predicts RAPM)
    make sense)                              │
                                             ├── Main effects (each trait)
                                             ├── Interactions (trait pairs)
                                             └── Residual (unmeasured)
                                                  │
                                                  ▼
                                          RAPM DECOMPOSITION
                                          "This trait gave +0.5,
                                           that interaction gave +0.2"
```

---

## Why This Matters for Scouting

Instead of: *"Our model says Cooper Flagg will have +1.8 RAPM"*

We can say: *"Cooper Flagg projects to +1.8 RAPM because:*
- *His rim pressure (+1.8σ) alone is worth +0.72 RAPM*
- *His defense (+1.2σ) adds +0.41 RAPM*
- *The combination of rim pressure AND passing creates an extra +0.19 RAPM (lob threat)*
- *Uncertainty: could be anywhere from +1.1 to +2.4 RAPM"*

This is **actionable**. You can see WHERE the value comes from and WHAT could go wrong.
