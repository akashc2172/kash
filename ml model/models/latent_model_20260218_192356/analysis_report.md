# Latent Space Model Analysis Report

**Generated**: 2026-02-18 19:23
**Model**: latent_model_20260218_192356

## Training Results

- **Best Epoch**: 2
- **Best Val Loss**: 4.2611
- **Test RMSE**: 1.592339038848877
- **Test Correlation**: 0.03216311131018578

## Discovered Archetypes

### Low Dev Quality Weight Type (Archetype 0)

- **Players**: 79
- **Avg RAPM**: 0.47
- **Survival Rate**: 82.3%
- **Description**: Characterized by low Dev Quality Weight
- **Examples**: P0, P3, P5

### High Dev Quality Weight Type (Archetype 1)

- **Players**: 135
- **Avg RAPM**: 0.16
- **Survival Rate**: 77.0%
- **Description**: Characterized by high Dev Quality Weight
- **Examples**: P70, P71, P72

### High Three Share Type (Archetype 2)

- **Players**: 63
- **Avg RAPM**: -0.17
- **Survival Rate**: 63.5%
- **Description**: Characterized by high Three Share
- **Examples**: P63, P85, P101

### High Career Wt Trueshootingpct Type (Archetype 3)

- **Players**: 2
- **Avg RAPM**: -2.15
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Career Wt Trueshootingpct
- **Examples**: P6, P411

### High Career Years Type (Archetype 4)

- **Players**: 53
- **Avg RAPM**: 0.23
- **Survival Rate**: 64.2%
- **Description**: Characterized by high Career Years
- **Examples**: P1, P27, P128

### High Career Years Type (Archetype 5)

- **Players**: 112
- **Avg RAPM**: 0.38
- **Survival Rate**: 67.9%
- **Description**: Characterized by high Career Years
- **Examples**: P122, P136, P139

### Low Dev Quality Weight Type (Archetype 6)

- **Players**: 87
- **Avg RAPM**: 0.62
- **Survival Rate**: 69.0%
- **Description**: Characterized by low Dev Quality Weight
- **Examples**: P4, P8, P9

### Low Career Wt Trueshootingpct Type (Archetype 7)

- **Players**: 7
- **Avg RAPM**: -0.42
- **Survival Rate**: 71.4%
- **Description**: Characterized by low Career Wt Trueshootingpct
- **Examples**: P2, P7, P244

## Usage

```python
from models import ProspectModel, ArchetypeAnalyzer

# Load model
model = ProspectModel(latent_dim=32, n_archetypes=8)
model.load_state_dict(torch.load('model.pt'))

# Analyze a prospect
analysis = analyzer.analyze_player(tier1, tier2, career, tier2_mask)
print(analysis.narrative)
```
