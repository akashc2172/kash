# Latent Space Model Analysis Report

**Generated**: 2026-02-18 19:03
**Model**: latent_model_20260218_190336

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 5.5604
- **Test RMSE**: 1.5855181217193604
- **Test Correlation**: 0.016497886708863648

## Discovered Archetypes

### Low Dev Quality Weight Type (Archetype 0)

- **Players**: 123
- **Avg RAPM**: 0.50
- **Survival Rate**: 80.5%
- **Description**: Characterized by low Dev Quality Weight
- **Examples**: P0, P3, P4

### Low Team Pace Type (Archetype 1)

- **Players**: 34
- **Avg RAPM**: 0.00
- **Survival Rate**: 73.5%
- **Description**: Characterized by low Team Pace
- **Examples**: P85, P131, P133

### High Career Wt Trueshootingpct Type (Archetype 2)

- **Players**: 2
- **Avg RAPM**: -2.15
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Career Wt Trueshootingpct
- **Examples**: P6, P411

### Low Career Wt Trueshootingpct Type (Archetype 3)

- **Players**: 6
- **Avg RAPM**: -0.16
- **Survival Rate**: 66.7%
- **Description**: Characterized by low Career Wt Trueshootingpct
- **Examples**: P2, P7, P299

### Low Team Pace Type (Archetype 4)

- **Players**: 13
- **Avg RAPM**: 0.33
- **Survival Rate**: 84.6%
- **Description**: Characterized by low Team Pace
- **Examples**: P1, P27, P135

### High Delta Trueshootingpct Type (Archetype 5)

- **Players**: 9
- **Avg RAPM**: 0.40
- **Survival Rate**: 77.8%
- **Description**: Characterized by high Delta Trueshootingpct
- **Examples**: P63, P140, P164

### High Dev Quality Weight Type (Archetype 6)

- **Players**: 175
- **Avg RAPM**: 0.29
- **Survival Rate**: 68.0%
- **Description**: Characterized by high Dev Quality Weight
- **Examples**: P70, P71, P72

### High Ft Att Type (Archetype 7)

- **Players**: 176
- **Avg RAPM**: 0.19
- **Survival Rate**: 67.6%
- **Description**: Characterized by high Ft Att
- **Examples**: P17, P19, P21

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
