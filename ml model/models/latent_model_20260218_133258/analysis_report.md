# Latent Space Model Analysis Report

**Generated**: 2026-02-18 13:32
**Model**: latent_model_20260218_133258

## Training Results

- **Best Epoch**: 0
- **Best Val Loss**: 4.6456
- **Test RMSE**: 1.5845723152160645
- **Test Correlation**: -0.02238281444715621

## Discovered Archetypes

### Low Ft Pct Type (Archetype 0)

- **Players**: 6
- **Avg RAPM**: 0.62
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Ft Pct
- **Examples**: P2, P219, P244

### High Career Years Type (Archetype 1)

- **Players**: 170
- **Avg RAPM**: 0.26
- **Survival Rate**: 71.2%
- **Description**: Characterized by high Career Years
- **Examples**: P70, P71, P72

### High Career Wt Trueshootingpct Type (Archetype 2)

- **Players**: 2
- **Avg RAPM**: -2.15
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Career Wt Trueshootingpct
- **Examples**: P6, P411

### Low Career Years Type (Archetype 3)

- **Players**: 122
- **Avg RAPM**: 0.44
- **Survival Rate**: 78.7%
- **Description**: Characterized by low Career Years
- **Examples**: P0, P3, P7

### Low Three Share Z Type (Archetype 4)

- **Players**: 111
- **Avg RAPM**: 0.12
- **Survival Rate**: 73.0%
- **Description**: Characterized by low Three Share Z
- **Examples**: P8, P74, P75

### Low Three Share Type (Archetype 5)

- **Players**: 43
- **Avg RAPM**: 0.89
- **Survival Rate**: 65.1%
- **Description**: Characterized by low Three Share
- **Examples**: P1, P4, P19

### Low Career Wt Trueshootingpct Type (Archetype 6)

- **Players**: 1
- **Avg RAPM**: 0.50
- **Survival Rate**: 100.0%
- **Description**: Characterized by low Career Wt Trueshootingpct
- **Examples**: P27

### High Three Share Type (Archetype 7)

- **Players**: 83
- **Avg RAPM**: -0.01
- **Survival Rate**: 61.4%
- **Description**: Characterized by high Three Share
- **Examples**: P5, P63, P76

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
