# Latent Space Model Analysis Report

**Generated**: 2026-02-18 20:53
**Model**: latent_model_20260218_205306

## Training Results

- **Best Epoch**: 22
- **Best Val Loss**: 1.6405
- **Test RMSE**: 1.856553554534912
- **Test Correlation**: 0.16173150378704945

## Discovered Archetypes

### High Transfer Max Shock Type (Archetype 0)

- **Players**: 47
- **Avg RAPM**: -0.17
- **Survival Rate**: 48.9%
- **Description**: Characterized by high Transfer Max Shock
- **Examples**: Kendall Marshall, Alex Len, Jordan Clarkson

### High Minutes Total Type (Archetype 1)

- **Players**: 93
- **Avg RAPM**: 0.37
- **Survival Rate**: 71.0%
- **Description**: Characterized by high Minutes Total
- **Examples**: Marquis Teague, Austin Rivers, Ben McLemore

### Low Recruiting Rating Type (Archetype 2)

- **Players**: 178
- **Avg RAPM**: 0.59
- **Survival Rate**: 68.0%
- **Description**: Characterized by low Recruiting Rating
- **Examples**: Jimmy Butler III, Kemba Walker, Cory Higgins

### High Shot Dist Var Type (Archetype 3)

- **Players**: 185
- **Avg RAPM**: -0.09
- **Survival Rate**: 65.4%
- **Description**: Characterized by high Shot Dist Var
- **Examples**: Meyers Leonard, Chris Babb, Jordan Adams

### Low Team Pace Type (Archetype 4)

- **Players**: 38
- **Avg RAPM**: 0.62
- **Survival Rate**: 73.7%
- **Description**: Characterized by low Team Pace
- **Examples**: Ivan Johnson, Brandon Knight, Josh Harrellson

### High Recruiting Rating Type (Archetype 5)

- **Players**: 143
- **Avg RAPM**: 0.09
- **Survival Rate**: 69.2%
- **Description**: Characterized by high Recruiting Rating
- **Examples**: JaJuan Johnson, MarShon Brooks, Jordan Williams

### Low Team Pace Type (Archetype 6)

- **Players**: 32
- **Avg RAPM**: -0.57
- **Survival Rate**: 75.0%
- **Description**: Characterized by low Team Pace
- **Examples**: Jordan Hamilton, Terrel Harris, Michael Kidd-Gilchrist

### Low Shot Dist Var Type (Archetype 7)

- **Players**: 101
- **Avg RAPM**: 0.13
- **Survival Rate**: 75.2%
- **Description**: Characterized by low Shot Dist Var
- **Examples**: E'Twaun Moore, Josh Selby, Norris Cole

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
