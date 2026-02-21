# Latent Space Model Analysis Report

**Generated**: 2026-02-18 19:57
**Model**: latent_model_20260218_195654

## Training Results

- **Best Epoch**: 29
- **Best Val Loss**: 2.3945
- **Test RMSE**: 1.617095708847046
- **Test Correlation**: -0.09280275525979977

## Discovered Archetypes

### High Three Share Z Type (Archetype 0)

- **Players**: 122
- **Avg RAPM**: 0.04
- **Survival Rate**: 65.6%
- **Description**: Characterized by high Three Share Z
- **Examples**: E'Twaun Moore, MarShon Brooks, Mychel Thompson

### Low Ft Pct Type (Archetype 1)

- **Players**: 8
- **Avg RAPM**: 0.47
- **Survival Rate**: 87.5%
- **Description**: Characterized by low Ft Pct
- **Examples**: Kendall Marshall, Mason Plumlee, Victor Oladipo

### High Mid Share Type (Archetype 2)

- **Players**: 1
- **Avg RAPM**: 1.75
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Mid Share
- **Examples**: Cameron Johnson

### Low Three Share Z Type (Archetype 3)

- **Players**: 100
- **Avg RAPM**: 0.16
- **Survival Rate**: 66.0%
- **Description**: Characterized by low Three Share Z
- **Examples**: Jordan Williams, Jimmy Butler III, Kyrie Irving

### Low Usage Team Resid Type (Archetype 4)

- **Players**: 5
- **Avg RAPM**: 0.08
- **Survival Rate**: 80.0%
- **Description**: Characterized by low Usage Team Resid
- **Examples**: Festus Ezeli, Doron Lamb, Ryan Kelly

### Low Team Pace Type (Archetype 5)

- **Players**: 13
- **Avg RAPM**: 1.11
- **Survival Rate**: 76.9%
- **Description**: Characterized by low Team Pace
- **Examples**: Fab Melo, Terrence Jones, Darius Johnson-Odom

### High Shots Total Type (Archetype 6)

- **Players**: 199
- **Avg RAPM**: 0.42
- **Survival Rate**: 75.4%
- **Description**: Characterized by high Shots Total
- **Examples**: Ivan Johnson, JaJuan Johnson, Dennis Horner

### Low Fga Total Type (Archetype 7)

- **Players**: 31
- **Avg RAPM**: 0.25
- **Survival Rate**: 71.0%
- **Description**: Characterized by low Fga Total
- **Examples**: Jordan Hamilton, Derrick Williams, Michael Kidd-Gilchrist

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
