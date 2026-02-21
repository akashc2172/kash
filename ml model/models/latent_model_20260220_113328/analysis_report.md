# Latent Space Model Analysis Report

**Generated**: 2026-02-20 11:33
**Model**: latent_model_20260220_113328

## Training Results

- **Best Epoch**: 66
- **Best Val Loss**: 3.8258
- **Test RAPM RMSE**: 1.7031580209732056
- **Test RAPM Correlation**: 0.025034816702935198
- **Test EPM RMSE**: 1.6896851062774658
- **Test EPM Correlation**: -0.039239185337764125

## Discovered Archetypes

### High Fga Total Type (Archetype 0)

- **Players**: 140
- **Avg RAPM**: -0.15
- **Survival Rate**: 67.9%
- **Description**: Characterized by high Fga Total
- **Examples**: Erik McCree, Josh Hart, Edmond Sumner

### Low Three Share Type (Archetype 1)

- **Players**: 61
- **Avg RAPM**: 0.60
- **Survival Rate**: 62.3%
- **Description**: Characterized by low Three Share
- **Examples**: Mangok Mathiang, Ike Anigbogu, Amile Jefferson

### High Blk Total Per100Poss Type (Archetype 2)

- **Players**: 12
- **Avg RAPM**: 1.12
- **Survival Rate**: 66.7%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Anthony Davis, Michael Kidd-Gilchrist, Andre Drummond

### High Rim Pressure Index Type (Archetype 3)

- **Players**: 59
- **Avg RAPM**: 0.39
- **Survival Rate**: 69.5%
- **Description**: Characterized by high Rim Pressure Index
- **Examples**: Zach Collins, Luke Kornet, Charles Cooke

### Low Team Srs Type (Archetype 4)

- **Players**: 16
- **Avg RAPM**: -0.72
- **Survival Rate**: 68.8%
- **Description**: Characterized by low Team Srs
- **Examples**: Jaylen Morris, Darius Johnson-Odom, Shane Larkin

### High Blk Total Per100Poss Type (Archetype 5)

- **Players**: 20
- **Avg RAPM**: 1.12
- **Survival Rate**: 60.0%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Rondae Hollis-Jefferson, Cliff Alexander, Willie Cauley-Stein

### Low Ft Att Type (Archetype 6)

- **Players**: 85
- **Avg RAPM**: 0.16
- **Survival Rate**: 75.3%
- **Description**: Characterized by low Ft Att
- **Examples**: PJ Dozier, London Perrantes, Monte Morris

### High Ft Att Type (Archetype 7)

- **Players**: 103
- **Avg RAPM**: 0.42
- **Survival Rate**: 77.7%
- **Description**: Characterized by high Ft Att
- **Examples**: Donovan Mitchell, Caleb Swanigan, Sindarius Thornwell

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
