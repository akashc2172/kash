# Latent Space Model Analysis Report

**Generated**: 2026-02-20 21:43
**Model**: latent_model_20260220_214321

## Training Results

- **Best Epoch**: 7
- **Best Val Loss**: 5.3116
- **Test RAPM RMSE**: 1.9421546459197998
- **Test RAPM Correlation**: -0.08250267981344707
- **Test EPM RMSE**: 1.8192992210388184
- **Test EPM Correlation**: 0.060431274875546316

## Discovered Archetypes

### High Blk Total Per100Poss Type (Archetype 0)

- **Players**: 21
- **Avg RAPM**: 1.09
- **Survival Rate**: 66.7%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Elias Harris, Quincy Acy, Rondae Hollis-Jefferson

### High Recruiting Rating Type (Archetype 1)

- **Players**: 74
- **Avg RAPM**: 0.60
- **Survival Rate**: 75.7%
- **Description**: Characterized by high Recruiting Rating
- **Examples**: Alec Peters, Devyn Marble, DeAndre' Bembry

### Low Recruiting Rating Type (Archetype 2)

- **Players**: 145
- **Avg RAPM**: 0.47
- **Survival Rate**: 71.7%
- **Description**: Characterized by low Recruiting Rating
- **Examples**: Buddy Hield, Shawn Long, Sindarius Thornwell

### High Recruiting Rank Type (Archetype 3)

- **Players**: 63
- **Avg RAPM**: 0.51
- **Survival Rate**: 76.2%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Kay Felder, Antonius Cleveland, Mike Muscala

### High Rim Pressure Index Type (Archetype 4)

- **Players**: 77
- **Avg RAPM**: 0.98
- **Survival Rate**: 64.9%
- **Description**: Characterized by high Rim Pressure Index
- **Examples**: Charles Cooke, Marcus Georges-Hunt, Brice Johnson

### Low Team Srs Type (Archetype 5)

- **Players**: 10
- **Avg RAPM**: 0.39
- **Survival Rate**: 80.0%
- **Description**: Characterized by low Team Srs
- **Examples**: Reggie Bullock Jr., Kendall Marshall, Michael Carter-Williams

### High Blk Total Per100Poss Type (Archetype 6)

- **Players**: 4
- **Avg RAPM**: 0.88
- **Survival Rate**: 50.0%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Jeff Withey, Fab Melo, Anthony Davis

### High Three Share Type (Archetype 7)

- **Players**: 102
- **Avg RAPM**: 0.46
- **Survival Rate**: 66.7%
- **Description**: Characterized by high Three Share
- **Examples**: Joe Young, Frank Mason III, Josh Hart

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
