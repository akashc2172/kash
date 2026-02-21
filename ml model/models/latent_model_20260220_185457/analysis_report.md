# Latent Space Model Analysis Report

**Generated**: 2026-02-20 18:55
**Model**: latent_model_20260220_185457

## Training Results

- **Best Epoch**: 23
- **Best Val Loss**: 3.7384
- **Test RAPM RMSE**: 1.6511383056640625
- **Test RAPM Correlation**: -0.01696163683527814
- **Test EPM RMSE**: 1.6767103672027588
- **Test EPM Correlation**: -1.952049869570864e-05

## Discovered Archetypes

### Low Three Share Type (Archetype 0)

- **Players**: 64
- **Avg RAPM**: 0.40
- **Survival Rate**: 60.9%
- **Description**: Characterized by low Three Share
- **Examples**: Marcus Georges-Hunt, Elfrid Payton, Sheldon Mac

### High Fga Total Type (Archetype 1)

- **Players**: 161
- **Avg RAPM**: 0.15
- **Survival Rate**: 70.2%
- **Description**: Characterized by high Fga Total
- **Examples**: Joe Young, Buddy Hield, Josh Hart

### High Blk Total Per100Poss Type (Archetype 2)

- **Players**: 5
- **Avg RAPM**: 1.16
- **Survival Rate**: 60.0%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Mason Plumlee, Jeff Withey, Fab Melo

### Low Usage Team Resid Type (Archetype 3)

- **Players**: 6
- **Avg RAPM**: 1.22
- **Survival Rate**: 66.7%
- **Description**: Characterized by low Usage Team Resid
- **Examples**: Festus Ezeli, Ryan Kelly, Ike Anigbogu

### High Ft Att Type (Archetype 4)

- **Players**: 111
- **Avg RAPM**: 0.38
- **Survival Rate**: 71.2%
- **Description**: Characterized by high Ft Att
- **Examples**: Shawn Long, Sindarius Thornwell, Tyrone Wallace

### Low Team Srs Type (Archetype 5)

- **Players**: 15
- **Avg RAPM**: -0.58
- **Survival Rate**: 66.7%
- **Description**: Characterized by low Team Srs
- **Examples**: Darius Johnson-Odom, DeQuan Jones, Reggie Bullock Jr.

### High Blk Total Per100Poss Type (Archetype 6)

- **Players**: 11
- **Avg RAPM**: 0.70
- **Survival Rate**: 63.6%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Quincy Acy, Rondae Hollis-Jefferson, Terrence Jones

### High Three Share Type (Archetype 7)

- **Players**: 123
- **Avg RAPM**: 0.12
- **Survival Rate**: 77.2%
- **Description**: Characterized by high Three Share
- **Examples**: Alec Peters, Kay Felder, Frank Mason III

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
