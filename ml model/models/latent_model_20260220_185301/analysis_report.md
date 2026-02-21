# Latent Space Model Analysis Report

**Generated**: 2026-02-20 18:53
**Model**: latent_model_20260220_185301

## Training Results

- **Best Epoch**: 23
- **Best Val Loss**: 3.6041
- **Test RAPM RMSE**: 1.6706531047821045
- **Test RAPM Correlation**: -0.006654996939425079
- **Test EPM RMSE**: 1.6632174253463745
- **Test EPM Correlation**: 0.048494335269348605

## Discovered Archetypes

### Low Recruiting Rank Type (Archetype 0)

- **Players**: 83
- **Avg RAPM**: 0.13
- **Survival Rate**: 71.1%
- **Description**: Characterized by low Recruiting Rank
- **Examples**: Justin Jackson, Georges Niang, Tyler Cavanaugh

### Low Three Share Type (Archetype 1)

- **Players**: 39
- **Avg RAPM**: 0.46
- **Survival Rate**: 61.5%
- **Description**: Characterized by low Three Share
- **Examples**: Marcus Georges-Hunt, Elfrid Payton, Sheldon Mac

### High Recruiting Rank Type (Archetype 2)

- **Players**: 164
- **Avg RAPM**: 0.10
- **Survival Rate**: 72.0%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Alec Peters, Kay Felder, Antonius Cleveland

### High Shots Total Type (Archetype 3)

- **Players**: 68
- **Avg RAPM**: 0.22
- **Survival Rate**: 79.4%
- **Description**: Characterized by high Shots Total
- **Examples**: Buddy Hield, Shawn Long, Sindarius Thornwell

### High Blk Total Per100Poss Type (Archetype 4)

- **Players**: 9
- **Avg RAPM**: 1.53
- **Survival Rate**: 77.8%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Terrence Jones, Mason Plumlee, Ryan Kelly

### High Ft Att Type (Archetype 5)

- **Players**: 45
- **Avg RAPM**: 0.72
- **Survival Rate**: 64.4%
- **Description**: Characterized by high Ft Att
- **Examples**: Charles Cooke, Brice Johnson, Larry Nance Jr.

### High Three Share Type (Archetype 6)

- **Players**: 78
- **Avg RAPM**: 0.01
- **Survival Rate**: 67.9%
- **Description**: Characterized by high Three Share
- **Examples**: Joe Young, Frank Mason III, Josh Hart

### High Blk Total Per100Poss Type (Archetype 7)

- **Players**: 10
- **Avg RAPM**: 1.21
- **Survival Rate**: 60.0%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Quincy Acy, Rondae Hollis-Jefferson, Festus Ezeli

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
