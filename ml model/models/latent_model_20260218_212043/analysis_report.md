# Latent Space Model Analysis Report

**Generated**: 2026-02-18 21:20
**Model**: latent_model_20260218_212043

## Training Results

- **Best Epoch**: 45
- **Best Val Loss**: 1.5659
- **Test RMSE**: 1.625452995300293
- **Test Correlation**: -0.022770996144527583

## Discovered Archetypes

### High Fga Total Type (Archetype 0)

- **Players**: 109
- **Avg RAPM**: 0.12
- **Survival Rate**: 74.3%
- **Description**: Characterized by high Fga Total
- **Examples**: Erik McCree, Caleb Swanigan, T.J. Leaf

### High Three Share Type (Archetype 1)

- **Players**: 113
- **Avg RAPM**: -0.16
- **Survival Rate**: 62.8%
- **Description**: Characterized by high Three Share
- **Examples**: PJ Dozier, Josh Hart, London Perrantes

### Low Fga Total Type (Archetype 2)

- **Players**: 40
- **Avg RAPM**: 1.15
- **Survival Rate**: 65.0%
- **Description**: Characterized by low Fga Total
- **Examples**: Amile Jefferson, Skal Labissiere, Alex Poythress

### High Blk Total Per100Poss Type (Archetype 3)

- **Players**: 2
- **Avg RAPM**: 1.15
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Jeff Withey, Mason Plumlee

### High Recruiting Rank Type (Archetype 4)

- **Players**: 33
- **Avg RAPM**: 0.41
- **Survival Rate**: 75.8%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: OG Anunoby, Briante Weber, Tarik Black

### Low Games Played Type (Archetype 5)

- **Players**: 28
- **Avg RAPM**: -0.23
- **Survival Rate**: 75.0%
- **Description**: Characterized by low Games Played
- **Examples**: Jaylen Morris, Travis Wear, Solomon Hill

### High Recruiting Rank Type (Archetype 6)

- **Players**: 69
- **Avg RAPM**: 0.24
- **Survival Rate**: 75.4%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Edmond Sumner, Luke Kornet, Jamel Artis

### High Ft Att Type (Archetype 7)

- **Players**: 89
- **Avg RAPM**: 0.66
- **Survival Rate**: 73.0%
- **Description**: Characterized by high Ft Att
- **Examples**: Mangok Mathiang, Zach Collins, Charles Cooke

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
