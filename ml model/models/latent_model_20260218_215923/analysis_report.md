# Latent Space Model Analysis Report

**Generated**: 2026-02-18 21:59
**Model**: latent_model_20260218_215923

## Training Results

- **Best Epoch**: 54
- **Best Val Loss**: 1.5988
- **Test RMSE**: 1.5923079252243042
- **Test Correlation**: -0.04609996004419374

## Discovered Archetypes

### High Career Wt Usage Type (Archetype 0)

- **Players**: 70
- **Avg RAPM**: 0.35
- **Survival Rate**: 68.6%
- **Description**: Characterized by high Career Wt Usage
- **Examples**: Zach Collins, Luke Kornet, Dennis Smith Jr.

### High Blk Total Per100Poss Type (Archetype 1)

- **Players**: 4
- **Avg RAPM**: 0.81
- **Survival Rate**: 75.0%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Jeff Withey, Mason Plumlee, Ryan Kelly

### High Ast Total Per100Poss Type (Archetype 2)

- **Players**: 152
- **Avg RAPM**: -0.04
- **Survival Rate**: 73.7%
- **Description**: Characterized by high Ast Total Per100Poss
- **Examples**: Jaylen Morris, Josh Hart, Edmond Sumner

### Low Three Share Type (Archetype 3)

- **Players**: 44
- **Avg RAPM**: 0.81
- **Survival Rate**: 63.6%
- **Description**: Characterized by low Three Share
- **Examples**: Mangok Mathiang, Amile Jefferson, Bam Adebayo

### High Tov Total Per100Poss Type (Archetype 4)

- **Players**: 9
- **Avg RAPM**: 1.40
- **Survival Rate**: 55.6%
- **Description**: Characterized by high Tov Total Per100Poss
- **Examples**: Eric Moreland, Anthony Davis, Michael Kidd-Gilchrist

### High Transfer Pace Delta Mean Type (Archetype 5)

- **Players**: 140
- **Avg RAPM**: 0.20
- **Survival Rate**: 71.4%
- **Description**: Characterized by high Transfer Pace Delta Mean
- **Examples**: Erik McCree, PJ Dozier, Caleb Swanigan

### Low Recruiting Rating Type (Archetype 6)

- **Players**: 49
- **Avg RAPM**: 0.20
- **Survival Rate**: 75.5%
- **Description**: Characterized by low Recruiting Rating
- **Examples**: Tyler Cavanaugh, Charles Cooke, Derrick White

### High Blk Total Per100Poss Type (Archetype 7)

- **Players**: 15
- **Avg RAPM**: 1.10
- **Survival Rate**: 66.7%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Skal Labissiere, Cliff Alexander, Dakari Johnson

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
