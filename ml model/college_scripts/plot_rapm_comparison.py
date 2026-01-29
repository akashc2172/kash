import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
HML_2018_PATH = '/Users/akashc/my-trankcopy/my-trank/data/raw/players_2018_HML.csv'
HISTORICAL_2015_PATH = 'data/historical_rapm_results_lambda1000.csv'
OUTPUT_PLOT = 'data/rapm_distribution_comparison.png'

def main():
    print("Loading datasets...")
    
    # Load 2018 HML
    df_2018 = pd.read_csv(HML_2018_PATH)
    # Total RApM = Off + Def (Def is usually negative in this schema to represent 'points allowed', but let's check)
    # Looking at the sample: CaJohnson off=8.75, def=-4.39. 
    # Usually RApM = Off - Def if Def is 'points allowed', or Off + Def if Def is 'impact on defense'.
    # In my-trank schema, usually it's Off + Def where Def impact is signed.
    df_2018['total_rapm'] = df_2018['off_adj_rapm.value'] + df_2018['def_adj_rapm.value']
    
    # Load 2015 Historical
    df_2015 = pd.read_csv(HISTORICAL_2015_PATH)
    df_2015 = df_2015[df_2015['season'] == 2015]
    
    # Prepare data for plotting
    data_2018 = pd.DataFrame({'RApM': df_2018['total_rapm'], 'Source': '2018 HML (Baseline)'})
    data_2015 = pd.DataFrame({'RApM': df_2015['rapm'], 'Source': '2015 Historical (Lambda=1000)'})
    
    combined = pd.concat([data_2018, data_2015])
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Source', y='RApM', data=combined, inner='quartile', palette='muted')
    plt.title('Distribution Comparison: Historical RApM vs. HML Baseline')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add summary stats box
    stats_text = (
        f"2018 HML: Mean={df_2018['total_rapm'].mean():.2f}, Std={df_2018['total_rapm'].std():.2f}\n"
        f"2015 Hist: Mean={df_2015['rapm'].mean():.2f}, Std={df_2015['rapm'].std():.2f}"
    )
    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"Plot saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()
