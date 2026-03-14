#RMSE calculation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from main import find_sim_history, OLD_pred_ret, DEFAULT_TICKER
import z_util as zu

def calculate_single_error(returns_df):
    """Pick a random date/time and calculate prediction error."""
    random_row = returns_df.sample(1).iloc[0]
    date = random_row['date']
    time = random_row['time']
    actual_ret = random_row['ret_to_close']
    
    current_day = returns_df[(returns_df['date'] == date) & (returns_df['time'] == time)].copy()
    similar_history, _ = find_sim_history(current_day, DEFAULT_TICKER)
    avg_ret, _, _ = OLD_pred_ret(similar_history)
    
    return (actual_ret - avg_ret) ** 2

def RMSE_prob_dist(n_outer=30, n_inner=1):
    """Calculate RMSE distribution over multiple trials."""
    returns_df = zu.load_parquet('returns')
    rmse_list = []
    
    for i in range(n_outer):
        squared_errors = []
        for j in range(n_inner):
            sq_error = calculate_single_error(returns_df)
            if not np.isnan(sq_error):
                squared_errors.append(sq_error)
        if squared_errors:
            rmse = np.sqrt(np.mean(squared_errors))
            rmse_list.append(rmse)
            print(f"Trial {i+1}/{n_outer}: RMSE = {rmse:.6f}")
        else:
            print(f"Trial {i+1}/{n_outer}: RMSE = skipped (no valid errors)")
    
    rmse_arr = np.array(rmse_list)
    rmse_arr = rmse_arr[~np.isnan(rmse_arr)]  # Filter out any NaN
    return rmse_arr, np.mean(rmse_arr), np.std(rmse_arr)

def RMSE_prob_dist_plot():
    """Calculate and plot RMSE distribution."""
    rmse_list, avg_RMSE, std_RMSE = RMSE_prob_dist()
    bin_num = 25
    
    print(f"\n{'='*50}")
    print(f"Average RMSE: {avg_RMSE:.6f}")
    print(f"Std Dev RMSE: {std_RMSE:.6f}")
    print(f"95% CI: [{avg_RMSE - 2*std_RMSE:.6f}, {avg_RMSE + 2*std_RMSE:.6f}]")
    print(f"{'='*50}\n")
    
    if len(rmse_list) == 0:
        print("ERROR: No valid data to plot!")
        return
    
    # Plot 1: Histogram only (raw frequency)
    plt.figure(figsize=(10, 6))
    plt.hist(rmse_list, bins=bin_num, edgecolor='black', alpha=0.7)
    plt.axvline(x=avg_RMSE, color='red', linestyle='--', linewidth=2, label=f'Avg: {avg_RMSE:.4f}')
    median = np.median(rmse_list)
    q2_5 = np.percentile(rmse_list, 2.5)
    q97_5 = np.percentile(rmse_list, 97.5)
    plt.axvline(x=median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.4f}')
    plt.axvline(x=q2_5, color='black', linestyle=':', linewidth=2, label=f'2.5%: {q2_5:.4f}')
    plt.axvline(x=q97_5, color='black', linestyle=':', linewidth=2, label=f'97.5%: {q97_5:.4f}')
    plt.legend()
    plt.xlabel('RMSE')
    plt.ylabel('Frequency')
    plt.title(f'RMSE Distribution (n={len(rmse_list)})')
    plt.tight_layout()
    plt.savefig('RMSE_histogram.png', dpi=150)
    plt.show()
    
    # Plot 2: Histogram with KDE overlay
    plt.figure(figsize=(10, 6))
    plt.hist(rmse_list, bins=bin_num, edgecolor='black', alpha=0.7, density=True)
    sns.kdeplot(rmse_list, color='blue', linewidth=2, label='KDE')
    plt.axvline(x=avg_RMSE, color='red', linestyle='--', linewidth=2, label=f'Avg: {avg_RMSE:.4f}')
    median = np.median(rmse_list)
    q2_5 = np.percentile(rmse_list, 2.5)
    q97_5 = np.percentile(rmse_list, 97.5)
    plt.axvline(x=median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.4f}')
    plt.axvline(x=q2_5, color='black', linestyle=':', linewidth=2, label=f'2.5%: {q2_5:.4f}')
    plt.axvline(x=q97_5, color='black', linestyle=':', linewidth=2, label=f'97.5%: {q97_5:.4f}')
    plt.legend()
    plt.xlabel('RMSE')
    plt.ylabel('Frequency')
    plt.title(f'RMSE Distribution (n={len(rmse_list)})')
    plt.tight_layout()
    plt.savefig('RMSE_histogram_kde.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    RMSE_prob_dist_plot()
