import pandas as pd
import numpy as np
from scipy.stats import gmean
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

def gquant(df1):
    """
    This function takes a DataFrame (df1) as input, processes it, and ranks genes based on their stability 
    using multiple statistical metrics (Standard Deviation, Geometric Mean, Covariance, and Kernel Density Estimation).
    
    Parameters:
    df1 (pd.DataFrame): DataFrame where the first column is non-numerical (e.g., gene names) and the remaining columns are numerical gene expression data.
    
    Returns:
    pd.DataFrame: A DataFrame containing the ranked genes and their corresponding rankings.
    """

    def standard_deviation(input_data):
        """Calculate the standard deviation for each column of the input data."""
        return np.std(input_data, axis=0)

    def geometric_mean(input_data):
        """Calculate the geometric mean for each column of the input data."""
        return gmean(input_data, axis=0)

    def covariance(input_data):
        """Calculate the covariance for each gene column and return the mean of covariances."""
        cov_matrix = np.cov(input_data, rowvar=False)
        return np.mean(cov_matrix, axis=0)

    def kernel_density_function(column_data, bandwidth=1.0):
        """Estimate the kernel density for the column's mean value."""
        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(column_data.values.reshape(-1, 1))
        log_density = kde.score_samples([[column_data.mean()]])
        return np.exp(log_density)[0]

    def scale_std(input_data):
        """Scale the standard deviation values to the range [0, 1]."""
        input_data_np = np.array(input_data)
        scaler = MinMaxScaler()
        return scaler.fit_transform(input_data_np.reshape(-1, 1)).flatten()

    # Extract gene names and numerical data
    gene_names = df1.columns[1:]
    numerical_data = df1.iloc[:, 1:].astype(float)
    remaining_genes = list(gene_names)
    ranking_index = []
    ranking_value = 1

    while numerical_data.shape[1] > 1:
        print(f'Current shape of numerical_data: {numerical_data.shape}')

        # Calculate metrics for each column in the numerical data
        std_result = scale_std(standard_deviation(numerical_data))
        gm_result = geometric_mean(numerical_data)
        cov_result = covariance(numerical_data)
        kde_results = [kernel_density_function(numerical_data[column_name]) for column_name in numerical_data.columns]

        # Identify the top gene for each metric
        min_std_col = remaining_genes[np.argmin(std_result)]
        min_gm_col = remaining_genes[np.argmin(gm_result)]
        min_cov_col = remaining_genes[np.argmin(cov_result)]
        max_kde_col = remaining_genes[np.argmax(kde_results)]

        # Collect votes for each gene based on the metric results
        votes = [min_std_col, min_gm_col, min_cov_col, max_kde_col]
        votes_count = Counter(votes)
        max_vote_count = votes_count.most_common(1)[0][1]
        tied_genes = [gene for gene, count in votes_count.items() if count == max_vote_count]

        if len(tied_genes) > 1:  # If there is a tie in the votes
            for gene in tied_genes:
                if gene in numerical_data.columns:
                    ranking_index.append((gene, ranking_value))
                    numerical_data = numerical_data.drop(columns=gene)
                    remaining_genes.remove(gene)
            ranking_value += 1
        else:  # If there is a majority vote
            majority_vote_gene = tied_genes[0]
            if majority_vote_gene in numerical_data.columns:
                ranking_index.append((majority_vote_gene, ranking_value))
                numerical_data = numerical_data.drop(columns=majority_vote_gene)
                remaining_genes.remove(majority_vote_gene)
            ranking_value += 1

        print(f'Votes: {votes}, Majority Vote: {majority_vote_gene if len(tied_genes) == 1 else "Tie"}, Remaining genes: {len(remaining_genes)}')

    print('Final Ranking index:', ranking_index)
    
    ranking_df = pd.DataFrame(ranking_index, columns=['Gene', 'Rank'])
    return ranking_df
