import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from sklearn.metrics import silhouette_score


file_path = 'main_dataset.xlsx'
df = pd.read_excel(file_path)

# Replacing blanks with NaNs for easy processing and handle 'Indigenous_pct' column
df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)  # Replace blanks with NaN

# Convert 'Indigenous_pct' to numeric and fill NaN with mean
if 'Indigenous_pct' in df.columns:
    df['Indigenous_pct'] = pd.to_numeric(df['Indigenous_pct'], errors='coerce')
    df['Indigenous_pct'].fillna(df['Indigenous_pct'].mean(), inplace=True)

# Extracting 'School_subtype' ends with 'Year 12's (Only considering High Schools)
if 'School_subtype' in df.columns:
    year_12_schools = df[df['School_subtype'].str.endswith('Year 12', na=False)]
else:
    year_12_schools = df  # Proceed with the entire dataset if 'School_subtype' doesn't exist

# Feature selection
relevant_columns = ['School_code', 'School_name', 'Postcode', 'School_subtype', 'ICSEA_value', 'LBOTE_pct', 
                    'Indigenous_pct', 'FOEI_Value', 'LGA', 'SA4', 'AECG_region', 
                    'Local_health_district', 'Latitude', 'Longitude']

selected_columns = [col for col in relevant_columns if col in year_12_schools.columns]
features = year_12_schools[selected_columns].dropna()

# Handling missing values in 'LBOTE_pct' and numeric columns
numeric_columns = ['ICSEA_value', 'LBOTE_pct', 'Indigenous_pct', 'FOEI_Value']

# Convert 'LBOTE_pct' to numeric, replacing 'np' with NaN, and fill NaN with the mean
features['LBOTE_pct'] = features['LBOTE_pct'].replace('np', pd.NA)
features['LBOTE_pct'] = pd.to_numeric(features['LBOTE_pct'], errors='coerce')
features['LBOTE_pct'].fillna(features['LBOTE_pct'].mean(), inplace=True)

# Standardizing the numeric features for K-Means
scaler = StandardScaler()
numeric_features_scaled = scaler.fit_transform(features[numeric_columns])

# Applying K-means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
features['Cluster'] = kmeans.fit_predict(numeric_features_scaled)

# Writing the clustered data to a CSV file
results = 'clustered_schools.csv'
features.to_csv(results, index=False)
print(f"Clustered data written to {results}")

web_map = "map_data.csv"
features[['School_name', 'Longitude', 'Latitude', 'Cluster']].to_csv(web_map, index=False);
print(f"Data with selected columns written to {web_map}")

# Removing NaNs in LBOTE_pct
print(f"NaNs in 'LBOTE_pct': {features['LBOTE_pct'].isna().sum()}")
print(f"Unique 'LBOTE_pct' values per cluster:\n{features.groupby('Cluster')['LBOTE_pct'].nunique()}")

# ANOVA analysis to find correlation
features_clean = features.dropna(subset=numeric_columns)

# Data Analysis
# Feature Importance
cluster_variability = features.groupby('Cluster')[numeric_columns].std()
print(cluster_variability)

# Silhouette_score
silhouette_avg = silhouette_score(numeric_features_scaled, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg}")

# ANOVA
for feature in numeric_columns:
    anova_result = stats.f_oneway(
        features_clean[features_clean['Cluster'] == 0][feature],
        features_clean[features_clean['Cluster'] == 1][feature],
        features_clean[features_clean['Cluster'] == 2][feature]
    )
    print(f"ANOVA result for {feature}: p-value = {anova_result.pvalue}")

# Visualize the clusters using 'ICSEA_value' and 'FOEI_Value'
plt.figure(figsize=(10, 6))
plt.scatter(features['ICSEA_value'], features['FOEI_Value'], 
            c=features['Cluster'], cmap='viridis', marker='o', 
            s=100, alpha=0.6, edgecolor='k')
plt.colorbar(label='Cluster')
plt.xlabel('ICSEA Value', fontsize=12)
plt.ylabel('FOEI Value', fontsize=12)
plt.title('School Clusters Based on ICSEA and FOEI Values', fontsize=14)
plt.grid(True)
plt.show()

sns.pairplot(features, hue='Cluster', vars=['ICSEA_value', 'LBOTE_pct', 'Indigenous_pct', 'FOEI_Value'])
plt.show()
