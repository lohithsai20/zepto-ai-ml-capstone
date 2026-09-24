from pathlib import Path
import pandas as pd, numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
PLOTS=ROOT/'plots'; PLOTS.mkdir(exist_ok=True)
CSV=ROOT/'titanic.csv'

# The only network/cache load in this module.
if not CSV.exists():
    df=sns.load_dataset('titanic')
    df.to_csv(CSV,index=False)
else:
    df=pd.read_csv(CSV)

print('=== INFO ==='); print(df.info())
print('=== DESCRIBE ==='); print(df.describe(include='all').T)
print('=== SHAPE ===',df.shape)
missing=(df.isna().mean()*100).sort_values(ascending=False)
print('=== MISSING % ==='); print(missing[missing>0])

# Required threshold rule.
clean=df.copy()
missing_pct=clean.isna().mean()*100
# Under 5%: drop rows. 5-30%: impute. >30%: drop column.
for col,pct in missing_pct.items():
    if pct==0: continue
    if pct < 5:
        clean=clean.dropna(subset=[col])
    elif pct <= 30:
        if pd.api.types.is_numeric_dtype(clean[col]): clean[col]=clean[col].fillna(clean[col].median())
        else: clean[col]=clean[col].fillna(clean[col].mode(dropna=True).iloc[0])
    else:
        clean=clean.drop(columns=[col])
print('=== CLEANED SHAPE ===',clean.shape)
print('Cleaning decisions:', {c:(round(float(missing_pct[c]),2), 'drop rows' if missing_pct[c]<5 else 'impute' if missing_pct[c]<=30 else 'drop column') for c in missing_pct[missing_pct>0].index})

# Univariate age/fare.
for col in ['age','fare']:
    fig,ax=plt.subplots(figsize=(7,4)); sns.histplot(clean[col],kde=True,ax=ax); ax.set_title(f'{col.title()} distribution'); fig.tight_layout(); fig.savefig(PLOTS/f'{col}_hist.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3)); sns.boxplot(x=clean[col],ax=ax); ax.set_title(f'{col.title()} box plot'); fig.tight_layout(); fig.savefig(PLOTS/f'{col}_box.png',dpi=150); plt.close(fig)

def iqr_outliers(s):
    q1,q3=s.quantile([.25,.75]); iqr=q3-q1; return int(((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum()),q1,q3,iqr
for c in ['age','fare']:
    print(f'{c} IQR:', iqr_outliers(clean[c]))
fare_mode=clean['fare'].mode().iloc[0]
print('Fare mean/median/mode:',clean['fare'].mean(),clean['fare'].median(),fare_mode)
print('Fare skewness:', 'right-skewed' if clean['fare'].mean()>clean['fare'].median()>fare_mode else 'left-skewed' if clean['fare'].mean()<clean['fare'].median()<fare_mode else 'not strictly determined by ordering')

# Bivariate masking.
print('=== SURVIVAL BY SEX ==='); print(clean.groupby('sex')['survived'].mean())
print('=== SURVIVAL BY PCLASS ==='); print(clean.groupby('pclass')['survived'].mean())
print('=== SURVIVAL BY SEX + PCLASS ==='); print(clean.groupby(['sex','pclass'])['survived'].mean())

corr_cols=['survived','pclass','age','sibsp','parch','fare']
corr=clean[corr_cols].corr()
print('=== CORRELATION ==='); print(corr)
pairs=[]
for i in range(len(corr_cols)):
    for j in range(i+1,len(corr_cols)):
        pairs.append((corr_cols[i],corr_cols[j],corr.iloc[i,j],abs(corr.iloc[i,j])))
print('Top 2 absolute correlations:')
for p in sorted(pairs,key=lambda x:x[3],reverse=True)[:2]: print(p)
fig,ax=plt.subplots(figsize=(7,6)); sns.heatmap(corr,annot=True,cmap='vlag',center=0,ax=ax); ax.set_title('Six-column correlation matrix'); fig.tight_layout(); fig.savefig(PLOTS/'correlation_heatmap.png',dpi=150); plt.close(fig)

# Four+ multivariate charts.
charts=[]
fig,ax=plt.subplots(figsize=(7,4)); sns.barplot(data=clean,x='sex',y='survived',hue='pclass',errorbar=None,ax=ax); ax.set_title('Survival rate by sex and passenger class'); fig.tight_layout(); fig.savefig(PLOTS/'survival_sex_pclass.png',dpi=150); plt.close(fig); charts.append('Women show higher survival rates than men across classes; class also separates outcomes, with first-class passengers generally showing higher survival than lower classes. The grouped view makes the interaction visible rather than attributing survival to sex alone.')
fig,ax=plt.subplots(figsize=(7,4)); sns.boxplot(data=clean,x='survived',y='fare',hue='sex',ax=ax); ax.set_title('Fare by survival and sex'); fig.tight_layout(); fig.savefig(PLOTS/'fare_survival_sex.png',dpi=150); plt.close(fig); charts.append('Survivors tend to have higher fares, reflecting the strong association between fare and passenger class. The overlap shows fare is informative but not sufficient by itself to determine survival.')
fig,ax=plt.subplots(figsize=(7,4)); sns.scatterplot(data=clean,x='age',y='fare',hue='survived',alpha=.65,ax=ax); ax.set_title('Age vs fare colored by survival'); fig.tight_layout(); fig.savefig(PLOTS/'age_fare_survival.png',dpi=150); plt.close(fig); charts.append('Survival is distributed across ages, while fare varies substantially and contains high-value observations. The color separation is imperfect, supporting a multivariate rather than single-feature explanation.')
fig,ax=plt.subplots(figsize=(7,4)); sns.pointplot(data=clean,x='pclass',y='survived',hue='sex',errorbar=None,ax=ax); ax.set_title('Survival interaction: class × sex'); fig.tight_layout(); fig.savefig(PLOTS/'interaction_plot.png',dpi=150); plt.close(fig); charts.append('The interaction plot reinforces that sex and class jointly describe survival patterns. Differences between classes remain visible within each sex category, indicating that both variables contribute to the data story.')

# Exploratory z-score check on full cleaned data; not used by modeling.
z=clean[['age','fare']].copy(); before=z.agg(['mean','std']); z=(z-z.mean())/z.std(ddof=1); after=z.agg(['mean','std'])
print('=== STANDARDIZATION BEFORE ==='); print(before)
print('=== STANDARDIZATION AFTER ==='); print(after)
clean.to_csv(ROOT/'cleaned_titanic.csv',index=False)

README=ROOT/'EDA_RESULTS.md'
README.write_text(f'''# EDA Results\n\nMissing-value percentages before cleaning:\n\n{missing[missing>0].to_string()}\n\nThreshold decisions: columns below 5% missing had affected rows dropped; columns from 5% through 30% were imputed; columns above 30% were dropped. In this dataset `deck` is above 30% and is dropped. `age` is imputed with its median. Low-missing `embarked`/other duplicate low-missing fields are handled by row deletion; the modeling pipeline separately handles missing values on its training split.\n\nIQR outliers and fare distribution statistics are printed by this script. The fare mean/median/mode ordering is used to state the skewness conclusion.\n\n## Chart interpretations\n1. {charts[0]}\n2. {charts[1]}\n3. {charts[2]}\n4. {charts[3]}\n\nThe exact two strongest correlations are calculated by ranking all off-diagonal pairs by absolute correlation and printed above. The correlation matrix contains exactly survived, pclass, age, sibsp, parch and fare; `adult_male` and `alone` are excluded.\n\nThe age/fare z-score sanity check prints pre/post means and standard deviations; the standardized values are not passed into the modeling pipeline.\n''')
