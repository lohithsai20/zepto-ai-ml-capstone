# EDA Results

## Dataset Overview

- Original shape: **891 rows × 15 columns**
- Cleaned shape: **889 rows × 14 columns**

The script prints df.info(), df.describe(include='all'), and the dataset shape during execution.

## Missing-Value Analysis

Missing-value percentages before cleaning:

| Column | Missing % | Decision |
|---|---:|---|
| deck | 77.2166% | Dropped because missingness is above 30% |
| age | 19.8653% | Median imputed because missingness is between 5% and 30% |
| embarked | 0.2245% | Affected rows dropped because missingness is below 5% |
| embark_town | 0.2245% | Affected rows dropped because missingness is below 5% |

The threshold used is:
- Below 5%: drop affected rows
- 5%–30%: impute
- Above 30%: drop or explicitly encode missingness with justification

deck is dropped because 77.2166% of its values are missing. age is median-imputed because 19.8653% is missing. Low-missing fields are handled through row deletion. The modeling pipeline separately performs preprocessing on the training data only.

## Outlier Analysis

IQR-based outlier counts:

- Age: **65**
- Fare: **114**

## Fare Distribution

- Mean: **32.0967**
- Median: **14.4542**
- Mode: **8.05**

Because **mean > median > mode**, the fare distribution is **right-skewed**.

## Survival Analysis

Survival rates are calculated separately by:

- Sex
- Passenger class (pclass)
- Sex + passenger class

The grouped analysis shows differences in survival across both sex and passenger class.

## Correlation Analysis

The required six-column correlation matrix contains:

survived, pclass, age, sibsp, parch, fare

The columns adult_male and alone are excluded.

The two strongest absolute off-diagonal correlations are:

1. **pclass vs fare: -0.5482**
   - The negative relationship indicates that higher passenger class numbers are associated with lower fares.

2. **sibsp vs parch: 0.4145**
   - This indicates a moderate positive relationship between the number of siblings/spouses and parents/children travelling with a passenger.

A heatmap of the six-column correlation matrix is saved as plots/correlation_heatmap.png.

## Chart Interpretations

1. Women show higher survival rates than men across classes; class also separates outcomes, with first-class passengers generally showing higher survival than lower classes. The grouped view makes the interaction visible rather than attributing survival to sex alone.

2. Survivors tend to have higher fares, reflecting the strong association between fare and passenger class. The overlap shows fare is informative but not sufficient by itself to determine survival.

3. Survival is distributed across ages, while fare varies substantially and contains high-value observations. The color separation is imperfect, supporting a multivariate rather than single-feature explanation.

4. The interaction plot reinforces that sex and class jointly describe survival patterns. Differences between classes remain visible within each sex category, indicating that both variables contribute to the data story.

## Standardization Check

Age and fare were standardized using z-scores as an exploratory analysis.

The script calculates the mean and standard deviation before and after standardization to verify the transformation.

The standardized values are **not used as inputs to the modeling pipeline**.

## Generated Outputs

The EDA process generates:

- Age histogram
- Age box plot
- Fare histogram
- Fare box plot
- Survival by sex and passenger class
- Fare by survival and sex
- Age vs fare colored by survival
- Sex/class interaction plot
- Six-column correlation heatmap
