# EDA Results

Missing-value percentages before cleaning:

deck           77.216611
age            19.865320
embarked        0.224467
embark_town     0.224467

Threshold decisions: columns below 5% missing had affected rows dropped; columns from 5% through 30% were imputed; columns above 30% were dropped. In this dataset `deck` is above 30% and is dropped. `age` is imputed with its median. Low-missing `embarked`/other duplicate low-missing fields are handled by row deletion; the modeling pipeline separately handles missing values on its training split.

IQR outliers and fare distribution statistics are printed by this script. The fare mean/median/mode ordering is used to state the skewness conclusion.

## Chart interpretations
1. Women show higher survival rates than men across classes; class also separates outcomes, with first-class passengers generally showing higher survival than lower classes. The grouped view makes the interaction visible rather than attributing survival to sex alone.
2. Survivors tend to have higher fares, reflecting the strong association between fare and passenger class. The overlap shows fare is informative but not sufficient by itself to determine survival.
3. Survival is distributed across ages, while fare varies substantially and contains high-value observations. The color separation is imperfect, supporting a multivariate rather than single-feature explanation.
4. The interaction plot reinforces that sex and class jointly describe survival patterns. Differences between classes remain visible within each sex category, indicating that both variables contribute to the data story.

The exact two strongest correlations are calculated by ranking all off-diagonal pairs by absolute correlation and printed above. The correlation matrix contains exactly survived, pclass, age, sibsp, parch and fare; `adult_male` and `alone` are excluded.

The age/fare z-score sanity check prints pre/post means and standard deviations; the standardized values are not passed into the modeling pipeline.
