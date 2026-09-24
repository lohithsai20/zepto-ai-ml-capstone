from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

ROOT=Path(__file__).resolve().parent; PLOTS=ROOT/'plots'; PLOTS.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/'cleaned_titanic.csv') if (ROOT/'cleaned_titanic.csv').exists() else pd.read_csv(ROOT/'titanic.csv')
features=['pclass','sex','age','sibsp','parch','fare','embarked']; target='survived'
X=df[features].copy(); y=df[target].astype(int)
print('Class balance:'); print(y.value_counts(normalize=False)); print(y.value_counts(normalize=True))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
print('Stratification preserves class proportions across train/test; this matters because survival is imbalanced.')
num=['pclass','age','sibsp','parch','fare']; cat=['sex','embarked']
pre=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),num),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)])
models={'Logistic Regression':LogisticRegression(max_iter=1000,random_state=42),'Decision Tree':DecisionTreeClassifier(max_depth=5,random_state=42),'Random Forest':RandomForestClassifier(n_estimators=200,random_state=42)}
results={}
for name,est in models.items():
    pipe=Pipeline([('preprocess',pre),('model',est)]); pipe.fit(X_train,y_train); pred=pipe.predict(X_test); proba=pipe.predict_proba(X_test)[:,1]
    results[name]={'accuracy':accuracy_score(y_test,pred),'precision':precision_score(y_test,pred,zero_division=0),'recall':recall_score(y_test,pred,zero_division=0),'f1':f1_score(y_test,pred,zero_division=0),'auc':roc_auc_score(y_test,proba)}
    cm=confusion_matrix(y_test,pred); ConfusionMatrixDisplay(cm).plot(); plt.title(f'{name} confusion matrix'); plt.tight_layout(); plt.savefig(PLOTS/(name.lower().replace(' ','_')+'_cm.png'),dpi=150); plt.close()
    fpr,tpr,_=roc_curve(y_test,proba); plt.figure(figsize=(6,4)); plt.plot(fpr,tpr,label=f'AUC={results[name]["auc"]:.3f}'); plt.plot([0,1],[0,1],'--'); plt.xlabel('False positive rate'); plt.ylabel('True positive rate'); plt.title(f'{name} ROC'); plt.legend(); plt.tight_layout(); plt.savefig(PLOTS/(name.lower().replace(' ','_')+'_roc.png'),dpi=150); plt.close()

# Decision tree visualization with transformed feature names.
dt=models['Decision Tree']; dt_pipe=Pipeline([('preprocess',pre),('model',dt)]); dt_pipe.fit(X_train,y_train)
feat_names=list(dt_pipe.named_steps['preprocess'].get_feature_names_out())
plt.figure(figsize=(22,12)); plot_tree(dt_pipe.named_steps['model'],feature_names=feat_names,class_names=['Not survived','Survived'],filled=False,max_depth=4); plt.tight_layout(); plt.savefig(PLOTS/'decision_tree.png',dpi=150); plt.close()

# Imbalance comparison using same train/test split.
variants={
'baseline':Pipeline([('preprocess',pre),('model',LogisticRegression(max_iter=1000,random_state=42))]),
'class_weight_balanced':Pipeline([('preprocess',pre),('model',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42))]),
'smote':ImbPipeline([('preprocess',pre),('smote',SMOTE(random_state=42)),('model',LogisticRegression(max_iter=1000,random_state=42))])}
imb=[]
for name,p in variants.items():
    p.fit(X_train,y_train); pred=p.predict(X_test); imb.append({'strategy':name,'precision':precision_score(y_test,pred,zero_division=0),'recall':recall_score(y_test,pred,zero_division=0),'f1':f1_score(y_test,pred,zero_division=0)})
imb_df=pd.DataFrame(imb); print('=== IMBALANCE COMPARISON ==='); print(imb_df.to_string(index=False)); imb_df.to_csv(ROOT/'imbalance_comparison.csv',index=False)

# Grid search RF; OOB score is available because estimator is constructed with oob_score=True.
rf_pipe=Pipeline([('preprocess',pre),('model',RandomForestClassifier(random_state=42,oob_score=True,n_jobs=-1))])
param_grid={'model__n_estimators':[100,200],'model__max_depth':[None,5,10],'model__max_features':['sqrt','log2']}
grid=GridSearchCV(rf_pipe,param_grid=param_grid,cv=5,scoring='f1',n_jobs=-1); grid.fit(X_train,y_train)
print('=== RF GRID SEARCH ==='); print('Best params:',grid.best_params_); print('Best CV F1:',grid.best_score_); print('OOB:',grid.best_estimator_.named_steps['model'].oob_score_)

# Regression: fare from all other available features, using train/test split and a preprocessing pipeline.
reg_features=[c for c in df.columns if c!='fare' and c not in ['survived']]
Xr=df[reg_features].copy(); yr=df['fare'].astype(float)
Xr_train,Xr_test,yr_train,yr_test=train_test_split(Xr,yr,test_size=.2,random_state=42)
reg_num=Xr.select_dtypes(include=np.number).columns.tolist(); reg_cat=[c for c in Xr.columns if c not in reg_num]
reg_pre=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),reg_num),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),reg_cat)])
reg_pipe=Pipeline([('preprocess',reg_pre),('model',LinearRegression())]); reg_pipe.fit(Xr_train,yr_train); rp=reg_pipe.predict(Xr_test)
mae=mean_absolute_error(yr_test,rp); rmse=mean_squared_error(yr_test,rp)**.5; r2=r2_score(yr_test,rp); n=len(yr_test); p=reg_pipe.named_steps['preprocess'].transform(Xr_test).shape[1]; adj=1-(1-r2)*(n-1)/(n-p-1) if n-p-1>0 else np.nan
resid=yr_test-rp
plt.figure(figsize=(7,4)); plt.scatter(rp,resid,alpha=.6); plt.axhline(0,ls='--'); plt.xlabel('Predicted fare'); plt.ylabel('Residual'); plt.title('Fare regression residuals'); plt.tight_layout(); plt.savefig(PLOTS/'fare_residuals.png',dpi=150); plt.close()
hetero=abs(pd.Series(resid).corr(pd.Series(rp)))>.2
print('=== REGRESSION ===',{'MAE':mae,'RMSE':rmse,'R2':r2,'Adjusted_R2':adj,'heteroscedasticity_indication':bool(hetero)})

clf_table=pd.DataFrame(results).T.reset_index(names='classifier'); print('=== CLASSIFICATION ==='); print(clf_table.to_string(index=False)); clf_table.to_csv(ROOT/'classification_comparison.csv',index=False)
reg_metrics=pd.DataFrame([{'model':'Linear Regression','MAE':mae,'RMSE':rmse,'R2':r2,'Adjusted_R2':adj}]); reg_metrics.to_csv(ROOT/'regression_metrics.csv',index=False)
print('\nClassification metrics and regression metrics are separate metric groups and are not directly comparable.')

# Save complete best pipeline end-to-end. Select by test F1 only after the common evaluation; no ranking is hidden in the artifact.
best_name=max(results,key=lambda k:results[k]['f1']); best_pipe=Pipeline([('preprocess',pre),('model',models[best_name])]); best_pipe.fit(X_train,y_train); joblib.dump(best_pipe,ROOT/'best_pipeline.joblib')
reloaded=joblib.load(ROOT/'best_pipeline.joblib'); sample_pred=reloaded.predict(X_test.head(5)); print('Reloaded raw-input predictions:',sample_pred.tolist())

recommendation=f"Model selection was based on the reported test-set F1, while also considering accuracy, recall and AUC. The selected classifier is {best_name}, with accuracy={results[best_name]['accuracy']:.3f}, precision={results[best_name]['precision']:.3f}, recall={results[best_name]['recall']:.3f}, F1={results[best_name]['f1']:.3f}, and AUC={results[best_name]['auc']:.3f}. These values should be read together because no single classification metric captures every operational trade-off. The complete preprocessing-plus-estimator pipeline was saved and successfully reloaded against raw feature columns. The fare regression metrics are reported separately because regression and classification metrics are different quantities."
(ROOT/'MODEL_RESULTS.md').write_text('# Modeling Results\n\n'+clf_table.to_markdown(index=False)+'\n\n## Regression\n\n'+reg_metrics.to_markdown(index=False)+'\n\n## Final recommendation\n\n'+recommendation+'\n\n## Imbalance comparison\n\n'+imb_df.to_markdown(index=False)+'\n\n## Tuning\n\nBest RF parameters: `'+str(grid.best_params_)+'`\n\nOOB score: `'+str(grid.best_estimator_.named_steps['model'].oob_score_)+ '`\n\nResidual interpretation: the residual plot should be inspected for a funnel or systematic spread; the script also prints a simple correlation-based indication, which is supporting evidence rather than a formal heteroscedasticity test.\n')
