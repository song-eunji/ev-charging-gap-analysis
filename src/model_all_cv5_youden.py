# -*- coding: utf-8 -*-
"""
결핍지역 예측모델 - 5-fold CV 통일 + Youden's J 임계값으로 재평가.

기존 방식(단일 train/test split 25% + 유병률 분위수 임계값)의 문제:
  N=246, 양성(결핍) 49개뿐 -> 단일 split이면 test set에 양성이 12~13개밖에 안 남아
  모델 간 소수점 셋째자리 순위 비교가 노이즈에 가까움.

개선:
  1) 모든 모델을 동일한 StratifiedKFold(5-fold)로 cross_val_predict
     -> 전체 246개 표본 전부가 한 번씩은 test로 쓰이고, out-of-fold 확률을 모아(pooled) 평가.
  2) Threshold는 pooled out-of-fold 확률에서 Youden's J(=Sensitivity+Specificity-1 최대화)
     로 "한 번만" 결정. (TunedThresholdClassifierCV처럼 nested로 또 쪼개지 않음 -
      표본이 작아서 추가로 쪼개면 더 불안정해지기 때문 - 사용자와 논의 후 결정한 방식)
  3) AUROC/AUPRC는 threshold-free라 pooled 확률로 바로 계산.
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, VotingClassifier, StackingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, f1_score, roc_curve,
    confusion_matrix, balanced_accuracy_score, average_precision_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.width', 200)

# =============================================================================
# 0. 데이터 준비 (동일)
# =============================================================================
fresh = pd.read_csv('data/scratch_ev_charger3_fresh.csv', encoding='utf-8-sig')
housing = pd.read_csv('data/scratch_housing_indicators.csv', encoding='utf-8-sig')
pop = pd.read_csv('data/scratch_ev_full_merged.csv', encoding='utf-8-sig')

df = fresh.merge(housing[['sido_short', 'sigungu', '아파트비율', '노후비율', '자가주차장확보율']],
                  on=['sido_short', 'sigungu'], how='left')
df = df.merge(pop[['sido_short', 'sigungu', '인구수']], on=['sido_short', 'sigungu'], how='left')
df['log_pop'] = np.log(df['인구수'])

sub = pd.read_excel('data/무공해차_보조금현황_2026_2026-08-17.xlsx', sheet_name='요약')[
    ['시도', '지역구분', '접수율(%)', '예산소진율(%)']]
METRO = {'서울', '부산', '대구', '인천', '광주', '대전', '울산'}
MULTI_GU = {'수원시', '성남시', '안산시', '안양시', '용인시', '고양시', '창원시', '청주시', '천안시', '전주시', '포항시'}
metro_full = {'서울': '서울특별시', '부산': '부산광역시', '대구': '대구광역시', '인천': '인천광역시',
              '광주': '광주광역시', '대전': '대전광역시', '울산': '울산광역시'}

def lookup_key(row):
    sido, sgg = row['sido_short'], row['sigungu']
    if sido in METRO:
        return sido, metro_full[sido]
    for city in MULTI_GU:
        if sgg.startswith(city):
            return sido, city
    return sido, sgg

keys = df.apply(lookup_key, axis=1)
df['lk_sido'] = keys.apply(lambda t: t[0])
df['lk_region'] = keys.apply(lambda t: t[1])
df = df.merge(sub, left_on=['lk_sido', 'lk_region'], right_on=['시도', '지역구분'], how='left')

cutoff = df['pct_total'].quantile(0.20)
df['is_deficit'] = (df['pct_total'] <= cutoff).astype(int)

FEATURES = ['log_reg', 'log_pop', '아파트비율', '노후비율', '자가주차장확보율', '접수율(%)', '예산소진율(%)']
model_df = df.dropna(subset=FEATURES).copy()
X = model_df[FEATURES].values
y = model_df['is_deficit'].values
prevalence = y.mean()
print(f'N = {len(y)}, 결핍비율 = {prevalence:.3f} (양성 {y.sum()}개)')

# 모든 모델 공통 - 동일한 5-fold 분할 사용 (shuffle+seed 고정으로 재현 가능)
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def cv_evaluate(name, make_clf):
    """5-fold pooled out-of-fold 확률 -> Youden's J로 단일 threshold 결정.
    make_clf: 인자 없는 콜러블 -> 매 fold마다 새 인스턴스 생성 (CatBoost가 sklearn
    clone()과 호환 안 되는 문제가 있어, cross_val_predict 대신 수동 CV 루프 사용)."""
    n = len(y)
    proba = np.zeros(n)
    for train_idx, test_idx in SKF.split(X, y):
        clf = make_clf()
        clf.fit(X[train_idx], y[train_idx])
        proba[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

    auroc = roc_auc_score(y, proba)
    auprc = average_precision_score(y, proba)

    # ---- Youden's J: ROC curve 상에서 (TPR - FPR)이 최대인 지점의 threshold 선택 ----
    fpr, tpr, thresholds = roc_curve(y, proba)
    j_scores = tpr - fpr
    best_thresh = thresholds[np.argmax(j_scores)]

    pred = (proba >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        '모델': name,
        'AUROC': auroc,
        'BalancedAccuracy': balanced_accuracy_score(y, pred),
        'Accuracy': accuracy_score(y, pred),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        'AUPRC': auprc,
        'F1': f1_score(y, pred, zero_division=0),
        'Precision': precision_score(y, pred, zero_division=0),
        "Youden_threshold": best_thresh,
    }


results = []

# =============================================================================
# ========== 1. DecisionTree ==========
# 하이퍼파라미터: max_depth=5, min_samples_leaf=5, class_weight='balanced'
# =============================================================================
make_dt = lambda: DecisionTreeClassifier(
    max_depth=5, min_samples_leaf=5, random_state=42, class_weight='balanced',
)
results.append(cv_evaluate('DecisionTree', make_dt))

# =============================================================================
# ========== 2. XGBoost ==========
# 하이퍼파라미터: n_estimators=200, max_depth=3, learning_rate=0.05
# scale_pos_weight=(1-prevalence)/prevalence로 불균형 보정 (전체 유병률 기준 고정값)
# =============================================================================
make_xgb = lambda: XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
    scale_pos_weight=(1 - prevalence) / prevalence, eval_metric='logloss',
)
results.append(cv_evaluate('XGBoost', make_xgb))

# =============================================================================
# ========== 3. RandomForest ==========
# 하이퍼파라미터: n_estimators=300, max_depth=5, min_samples_leaf=5, class_weight='balanced'
# =============================================================================
make_rf = lambda: RandomForestClassifier(
    n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=42, class_weight='balanced',
)
results.append(cv_evaluate('RandomForest', make_rf))

# =============================================================================
# ========== 4. GBM (GradientBoostingClassifier) ==========
# 하이퍼파라미터: n_estimators=150, max_depth=3, learning_rate=0.05
# (class_weight 인자 미지원 -> 보정 없이 기본값)
# =============================================================================
make_gbm = lambda: GradientBoostingClassifier(
    n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42,
)
results.append(cv_evaluate('GBM', make_gbm))

# =============================================================================
# ========== 5. MLP ==========
# 하이퍼파라미터: hidden_layer_sizes=(16, 8), max_iter=2000, early_stopping=True
# Pipeline(StandardScaler+clf)로 감싸서 CV 각 fold마다 fold 내부에서만 스케일링
# (fold 밖 정보가 새어들어가지 않도록 - 이전 단일split 버전은 이 부분이 느슨했음)
# =============================================================================
make_mlp = lambda: Pipeline([
    ('scaler', StandardScaler()),
    ('clf', MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000,
                           random_state=42, early_stopping=True)),
])
results.append(cv_evaluate('MLP', make_mlp))

# =============================================================================
# ========== 6. Logistic regression ==========
# 하이퍼파라미터: max_iter=1000, class_weight='balanced'
# 마찬가지로 Pipeline으로 감싸 fold별 스케일링 누수 방지
# =============================================================================
make_logreg = lambda: Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced')),
])
results.append(cv_evaluate('Logistic regression', make_logreg))

# =============================================================================
# ========== 7. Extra trees ==========
# 하이퍼파라미터: n_estimators=300, max_depth=5, min_samples_leaf=5, class_weight='balanced'
# =============================================================================
make_et = lambda: ExtraTreesClassifier(
    n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=42, class_weight='balanced',
)
results.append(cv_evaluate('Extra trees', make_et))

# =============================================================================
# ========== 8. LightGBM ==========
# 하이퍼파라미터: n_estimators=200, max_depth=3, learning_rate=0.05, class_weight='balanced'
# =============================================================================
make_lgbm = lambda: LGBMClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
    class_weight='balanced', verbose=-1,
)
results.append(cv_evaluate('LightGBM', make_lgbm))

# =============================================================================
# ========== 9. AdaBoost ==========
# 하이퍼파라미터: n_estimators=150, learning_rate=0.05 (class_weight 미지원)
# =============================================================================
make_ada = lambda: AdaBoostClassifier(
    n_estimators=150, learning_rate=0.05, random_state=42,
)
results.append(cv_evaluate('AdaBoost', make_ada))

# =============================================================================
# ========== 10. CatBoost ==========
# 하이퍼파라미터: iterations=200, depth=3, learning_rate=0.05
# scale_pos_weight=(1-prevalence)/prevalence로 불균형 보정
# (CatBoost sklearn wrapper가 clone()과 호환 안 돼 매 fold 새 인스턴스 생성 방식 사용)
# =============================================================================
make_cat = lambda: CatBoostClassifier(
    iterations=200, depth=3, learning_rate=0.05, random_state=42,
    scale_pos_weight=(1 - prevalence) / prevalence, verbose=False,
)
results.append(cv_evaluate('CatBoost', make_cat))

single_df = pd.DataFrame(results).set_index('모델').round(3).sort_values('AUROC', ascending=False)
print('\n=== 단일모델 10종 (5-fold CV, Youden\'s J) ===')
print(single_df.to_string())

# =============================================================================
# 앙상블: 새로 구한 5-fold pooled AUROC 순위를 기준으로 상위 2/3/4개 구성
# =============================================================================
individual_auroc = single_df['AUROC'].to_dict()


def make_lr():
    return Pipeline([('scaler', StandardScaler()),
                      ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))])

def make_et():
    return ExtraTreesClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5,
                                 random_state=42, class_weight='balanced')

def make_lgbm():
    return LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
                           class_weight='balanced', verbose=-1)

def make_rf():
    return RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5,
                                   random_state=42, class_weight='balanced')

BASE_MODEL_MAP = {'lr': make_lr, 'et': make_et, 'lgbm': make_lgbm, 'rf': make_rf}
BASE_MODEL_NAME = {'lr': 'Logistic regression', 'et': 'Extra trees',
                    'lgbm': 'LightGBM', 'rf': 'RandomForest'}


def build_ensembles(estimator_keys):
    """각 앙상블마다 '팩토리 함수'를 반환 -> cv_evaluate가 fold마다 새로 생성해서 씀."""
    weights = [individual_auroc[BASE_MODEL_NAME[k]] for k in estimator_keys]
    weights = [w / sum(weights) for w in weights]

    def make_soft():
        estimators = [(k, BASE_MODEL_MAP[k]()) for k in estimator_keys]
        return VotingClassifier(estimators=estimators, voting='soft')

    def make_weighted():
        estimators = [(k, BASE_MODEL_MAP[k]()) for k in estimator_keys]
        return VotingClassifier(estimators=estimators, voting='soft', weights=weights)

    def make_stacking():
        estimators = [(k, BASE_MODEL_MAP[k]()) for k in estimator_keys]
        # Stacking 내부 cv=5는 메타모델용 OOF 피처 생성을 위한 것 (임계값 튜닝이 아님 -> 유지)
        return StackingClassifier(
            estimators=estimators, final_estimator=LogisticRegression(max_iter=1000),
            cv=5, stack_method='predict_proba',
        )

    return {
        'Soft Voting': make_soft,
        'Weighted Soft Voting': make_weighted,
        'Stacking': make_stacking,
    }


ensemble_results = []

# =============================================================================
# ========== 11. 앙상블 - 상위 2개 (Logistic regression + Extra trees) ==========
# =============================================================================
for name, make_clf in build_ensembles(['lr', 'et']).items():
    ensemble_results.append(cv_evaluate(f'[상위2] {name}', make_clf))

# =============================================================================
# ========== 12. 앙상블 - 상위 3개 (+LightGBM) ==========
# =============================================================================
for name, make_clf in build_ensembles(['lr', 'et', 'lgbm']).items():
    ensemble_results.append(cv_evaluate(f'[상위3] {name}', make_clf))

# =============================================================================
# ========== 13. 앙상블 - 상위 4개 (+RandomForest) ==========
# =============================================================================
for name, make_clf in build_ensembles(['lr', 'et', 'lgbm', 'rf']).items():
    ensemble_results.append(cv_evaluate(f'[상위4] {name}', make_clf))

ensemble_df = pd.DataFrame(ensemble_results).set_index('모델').round(3)
print('\n=== 앙상블 (5-fold CV, Youden\'s J) ===')
print(ensemble_df.to_string())

combined = pd.concat([single_df, ensemble_df]).sort_values('AUROC', ascending=False)
print('\n=== 전체 통합 순위 (5-fold CV + Youden\'s J 통일) ===')
print(combined.to_string())

combined.to_csv('data/scratch_model_all_cv5_youden.csv', encoding='utf-8-sig')
print('\n[저장] data/scratch_model_all_cv5_youden.csv')
