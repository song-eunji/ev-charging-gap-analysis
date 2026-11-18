# -*- coding: utf-8 -*-
"""
전기차 충전 인프라 갭 분석 - 통합 파이프라인
기존에 4개 파일로 나뉘어 있던 걸 한 파일에서 순서대로 실행되도록 통합.
로직/파라미터는 전혀 안 바꿈 -> 결과는 기존과 100% 동일.

PART 1. 데이터 준비        (등록현황 + ev_charger_3 파싱/집계/조인)
PART 2. 개수 기반 갭분석    (로그-로그 회귀 -> pct_total/pct_fast/pct_slow)
PART 3. 가동률 분석         (ev_charger_1 STAT 코드 -> ok_rate)
PART 4. 용량 + 혼잡도 분석  (급속충전량kW, STAT=3 busy_rate)
PART 5. 4개 축 통합 테이블
PART 6. 결핍지역 예측모델   (5-fold CV + Youden's J)
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

pd.set_option('display.max_rows', 300)
pd.set_option('display.width', 200)

REG_FILE = 'data/한국교통안전공단_전국_전기차_차종별_용도별_차량_등록대수&#40;운행차량기준&#41;_20250407.csv'
CHG3_FILE = 'data/ev_charger_3.csv'
CHG1_FILE = 'data/ev_charger_1.csv'

# =============================================================================
# STEP 0. 공통 유틸 - 주소 파싱 (등록현황 제외 전부 재사용)
# =============================================================================
sido_full_set = {'서울특별시', '인천광역시', '경기도', '강원도', '강원특별자치도', '충청북도', '충청남도',
                  '대전광역시', '세종특별자치시', '경상북도', '대구광역시', '전라북도', '전북특별자치도',
                  '전라남도', '광주광역시', '경상남도', '부산광역시', '울산광역시', '제주특별자치도'}
multi_gu_cities = {'고양시', '성남시', '수원시', '안산시', '안양시', '용인시',
                    '창원시', '청주시', '천안시', '전주시', '포항시'}
sido_map_full_to_short = {
    '서울특별시': '서울', '인천광역시': '인천', '경기도': '경기', '강원도': '강원', '강원특별자치도': '강원',
    '충청북도': '충북', '충청남도': '충남', '대전광역시': '대전', '세종특별자치시': '세종',
    '경상북도': '경북', '대구광역시': '대구', '전라북도': '전북', '전북특별자치도': '전북', '전라남도': '전남',
    '광주광역시': '광주', '경상남도': '경남', '부산광역시': '부산', '울산광역시': '울산', '제주특별자치도': '제주',
}

def parse_addr(tok_list):
    if len(tok_list) < 2 or tok_list[0] == '-':
        return None, None
    t0 = tok_list[0]
    if t0 not in sido_full_set:
        return None, None
    if t0 == '세종특별자치시':
        return t0, '세종특별자치시'
    if len(tok_list) >= 3 and tok_list[1] in multi_gu_cities and tok_list[2].endswith('구'):
        return t0, tok_list[1] + ' ' + tok_list[2]
    return t0, tok_list[1]

def clean_addr_df(raw_df, addr_col):
    """주소 파싱 + 군위군(2023 경북->대구 편입) 재매핑까지 공통 처리."""
    tokens = raw_df[addr_col].astype(str).str.split()
    parsed = tokens.apply(parse_addr)
    raw_df = raw_df.copy()
    raw_df['sido_full'] = parsed.apply(lambda t: t[0])
    raw_df['sigungu'] = parsed.apply(lambda t: t[1])
    valid = raw_df[raw_df['sido_full'].notna()].copy()
    valid['sido_short'] = valid['sido_full'].map(sido_map_full_to_short)
    gunwi_mask = (valid['sido_short'] == '경북') & (valid['sigungu'] == '군위군')
    valid.loc[gunwi_mask, 'sido_short'] = '대구'
    return valid


# =============================================================================
# PART 1. 데이터 준비
# =============================================================================
print('=' * 70); print('PART 1. 데이터 준비'); print('=' * 70)

# ---- STEP 1-1. 등록현황 로드 ----
reg = pd.read_csv(REG_FILE, encoding='cp949')
reg = reg[reg['연료별'] == '전기'].copy()
sido_alias = {'경상북도': '경북'}

def split_key(s):
    tok1, rest = s.split(' ', 1)
    return sido_alias.get(tok1, tok1), rest

parsed = reg['시군구별'].apply(split_key)
reg['sido_short'] = parsed.apply(lambda t: t[0])
reg['sigungu_raw'] = parsed.apply(lambda t: t[1])
reg.loc[reg['sigungu_raw'].str.startswith('부천시'), 'sigungu_raw'] = '부천시'
reg_agg = reg.groupby(['sido_short', 'sigungu_raw'])['계'].sum().reset_index()
reg_agg = reg_agg.rename(columns={'sigungu_raw': 'sigungu', '계': 'registration'})
print(f'등록현황 시군구 수: {len(reg_agg)}')

# ---- STEP 1-2. ev_charger_3 로드 + 파싱 + 집계(개수/급속/완속/아파트비율) ----
chg3 = pd.read_csv(CHG3_FILE, encoding='cp949', low_memory=False)
chg3_valid = clean_addr_df(chg3, '주소')
chg3_valid['is_fast'] = (chg3_valid['기종(대)'] == '급속').astype(int)
chg3_valid['is_slow'] = (chg3_valid['기종(대)'] == '완속').astype(int)
chg3_valid['is_apt'] = (chg3_valid['시설구분(대)'] == '공동주택시설').astype(int)

chg_agg = chg3_valid.groupby(['sido_short', 'sigungu']).agg(
    total_chargers=('충전기ID', 'count'),
    fast_chargers=('is_fast', 'sum'),
    slow_chargers=('is_slow', 'sum'),
    apt_facility_count=('is_apt', 'sum'),
).reset_index()
chg_agg['apt_ratio_official'] = chg_agg['apt_facility_count'] / chg_agg['total_chargers']
print(f'ev_charger_3 유효 레코드: {len(chg3_valid):,} / {len(chg3):,}, 시군구 수: {len(chg_agg)}')

# ---- STEP 1-3. 등록현황 <-> 충전소 조인 ----
merged_outer = reg_agg.merge(chg_agg, on=['sido_short', 'sigungu'], how='outer', indicator=True)
merged = merged_outer[merged_outer['_merge'] == 'both'].drop(columns='_merge').copy()
merged = merged[merged['registration'] > 0].copy()
print(f'*** 최종 표본 N = {len(merged)} ***')


# =============================================================================
# PART 2. 개수 기반 갭분석 - 로그-로그 회귀
# =============================================================================
print(); print('=' * 70); print('PART 2. 개수 기반 갭분석'); print('=' * 70)

# ---- STEP 2-1. log-log 회귀: log(충전기수) ~ log(등록대수) ----
# (선형-원본카운트 회귀는 소규모지역 기대값이 음수가 되는 문제가 있어 로그-로그로 교체)
merged['log_reg'] = np.log(merged['registration'])

def loglog_fit(y_col):
    y = np.log(merged[y_col].clip(lower=0.5))
    x = merged['log_reg']
    X = np.column_stack([np.ones(len(x)), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted_log = X @ beta
    expected = np.exp(fitted_log)
    pct = (merged[y_col] / expected - 1) * 100
    return expected, pct

# ---- STEP 2-2. total/fast/slow 각각 기대치대비% 산출 ----
merged['expected_total_ll'], merged['pct_total'] = loglog_fit('total_chargers')
merged['expected_fast_ll'], merged['pct_fast'] = loglog_fit('fast_chargers')
merged['expected_slow_ll'], merged['pct_slow'] = loglog_fit('slow_chargers')

r_pct_vs_reg = merged['pct_total'].corr(merged['log_reg'])
print(f'log(등록대수) vs pct_total 상관계수: {r_pct_vs_reg:.4f}  (0에 가까움 -> "규모효과" 가설 기각)')
print()
print('-- 전체(급속+완속) 기준 기대치대비% 하위 10 --')
print(merged.nsmallest(10, 'pct_total')[['sido_short', 'sigungu', 'registration',
      'total_chargers', 'pct_total']].round(1).to_string(index=False))


# =============================================================================
# PART 3. 가동률 분석 (ev_charger_1 STAT 코드)
# =============================================================================
print(); print('=' * 70); print('PART 3. 가동률 분석'); print('=' * 70)

# ---- STEP 3-1. ev_charger_1 로드 + 파싱 ----
cols1 = ['STATID', 'CHGERID', 'STAT', 'ADDR', 'STATUPDDT', 'YEAR', 'BUSINM']
chg1 = pd.read_csv(CHG1_FILE, usecols=cols1)
chg1_valid = clean_addr_df(chg1, 'ADDR')

# ---- STEP 3-2. STAT 코드 분류 -> ok/broken/unknown ----
# STAT: 2=충전대기, 3=충전중 (정상) / 1=통신이상, 4=운영중지, 5=점검중 (고장) / 9=상태미확인
chg1_valid['is_ok'] = chg1_valid['STAT'].isin([2, 3]).astype(int)
chg1_valid['is_broken'] = chg1_valid['STAT'].isin([1, 4, 5]).astype(int)
chg1_valid['is_unknown'] = (chg1_valid['STAT'] == 9).astype(int)

# ---- STEP 3-3. 시군구별 가동률 집계 ----
util_agg = chg1_valid.groupby(['sido_short', 'sigungu']).agg(
    total_reported=('STATID', 'count'),
    n_ok=('is_ok', 'sum'),
    n_broken=('is_broken', 'sum'),
).reset_index()
util_agg['ok_rate'] = util_agg['n_ok'] / util_agg['total_reported'] * 100
util_agg['broken_rate'] = util_agg['n_broken'] / util_agg['total_reported'] * 100
util_stable = util_agg[util_agg['total_reported'] >= 10].copy()  # 10대 미만은 % 불안정 -> 제외
print(f'전국 평균 가동률(정상): {util_stable["ok_rate"].mean():.1f}%')
print()
print('-- 가동률 하위 10 --')
print(util_stable.nsmallest(10, 'ok_rate')[['sido_short', 'sigungu', 'total_reported',
      'ok_rate', 'broken_rate']].round(1).to_string(index=False))


# =============================================================================
# PART 4. 용량(kW) + 혼잡도 분석
# =============================================================================
print(); print('=' * 70); print('PART 4. 용량 + 혼잡도 분석'); print('=' * 70)

# ---- STEP 4-1. [용량] "급속(100kW단독)" 문자열에서 숫자만 정규식 추출 ----
kw_extract = chg3_valid['급속충전량'].astype(str).str.extract(r'(\d+)kW')[0]
chg3_valid['kw'] = pd.to_numeric(kw_extract, errors='coerce')
fast_with_kw = chg3_valid[(chg3_valid['기종(대)'] == '급속') & chg3_valid['kw'].notna()]
n_fast_total = (chg3_valid['기종(대)'] == '급속').sum()
print(f'[용량] 급속충전기 {n_fast_total:,}개 중 kW정보 있는 것: {len(fast_with_kw):,}개 '
      f'({len(fast_with_kw)/n_fast_total*100:.1f}%, 결측 다수 -> 참고용)')

kw_agg = fast_with_kw.groupby(['sido_short', 'sigungu']).agg(
    n_kw_reported=('kw', 'count'), avg_kw=('kw', 'mean')).reset_index()

# ---- STEP 4-2. [혼잡도] STAT==3(충전중) 비율 = 정상보고 대비 지금 사용중 비율 ----
chg1_valid['is_busy'] = (chg1_valid['STAT'] == 3).astype(int)
chg1_valid['is_reported'] = chg1_valid['STAT'].isin([2, 3]).astype(int)
busy_agg = chg1_valid.groupby(['sido_short', 'sigungu']).agg(
    n_busy=('is_busy', 'sum'), n_normal=('is_reported', 'sum')).reset_index()
busy_agg['busy_rate'] = busy_agg['n_busy'] / busy_agg['n_normal'] * 100
busy_stable = busy_agg[busy_agg['n_normal'] >= 10].copy()
print(f'[혼잡도] 전국 평균: {busy_stable["busy_rate"].mean():.1f}%')


# =============================================================================
# PART 5. 4개 축 통합 테이블
# =============================================================================
print(); print('=' * 70); print('PART 5. 4개 축(개수/가동률/용량/혼잡도) 통합'); print('=' * 70)

full = merged.merge(util_stable[['sido_short', 'sigungu', 'ok_rate', 'broken_rate']],
                     on=['sido_short', 'sigungu'], how='left')
full = full.merge(kw_agg, on=['sido_short', 'sigungu'], how='left')
full = full.merge(busy_stable[['sido_short', 'sigungu', 'n_normal', 'busy_rate']],
                   on=['sido_short', 'sigungu'], how='left')

# ---- STEP 5-1. "3중고" 지역 스캔 = 개수부족 + 가동률낮음 + 혼잡도높음 동시 충족 ----
deficit_cut = full['pct_total'].quantile(0.20)
ok_median = full['ok_rate'].median()
busy_median = full['busy_rate'].median()
triple = full[(full['pct_total'] <= deficit_cut) &
              (full['ok_rate'] <= ok_median) &
              (full['busy_rate'] >= busy_median)]
print('-- 3중고 지역 (개수부족 하위20% + 가동률 중앙값 이하 + 혼잡도 중앙값 이상) --')
print(triple[['sido_short', 'sigungu', 'pct_total', 'ok_rate', 'busy_rate']]
      .round(1).sort_values('pct_total').to_string(index=False))

full.to_csv('data/scratch_ev_full_4axis.csv', index=False, encoding='utf-8-sig')
print('\n[저장] data/scratch_ev_full_4axis.csv')


# =============================================================================
# PART 6. 결핍지역 예측모델 (5-fold CV + Youden's J)
# =============================================================================
print(); print('=' * 70); print('PART 6. 결핍지역 예측모델'); print('=' * 70)

# ---- STEP 6-1. 주택/인구/보조금 병합 (분류모델 전용 피처) ----
housing = pd.read_csv('data/scratch_housing_indicators.csv', encoding='utf-8-sig')
pop = pd.read_csv('data/scratch_ev_full_merged.csv', encoding='utf-8-sig')

df = merged.merge(housing[['sido_short', 'sigungu', '아파트비율', '노후비율', '자가주차장확보율']],
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

# ---- STEP 6-2. 라벨: pct_total 하위 20% = 결핍 (비순환적 - 로그로그 잔차 기반) ----
cutoff = df['pct_total'].quantile(0.20)
df['is_deficit'] = (df['pct_total'] <= cutoff).astype(int)

FEATURES = ['log_reg', 'log_pop', '아파트비율', '노후비율', '자가주차장확보율', '접수율(%)', '예산소진율(%)']
model_df = df.dropna(subset=FEATURES).copy()
X = model_df[FEATURES].values
y = model_df['is_deficit'].values
prevalence = y.mean()
print(f'N = {len(y)}, 결핍비율 = {prevalence:.3f} (양성 {y.sum()}개)')

SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cv_evaluate(name, make_clf):
    """5-fold pooled out-of-fold 확률 -> Youden's J로 단일 threshold 결정 (nested 튜닝 없음)."""
    proba = np.zeros(len(y))
    for train_idx, test_idx in SKF.split(X, y):
        clf = make_clf()
        clf.fit(X[train_idx], y[train_idx])
        proba[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    auroc = roc_auc_score(y, proba)
    auprc = average_precision_score(y, proba)
    fpr, tpr, thresholds = roc_curve(y, proba)
    best_thresh = thresholds[np.argmax(tpr - fpr)]  # Youden's J
    pred = (proba >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        '모델': name, 'AUROC': auroc,
        'BalancedAccuracy': balanced_accuracy_score(y, pred),
        'Accuracy': accuracy_score(y, pred),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        'AUPRC': auprc, 'F1': f1_score(y, pred, zero_division=0),
        'Precision': precision_score(y, pred, zero_division=0),
    }

results = []
# ========== 1. DecisionTree ==========
make_dt = lambda: DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42, class_weight='balanced')
results.append(cv_evaluate('DecisionTree', make_dt))
# ========== 2. XGBoost ==========
make_xgb = lambda: XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
                                  scale_pos_weight=(1 - prevalence) / prevalence, eval_metric='logloss')
results.append(cv_evaluate('XGBoost', make_xgb))
# ========== 3. RandomForest ==========
make_rf = lambda: RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5,
                                          random_state=42, class_weight='balanced')
results.append(cv_evaluate('RandomForest', make_rf))
# ========== 4. GBM ==========
make_gbm = lambda: GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
results.append(cv_evaluate('GBM', make_gbm))
# ========== 5. MLP ==========
make_mlp = lambda: Pipeline([('scaler', StandardScaler()),
    ('clf', MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=42, early_stopping=True))])
results.append(cv_evaluate('MLP', make_mlp))
# ========== 6. Logistic regression ==========
make_logreg = lambda: Pipeline([('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))])
results.append(cv_evaluate('Logistic regression', make_logreg))
# ========== 7. Extra trees ==========
make_et = lambda: ExtraTreesClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5,
                                        random_state=42, class_weight='balanced')
results.append(cv_evaluate('Extra trees', make_et))
# ========== 8. LightGBM ==========
make_lgbm = lambda: LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
                                    class_weight='balanced', verbose=-1)
results.append(cv_evaluate('LightGBM', make_lgbm))
# ========== 9. AdaBoost ==========
make_ada = lambda: AdaBoostClassifier(n_estimators=150, learning_rate=0.05, random_state=42)
results.append(cv_evaluate('AdaBoost', make_ada))
# ========== 10. CatBoost ==========
make_cat = lambda: CatBoostClassifier(iterations=200, depth=3, learning_rate=0.05, random_state=42,
                                       scale_pos_weight=(1 - prevalence) / prevalence, verbose=False)
results.append(cv_evaluate('CatBoost', make_cat))

single_df = pd.DataFrame(results).set_index('모델').round(3).sort_values('AUROC', ascending=False)
print('\n-- 단일모델 10종 (5-fold CV, Youden\'s J) --')
print(single_df.to_string())

# ---- STEP 6-3. 앙상블 (상위2/3/4, 개별 AUROC 기준) ----
individual_auroc = single_df['AUROC'].to_dict()

def make_lr(): return Pipeline([('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))])
def make_et2(): return ExtraTreesClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5,
    random_state=42, class_weight='balanced')
def make_lgbm2(): return LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
    random_state=42, class_weight='balanced', verbose=-1)
def make_rf2(): return RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5,
    random_state=42, class_weight='balanced')

BASE_MODEL_MAP = {'lr': make_lr, 'et': make_et2, 'lgbm': make_lgbm2, 'rf': make_rf2}
BASE_MODEL_NAME = {'lr': 'Logistic regression', 'et': 'Extra trees', 'lgbm': 'LightGBM', 'rf': 'RandomForest'}

def build_ensembles(estimator_keys):
    weights = [individual_auroc[BASE_MODEL_NAME[k]] for k in estimator_keys]
    weights = [w / sum(weights) for w in weights]
    def make_soft():
        return VotingClassifier(estimators=[(k, BASE_MODEL_MAP[k]()) for k in estimator_keys], voting='soft')
    def make_weighted():
        return VotingClassifier(estimators=[(k, BASE_MODEL_MAP[k]()) for k in estimator_keys],
                                 voting='soft', weights=weights)
    def make_stacking():
        return StackingClassifier(estimators=[(k, BASE_MODEL_MAP[k]()) for k in estimator_keys],
                                   final_estimator=LogisticRegression(max_iter=1000), cv=5,
                                   stack_method='predict_proba')
    return {'Soft Voting': make_soft, 'Weighted Soft Voting': make_weighted, 'Stacking': make_stacking}

ensemble_results = []
# ========== 11~13. 앙상블 - 상위2/3/4 ==========
for label, keys in [('상위2', ['lr', 'et']), ('상위3', ['lr', 'et', 'lgbm']), ('상위4', ['lr', 'et', 'lgbm', 'rf'])]:
    for name, make_clf in build_ensembles(keys).items():
        ensemble_results.append(cv_evaluate(f'[{label}] {name}', make_clf))

ensemble_df = pd.DataFrame(ensemble_results).set_index('모델').round(3)
combined = pd.concat([single_df, ensemble_df]).sort_values('AUROC', ascending=False)
print('\n-- 전체 통합 순위 --')
print(combined.to_string())

combined.to_csv('data/scratch_model_all_final_pipeline.csv', encoding='utf-8-sig')
print('\n[저장] data/scratch_model_all_final_pipeline.csv')
print('\n=== 파이프라인 완료 ===')
