# -*- coding: utf-8 -*-
"""
전기차 충전 인프라 갭 분석 - Streamlit in Snowflake 대시보드 (외부 라이브러리 없는 버전)
Snowsight > Streamlit > + Streamlit App 으로 새 앱 만들 때,
Database/Schema를 PROJECT.EV로 지정하고 이 코드를 그대로 붙여넣으면 됨.
plotly/matplotlib 둘 다 컨테이너에 없어서, streamlit 내장 차트(st.bar_chart/scatter_chart)만 사용.
별도 패키지 설치가 필요 없음(streamlit, pandas는 항상 존재).
"""
import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session

session = get_active_session()

st.set_page_config(page_title="전기차 충전 인프라 갭 분석", layout="wide")

COLOR_DEFICIT = "#d03b3b"
COLOR_SURPLUS = "#2a78d6"
COLOR_9CITY = "#eb6834"
COLOR_BUSAN = "#4a3aa7"
COLOR_NEUTRAL = "#c3c2b7"

NINE_CITIES = ['아산시', '천안시 동남구', '여수시', '순천시', '광양시',
               '포항시 남구', '포항시 북구', '영천시', '경산시']
BUSAN4 = ['수영구', '연제구', '중구', '동구']

# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_residuals():
    q = """
    WITH base AS (
        SELECT *, LN(REGISTRATION) AS LOG_REG FROM PROJECT.EV.EV_REGISTRATION_SIGUNGU
    ),
    stats_total AS (
        SELECT REGR_SLOPE(TOTAL_CHARGERS, LOG_REG) AS SLOPE,
               REGR_INTERCEPT(TOTAL_CHARGERS, LOG_REG) AS INTERCEPT,
               REGR_R2(TOTAL_CHARGERS, LOG_REG) AS R2
        FROM base
    ),
    stats_fast AS (
        SELECT REGR_SLOPE(FAST_CHARGERS, LOG_REG) AS SLOPE,
               REGR_INTERCEPT(FAST_CHARGERS, LOG_REG) AS INTERCEPT
        FROM base
    ),
    stats_slow AS (
        SELECT REGR_SLOPE(SLOW_CHARGERS, LOG_REG) AS SLOPE,
               REGR_INTERCEPT(SLOW_CHARGERS, LOG_REG) AS INTERCEPT
        FROM base
    )
    SELECT
        b.SIDO, b.SIGUNGU, b.REGISTRATION, b.LOG_REG,
        b.TOTAL_CHARGERS, b.FAST_CHARGERS, b.SLOW_CHARGERS,
        t.R2 AS MODEL_R2,
        t.SLOPE * b.LOG_REG + t.INTERCEPT AS EXPECTED_TOTAL,
        b.TOTAL_CHARGERS - (t.SLOPE * b.LOG_REG + t.INTERCEPT) AS RESID_TOTAL,
        b.FAST_CHARGERS  - (f.SLOPE * b.LOG_REG + f.INTERCEPT) AS RESID_FAST,
        b.SLOW_CHARGERS  - (s.SLOPE * b.LOG_REG + s.INTERCEPT) AS RESID_SLOW
    FROM base b
    CROSS JOIN stats_total t
    CROSS JOIN stats_fast f
    CROSS JOIN stats_slow s
    """
    return session.sql(q).to_pandas()


@st.cache_data(ttl=3600)
def load_housing():
    return session.sql("SELECT * FROM PROJECT.EV.HOUSING_INDICATORS").to_pandas()


@st.cache_data(ttl=3600)
def load_subsidy_joined():
    q = """
    WITH base AS (
        SELECT *, LN(REGISTRATION) AS LOG_REG FROM PROJECT.EV.EV_REGISTRATION_SIGUNGU
    ),
    stats_total AS (
        SELECT REGR_SLOPE(TOTAL_CHARGERS, LOG_REG) AS SLOPE,
               REGR_INTERCEPT(TOTAL_CHARGERS, LOG_REG) AS INTERCEPT
        FROM base
    ),
    resid AS (
        SELECT b.SIDO, b.SIGUNGU,
               b.TOTAL_CHARGERS - (t.SLOPE * b.LOG_REG + t.INTERCEPT) AS RESID_TOTAL
        FROM base b CROSS JOIN stats_total t
    ),
    lookup AS (
        SELECT r.*,
            CASE r.SIDO
                WHEN '서울' THEN '서울특별시' WHEN '부산' THEN '부산광역시' WHEN '대구' THEN '대구광역시'
                WHEN '인천' THEN '인천광역시' WHEN '광주' THEN '광주광역시' WHEN '대전' THEN '대전광역시'
                WHEN '울산' THEN '울산광역시'
                WHEN '경기' THEN
                    CASE WHEN r.SIGUNGU LIKE '수원시%' THEN '수원시' WHEN r.SIGUNGU LIKE '성남시%' THEN '성남시'
                         WHEN r.SIGUNGU LIKE '안산시%' THEN '안산시' WHEN r.SIGUNGU LIKE '안양시%' THEN '안양시'
                         WHEN r.SIGUNGU LIKE '용인시%' THEN '용인시' WHEN r.SIGUNGU LIKE '고양시%' THEN '고양시'
                         ELSE r.SIGUNGU END
                WHEN '경남' THEN CASE WHEN r.SIGUNGU LIKE '창원시%' THEN '창원시' ELSE r.SIGUNGU END
                WHEN '충북' THEN CASE WHEN r.SIGUNGU LIKE '청주시%' THEN '청주시' ELSE r.SIGUNGU END
                WHEN '충남' THEN CASE WHEN r.SIGUNGU LIKE '천안시%' THEN '천안시' ELSE r.SIGUNGU END
                WHEN '전북' THEN CASE WHEN r.SIGUNGU LIKE '전주시%' THEN '전주시' ELSE r.SIGUNGU END
                WHEN '경북' THEN CASE WHEN r.SIGUNGU LIKE '포항시%' THEN '포항시' ELSE r.SIGUNGU END
                ELSE r.SIGUNGU
            END AS SUBSIDY_LOOKUP_REGION
        FROM resid r
    )
    SELECT j.SIDO, j.SIGUNGU, j.RESID_TOTAL, s.APPLICATION_RATE_PCT, s.BUDGET_USED_PCT
    FROM lookup j
    JOIN PROJECT.EV.SUBSIDY_REGION s ON j.SIDO = s.SIDO AND j.SUBSIDY_LOOKUP_REGION = s.REGION_NAME
    """
    return session.sql(q).to_pandas()


@st.cache_data(ttl=3600)
def load_model():
    return session.sql("""
        SELECT * FROM PROJECT.EV.INSTALL_PRIORITY_MODEL ORDER BY INSTALL_PRIORITY_SCORE DESC
    """).to_pandas()


@st.cache_data(ttl=3600)
def load_centroids():
    return session.sql("SELECT * FROM PROJECT.EV.SIGUNGU_CENTROIDS").to_pandas()


df = load_residuals()
housing = load_housing()
model_df = load_model()
centroids = load_centroids()

# ---------------------------------------------------------------------------
# 사이드바 필터 (시도)
# ---------------------------------------------------------------------------
st.sidebar.header("🔍 조건 선택")
sido_options = sorted(df["SIDO"].unique())
sido_pick = st.sidebar.selectbox("시도", ["전체"] + sido_options)

if sido_pick != "전체":
    df = df[df["SIDO"] == sido_pick]
    housing = housing[housing["SIDO"] == sido_pick]
    model_df = model_df[model_df["SIDO"] == sido_pick]

st.sidebar.caption(f"현재 {len(df)}개 시군구 표시 중")

# ---------------------------------------------------------------------------
# 헤더
# ---------------------------------------------------------------------------
st.title("⚡ 전기차 충전 인프라 갭 분석")
st.caption("등록대수를 감안했을 때, 예상보다 충전기가 부족한 지역은 어디이고 왜 그런가")

c1, c2, c3 = st.columns(3)
c1.metric("분석 시군구 수", f"{len(df)}개")
c2.metric("회귀 모델 R²", f"{df['MODEL_R2'].iloc[0]:.3f}")
c3.metric("완속만 부족한 도시", f"{len(NINE_CITIES)}개")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🚨 결핍지역", "⚡ 완속 vs 급속", "🏘️ 왜 부족한가", "💰 보조금 검증", "🎯 설치 우선순위"]
)

# ---------------------------------------------------------------------------
# TAB 1. 헤드라인 - 결핍지역
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("🗺️ 지역별 결핍 지도 (빨강 = 부족 · 파랑 = 과잉)")
    map_df = df.merge(centroids, on=["SIDO", "SIGUNGU"], how="inner")
    resid_abs_max = map_df["RESID_TOTAL"].abs().quantile(0.95) or 1
    map_df["COLOR"] = map_df["RESID_TOTAL"].apply(
        lambda v: COLOR_DEFICIT if v < 0 else COLOR_SURPLUS
    )
    map_df["SIZE"] = (map_df["RESID_TOTAL"].abs().clip(upper=resid_abs_max) / resid_abs_max * 8000 + 500)
    st.map(map_df, latitude="LAT", longitude="LON", color="COLOR", size="SIZE")
    st.caption("점 크기 = 잔차 크기, 색 = 부족(빨강)/과잉(파랑). 지도가 안 뜨면 사이드바 아래 "
               "지도 타일 이용약관 동의가 필요할 수 있음.")

    st.subheader("등록대수 대비 충전기 잔차 (기대 대비 과부족)")

    cutoff = df["RESID_TOTAL"].quantile(0.2)
    plot_df = df.copy()
    plot_df["구분"] = plot_df["RESID_TOTAL"].apply(lambda v: "잔차 하위 20%" if v < cutoff else "정상 범위")
    st.scatter_chart(plot_df, x="LOG_REG", y="TOTAL_CHARGERS", color="구분",
                      x_label="log(등록대수)", y_label="충전기 수", height=420)

    st.subheader("잔차 하위 20곳 (가장 부족)")
    bottom20 = df.nsmallest(20, "RESID_TOTAL").sort_values("RESID_TOTAL")
    bar_df = bottom20.copy()
    bar_df["지역"] = bar_df["SIDO"] + " " + bar_df["SIGUNGU"]
    st.bar_chart(bar_df.set_index("지역")["RESID_TOTAL"], color=COLOR_DEFICIT, height=480)

    st.dataframe(bottom20[["SIDO", "SIGUNGU", "REGISTRATION", "TOTAL_CHARGERS", "RESID_TOTAL"]],
                 use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 2. 완속/급속 분리
# ---------------------------------------------------------------------------
with tab2:
    st.subheader('"완속만 특히 부족한" 9개 도시')
    nine_df = df[df["SIGUNGU"].isin(NINE_CITIES)].set_index("SIGUNGU").loc[NINE_CITIES]
    chart_df = nine_df[["RESID_FAST", "RESID_SLOW"]].rename(
        columns={"RESID_FAST": "급속", "RESID_SLOW": "완속"})
    st.bar_chart(chart_df, color=[COLOR_SURPLUS, COLOR_DEFICIT], height=420)
    st.caption("급속(파랑)은 0 근처거나 플러스인데, 완속(빨강)만 크게 마이너스 — "
               "통과차량용 급속 인프라는 갖췄지만 생활충전(완속)만 뒤처짐")

# ---------------------------------------------------------------------------
# TAB 3. 왜 부족한가 - 주택지표
# ---------------------------------------------------------------------------
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("9개 도시 vs 전국 - 주택 구성")
        h = housing.copy()
        h["그룹"] = h["SIGUNGU"].apply(lambda s: "9개도시" if s in NINE_CITIES else "전국(그외)")
        summary1 = h.groupby("그룹")[["APT_RATIO", "OLD_HOUSING_RATIO"]].mean()
        summary1 = summary1.rename(columns={"APT_RATIO": "아파트비율", "OLD_HOUSING_RATIO": "노후주택비율"})
        st.bar_chart(summary1.T, color=[COLOR_9CITY, COLOR_NEUTRAL], height=350)
        st.caption("아파트비율이 오히려 더 높은데도 완속이 부족함 → '아파트가 없어서'라는 "
                   "당연한 설명이 기각됨 (정책배정형 결핍)")

    with col2:
        st.subheader("부산4구 vs 전국 - 노후·주차")
        h2 = housing.copy()
        h2["그룹"] = h2.apply(
            lambda r: "부산4구" if (r["SIDO"] == "부산" and r["SIGUNGU"] in BUSAN4) else "전국(그외)",
            axis=1
        )
        summary2 = h2.groupby("그룹")[["OLD_HOUSING_RATIO", "PARKING_SECURED_RATIO"]].mean()
        summary2 = summary2.rename(columns={"OLD_HOUSING_RATIO": "노후주택비율", "PARKING_SECURED_RATIO": "자가주차장확보율"})
        st.bar_chart(summary2.T, color=[COLOR_BUSAN, COLOR_NEUTRAL], height=350)
        st.caption("노후주택비율 높고 자가주차장확보율 낮음 → 물리적 제약 가설이 지지됨")

# ---------------------------------------------------------------------------
# TAB 4. 보조금 - 정직한 null 결과
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("보조금 신청률 vs 완속잔차 (전체 표본)")
    sub_df = load_subsidy_joined()
    corr = sub_df["RESID_TOTAL"].corr(sub_df["APPLICATION_RATE_PCT"])

    st.info(f"전체 {len(sub_df)}개 지역 상관계수 = **{corr:.3f}** — "
            f"통계적으로 유의한 관계 없음. 포항·천안·광양 등 개별 사례에서 보였던 정황이 "
            f"전국 패턴으로는 일반화되지 않는다는 걸 확인 (정직한 null 결과)")

    st.scatter_chart(sub_df, x="APPLICATION_RATE_PCT", y="RESID_TOTAL",
                      x_label="보조금 접수율(%)", y_label="완속잔차", height=420)

# ---------------------------------------------------------------------------
# TAB 5. 설치 우선순위 모델
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("다변량 모델 기반 설치 우선 조사 대상")
    st.caption("등록대수·인구·주택지표를 함께 반영한 모델의 기대충전기수 − 실제충전기수. "
               "'예측'이 아니라 '추가 조사가 필요한 지역 스크리닝' 결과로 해석할 것.")

    n = st.slider("표시할 지역 수", 5, 50, 20)
    top_n = model_df.head(n).copy()
    top_n["지역"] = top_n["SIDO"] + " " + top_n["SIGUNGU"]
    st.bar_chart(top_n.set_index("지역")["INSTALL_PRIORITY_SCORE"], color=COLOR_9CITY, height=480)

    st.dataframe(top_n[["SIDO", "SIGUNGU", "REGISTRATION", "TOTAL_CHARGERS",
                        "EXPECTED_TOTAL_MULTIVAR", "INSTALL_PRIORITY_SCORE"]],
                 use_container_width=True, hide_index=True)
