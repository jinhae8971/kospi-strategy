"""
run_strategy.py — 내일 코스피 대응전략 파이프라인
================================================
실행 순서:
  1. 코스피/코스닥 시장 데이터 수집 (yfinance + web scraping)
  2. 글로벌 매크로 지표 수집
  3. Claude AI 분석 → 내일 대응전략 생성
  4. HTML 보고서 생성 (GitHub Pages)
  5. JSON 데이터 저장 (대시보드 연동)
  6. Telegram 알림 발송

스케줄: 일~목 KST 21:00 (다음 거래일 대응전략)
"""
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ─── 로깅 ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))
NOW_KST = datetime.now(KST)
TODAY_STR = NOW_KST.strftime("%Y-%m-%d")

# 다음 거래일 계산
def next_trading_day(dt):
    """다음 거래일 (토/일 건너뛰기)"""
    nd = dt + timedelta(days=1)
    while nd.weekday() >= 5:  # 5=토, 6=일
        nd += timedelta(days=1)
    return nd

NEXT_DAY = next_trading_day(NOW_KST)
NEXT_DAY_STR = NEXT_DAY.strftime("%Y-%m-%d")
NEXT_DAY_KR  = NEXT_DAY.strftime("%m월 %d일 (%a)")

DOCS_DIR     = ROOT / "docs" / "data"
REPORT_JSON  = DOCS_DIR / "daily_report.json"
REPORTS_DIR  = ROOT / "reports"
HISTORY_DIR  = ROOT / "data" / "history"

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")


# ─── 설정 ─────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    cfg = {
        "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "telegram_token":    os.environ.get("TELEGRAM_TOKEN", ""),
        "telegram_chat_id":  os.environ.get("TELEGRAM_CHAT_ID", ""),
    }
    config_path = ROOT / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            for k, v in json.load(f).items():
                if not cfg.get(k):
                    cfg[k] = v
    if not cfg["anthropic_api_key"]:
        raise ValueError("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
    return cfg


# ─── 시장 데이터 수집 ─────────────────────────────────────────────────────────

def collect_market_data() -> dict:
    """yfinance로 시장 데이터 수집"""
    import yfinance as yf
    import numpy as np

    logger.info("시장 데이터 수집 시작...")

    tickers = {
        "KOSPI":    "^KS11",
        "KOSDAQ":   "^KQ11",
        "S&P500":   "^GSPC",
        "NASDAQ":   "^IXIC",
        "DOW":      "^DJI",
        "NIKKEI":   "^N225",
        "VIX":      "^VIX",
        "DXY":      "DX-Y.NYB",
        "WTI":      "CL=F",
        "GOLD":     "GC=F",
        "USD_KRW":  "KRW=X",
        "US10Y":    "^TNX",
        "BTC":      "BTC-USD",
    }

    market = {}
    for name, ticker in tickers.items():
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if len(hist) >= 2:
                close  = float(hist["Close"].iloc[-1])
                prev   = float(hist["Close"].iloc[-2])
                change = ((close - prev) / prev) * 100
                high5d = float(hist["High"].max())
                low5d  = float(hist["Low"].min())
                vol    = int(hist["Volume"].iloc[-1]) if "Volume" in hist else 0
                market[name] = {
                    "close": round(close, 2),
                    "change_pct": round(change, 2),
                    "high_5d": round(high5d, 2),
                    "low_5d": round(low5d, 2),
                    "volume": vol,
                }
            elif len(hist) == 1:
                market[name] = {
                    "close": round(float(hist["Close"].iloc[-1]), 2),
                    "change_pct": 0.0,
                }
            logger.info(f"  ✅ {name}: {market.get(name, {}).get('close', '?')}")
        except Exception as e:
            logger.warning(f"  ⚠️ {name}: {e}")
            market[name] = {"close": 0, "change_pct": 0, "error": str(e)}

    # KOSPI 추가 지표: 이동평균선
    try:
        kospi_hist = yf.Ticker("^KS11").history(period="120d")
        if len(kospi_hist) >= 20:
            closes = kospi_hist["Close"]
            market["KOSPI"]["ma5"]   = round(float(closes.rolling(5).mean().iloc[-1]), 2)
            market["KOSPI"]["ma20"]  = round(float(closes.rolling(20).mean().iloc[-1]), 2)
            market["KOSPI"]["ma60"]  = round(float(closes.rolling(60).mean().iloc[-1]), 2) if len(closes) >= 60 else None
            market["KOSPI"]["ma120"] = round(float(closes.rolling(120).mean().iloc[-1]), 2) if len(closes) >= 120 else None
            # RSI 14
            delta = closes.diff()
            gain  = delta.where(delta > 0, 0).rolling(14).mean()
            loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
            market["KOSPI"]["rsi14"] = round(100 - (100 / (1 + rs)), 1)
    except Exception as e:
        logger.warning(f"  ⚠️ KOSPI 기술 지표 수집 실패: {e}")

    # 한국 ETF 섹터별 데이터
    sector_etfs = {
        "반도체": "091160.KS",
        "2차전지": "305720.KS",
        "바이오": "244580.KS",
        "자동차": "091170.KS",
        "금융": "091180.KS",
        "철강": "117680.KS",
        "건설": "117700.KS",
    }
    sectors = {}
    for sname, sticker in sector_etfs.items():
        try:
            sh = yf.Ticker(sticker).history(period="5d")
            if len(sh) >= 2:
                sc = float(sh["Close"].iloc[-1])
                sp = float(sh["Close"].iloc[-2])
                sectors[sname] = {
                    "close": round(sc, 0),
                    "change_pct": round(((sc - sp) / sp) * 100, 2),
                }
        except:
            pass
    market["sectors"] = sectors

    logger.info(f"시장 데이터 수집 완료: {len(market)}개 항목")
    return market


# ─── Claude AI 분석 ───────────────────────────────────────────────────────────

def analyze_with_claude(market_data: dict, api_key: str) -> dict:
    """Claude API로 내일 코스피 대응전략 분석"""
    import anthropic

    logger.info("Claude AI 분석 시작...")

    # 시장 데이터 포맷팅
    market_summary = json.dumps(market_data, ensure_ascii=False, indent=2)

    prompt = f"""당신은 20년 경력의 한국 증시 전문 애널리스트입니다.
아래 실시간 시장 데이터를 기반으로 **{NEXT_DAY_KR}(다음 거래일) 코스피 대응전략**을 작성해주세요.

## 오늘({TODAY_STR}) 시장 데이터
{market_summary}

## 분석 요청사항
아래 JSON 형식으로만 응답해주세요. 다른 텍스트는 절대 포함하지 마세요.

{{
  "market_overview": {{
    "kospi_close": "코스피 종가",
    "kospi_change": "등락률",
    "market_sentiment": "매우긍정/긍정/중립/부정/매우부정 중 택1",
    "sentiment_score": 0~100 (0=극도의 공포, 100=극도의 탐욕)
  }},
  "key_factors": [
    {{
      "factor": "핵심 요인명",
      "impact": "긍정/부정/중립",
      "detail": "상세 설명 (2~3문장)"
    }}
  ],
  "tomorrow_strategy": {{
    "stance": "적극매수/매수/관망/매도/적극매도 중 택1",
    "confidence": 0~100,
    "kospi_range": {{
      "support": "예상 지지선",
      "resistance": "예상 저항선"
    }},
    "summary": "내일 전략 핵심 요약 (3~5문장)",
    "action_items": [
      "구체적 실행 항목 1",
      "구체적 실행 항목 2",
      "구체적 실행 항목 3"
    ]
  }},
  "sector_analysis": [
    {{
      "sector": "섹터명",
      "outlook": "긍정/중립/부정",
      "reason": "이유 (1~2문장)",
      "top_pick": "대표 종목명"
    }}
  ],
  "risk_factors": [
    {{
      "risk": "리스크명",
      "probability": "높음/중간/낮음",
      "detail": "설명"
    }}
  ],
  "global_impact": {{
    "us_market": "미국 시장 영향 요약",
    "fx_outlook": "환율 전망",
    "commodity": "원자재/유가 전망"
  }},
  "technical_analysis": {{
    "trend": "상승/횡보/하락",
    "support_levels": ["지지선1", "지지선2"],
    "resistance_levels": ["저항선1", "저항선2"],
    "key_indicator": "핵심 기술적 시그널 설명"
  }}
}}"""

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    # JSON 추출 (```json ... ``` 래핑 제거)
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    analysis = json.loads(raw)
    logger.info("Claude AI 분석 완료")
    return analysis


# ─── HTML 보고서 생성 ─────────────────────────────────────────────────────────

def generate_html_report(analysis: dict, market_data: dict) -> str:
    """GitHub Pages용 HTML 보고서 생성"""

    stance = analysis.get("tomorrow_strategy", {}).get("stance", "관망")
    stance_colors = {
        "적극매수": ("#059669", "#ecfdf5", "🟢"),
        "매수":     ("#10b981", "#ecfdf5", "🟢"),
        "관망":     ("#f59e0b", "#fffbeb", "🟡"),
        "매도":     ("#ef4444", "#fef2f2", "🔴"),
        "적극매도": ("#dc2626", "#fef2f2", "🔴"),
    }
    sc = stance_colors.get(stance, stance_colors["관망"])
    confidence = analysis.get("tomorrow_strategy", {}).get("confidence", 50)
    sentiment  = analysis.get("market_overview", {}).get("sentiment_score", 50)

    # f-string 안에서 dict 리터럴 {{}} 충돌 방지: 미리 추출
    kospi_data = market_data.get("KOSPI", {})
    kospi_close = kospi_data.get("close", "—")
    kospi_change = kospi_data.get("change_pct", 0)
    kospi_change_color = "#059669" if kospi_change >= 0 else "#ef4444"
    kospi_change_sign = "+" if kospi_change >= 0 else ""
    market_sentiment_str = analysis.get("market_overview", {}).get("market_sentiment", "—")

    # Key factors HTML
    factors_html = ""
    for f in analysis.get("key_factors", []):
        impact_color = {"긍정": "#059669", "부정": "#ef4444", "중립": "#6b7280"}.get(f.get("impact", "중립"), "#6b7280")
        impact_bg    = {"긍정": "#ecfdf5", "부정": "#fef2f2", "중립": "#f3f4f6"}.get(f.get("impact", "중립"), "#f3f4f6")
        factors_html += f'''
        <div style="display:flex;gap:12px;align-items:flex-start;padding:14px 16px;background:#fff;border:1px solid #e5e7eb;border-radius:10px;">
          <span style="display:inline-block;padding:3px 10px;border-radius:6px;font-size:12px;font-weight:600;background:{impact_bg};color:{impact_color};white-space:nowrap;">{f.get("impact","")}</span>
          <div>
            <div style="font-weight:600;color:#111827;margin-bottom:4px;">{f.get("factor","")}</div>
            <div style="font-size:13px;color:#6b7280;line-height:1.6;">{f.get("detail","")}</div>
          </div>
        </div>'''

    # Sector analysis HTML
    sectors_html = ""
    for s in analysis.get("sector_analysis", []):
        ol_color = {"긍정": "#059669", "부정": "#ef4444", "중립": "#6b7280"}.get(s.get("outlook", "중립"), "#6b7280")
        ol_emoji = {"긍정": "📈", "부정": "📉", "중립": "➡️"}.get(s.get("outlook", "중립"), "➡️")
        sectors_html += f'''
        <div style="padding:14px 16px;background:#fff;border:1px solid #e5e7eb;border-radius:10px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-weight:700;color:#111827;">{ol_emoji} {s.get("sector","")}</span>
            <span style="font-size:12px;font-weight:600;color:{ol_color};">{s.get("outlook","")}</span>
          </div>
          <div style="font-size:13px;color:#6b7280;line-height:1.5;">{s.get("reason","")}</div>
          <div style="font-size:12px;color:#3b82f6;margin-top:6px;">대표종목: {s.get("top_pick","")}</div>
        </div>'''

    # Risk factors HTML
    risks_html = ""
    for r in analysis.get("risk_factors", []):
        prob_color = {"높음": "#ef4444", "중간": "#f59e0b", "낮음": "#6b7280"}.get(r.get("probability", "중간"), "#6b7280")
        risks_html += f'''
        <div style="display:flex;gap:10px;align-items:flex-start;padding:12px 14px;background:#fff;border:1px solid #e5e7eb;border-radius:10px;">
          <span style="font-size:12px;font-weight:600;color:{prob_color};white-space:nowrap;">⚠️ {r.get("probability","")}</span>
          <div>
            <div style="font-weight:600;color:#111827;font-size:14px;">{r.get("risk","")}</div>
            <div style="font-size:13px;color:#6b7280;margin-top:2px;">{r.get("detail","")}</div>
          </div>
        </div>'''

    # Action items HTML
    actions_html = ""
    for i, item in enumerate(analysis.get("tomorrow_strategy", {}).get("action_items", []), 1):
        actions_html += f'<div style="display:flex;gap:10px;align-items:flex-start;padding:10px 14px;background:#f0fdf4;border-radius:8px;"><span style="font-weight:700;color:#059669;font-size:14px;">{i}</span><span style="color:#111827;font-size:14px;line-height:1.5;">{item}</span></div>'

    # Technical analysis
    ta = analysis.get("technical_analysis", {})
    gl = analysis.get("global_impact", {})
    ts = analysis.get("tomorrow_strategy", {})
    kospi_range = ts.get("kospi_range", {})

    html = f'''<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>내일 코스피 대응전략 — {NEXT_DAY_STR}</title>
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; font-family:'Noto Sans KR',sans-serif; }}
    body {{ background:#f8fafc; color:#111827; line-height:1.6; }}
    .container {{ max-width:760px; margin:0 auto; padding:24px 20px 60px; }}
    .mono {{ font-family:'JetBrains Mono',monospace; }}
    h2 {{ font-size:17px; font-weight:700; color:#111827; margin:28px 0 14px; padding-bottom:8px; border-bottom:2px solid #e5e7eb; }}
    .badge {{ display:inline-block; padding:4px 12px; border-radius:6px; font-size:12px; font-weight:600; }}
  </style>
</head>
<body>
<div class="container">

  <!-- 헤더 -->
  <div style="text-align:center;padding:32px 0 24px;">
    <div style="font-size:12px;letter-spacing:3px;color:#6b7280;text-transform:uppercase;margin-bottom:8px;">KOSPI STRATEGY</div>
    <h1 style="font-size:26px;font-weight:900;color:#111827;margin-bottom:4px;">내일 코스피 대응전략</h1>
    <div style="font-size:15px;color:#6b7280;">{NEXT_DAY_KR} | 발행: {NOW_KST.strftime("%Y-%m-%d %H:%M")} KST</div>
  </div>

  <!-- 메인 판단 카드 -->
  <div style="background:{sc[1]};border:2px solid {sc[0]};border-radius:16px;padding:28px;text-align:center;margin-bottom:20px;">
    <div style="font-size:48px;margin-bottom:8px;">{sc[2]}</div>
    <div style="font-size:28px;font-weight:900;color:{sc[0]};margin-bottom:4px;">{stance}</div>
    <div style="font-size:14px;color:#6b7280;">확신도 {confidence}%</div>
    <div style="width:200px;margin:12px auto 0;height:8px;background:#e5e7eb;border-radius:4px;overflow:hidden;">
      <div style="height:100%;width:{confidence}%;background:{sc[0]};border-radius:4px;"></div>
    </div>
  </div>

  <!-- 시장 요약 4칸 -->
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:24px;">
    <div style="background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px;text-align:center;">
      <div style="font-size:11px;color:#6b7280;">KOSPI</div>
      <div class="mono" style="font-size:18px;font-weight:700;color:#111827;">{kospi_close}</div>
      <div class="mono" style="font-size:12px;color:{kospi_change_color};">{kospi_change_sign}{kospi_change}%</div>
    </div>
    <div style="background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px;text-align:center;">
      <div style="font-size:11px;color:#6b7280;">심리지수</div>
      <div class="mono" style="font-size:18px;font-weight:700;color:#111827;">{sentiment}</div>
      <div style="font-size:12px;color:#6b7280;">{market_sentiment_str}</div>
    </div>
    <div style="background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px;text-align:center;">
      <div style="font-size:11px;color:#6b7280;">예상 지지</div>
      <div class="mono" style="font-size:18px;font-weight:700;color:#3b82f6;">{kospi_range.get("support","—")}</div>
    </div>
    <div style="background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px;text-align:center;">
      <div style="font-size:11px;color:#6b7280;">예상 저항</div>
      <div class="mono" style="font-size:18px;font-weight:700;color:#ef4444;">{kospi_range.get("resistance","—")}</div>
    </div>
  </div>

  <!-- 전략 요약 -->
  <div style="background:linear-gradient(135deg,#eff6ff,#f0fdf4);border:1px solid #bfdbfe;border-radius:12px;padding:20px;margin-bottom:24px;">
    <div style="font-size:13px;font-weight:700;color:#1d4ed8;margin-bottom:8px;">📌 전략 요약</div>
    <p style="font-size:14px;color:#1e3a5f;line-height:1.8;">{ts.get("summary","")}</p>
  </div>

  <!-- 실행 항목 -->
  <h2>🎯 실행 항목</h2>
  <div style="display:flex;flex-direction:column;gap:8px;margin-bottom:8px;">
    {actions_html}
  </div>

  <!-- 핵심 요인 -->
  <h2>📊 핵심 요인 분석</h2>
  <div style="display:flex;flex-direction:column;gap:10px;margin-bottom:8px;">
    {factors_html}
  </div>

  <!-- 섹터 분석 -->
  <h2>🏭 섹터별 전망</h2>
  <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:8px;">
    {sectors_html}
  </div>

  <!-- 글로벌 영향 -->
  <h2>🌍 글로벌 영향</h2>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:8px;">
    <div style="background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px;">
      <div style="font-size:12px;font-weight:600;color:#6b7280;margin-bottom:6px;">🇺🇸 미국 시장</div>
      <div style="font-size:13px;color:#111827;line-height:1.5;">{gl.get("us_market","")}</div>
    </div>
    <div style="background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px;">
      <div style="font-size:12px;font-weight:600;color:#6b7280;margin-bottom:6px;">💱 환율 전망</div>
      <div style="font-size:13px;color:#111827;line-height:1.5;">{gl.get("fx_outlook","")}</div>
    </div>
    <div style="background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px;">
      <div style="font-size:12px;font-weight:600;color:#6b7280;margin-bottom:6px;">🛢️ 원자재</div>
      <div style="font-size:13px;color:#111827;line-height:1.5;">{gl.get("commodity","")}</div>
    </div>
  </div>

  <!-- 기술적 분석 -->
  <h2>📐 기술적 분석</h2>
  <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:18px;margin-bottom:8px;">
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px;">
      <span class="badge" style="background:#f3f4f6;color:#111827;">추세: {ta.get("trend","—")}</span>
      <span class="badge" style="background:#eff6ff;color:#1d4ed8;">지지: {', '.join(ta.get("support_levels",[]))}</span>
      <span class="badge" style="background:#fef2f2;color:#ef4444;">저항: {', '.join(ta.get("resistance_levels",[]))}</span>
    </div>
    <div style="font-size:13px;color:#374151;line-height:1.6;">{ta.get("key_indicator","")}</div>
  </div>

  <!-- 리스크 요인 -->
  <h2>⚠️ 리스크 요인</h2>
  <div style="display:flex;flex-direction:column;gap:8px;margin-bottom:24px;">
    {risks_html}
  </div>

  <!-- 면책조항 -->
  <div style="text-align:center;padding:20px 0;border-top:1px solid #e5e7eb;margin-top:20px;">
    <div style="font-size:11px;color:#9ca3af;line-height:1.8;">
      본 보고서는 AI 분석 결과이며 투자 권유가 아닙니다. 투자 판단은 본인의 책임입니다.<br>
      Generated by KOSPI Strategy Agent | Powered by Claude AI
    </div>
  </div>

</div>

<script>
  // 자동 새로고침 (5분마다 데이터 확인)
  // setTimeout(() => location.reload(), 300000);
</script>
</body>
</html>'''
    return html


# ─── Telegram 알림 ────────────────────────────────────────────────────────────

def send_telegram(analysis: dict, token: str, chat_id: str):
    """텔레그램으로 전략 요약 + GitHub Pages 링크 발송"""
    ts = analysis.get("tomorrow_strategy", {})
    mo = analysis.get("market_overview", {})
    stance = ts.get("stance", "관망")
    stance_emoji = {
        "적극매수": "🟢🟢", "매수": "🟢", "관망": "🟡",
        "매도": "🔴", "적극매도": "🔴🔴",
    }.get(stance, "⚪")

    sectors = analysis.get("sector_analysis", [])
    sector_text = "\n".join(
        f"  {'📈' if s.get('outlook')=='긍정' else '📉' if s.get('outlook')=='부정' else '➡️'} {s.get('sector','')} → {s.get('top_pick','')}"
        for s in sectors[:5]
    )

    actions = ts.get("action_items", [])
    actions_text = "\n".join(f"  {i}. {a}" for i, a in enumerate(actions, 1))

    risks = analysis.get("risk_factors", [])
    risk_text = "\n".join(f"  ⚠️ {r.get('risk','')}" for r in risks[:3])

    kospi_range = ts.get("kospi_range", {})

    msg = (
        f"📊 <b>내일 코스피 대응전략</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📅 <b>{NEXT_DAY_KR}</b>\n\n"
        f"{stance_emoji} 전략: <b>{stance}</b> (확신도 {ts.get('confidence',50)}%)\n"
        f"📍 예상 범위: {kospi_range.get('support','?')} ~ {kospi_range.get('resistance','?')}\n\n"
        f"📌 <b>요약</b>\n{ts.get('summary','')[:400]}\n\n"
        f"🎯 <b>실행 항목</b>\n{actions_text}\n\n"
        f"🏭 <b>주목 섹터</b>\n{sector_text}\n\n"
        f"⚠️ <b>리스크</b>\n{risk_text}\n\n"
        f"📎 상세 보고서: https://jinhae8971.github.io/kospi-strategy/"
    )

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, json={
        "chat_id": chat_id,
        "text": msg,
        "parse_mode": "HTML",
    }, timeout=20)

    if r.status_code == 200:
        logger.info("✅ Telegram 발송 성공")
    else:
        logger.error(f"❌ Telegram 발송 실패: {r.status_code} {r.text}")


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    logger.info(f"{'='*55}")
    logger.info(f"  내일 코스피 대응전략 파이프라인")
    logger.info(f"  실행: {NOW_KST.strftime('%Y-%m-%d %H:%M KST')}")
    logger.info(f"  대상: {NEXT_DAY_KR}")
    logger.info(f"{'='*55}")

    # 설정
    cfg = load_config()

    # Step 1: 시장 데이터 수집
    market_data = collect_market_data()

    # Step 2: Claude AI 분석
    analysis = analyze_with_claude(market_data, cfg["anthropic_api_key"])

    # Step 3: HTML 보고서 생성
    html = generate_html_report(analysis, market_data)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    # docs/index.html 저장 (GitHub Pages)
    with open(ROOT / "docs" / "index.html", "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("✅ docs/index.html 저장 완료")

    # JSON 데이터 저장 (대시보드 연동)
    report_data = {
        "generated_at": NOW_KST.isoformat(),
        "target_date": NEXT_DAY_STR,
        "market_data": market_data,
        "analysis": analysis,
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    logger.info("✅ docs/data/daily_report.json 저장 완료")

    # reports/latest.json (대시보드 오케스트레이터용)
    with open(REPORTS_DIR / "latest.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    # 히스토리 저장
    with open(HISTORY_DIR / f"{TODAY_STR}.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    # Step 4: Telegram 발송
    if cfg.get("telegram_token") and cfg.get("telegram_chat_id"):
        send_telegram(analysis, cfg["telegram_token"], cfg["telegram_chat_id"])
    else:
        logger.warning("⚠️ Telegram 설정 없음 — 알림 건너뜀")

    logger.info(f"\n{'='*55}")
    logger.info(f"✅ 파이프라인 완료!")
    logger.info(f"   보고서: https://jinhae8971.github.io/kospi-strategy/")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"파이프라인 실패: {e}")
        traceback.print_exc()
        sys.exit(1)
