import yfinance as yf
import pandas as pd
import ta
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# Matplotlib 다크 스타일
plt.style.use("dark_background")

# CustomTkinter 전역 설정
ctk.set_appearance_mode("dark")          # 'system', 'light', 'dark'
ctk.set_default_color_theme("green")     # 'blue', 'dark-blue', 'green' 등

# =========================
# 유틸 함수
# =========================
def format_market_cap(mc, currency="USD"):
    if mc is None or (isinstance(mc, float) and np.isnan(mc)):
        return "N/A"
    if currency in ("KRW", "원"):
        if mc >= 10**12:
            return f"{mc / 10**12:.2f}조"
        elif mc >= 10**8:
            return f"{mc / 10**8:.2f}억"
        else:
            return f"{mc:,.0f}원"
    else:
        if mc >= 10**9:
            return f"{mc / 10**9:.2f}B"
        elif mc >= 10**6:
            return f"{mc / 10**6:.2f}M"
        else:
            return f"{mc:,.0f}"

def format_price(value, currency="USD"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if currency in ("KRW", "원"):
        return f"{value:,.0f}원"
    return f"{value:,.2f}"

def infer_currency(info):
    c = info.get("currency")
    if not c:
        try:
            c = info.get("fast_info", {}).get("currency")
        except Exception:
            c = None
    return c or "USD"

# =========================
# 1. 데이터 수집
# =========================
def fetch_data(ticker: str, years: int = 3):
    end = datetime.today()
    start = end - timedelta(days=365 * years)
    data = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False)

    if data is None or data.empty:
        raise ValueError(f"{ticker} : 야후 파이낸스에서 데이터를 가져올 수 없습니다.")

    # 멀티인덱스 컬럼이면 바깥 레벨만 사용
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    t = yf.Ticker(ticker)
    try:
        info = t.info
    except Exception:
        info = {}
    return data, info, t

# =========================
# 2. 실적 일정 (yfinance만 사용)
# =========================
def fetch_earnings_info_free(yf_ticker_obj):
    """
    yfinance Ticker 객체에서 calendar/earnings 정보를 최대한 안전하게 뽑아
    '실적 일정' 한 줄로 요약해서 반환.
    """
    try:
        cal = yf_ticker_obj.calendar

        # calendar가 dict로 들어오는 경우도 방어
        if isinstance(cal, dict):
            # 예: {'Earnings Date': [Timestamp(...), ...]}
            ed = cal.get("Earnings Date") or cal.get("Earnings date") \
                 or cal.get("earningsDate") or cal.get("Earnings")
            if isinstance(ed, (list, tuple)) and ed:
                date_text = str(ed[0])[:10]
                return f"다음/최근 실적 일자(야후 캘린더 기준): {date_text}"
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            # DataFrame 형태인 경우
            date_text = None
            for idx in cal.index:
                if "Earnings" in str(idx):
                    val = cal.loc[idx].iloc[0]
                    date_text = str(val)[:10]
                    break
            if date_text:
                return f"다음/최근 실적 일자(야후 캘린더 기준): {date_text}"

        # fallback: quarterly_earnings 마지막 날짜
        qe = getattr(yf_ticker_obj, "quarterly_earnings", None)
        if isinstance(qe, pd.DataFrame) and not qe.empty:
            last_q_date = str(qe.index[-1])[:10]
            return f"최근 분기 실적 발표 일자(추정): {last_q_date}"

        return "실적 일정: yfinance에서 정보를 찾기 어렵습니다."
    except Exception as e:
        return f"실적 일정: 오류 ({e})"

# =========================
# 3. 시장 지수 상황 (3개월 수익률)
# =========================
def fetch_market_context():
    indices = {
        "S&P500": "^GSPC",
        "NASDAQ": "^IXIC",
        "KOSPI": "^KS11",
    }
    end = datetime.today()
    start = end - timedelta(days=120)  # 여유 있게 4개월
    lines = []
    for name, code in indices.items():
        try:
            df = yf.download(code, start=start, end=end, interval="1d")
            # 최소 30개 이상 데이터가 있어야 3개월 대략 계산 가능
            if df is None or df.empty or len(df) < 30:
                continue
            close = df["Close"].iloc[-1]
            base = df["Close"].iloc[0]
            if base == 0 or np.isnan(base) or np.isnan(close):
                continue
            ret_3m = (close / base - 1) * 100
            lines.append(f"{name} 3개월 수익률(대략): {ret_3m:.1f}%")
        except Exception:
            continue

    if not lines:
        lines.append("시장 지수: 야후 데이터 수집에 실패했습니다.")
    return lines

# =========================
# 4. 분석 로직 (차트/트레이딩)
# =========================
def analyze_stock(data: pd.DataFrame, info: dict, hold_days: int = 10):
    df = data.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=1).mean()
    df["MA200"] = df["Close"].rolling(200, min_periods=1).mean()

    required = {"High", "Low", "Close"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"필수 컬럼(High, Low, Close)이 없습니다. 현재 컬럼: {df.columns.tolist()}"
        )

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close_series = df["Close"].astype(float)

    atr_indicator = ta.volatility.AverageTrueRange(
        high=high,
        low=low,
        close=close_series,
        window=14,
        fillna=True
    )
    df["ATR14"] = atr_indicator.average_true_range()

    last = df.iloc[-1]
    close = float(last["Close"])
    ma20 = float(last["MA20"])
    ma50 = float(last["MA50"])
    ma200 = float(last["MA200"])
    atr14 = float(last["ATR14"])

    if atr14 == 0 or np.isnan(atr14):
        atr14 = close * 0.01

    def pct_change_days(n):
        if len(df) <= n:
            return None
        base = df["Close"].iloc[-n]
        if base == 0 or np.isnan(base):
            return None
        return (close / base - 1) * 100

    r_3y = pct_change_days(252 * 3)
    r_1y = pct_change_days(252)
    r_3m = pct_change_days(63)
    r_1m = pct_change_days(21)

    last_252 = df.tail(min(252, len(df)))
    high_52w = float(last_252["High"].max())
    low_52w = float(last_252["Low"].min())

    if not np.isnan(ma200) and not np.isnan(ma50):
        if close > ma200 and ma50 > ma200:
            long_trend = "장기 우상향 (강한 추세)"
        elif close > ma200 and ma50 <= ma200:
            long_trend = "장기 상승 추세지만 중기 조정"
        elif close < ma200 and ma50 < ma200:
            long_trend = "장기 하락 또는 약세 구간"
        else:
            long_trend = "추세 애매 (횡보/변곡)"
    else:
        long_trend = "데이터 부족으로 추세 판단이 애매함"

    pos_vs_ma = []
    if not np.isnan(ma20):
        pos_vs_ma.append(f"현재가는 20일선 대비 {'위' if close > ma20 else '아래'}")
    if not np.isnan(ma50):
        pos_vs_ma.append(f"50일선 대비 {'위' if close > ma50 else '아래'}")
    if not np.isnan(ma200):
        pos_vs_ma.append(f"200일선 대비 {'위' if close > ma200 else '아래'}")
    if not pos_vs_ma:
        pos_vs_ma.append("이동평균 계산에 필요한 데이터가 부족함")

    if hold_days <= 5:
        tp_mult = 1.5
        sl_mult = 1.0
    elif hold_days <= 10:
        tp_mult = 2.0
        sl_mult = 1.3
    else:
        tp_mult = 2.5
        sl_mult = 1.5

    entry_low = close - 1.0 * atr14
    entry_high = close - 0.5 * atr14
    stop_loss = close - sl_mult * atr14
    target_1 = close + tp_mult * atr14
    target_2 = close + (tp_mult + 1.0) * atr14

    currency = infer_currency(info)
    long_name = info.get("longName", "") or info.get("shortName", "")
    sector = info.get("sector", "") or info.get("industry", "")
    market_cap = info.get("marketCap", None)
    trailing_pe = info.get("trailingPE", None)

    def pct(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return f"{v:.1f}%"

    lines = []
    lines.append(f"[기본 정보]")
    lines.append(f"이름: {long_name or 'N/A'}")
    lines.append(f"섹터/업종: {sector or 'N/A'}")
    lines.append(f"통화: {currency}")
    if market_cap:
        lines.append(f"시가총액: {format_market_cap(market_cap, currency)}")
    if trailing_pe and not np.isnan(trailing_pe):
        lines.append(f"PER(추정): {trailing_pe:.1f}")
    lines.append("")

    lines.append("[현재 가격 및 추세]")
    lines.append(f"현재가: {format_price(close, currency)}")
    lines.append(
        "20일선 / 50일선 / 200일선: "
        f"{format_price(ma20, currency)} / "
        f"{format_price(ma50, currency)} / "
        f"{format_price(ma200, currency)}"
    )
    lines.append(
        f"52주 고가 / 저가: "
        f"{format_price(high_52w, currency)} / "
        f"{format_price(low_52w, currency)}"
    )
    if r_3y is not None:
        lines.append(f"3년 수익률(대략): {pct(r_3y)}")
    if r_1y is not None:
        lines.append(f"1년 수익률(대략): {pct(r_1y)}")
    if r_3m is not None:
        lines.append(f"3개월 수익률(대략): {pct(r_3m)}")
    if r_1m is not None:
        lines.append(f"1개월 수익률(대략): {pct(r_1m)}")
    lines.append(f"장기 추세 판단: {long_trend}")
    lines.append(" · " + ", ".join(pos_vs_ma))
    lines.append("")

    lines.append("[단기 변동성]")
    lines.append(
        f"14일 ATR: {format_price(atr14, currency)} "
        "(대략 일일 평균 변동폭)"
    )
    lines.append(f"단기 트레이딩 가정 보유 기간: {hold_days} 거래일")
    lines.append("")

    entry_low_pct = (entry_low / close - 1) * 100
    entry_high_pct = (entry_high / close - 1) * 100
    stop_loss_pct = (stop_loss / close - 1) * 100
    target_1_pct = (target_1 / close - 1) * 100
    target_2_pct = (target_2 / close - 1) * 100

    lines.append("[단기 트레이딩 레벨 (예시)]")
    lines.append(
        f"- 진입 구간(눌림 매수): "
        f"{format_price(entry_low, currency)} ({entry_low_pct:.1f}%) ~ "
        f"{format_price(entry_high, currency)} ({entry_high_pct:.1f}%)"
    )
    lines.append(
        f"- 손절 라인: {format_price(stop_loss, currency)} "
        f"({stop_loss_pct:.1f}%)"
    )
    lines.append(
        f"- 1차 목표가 (~{tp_mult:.1f} * ATR): "
        f"{format_price(target_1, currency)} ({target_1_pct:.1f}%)"
    )
    lines.append(
        f"- 2차 목표가 (~{tp_mult+1:.1f} * ATR): "
        f"{format_price(target_2, currency)} ({target_2_pct:.1f}%)"
    )
    lines.append("")
    lines.append("※ 위 수치는 과거 변동성을 단순 적용한 참고 값이며,")
    lines.append("   실제 매매 시에는 실적 일정/뉴스/전체 시장 상황을 반드시 함께 고려해야 합니다.")

    summary = {
        "이름": long_name or "N/A",
        "현재가": format_price(close, currency),
        "통화": currency,
        "시가총액": format_market_cap(market_cap, currency) if market_cap else "N/A",
        "52주고가": format_price(high_52w, currency),
        "52주저가": format_price(low_52w, currency),
        "3개월 수익률": pct(r_3m),
        "단기진입하단": format_price(entry_low, currency),
        "단기진입상단": format_price(entry_high, currency),
        "손절": format_price(stop_loss, currency),
        "1차목표": format_price(target_1, currency),
        "2차목표": format_price(target_2, currency),
    }

    return "\n".join(lines), df, summary

# =========================
# 5. CustomTkinter GUI
# =========================
def create_gui():
    app = ctk.CTk()
    app.title("단기 트레이딩 분석 도구 (미·한 공통)")
    app.geometry("1250x750")

    base_font = ("Segoe UI", 11)

    # 상단 입력 영역
    top_frame = ctk.CTkFrame(app, corner_radius=15)
    top_frame.pack(side="top", fill="x", padx=10, pady=10)

    ctk.CTkLabel(top_frame, text="티커/코드", font=("Segoe UI Semibold", 12)).pack(
        side="left", padx=(10, 5)
    )
    ticker_var = tk.StringVar()
    ctk.CTkEntry(
        top_frame,
        textvariable=ticker_var,
        width=140,
        font=base_font,
        corner_radius=10
    ).pack(side="left", padx=5)

    ctk.CTkLabel(top_frame, text="단기 보유일 (5~20일)", font=("Segoe UI Semibold", 12)).pack(
        side="left", padx=(20, 5)
    )
    hold_days_var = tk.IntVar(value=10)
    hold_slider = ctk.CTkSlider(
        top_frame,
        from_=5,
        to=20,
        number_of_steps=15,
        variable=hold_days_var,
        width=200
    )
    hold_slider.pack(side="left", padx=5)

    hold_value_label = ctk.CTkLabel(top_frame, text="10일", font=base_font)
    hold_value_label.pack(side="left", padx=5)

    def update_hold_label(value):
        hold_value_label.configure(text=f"{int(float(value))}일")

    hold_slider.configure(command=update_hold_label)

    analyze_button = ctk.CTkButton(
        top_frame,
        text="분석 실행",
        font=("Segoe UI Semibold", 12),
        corner_radius=10,
        fg_color="#00b894",
        hover_color="#019170"
    )
    analyze_button.pack(side="right", padx=15)

    # 중앙 레이아웃
    main_frame = ctk.CTkFrame(app, corner_radius=15)
    main_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # 좌: 차트
    chart_frame = ctk.CTkFrame(main_frame, corner_radius=15)
    chart_frame.pack(side="left", fill="both", expand=True, padx=(10, 5), pady=10)

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#111111")
    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    # 우: 요약 + 텍스트
    right_frame = ctk.CTkFrame(main_frame, corner_radius=15)
    right_frame.pack(side="right", fill="both", expand=True, padx=(5, 10), pady=10)

    # 요약 테이블
    summary_label = ctk.CTkLabel(
        right_frame, text="요약 정보", font=("Segoe UI Semibold", 12)
    )
    summary_label.pack(side="top", anchor="w", padx=10, pady=(10, 5))

    columns = ("키", "값")
    style = ttk.Style()
    style.configure("Custom.Treeview",
                    font=("Segoe UI", 10),
                    rowheight=22)
    style.configure("Custom.Treeview.Heading",
                    font=("Segoe UI Semibold", 10))
    style.map("Custom.Treeview", background=[("selected", "#00b894")])

    tree = ttk.Treeview(
        right_frame,
        columns=columns,
        show="headings",
        height=6,
        style="Custom.Treeview"
    )
    tree.heading("키", text="항목")
    tree.heading("값", text="값")
    tree.column("키", width=120, anchor="w")
    tree.column("값", width=220, anchor="w")
    tree.pack(side="top", fill="x", padx=10, pady=(0, 10))

    # 상세 분석 + 실적/시장 텍스트
    text_label = ctk.CTkLabel(
        right_frame, text="상세 분석 / 실적 / 시장", font=("Segoe UI Semibold", 12)
    )
    text_label.pack(side="top", anchor="w", padx=10, pady=(0, 5))

    text_frame = ctk.CTkFrame(right_frame, corner_radius=10)
    text_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 10))

    text_box = tk.Text(
        text_frame,
        bg="#111111",
        fg="#eeeeee",
        insertbackground="#ffffff",
        font=("Segoe UI", 10),
        wrap="word"
    )
    text_box.pack(side="left", fill="both", expand=True, padx=(5, 0), pady=5)

    text_scroll = ctk.CTkScrollbar(text_frame, command=text_box.yview)
    text_scroll.pack(side="right", fill="y", padx=(0, 5), pady=5)
    text_box.configure(yscrollcommand=text_scroll.set)

    def run_analysis():
        ticker = ticker_var.get().strip()
        text_box.delete("1.0", "end")
        for row in tree.get_children():
            tree.delete(row)

        if not ticker:
            text_box.insert("end", "티커/코드를 입력하세요.\n예: ANET, NVDA, 005930.KS")
            return

        hold_days = int(hold_days_var.get())
        try:
            data, info, t_obj = fetch_data(ticker, years=3)
            report, df, summary = analyze_stock(data, info, hold_days=hold_days)

            # 차트
            ax.clear()
            ax.set_facecolor("#111111")
            ax.plot(df.index, df["Close"], label="Close", color="#00ff88", linewidth=1.5)
            if "MA50" in df.columns and not df["MA50"].isna().all():
                ax.plot(df.index, df["MA50"], label="MA50", color="#61afef", alpha=0.9)
            if "MA200" in df.columns and not df["MA200"].isna().all():
                ax.plot(df.index, df["MA200"], label="MA200", color="#e06c75", alpha=0.9)

            ax.set_title(f"{ticker} 가격 차트 (3년)", color="#ffffff", fontsize=11)
            ax.set_xlabel("Date", color="#cccccc")
            ax.set_ylabel("Price", color="#cccccc")
            ax.tick_params(colors="#cccccc")
            for spine in ax.spines.values():
                spine.set_color("#555555")
            ax.legend(facecolor="#111111", edgecolor="#555555", labelcolor="#ffffff")
            ax.grid(True, alpha=0.25, color="#444444")
            fig.autofmt_xdate()
            canvas.draw()

            # 요약 테이블
            for k, v in summary.items():
                tree.insert("", "end", values=(k, v))

            # 상세 분석
            text_box.insert("end", report + "\n\n")

            # 실적 일정
            earnings_text = fetch_earnings_info_free(t_obj)
            text_box.insert("end", "[실적 일정 참고]\n")
            text_box.insert("end", earnings_text + "\n\n")

            # 시장 지수 상황
            market_lines = fetch_market_context()
            text_box.insert("end", "[시장 지수 3개월 흐름]\n")
            for line in market_lines:
                text_box.insert("end", line + "\n")

        except Exception as e:
            text_box.insert("end", f"에러 발생:\n{repr(e)}")

    analyze_button.configure(command=run_analysis)
    app.mainloop()

if __name__ == "__main__":
    create_gui()
