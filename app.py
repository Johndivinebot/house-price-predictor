import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="John's House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #f4f6f9; }

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 20px;
    padding: 40px 30px;
    text-align: center;
    margin-bottom: 30px;
}
.hero .tag {
    display: inline-block;
    background: #0f3460;
    color: #a8d8ea;
    font-size: 13px;
    font-weight: 600;
    padding: 5px 16px;
    border-radius: 20px;
    margin-bottom: 14px;
    letter-spacing: 1px;
}
.hero h1 {
    font-size: 42px;
    font-weight: 900;
    color: white;
    margin-bottom: 8px;
}
.hero h1 span { color: #e94560; }
.hero p { color: #a0aec0; font-size: 15px; margin-bottom: 16px; }
.accuracy {
    display: inline-block;
    background: rgba(104,211,145,0.1);
    border: 1px solid rgba(104,211,145,0.3);
    color: #68d391;
    padding: 8px 20px;
    border-radius: 30px;
    font-size: 14px;
    font-weight: 600;
}

.result-box {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    margin-top: 20px;
}
.result-label { color: #a0aec0; font-size: 13px; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px; }
.result-price { color: #68d391; font-size: 52px; font-weight: 900; margin-bottom: 8px; }
.result-sub { color: #718096; font-size: 14px; margin-bottom: 14px; }
.result-formula {
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 12px;
    font-family: monospace;
    font-size: 13px;
    color: #90cdf4;
    word-break: break-all;
}

.metric-row { display: flex; gap: 16px; margin-bottom: 24px; }
.metric {
    flex: 1;
    background: white;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}
.metric-val { font-size: 26px; font-weight: 800; color: #1a1a2e; }
.metric-lbl { font-size: 12px; color: #999; margin-top: 4px; }

.step-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    border-left: 4px solid #e94560;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    margin-bottom: 12px;
}
.step-num { font-size: 12px; font-weight: 700; color: #e94560; margin-bottom: 4px; }
.step-title { font-size: 16px; font-weight: 700; margin-bottom: 6px; }
.step-body { font-size: 14px; color: #555; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
data = {
    'size_sqft': [800,1000,1200,1500,1800,2000,2400,900,1100,1600,2200,1300,700,3000,2800],
    'num_rooms': [2,3,3,4,4,5,5,2,3,4,5,3,1,6,5],
    'location':  [3,4,5,6,7,8,9,2,3,5,7,6,2,9,8],
    'price':     [180000,240000,310000,390000,460000,550000,670000,
                  155000,210000,360000,520000,330000,130000,850000,720000],
}
df = pd.DataFrame(data)

# ── Train model ───────────────────────────────────────────────────────────────
X = df[['size_sqft', 'num_rooms', 'location']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
score = r2_score(y_test, y_pred_test)
coef = model.coef_
intercept = model.intercept_

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="tag">JOHN'S PROJECT</div>
  <h1>House Price <span>Predictor</span></h1>
  <p>Machine Learning · Linear Regression · Scikit-learn · Python</p>
  <div class="accuracy">Model Accuracy (R²): {score*100:.1f}%</div>
</div>
""", unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="metric-row">
  <div class="metric"><div class="metric-val">{score*100:.1f}%</div><div class="metric-lbl">R² Accuracy</div></div>
  <div class="metric"><div class="metric-val">15</div><div class="metric-lbl">Training Houses</div></div>
  <div class="metric"><div class="metric-val">3</div><div class="metric-lbl">Features Used</div></div>
</div>
""", unsafe_allow_html=True)

# ── Sliders ───────────────────────────────────────────────────────────────────
st.markdown("### 🏠 Enter House Features")

size = st.slider("House Size (sq ft)", 500, 4000, 1500, step=50)
rooms = st.slider("Number of Rooms", 1, 8, 3)
location = st.slider("Location Score (1 = rural · 10 = city center)", 1, 10, 5)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Price", use_container_width=True):
    predicted = model.predict([[size, rooms, location]])[0]
    formula = (f"price = ${intercept:,.0f} + {coef[0]:.1f}×size "
               f"+ {coef[1]:,.0f}×rooms + {coef[2]:,.0f}×location")
    st.markdown(f"""
    <div class="result-box">
      <div class="result-label">Predicted Price</div>
      <div class="result-price">${predicted:,.0f}</div>
      <div class="result-sub">{size:,} sq ft · {rooms} rooms · location {location}/10</div>
      <div class="result-formula">{formula}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Chart: Actual vs Predicted ────────────────────────────────────────────────
st.markdown("### 📊 Actual vs Predicted Prices")
st.caption("Blue = actual price · Green = model's prediction")

all_preds = model.predict(X)
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#f4f6f9')
ax.set_facecolor('#ffffff')

x_idx = np.arange(len(df))
width = 0.35
bars1 = ax.bar(x_idx - width/2, df['price'], width, color='#4299e1', alpha=0.85, label='Actual', zorder=3)
bars2 = ax.bar(x_idx + width/2, all_preds, width, color='#68d391', alpha=0.85, label='Predicted', zorder=3)

ax.set_xticks(x_idx)
ax.set_xticklabels([f"{s}sqft" for s in df['size_sqft']], rotation=45, ha='right', fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
ax.set_ylabel('Price', fontsize=12)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# ── Price Trend Chart ─────────────────────────────────────────────────────────
st.markdown("### 📈 Price Trend: How Size Affects Price")
st.caption("Keeping rooms=3 and location=5 fixed")

sizes = np.arange(500, 4001, 50)
trend_prices = model.predict([[s, 3, 5] for s in sizes])

fig2, ax2 = plt.subplots(figsize=(10, 4))
fig2.patch.set_facecolor('#f4f6f9')
ax2.set_facecolor('#ffffff')
ax2.plot(sizes, trend_prices, color='#e94560', linewidth=2.5, zorder=3)
ax2.scatter(df['size_sqft'], df['price'], color='#4299e1', zorder=5, s=60, label='Training data')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
ax2.set_xlabel('House Size (sq ft)', fontsize=12)
ax2.set_ylabel('Price', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(linestyle='--', alpha=0.4, zorder=0)
ax2.spines[['top','right']].set_visible(False)
plt.tight_layout()
st.pyplot(fig2)

st.markdown("---")

# ── How it works ──────────────────────────────────────────────────────────────
st.markdown("### 💡 How It Works")
steps = [
    ("STEP 01", "Collect Data", "We gather house data: size, rooms, location, and their actual sale prices."),
    ("STEP 02", "Train the Model", "Scikit-learn's LinearRegression learns a formula from the training data."),
    ("STEP 03", "Evaluate", f"We test on unseen data. R² = {score*100:.1f}% means the model is very accurate."),
    ("STEP 04", "Predict", "Enter any house features and the model predicts its price instantly."),
]
for num, title, body in steps:
    st.markdown(f"""
    <div class="step-card">
      <div class="step-num">{num}</div>
      <div class="step-title">{title}</div>
      <div class="step-body">{body}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#999;font-size:13px'>John's House Price Predictor · ML Project · Built with Python & Scikit-learn</p>", unsafe_allow_html=True)