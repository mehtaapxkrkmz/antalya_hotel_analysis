import json
import io
import base64
import re
import os

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import mtranslate                  # To instantly and freely translate every word the user types.
from openai import OpenAI          # For Groq LLM API connection

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# ─── Groq LLM API Client Initialization ──────────────────────────────────────
# Automatically retrieves the key added to the Secrets field.
GROQ_API_KEY = os.environ.get("groq", "")
client = None
if GROQ_API_KEY:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY
    )

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Antalya Hotel Recommendation System",
    page_icon="🏨",
    layout="wide",
)

st.markdown("""
<style>
html { scroll-behavior: smooth; }
.block-container { padding-top: 1.5rem; padding-bottom: 0; }
#MainMenu, footer { visibility: hidden; }

div[data-testid="stHorizontalBlock"] {
    align-items: flex-start !important;
    flex-wrap: nowrap !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] {
    min-width: 0 !important;
    overflow: hidden !important;
}
div[data-testid="stIFrame"] > iframe { display: block !important; width: 100% !important; }

/* All Plotly charts stay the same height — no layout jump */
div.stPlotlyChart { min-height: 340px; max-height: 340px; }
div[data-testid="stMetric"] { min-height: 80px; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────

CATEGORIES = {
    "Food":         ["food","breakfast","dinner","delicious","restaurant","buffet","meal","tasty","lunch"],
    "Cleaning":     ["clean","hygiene","room service","bathroom","tidy","spotless","linen","dirty","smell"],
    "Staff":        ["staff","personnel","friendly","helpful","reception","waiter","service","smiling","host"],
    "Pool & Beach": ["pool","sea","beach","sand","water","aqua","shores","sunbed","poolside"],
}

WC_STOPWORDS = set(STOPWORDS) | {
    "hotel","room","stay","day","one","great","excellent",
    "nice","holiday","beach","everything","hotels","antalya",
}

CHART_H = 340  # All right-column charts use this fixed height

def _empty_div(h: int, msg: str = "") -> str:
    return (
        f'<div style="height:{h}px;display:flex;align-items:center;'
        f'justify-content:center;color:#bbb;font-size:13px;'
        f'background:#f8f9fa;border-radius:8px;">{msg}</div>'
    )

def _empty_plotly(h: int = CHART_H, msg: str = "") -> go.Figure:
    """Empty Plotly figure used as a placeholder with fixed height."""
    fig = go.Figure()
    fig.update_layout(
        height=h, autosize=True,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(
            text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=13, color="#bbb"),
        )] if msg else [],
    )
    return fig

# ─── Cached calculations ────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("hotels_Vize_FINAL.csv")
    df = df[(df["latitude"].between(35.5, 37.5)) & (df["longitude"].between(29.0, 32.5))]
    if "sentiment_10" not in df.columns:
        df["sentiment_10"] = (df["sentiment_score"] + 1) * 5
    return df


@st.cache_data(show_spinner=False)
def get_hotel_stats(key: str, df_json: str) -> pd.DataFrame:
    fdf = pd.read_json(io.StringIO(df_json))
    stats = (
        fdf.groupby("hotel_name")
        .agg(
            hybrid_score=("hybrid_score", "mean"),
            review_rating=("review_rating", "mean"),
            sentiment_score=("sentiment_score", "mean"),
            rating=("rating", "first"),
            total_reviews=("review_text", "count"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
        )
        .reset_index()
    )
    return stats[stats["total_reviews"] >= 2].sort_values("hybrid_score", ascending=False)


@st.cache_data(show_spinner=False)
def get_aspects(hotel_name: str, reviews_json: str) -> dict:
    hotel_df = pd.read_json(io.StringIO(reviews_json))
    scores = {cat: [] for cat in CATEGORIES}
    for _, row in hotel_df.iterrows():
        text = str(row["translated_text"]).lower()
        s = row["sentiment_10"]
        for cat, kws in CATEGORIES.items():
            if any(k in text for k in kws):
                scores[cat].append(s)
    return {cat: (sum(v) / len(v)) / 10 if v else None for cat, v in scores.items()}


def _text_tokens(text: str) -> set:
    tokens = set(re.findall(r"\b\w+\b", str(text).lower()))
    return {t for t in tokens if t not in WC_STOPWORDS}


@st.cache_data(show_spinner=False)
def get_llm_summary(reviews_list: list) -> str:
    """Generates an abstractive summary using free Groq LLM API."""
    if not client or not reviews_list:
        return ""
    try:
        # To reduce server load, combine and send the top 10 most recent/popular comments.
        combined_text = " \n".join(reviews_list[:10])
        prompt = (
            "You are an AI assistant analyzing hotel reviews. Provide a concise, 2-3 sentence summary "
            "of the following guest reviews, highlighting the main positives and negatives mentioned. "
            "Write the summary in professional English:\n\n" + combined_text
        )
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        return ""          # It returns empty in case of an error; the rule in the code is dropped into the local digest.


@st.cache_data(show_spinner=False)
def get_extractive_summary(text: str, top_n: int = 1) -> list:
    """Lightweight extractive summarizer that selects top sentence(s) by token frequency."""
    if not text or not str(text).strip():
        return []
    txt = str(text).strip()
    sents = re.split(r'(?<=[.!?])\s+', txt)
    if not sents:
        return []

    words = re.findall(r"\b\w+\b", txt.lower())
    freqs = {}
    for w in words:
        if w in WC_STOPWORDS:
            continue
        freqs[w] = freqs.get(w, 0) + 1

    scored = []
    for s in sents:
        sw = re.findall(r"\b\w+\b", s.lower())
        if not sw:
            continue
        score = sum(freqs.get(w, 0) for w in sw) / len(sw)
        scored.append((score, s.strip()))

    if not scored:
        return []
    scored.sort(reverse=True, key=lambda x: x[0])
    return [s for _, s in scored[:top_n]]


@st.cache_data(show_spinner=False)
def get_similar_hotels(selected_hotel: str, df_json: str, top_n: int = 3) -> list:
    hotel_df = pd.read_json(io.StringIO(df_json))
    amenity_cols = ["pool", "spa", "parking", "breakfast"]
    for col in amenity_cols:
        if col not in hotel_df.columns:
            hotel_df[col] = 0

    if selected_hotel not in hotel_df["hotel_name"].unique():
        return []
    grouped = (
        hotel_df.groupby("hotel_name")["translated_text"]
        .apply(lambda texts: " ".join(texts.dropna().astype(str)))
    )
    if selected_hotel not in grouped:
        return []

    amenity_summary = (
        hotel_df.groupby("hotel_name")[amenity_cols]
        .max()
        .astype(int)
    )
    if selected_hotel not in amenity_summary.index:
        return []

    target_tokens = _text_tokens(grouped[selected_hotel])
    target_amenities = amenity_summary.loc[selected_hotel]

    similarity_scores = []
    for hotel_name, text in grouped.items():
        if hotel_name == selected_hotel:
            continue

        tokens = _text_tokens(text)
        intersection = target_tokens & tokens
        union = target_tokens | tokens
        text_score = len(intersection) / len(union) if union else 0.0

        other_amenities = amenity_summary.loc[hotel_name]
        amenity_score = (
            (target_amenities & other_amenities).sum() / len(amenity_cols)
        ) if len(amenity_cols) else 0.0

        score = 0.7 * text_score + 0.3 * amenity_score
        if score > 0:
            similarity_scores.append((score, hotel_name))

    similar = [
        (hotel_name, score)
        for score, hotel_name in sorted(similarity_scores, reverse=True)
    ]
    return similar[:top_n]


@st.cache_data(show_spinner=False)
def get_similarity_explanations(selected_hotel: str, df_json: str, max_tokens: int = 5) -> dict:
    """Return top shared tokens between selected_hotel and others as a short explanation."""
    hotel_df = pd.read_json(io.StringIO(df_json))
    if selected_hotel not in hotel_df["hotel_name"].unique():
        return {}
    grouped = (
        hotel_df.groupby("hotel_name")["translated_text"]
        .apply(lambda texts: " ".join(texts.dropna().astype(str)))
    )
    if selected_hotel not in grouped:
        return {}

    target_tokens = _text_tokens(grouped[selected_hotel])
    explanations = {}
    for hotel_name, text in grouped.items():
        if hotel_name == selected_hotel:
            continue
        tokens = _text_tokens(text)
        shared = list(target_tokens & tokens)
        explanations[hotel_name] = shared[:max_tokens]
    return explanations


@st.cache_data(show_spinner=False)
def get_radar_fig(hotel_name: str, aspect_json: str) -> "go.Figure":
    aspects = json.loads(aspect_json)
    cats = [c for c, v in aspects.items() if v is not None]
    vals = [aspects[c] for c in cats]
    if not cats:
        return _empty_plotly(CHART_H, "Insufficient data")
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself", line_color="#FF4B4B",
        fillcolor="rgba(255,75,75,0.15)",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            angularaxis=dict(rotation=90, direction="clockwise"),
        ),
        showlegend=False, height=CHART_H, autosize=True,
        margin=dict(l=50, r=50, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


@st.cache_data(show_spinner=False)
def get_wordcloud_fig(hotel_name: str, text: str) -> "go.Figure":
    """Embed WordCloud into a Plotly figure — fixed CHART_H height to avoid jump."""
    if not text.strip():
        return _empty_plotly(CHART_H, "Insufficient text data.")

    wc = WordCloud(
        width=900, height=300,
        background_color="white",
        colormap="magma",
        stopwords=WC_STOPWORDS,
    ).generate(text)
    fig_mpl, ax = plt.subplots(figsize=(9, 3), dpi=100)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig_mpl.tight_layout(pad=0)

    buf = io.BytesIO()
    fig_mpl.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig_mpl)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()

    fig = go.Figure()
    fig.add_layout_image(dict(
        source=f"data:image/png;base64,{img_b64}",
        xref="paper", yref="paper",
        x=0, y=1, sizex=1, sizey=1,
        xanchor="left", yanchor="top",
        sizing="stretch",
        layer="below",
    ))
    fig.update_layout(
        height=CHART_H, autosize=True,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
    )
    return fig


@st.cache_data(show_spinner=False)
def build_folium_map(hotel_name: str, map_json: str):
    map_df = pd.read_json(io.StringIO(map_json))
    if map_df.empty:
        return None
    clat = map_df["latitude"].mean()
    clon = map_df["longitude"].mean()
    zoom = 13 if hotel_name != "All Hotels" else 10
    m = folium.Map(location=[clat, clon], zoom_start=zoom,
                   tiles="CartoDB positron", prefer_canvas=True)
    for _, row in map_df.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8, color="#FF4B4B", fill=True,
            fill_color="#FF4B4B", fill_opacity=0.8,
            tooltip=row["hotel_name"],
        ).add_to(m)
    return m


# ─── Load data ───────────────────────────────────────────────────────────────

try:
    df = load_data()
except Exception as e:
    st.error(f"Data load failed: {e}")
    st.stop()

# ─── Session state ────────────────────────────────────────────────────────────

if "selected_hotel" not in st.session_state:
    st.session_state.selected_hotel = "All Hotels"
if "recommendation_target" not in st.session_state:
    st.session_state.recommendation_target = None

def _select_recommendation(hotel_name: str):
    st.session_state.recommendation_target = hotel_name

# ─── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.header("🔍 Filters")
st.sidebar.markdown("Select a hotel from the dropdown at the top to view details.")
search_query = ""

st.sidebar.subheader("Amenities")
filter_pool      = st.sidebar.checkbox("Pool")
filter_spa       = st.sidebar.checkbox("Spa")
filter_parking   = st.sidebar.checkbox("Parking")
filter_breakfast = st.sidebar.checkbox("Breakfast")
score_range = st.sidebar.slider("Hybrid Score Range", 0.0, 10.0, (8.0, 9.5), step=0.1)

# ─── Filtering ───────────────────────────────────────────────────────────────

fdf = df.copy()
if filter_pool:      fdf = fdf[fdf["pool"] == 1]
if filter_spa:       fdf = fdf[fdf["spa"] == 1]
if filter_parking:   fdf = fdf[fdf["parking"] == 1]
if filter_breakfast: fdf = fdf[fdf["breakfast"] == 1]
if search_query:
    fdf = fdf[fdf["hotel_name"].str.contains(search_query, case=False, na=False)]
fdf = fdf[fdf["hybrid_score"].between(score_range[0], score_range[1])]

fkey = f"{filter_pool}_{filter_spa}_{filter_parking}_{filter_breakfast}_{score_range}_{search_query}"
top_hotels = get_hotel_stats(fkey, fdf.to_json(orient="records"))

# ─── Header & Dropdown ───────────────────────────────────────────────────────

st.title("🏨 Antalya Hotel Recommendation System")

if top_hotels.empty:
    st.warning("No hotels found matching the selected criteria.")
    st.stop()

hotel_list = ["All Hotels"] + top_hotels["hotel_name"].tolist()

def _on_change():
    st.session_state.selected_hotel = st.session_state._hotel_pick

if st.session_state.recommendation_target:
    st.session_state._hotel_pick = st.session_state.recommendation_target
    st.session_state.selected_hotel = st.session_state.recommendation_target
    st.session_state.recommendation_target = None

st.selectbox(
    "Explore a hotel for insights:",
    hotel_list,
    key="_hotel_pick",
    on_change=_on_change,
    index=hotel_list.index(st.session_state.selected_hotel)
          if st.session_state.selected_hotel in hotel_list else 0,
)

if st.session_state.selected_hotel not in hotel_list:
    st.session_state.selected_hotel = "All Hotels"
selected = st.session_state.selected_hotel

# ─── Layout ───────────────────────────────────────────────────────────────────

col1, col2 = st.columns([1, 1.2])

# ── Left column ──

with col1:
    if selected == "All Hotels":
        st.subheader("📊 Global Tourism Insights")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Active Hotels", len(top_hotels))
        c2.metric("Avg Antalya Rating", f"{top_hotels['review_rating'].mean():.2f}")
        c3.metric("Avg Hybrid Score", f"{top_hotels['hybrid_score'].mean():.2f}")
        st.write("---")

    st.subheader("🏆 Top-Rated Selection")
    st.dataframe(
        top_hotels[["hotel_name", "hybrid_score"]].head(15),
        use_container_width=True, height=250,
    )

    st.subheader("📍 Location Map")
    map_data = top_hotels[["hotel_name", "latitude", "longitude"]].dropna()
    disp_map = (
        map_data[map_data["hotel_name"] == selected]
        if selected != "All Hotels" else map_data
    )

    if disp_map.empty:
        st.markdown(_empty_div(320, "Map data unavailable."), unsafe_allow_html=True)
    elif FOLIUM_AVAILABLE:
        fmap = build_folium_map(selected, disp_map.to_json(orient="records"))
        if fmap:
            st_folium(fmap, height=320, width=480,
                      returned_objects=[], key=f"fmap_{selected}")
    else:
        st.map(disp_map.rename(columns={"latitude": "lat", "longitude": "lon"}),
               zoom=9 if selected == "All Hotels" else 12)

    if selected != "All Hotels":
        st.write("---")
        st.markdown("##### 💡 Similar Recommendations")
        st.caption("You may also like these:")
        similar_hotels = get_similar_hotels(
            selected,
            fdf[["hotel_name", "translated_text", "pool", "spa", "parking", "breakfast"]].to_json(orient="records"),
            top_n=3,
        )
        if similar_hotels:
            explanations = get_similarity_explanations(
                selected,
                fdf[["hotel_name", "translated_text", "pool", "spa", "parking", "breakfast"]].to_json(orient="records"),
            )
            for hotel_name, score in similar_hotels:
                row1, row2, row3 = st.columns([3, 1, 1])
                row1.markdown(f"**{hotel_name}**")
                expl = explanations.get(hotel_name, [])
                if expl:
                    row1.caption("Shared keywords: " + ", ".join(expl))
                row2.markdown(f"{score:.2f}")
                row3.button(
                    "View",
                    key=f"rec_{selected}_{hotel_name}",
                    on_click=_select_recommendation,
                    args=(hotel_name,),
                )
            if max(score for _, score in similar_hotels) < 0.20:
                st.info("Similarity scores are relatively low because review text overlap is limited.")
        else:
            st.caption("No similar hotel recommendations are available.")

        # ─── Dynamic Chatbot Integration with LLM + Fallback Support (Left Column) ───
        
        st.write("---")
        st.markdown("##### 💬 Chat with Hotel Bot")
        st.caption("You can ask anything about the hotel in Turkish or English (e.g., pool, food, room, location, parking, views).")

        if f"chat_history_{selected}" not in st.session_state: 
            st.session_state[f"chat_history_{selected}"] = []
                    
        for message in st.session_state[f"chat_history_{selected}"]:
            with st.chat_message(message["role"]): 
                st.markdown(message["content"])

        if user_query := st.chat_input("Ask the bot about hotel amenities...", key=f"chat_in_{selected}_new"):
            with st.chat_message("user"): 
                st.markdown(user_query)
            st.session_state[f"chat_history_{selected}"].append({"role": "user", "content": user_query})

            hotel_reviews = df[df["hotel_name"] == selected]
            raw_text_reviews = hotel_reviews["translated_text"].dropna().tolist()
            
            bot_response = ""
            
            #  If the API Key is available, the actual LLM response will be generated.
            if client and raw_text_reviews:
                try:
                    context_reviews = " \n".join(raw_text_reviews[:12])
                    system_prompt = (
                        "You are a helpful hotel assistant chatbot. Answer the user's question based strictly "
                        "on the following customer reviews provided as context. Synthesize the feedback naturally. "
                        "If the answer isn't in the reviews, state that politely. Answer in clean, professional English.\n\n"
                        f"Context Reviews:\n{context_reviews}"
                    )
                    
                    completion = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"User question: {user_query}"}
                        ],
                        temperature=0.5,
                        max_tokens=200
                    )
                    bot_response = completion.choices[0].message.content.strip()
                except Exception:
                    bot_response = ""        # If an error occurs, the system automatically reverts to Option A (Fallback).

            #   FALLBACK: Option A is activated if the LLM API crashes or the internet connection is lost.
            if not bot_response:
                try:
                    translated_query = mtranslate.translate(user_query, "en", "auto").lower()
                except Exception:
                    translated_query = user_query.lower()

                raw_words = [w.lower() for w in re.findall(r"\b\w+\b", user_query)]
                search_words = set(re.findall(r"\b\w+\b", translated_query))
                search_words = {w for w in search_words if w not in WC_STOPWORDS and w not in ["hotel", "room", "stay"]}

                matched_reviews = []
                if search_words:
                    for _, r_row in hotel_reviews.iterrows():
                        rev_translated = str(r_row["translated_text"]).lower()
                        rev_original = str(r_row["review_text"]).lower()
                        
                        match_score = 0
                        for word in search_words:
                            if word in rev_translated:
                                match_score += rev_translated.count(word) * 2
                            for original_word in raw_words:
                                if original_word not in ["otel", "hotel", "hakkında", "nasıl", "var", "mı", "mu", "mi", "mü"] and len(original_word) > 1:
                                    if original_word in rev_original:
                                        match_score += rev_original.count(original_word)
                        
                        if match_score > 0: 
                            matched_reviews.append((match_score, r_row["review_text"]))
                            
                matched_reviews.sort(key=lambda x: x[0], reverse=True)
                all_matches = [text for _, text in matched_reviews[:15]]

                if all_matches:
                    bot_response = f"Here are guest reviews mentioning **'{user_query}'**:\n\n"
                    for match in all_matches: 
                        bot_response += f"👉 *\"{match}\"*\n\n"
                else:
                    bot_response = f"I couldn't find any specific details matching **'{user_query}'** in the reviews. Please try other terms like food, room, or pool!"

            with st.chat_message("assistant"): 
                st.markdown(bot_response)
            st.session_state[f"chat_history_{selected}"].append({"role": "assistant", "content": bot_response})

# ── Right column ────────────────────────────────────────────────────────────────

with col2:
    st.subheader("📊 Detailed Analysis")

    slot_metrics = st.empty()

    st.write("---")
    st.markdown("##### 🔎️ Aspect Satisfaction")
    aspect_slots: dict = {}
    for cat in CATEGORIES:
        lc, rc = st.columns([1, 3])
        lc.caption(cat)
        aspect_slots[cat] = rc.empty()

    st.write("---")
    st.markdown("##### 💬 Guest Keywords")
    slot_wc = st.empty()
    slot_wc.plotly_chart(
        _empty_plotly(CHART_H, "Loading keywords…"),
        use_container_width=True,
        config={"displayModeBar": False},
        key="wc_init",
    )

    # ── Fill slots ─────────────────────────────────────────────────────────────

    if selected == "All Hotels":
        slot_metrics.markdown(_empty_div(80, "Select a hotel"), unsafe_allow_html=True)
        for s in aspect_slots.values():
            s.progress(0.0)

    else:
        hotel_row     = top_hotels[top_hotels["hotel_name"] == selected].iloc[0]
        hotel_reviews = df[df["hotel_name"] == selected]

        with slot_metrics.container():
            c1, c2, c3 = st.columns(3)
            c1.metric("Hybrid Score",  f"{hotel_row['hybrid_score']:.2f}")
            c2.metric("Review Rating", f"{hotel_row['review_rating']:.1f}")
            c3.metric("Sentiment",     f"{hotel_row['sentiment_score']:.2f}")

        aspects = get_aspects(
            selected,
            hotel_reviews[["translated_text", "sentiment_10"]].to_json(orient="records"),
        )
        for cat, slot in aspect_slots.items():
            v = aspects.get(cat)
            if v is not None:
                slot.progress(min(v, 1.0))
            else:
                slot.caption("— no reviews found")

        raw_text = hotel_reviews["translated_text"].dropna().str.cat(sep=" ")
        slot_wc.plotly_chart(
            get_wordcloud_fig(selected, raw_text),
            use_container_width=True,
            config={"displayModeBar": False},
            key=f"wc_{selected}",
        )

        # ─── LLM + FALLBACK SUPPORTED AUTO SUMMARY AREA ───
        st.markdown("##### 🤖 Auto Summary")
        reviews_list_for_summary = hotel_reviews["translated_text"].dropna().tolist()
        
        summary_text = ""
        #  First, request an abstract summary from the generative AI (Groq).
        if client and reviews_list_for_summary:
            summary_text = get_llm_summary(reviews_list_for_summary)
            
        #  If the API is down or crashes, it will revert to the old frequency-based hash.
        if not summary_text:
            summary_fallback = get_extractive_summary(raw_text, top_n=1)
            if summary_fallback:
                summary_text = summary_fallback[0]
            else:
                summary_text = "Not enough review text to generate an automatic summary."
                
        st.info(summary_text)

        st.write("---")
        st.markdown("##### 📈 Review Summary")
        review_count = len(hotel_reviews)
        avg_rating = hotel_reviews["review_rating"].mean() if review_count else 0.0
        avg_sentiment = hotel_reviews["sentiment_score"].mean() if review_count else 0.0
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Reviews", review_count)
        c2.metric("Avg Review Rating", f"{avg_rating:.1f}")
        c3.metric("Avg Sentiment", f"{avg_sentiment:.2f}")

        sentiment_summary = (
            hotel_reviews["sentiment_label"]
            .fillna("Unknown")
            .value_counts(dropna=False)
            .rename_axis("Sentiment")
            .reset_index(name="Count")
        )
        sentiment_summary["Share"] = (
            sentiment_summary["Count"] / sentiment_summary["Count"].sum() * 100
        ).round(1).astype(str) + "%"
        st.dataframe(sentiment_summary, use_container_width=True, height=160)

        st.write("---")

        if hotel_reviews.empty:
            st.info("No reviews available for this hotel.")
        else:
            max_reviews = len(hotel_reviews)
            show_all = st.checkbox(
                "Show all reviews",
                value=False,
                key=f"review_all_{selected}",
            )
            if show_all:
                num_reviews = max_reviews
            else:
                num_reviews = st.number_input(
                    "Show latest reviews",
                    min_value=1,
                    max_value=max_reviews,
                    value=min(10, max_reviews),
                    step=1,
                    key=f"review_count_{selected}",
                )
            review_sample = (
                hotel_reviews.sort_values("review_date", ascending=False)
                .head(num_reviews)
                .reset_index(drop=True)
            )
            for _, row in review_sample.iterrows():
                header = (
                    f"{row['review_date']} — Rating {row['review_rating']:.1f} / "
                    f"{row.get('sentiment_label', 'N/A')} "
                    f"({row.get('review_language', 'N/A')})"
                )
                with st.expander(header):
                    st.write(row["review_text"])