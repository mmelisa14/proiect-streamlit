# Core
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Vizualizare
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Statistica
from scipy import stats
from scipy.stats import skew, kurtosis

# Setări
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Spotify AI Recommender",
    layout="wide"
)


@st.cache_data
def load_data():
    df = pd.read_csv("spotify_tracks.csv")
    return df

df = load_data()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.markdown("""
<div style="
font-size:22px;
font-weight:700;
color:#1f4e79;
margin-bottom:10px;
">
Navigare
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "",
    ["Acasă", "EDA","Preprocesare", "Model", "Recomandări"]
)

# -----------------------------
# HOME PAGE
# -----------------------------

if menu == "Acasă":
    st.title("🎵 Spotify AI Recommender System")

# -----------------------------
# EDA PAGE
# -----------------------------

if menu == "EDA":

    st.title("📊 Exploratory Data Analysis")

    # -----------------------------
    # EXPLICATIE EDA
    # -----------------------------

    st.markdown("""
    <div style="
    background-color:#F0F4FF;
    border-left:5px solid #4C6EF5;
    border-radius:12px;
    padding:26px;
    font-size:16px;
    line-height:1.8;
    color:#1a1a2e;
    ">

    <h4>🔍 Ce este EDA și de ce o facem?</h4>

    Imaginează-ți că primești un teanc de <b>114.000 de fișe</b> — fiecare fișă este o melodie
    și conține informații precum energie, tempo sau dispoziția melodiei. Înainte să facem orice analiză sau model Machine Learning,
    prima întrebare este: <i>ce avem în față?</i>

    Asta înseamnă EDA — explorezi datele ca un detectiv: cauți tipare, anomalii
        și relații ascunse între variabile. Fără pasul ăsta construim un sistem de recomandare pe date pe care nu le înțelegem —
        și la final nu vom ști de ce ne recomandă o melodie de Latino când noi am cerut ceva de Rock.

    <b>Concret, în această secțiune vom:</b>

    <ul>
    <li>Înțelege ce tip are fiecare coloană — număr, text sau adevărat/fals</li>
    <li>Verifica dacă datele sunt complete sau lipsesc valori</li>
    <li>Descoperi legături între caracteristicile muzicale</li>
    <li>Identifica transformările necesare înainte să construim modelul</li>
    <li>Vizualiza datele prin grafice pentru a observa tipare</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # -----------------------------
    # DATASET PREVIEW
    # -----------------------------

    st.subheader("📁 Vizualizare dataset")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🎵 Melodii", f"{df.shape[0]:,}")
    with col2:
        st.metric("📋 Coloane", df.shape[1])
    with col3:
        st.metric("🎸 Genuri unice", df["track_genre"].nunique() if "track_genre" in df.columns else "—")
    with col4:
        st.metric("⚠️ Valori lipsă", int(df.isnull().sum().sum()))

    st.dataframe(
        df[["track_name", "artists", "track_genre", "popularity",
            "energy", "danceability", "valence", "tempo"]].head(50),
        height=350,
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------
    # DESCRIERE COLOANE
    # -----------------------------

    st.subheader("📘 Ce înseamnă fiecare coloană?")

    descriere = pd.DataFrame({
        "Nume coloană": [
            "track_id", "track_name", "artists", "album_name",
            "popularity", "duration_ms", "explicit",
            "danceability", "energy", "key", "loudness",
            "mode", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence",
            "tempo", "time_signature", "track_genre"
        ],
        "Descriere": [
            "Codul unic al melodiei în Spotify — ca un CNP. Nu îl folosim în analiză.",
            "Numele melodiei. Ex: Blinding Lights. Doar pentru afișare.",
            "Artistul melodiei. Ex: The Weeknd. Doar pentru afișare.",
            "Albumul din care face parte. Nu îl folosim în model.",
            "Cât de populară e pe Spotify în momentul colectării. Scor 0–100.",
            "Durata în milisecunde. Ex: 200.000 ms = 3 minute și 20 secunde.",
            "Conține limbaj explicit? True = da, False = nu.",
            "Cât de ușor poți dansa pe ea — ritm stabil, beat regulat. 0 = greu, 1 = perfect pentru dans.",
            "Intensitatea percepută. Clasica are energy mic, metalul și EDM au energy mare. 0–1.",
            "Tonalitatea muzicală: 0 = Do, 1 = Do#, 2 = Re... Nu îl folosim în model.",
            "Volumul mediu în decibeli. Valori negative — -5 dB e mai tare decât -20 dB.",
            "Modul muzical: 1 = major (vesel, luminos), 0 = minor (trist, melancolic).",
            "Cât de multă vorbire conține. Rap = valoare mare, instrumental = aproape 0.",
            "Cât e de acustică vs electronică. 1 = chitară/pian, 0 = complet produsă electronic.",
            "Cât e de instrumentală — fără voce. Jazz instrumental sau clasică au valori mari.",
            "Probabilitatea că e înregistrată live. Valori mari = aplauze, zgomot de fond.",
            "Dispoziția melodiei. 0 = trist/furios, 1 = fericit/euforic.",
            "Viteza în BPM (bătăi pe minut). Lent ~60, dans ~128, metal poate depăși 180.",
            "Măsura muzicală. Aproape toate melodiile pop au 4 (4/4). Nu îl folosim în model.",
            "Genul muzical. Ex: pop, rock, latin. Avem 114 genuri diferite în dataset."
        ],
        "Tip": [
            "identificator", "text", "text", "text",
            "numeric 0–100", "numeric (ms)", "boolean",
            "numeric 0–1", "numeric 0–1", "numeric 0–11",
            "numeric (dB)", "boolean 0/1", "numeric 0–1",
            "numeric 0–1", "numeric 0–1", "numeric 0–1",
            "numeric 0–1", "numeric (BPM)", "numeric 3–7",
            "categoric"
        ]
    })

    st.dataframe(descriere, use_container_width=True, hide_index=True)

    st.markdown("---")

    # -----------------------------
    # GRAFIC 1 — POPULARITATE PE GEN
    # -----------------------------

    st.subheader("🎸 Grafic 1 — Ce gen muzical e cel mai popular?")

    st.info("💬 **Întrebare pentru voi înainte să vedeți graficul:** Care gen credeți că are popularitatea medie cea mai mare în dataset?")

    top_genuri = df["track_genre"].value_counts().head(12).index
    df_top = df[df["track_genre"].isin(top_genuri)]
    gen_pop = df_top.groupby("track_genre")["popularity"].mean().reset_index()
    gen_pop.columns = ["Gen", "Popularitate medie"]
    gen_pop = gen_pop.sort_values("Popularitate medie", ascending=False)

    fig1 = px.bar(
        gen_pop,
        x="Gen",
        y="Popularitate medie",
        color="Popularitate medie",
        color_continuous_scale="teal",
        text=gen_pop["Popularitate medie"].round(1),
        title="Popularitate medie pe gen muzical (top 12 genuri)"
    )
    fig1.update_traces(textposition="outside")
    fig1.update_layout(coloraxis_showscale=False, yaxis_range=[0, 70])
    st.plotly_chart(fig1, use_container_width=True)

    st.success("""
🔍 **Ce observăm?**

Analiza indică faptul că, la momentul colectării, cele mai populare genuri au fost **Anime (48.8), Brazil (44.7) și Ambient (44.2).**

Validitatea studiului este garantată de structura echilibrată a eșantionului (1000 de piese per gen), eliminând erorile de volum și oferind o bază solidă pentru compararea directă a popularității între cele 114 genuri.
""")

    st.markdown("---")

    # -----------------------------
    # GRAFIC 2 — TRIST VS FERICIT
    # -----------------------------

    st.subheader("😊 Grafic 2 — Melodiile fericite sunt mai populare?")

    st.info("💬 **Întrebare:** Voi ascultați mai mult muzică tristă sau fericită? Credeți că melodiile fericite sunt mai ascultate?")

    df["dispozitie"] = pd.cut(
        df["valence"],
        bins=[0, 0.33, 0.66, 1.0],
        labels=["😢 Trist", "😐 Neutru", "😊 Fericit"]
    )

    pop_disp = df.groupby("dispozitie", observed=True)["popularity"].mean().reset_index()
    pop_disp.columns = ["Dispoziție", "Popularitate medie"]

    fig2 = px.bar(
        pop_disp,
        x="Dispoziție",
        y="Popularitate medie",
        color="Dispoziție",
        color_discrete_map={
            "😢 Trist":  "#7F77DD",
            "😐 Neutru": "#888780",
            "😊 Fericit": "#EF9F27"
        },
        text=pop_disp["Popularitate medie"].round(1),
        title="Popularitate medie după dispoziția melodiei"
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(showlegend=False, yaxis_range=[0, 60])
    st.plotly_chart(fig2, use_container_width=True)

    st.success("🔍 În setul de date analizat, există o corelație inversă între nivelul de optimism al unei piese și popularitatea sa medie. Utilizatorii Spotify preferă conținutul care induce o stare de calm sau reflecție, în detrimentul melodiilor pur energice/fericite.")

    st.markdown("---")

    # -----------------------------
    # GRAFIC 3 — ENERGIE VS DANSABILITATE
    # -----------------------------

    st.subheader("⚡ Grafic 3 — Melodiile energice se dansează mai ușor?")

    st.info("💬 **Întrebare:** Dacă o melodie e mai intensă și energică, înseamnă automat că e mai ușor de dansat?")

    sample = df.sample(2000, random_state=42)

    fig3 = px.scatter(
        sample,
        x="energy",
        y="danceability",
        color="track_genre",
        opacity=0.5,
        hover_data=["track_name", "artists"],
        title="Energie vs Dansabilitate — fiecare punct e o melodie",
        labels={"energy": "Energie", "danceability": "Dansabilitate"}
    )
    fig3.update_layout(showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    st.success("🔍 **Ce observăm?** Nu există o corelație directă. O melodie foarte energică poate fi greu de dansat. Dansabilitatea depinde de regularitatea ritmului, nu de intensitate — de aceea modelul nostru are nevoie de ambele coloane separat.")

    st.markdown("---")

    st.info("✅ Am explorat datele. Acum știm ce conțin, ce lipsește și cum se leagă variabilele între ele. Mergem la pasul următor: preprocesare și construirea modelului de recomandare.")


# -----------------------------
# PREPROCESARE PAGE
# -----------------------------

# ==============================
# PREPROCESARE PAGE
# ==============================

if menu == "Preprocesare":

    st.title("⚙️ Preprocesarea datelor")

    st.subheader("📊 Variabile utilizate în model")

    numeric_cols = [
        "danceability",
        "energy",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "valence",
        "tempo"
    ]

    X_model = df[numeric_cols].copy()

    # =========================
    # AFISARE CHIP-URI COLOANE
    # =========================

    def render_feature_cards(features):

        cols_per_row = 4

        for i in range(0, len(features), cols_per_row):

            row = features[i:i+cols_per_row]

            cols = st.columns(cols_per_row)

            for j in range(cols_per_row):

                if j < len(row):

                    with cols[j]:

                        st.markdown(
                            f"""
                            <div style="
                            background:#E9EEF6;
                            border-radius:14px;
                            padding:12px;
                            text-align:center;
                            font-weight:500;
                            margin-bottom:12px;
                            ">
                            {row[j]}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                else:

                    cols[j].empty()


    render_feature_cards(numeric_cols)


    # =========================
    # EXPLICATIE
    # =========================

    st.markdown("""
    <div style="
    background:#EEF2F8;
    padding:18px;
    border-left:5px solid #5C7CFA;
    border-radius:10px;
    margin-top:20px;
    ">

În cadrul analizei au fost utilizate doar variabilele numerice menționate anterior, 
deoarece algoritmii de recomandare bazați pe calculul <b>distanței</b> și al 
<b>similarității</b> pot lucra exclusiv cu date în <b>format numeric</b>. Variabilele precum <b>titlul melodiei</b> și <b>artistul</b> nu au fost incluse, deoarece 
nu dorim ca recomandările să se bazeze pe similaritatea numelui sau a interpretului.

Variabila <b>gen muzical</b> nu a fost utilizată deoarece transformarea acesteia în format numeric:

• ar genera un număr mare de coloane suplimentare (prin <b>One-Hot Encoding</b>)  
• ar introduce relații numerice artificiale între genuri (prin <b>Label Encoding</b>)

</div>
""", unsafe_allow_html=True)


    st.markdown("---")


    # =========================
    # DATASET MODEL
    # =========================

    df_model = df[numeric_cols].copy()


    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style="
            background:#F5F7FB;
            border-radius:14px;
            padding:22px;
            text-align:center;
            box-shadow:0px 2px 6px rgba(0,0,0,0.05);
        ">
            <div style="font-size:32px;">📊</div>
            <div style="font-size:26px; font-weight:700;">
                {X_model.shape[0]:,}
            </div>
            <div style="font-size:15px; color:#555;">
                Număr observații (melodii)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="
            background:#F5F7FB;
            border-radius:14px;
            padding:22px;
            text-align:center;
            box-shadow:0px 2px 6px rgba(0,0,0,0.05);
        ">
            <div style="font-size:32px;">🧱</div>
            <div style="font-size:26px; font-weight:700;">
                {X_model.shape[1]}
            </div>
            <div style="font-size:15px; color:#555;">
                Număr variabile utilizate
            </div>
        </div>
        """, unsafe_allow_html=True)


    # =========================
    # VALORI LIPSA
    # =========================

    st.markdown("---")

    st.subheader("🔍 Verificare valori lipsă")

    missing = df_model.isnull().sum()


    col1, col2 = st.columns([1.5,1])


    with col1:

        import plotly.express as px

        missing_df = missing.reset_index()

        missing_df.columns = ["Feature","Missing"]

        fig = px.bar(
            missing_df,
            x="Feature",
            y="Missing",
            title="Valori lipsă pe variabile"
        )

        st.plotly_chart(fig,use_container_width=True)


    with col2:

        if missing.sum() == 0:

            st.markdown("""
            <div style="
            background:#E8F5E9;
            border-left:6px solid #2E7D32;
            padding:18px;
            border-radius:10px;
            line-height:1.6;
            font-size:15.5px;
            ">

            <b>Rezultat verificare:</b> În acest moment datasetul nu conține valori lipsă pentru variabilele utilizate în model, astfel analiza poate continua fără aplicarea unor metode suplimentare de corectare.


            <b>Dacă ar fi existat valori lipsă</b>, acestea ar fi trebuit tratate înainte de continuarea analizei deoarece pot afecta calculul similarității dintre melodii. Alegerea metodei depinde de context și nu există o soluție universal valabilă:

            <ul style="margin-top:6px;">
            <li><b>Eliminarea înregistrărilor</b> – recomandată când numărul valorilor lipsă este redus.</li>
            <li><b>Media (mean)</b> – potrivită pentru variabile numerice cu distribuție relativ simetrică.</li>
            <li><b>Mediana (median)</b> – valoarea centrală a distribuției, recomandată când există outlieri.</li>
            <li><b>Modul (mode)</b> – cea mai frecventă valoare, utilizată în special pentru variabile categoriale.</li>
            </ul>

            În cadrul acestui proiect nu este necesară aplicarea acestor metode deoarece datele analizate sunt complete.

            </div>
            """, unsafe_allow_html=True)

        else:

            st.warning("Există valori lipsă ce vor fi imputate.")


    # =========================
    # CONFIGURARI PREPROCESARE
    # =========================

    st.markdown("---")

    st.subheader("⚙️ Configurări preprocesare")


    col1,col2 = st.columns([1,1])


    with col1:

        imputation_method = st.selectbox(
            "Metodă imputare valori lipsă",
            ["Fără imputare", "mean", "median", "mode"]
        )


        scaling_method = st.selectbox(
            "Metodă scalare",
            ["Fără scalare","StandardScaler","MinMaxScaler"]
        )
        st.markdown("""
            <style>
            div.stButton > button {
                background-color: #1f77ff;
                color: white;
                border-radius: 8px;
            }
            </style>
            """, unsafe_allow_html=True)

        apply_preprocessing = st.button("Aplică preprocesarea datelor")

    with col2:

        st.markdown("""
        <div style="
        background:#EEF4FF;
        border-left:5px solid #4C6EF5;
        padding:16px;
        border-radius:10px;
        font-size:14.8px;
        line-height:1.6;
        ">

        <b>Ce reprezintă scalarea datelor?</b>

        Scalarea este procesul prin care variabilele numerice sunt aduse la aceeași scară de valori, astfel încât niciuna dintre ele să nu influențeze modelul mai mult decât celelalte doar din cauza unității de măsură.


        <b>Metode utilizate:</b>

        • <b>Fără scalare</b> – păstrăm valorile originale atunci când variabilele sunt deja comparabile ca interval  

        • <b>StandardScaler</b> – transformă datele astfel încât media devine 0 iar deviația standard devine 1  
        (se folosește frecvent în algoritmi bazați pe distanță)

        • <b>MinMaxScaler</b> – transformă valorile în intervalul [0, 1]  (util atunci când dorim compararea directă între caracteristici)

        <b>Exemplu intuitiv:</b>

        În datasetul Spotify, variabila <b>tempo</b> poate avea valori între 60 și 200,
        iar variabila <b>danceability</b> are valori între 0 și 1. Fără scalare, algoritmul ar considera tempo mult mai important doar pentru că are valori mai mari numeric.
        Prin scalare, toate variabilele devin comparabile și modelul poate calcula corect similaritatea dintre melodii.

        </div>
        """, unsafe_allow_html=True)


    # =========================
    # BUTON PREPROCESARE
    # =========================

    st.markdown("---")

    st.markdown("""
    <style>
    div.stButton > button {
        background-color: #1f77ff;
        color: white;
        border-radius: 8px;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #1f77ff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    if apply_preprocessing:

        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, MinMaxScaler


        X_model = df[numeric_cols].copy()

        # =========================
        # IMPUTARE
        # =========================

        if imputation_method == "mean":
            imputer = SimpleImputer(strategy="mean")
            X_model = imputer.fit_transform(X_model)

        elif imputation_method == "median":
            imputer = SimpleImputer(strategy="median")
            X_model = imputer.fit_transform(X_model)

        elif imputation_method == "mode":
            imputer = SimpleImputer(strategy="most_frequent")
            X_model = imputer.fit_transform(X_model)

        elif imputation_method == "Fără imputare":
            X_model = X_model.values

        # =========================
        # SCALARE
        # =========================

        if scaling_method == "StandardScaler":
            scaler = StandardScaler()
            X_model = scaler.fit_transform(X_model)

        elif scaling_method == "MinMaxScaler":
            scaler = MinMaxScaler()
            X_model = scaler.fit_transform(X_model)

        elif scaling_method == "Fără scalare":
            pass  # păstrăm valorile originale

        # =========================
        # SALVARE DATE MODEL
        # =========================

        st.session_state["X_model"] = X_model
        st.session_state["df_model"] = df.reset_index(drop=True)
        st.session_state["features"] = numeric_cols

        st.success("Preprocesarea a fost aplicată cu succes.")



    if "X_model" in st.session_state:
        df_scaled = pd.DataFrame(
            st.session_state["X_model"],
            columns=numeric_cols
        )
    else:
        df_scaled = df[numeric_cols].copy()

    # =========================
    # TABEL DESCRIBE STILIZAT
    # =========================

    describe_df = df_scaled.describe().T

    st.markdown("### 📋 Indicatori statistici principali")

    st.dataframe(
        describe_df.style.format("{:.3f}")
        .background_gradient(cmap="Blues"),
        use_container_width=True
    )

    # =========================
    # SELECTARE VARIABILĂ OUTLIERI
    # =========================

    st.markdown("### 🔎 Detectare outlieri pe variabile")

    selected_feature = st.selectbox(
        "Selectează variabila analizată",
        numeric_cols
    )

    col1, col2 = st.columns(2)

    # =========================
    # BOXPLOT OUTLIERI
    # =========================

    with col1:
        fig_box = px.box(
            df_scaled,
            y=selected_feature,
            title=f"Outlieri pentru {selected_feature}",
            color_discrete_sequence=["#1f77ff"]
        )

        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("""
        <div style="
        background:#EEF4FF;
        border-left:5px solid #1f77ff;
        padding:14px;
        border-radius:10px;
        font-size:14.5px;
        line-height:1.6;
        ">

        <b>Cum interpretăm boxplot-ul?</b>

        Boxplot-ul evidențiază distribuția valorilor unei variabile și permite identificarea valorilor extreme (outlieri):

        • linia din interiorul cutiei reprezintă <b>mediana</b>  
        • marginile cutiei reprezintă <b>quartilele Q1 și Q3</b>  
        • „mustățile” indică intervalul valorilor obișnuite  
        • punctele din exterior reprezintă <b>outlieri</b>

        În general, un număr mic de outlieri este normal. Dacă există foarte mulți, aceștia pot influența calculele de similaritate dintre melodii.

        </div>
        """, unsafe_allow_html=True)
    # =========================
    # HISTOGRAM DISTRIBUȚIE
    # =========================

    with col2:
        fig_hist = px.histogram(
            df_scaled,
            x=selected_feature,
            nbins=40,
            title=f"Distribuția variabilei {selected_feature}",
            color_discrete_sequence=["#4C6EF5"]
        )

        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("""
        <div style="
        background:#E8F5E9;
        border-left:5px solid #2E7D32;
        padding:14px;
        border-radius:10px;
        font-size:14.5px;
        line-height:1.6;
        ">

        <b>Cum interpretăm distribuția variabilei?</b>

        Histograma arată modul în care sunt distribuite valorile variabilei:

        • o distribuție relativ simetrică indică date echilibrate  
        • o distribuție asimetrică poate indica valori extreme  
        • concentrarea valorilor într-un interval restrâns arată consistență în caracteristica analizată

        Pentru algoritmii bazați pe distanță, este preferabil ca variabilele să aibă distribuții comparabile după scalare.

        </div>
        """, unsafe_allow_html=True)


if menu == "Model":

    from sklearn.neighbors import NearestNeighbors

    st.title("🧠 Model — Recomandare melodii similare")

    # =========================
    # METODE DE RECOMANDARE
    # =========================

    st.subheader("📚 Cum funcționează sistemele de recomandare?")
    st.write("Există mai multe metode prin care un sistem poate face recomandări. Fiecare are logica ei, avantaje și limitări:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="
            background: #EEF4FF;
            border-top: 4px solid #4C6EF5;
            border-radius: 12px;
            padding: 18px 20px;
            line-height: 1.75;
            font-size: 14px;
            min-height: 260px;
        ">
            <div style="font-size:16px; font-weight:700; color:#1a1a2e; margin-bottom:10px;">
                🎵 Content-Based Filtering
            </div>
            <div style="color:#333; margin-bottom:12px;">
                Analizează <b>caracteristicile obiectului</b> în sine — energie, tempo, dispoziție.
                Dacă îți place o melodie energică, îți recomandă alte melodii energice,
                indiferent de ce ascultă alții.
            </div>
            <div style="color:#555; font-size:13px; margin-bottom:12px;">
                <b>Folosit de:</b> Spotify (analiza audio), Netflix (gen și regie)
            </div>
            <div>
                <span style="background:#D3F9D8; color:#1a7431; padding:2px 8px; border-radius:4px; font-size:12px; display:inline-block; margin:2px;">✅ simplu de implementat</span>
                <span style="background:#D3F9D8; color:#1a7431; padding:2px 8px; border-radius:4px; font-size:12px; display:inline-block; margin:2px;">✅ nu ai nevoie de alți utilizatori</span>
                <span style="background:#FFE3E3; color:#c0392b; padding:2px 8px; border-radius:4px; font-size:12px; display:inline-block; margin:2px;">❌ recomandări mai puțin surprinzătoare</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
            background: #F3FFF3;
            border-top: 4px solid #2E7D32;
            border-radius: 12px;
            padding: 18px 20px;
            line-height: 1.75;
            font-size: 14px;
            min-height: 260px;
        ">
            <div style="font-size:16px; font-weight:700; color:#1a1a2e; margin-bottom:10px;">
                👥 Collaborative Filtering
            </div>
            <div style="color:#333; margin-bottom:12px;">
                Nu analizează conținutul — analizează <b>comportamentul utilizatorilor</b>.
                „Utilizatorii care au ascultat X au ascultat și Y."
                Funcționează pe baza similitudinii dintre preferințele oamenilor.
            </div>
            <div style="color:#555; font-size:13px; margin-bottom:12px;">
                <b>Folosit de:</b> Instagram (sugestii conturi), Amazon, TikTok
            </div>
            <div>
                <span style="background:#D3F9D8; color:#1a7431; padding:2px 8px; border-radius:4px; font-size:12px; display:inline-block; margin:2px;">✅ descoperă lucruri neașteptate</span>
                <span style="background:#D3F9D8; color:#1a7431; padding:2px 8px; border-radius:4px; font-size:12px; display:inline-block; margin:2px;">✅ foarte personalizat</span>
                <span style="background:#FFE3E3; color:#c0392b; padding:2px 8px; border-radius:4px; font-size:12px; display:inline-block; margin:2px;">❌ ai nevoie de istoric al multor utilizatori</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
        <div style="
            background: #FFF8F0;
            border-top: 4px solid #F08C00;
            border-radius: 12px;
            padding: 18px 20px;
            line-height: 1.75;
            font-size: 14px;
            min-height: 260px;
        ">
            <div style="font-size:16px; font-weight:700; color:#1a1a2e; margin-bottom:10px;">
                🔀 Hybrid Recommender
            </div>
            <div style="color:#333; margin-bottom:12px;">
                Combină content-based și collaborative filtering.
                Analizează atât caracteristicile melodiei cât și
                comportamentul utilizatorilor similari ție.
                Rezultatul e mai precis și mai variat.
            </div>
            <div style="color:#555; font-size:13px; margin-bottom:12px;">
                <b>Folosit de:</b> Spotify, YouTube, Netflix — majoritatea platformelor mari
            </div>
            <div>
                <span style="background:#D3F9D8; color:#1a7431; padding:2px 8px; border-radius:4px; font-size:12px; display:inline-block; margin:2px;">✅ cele mai bune rezultate</span>
                <span style="background:#FFE3E3; color:#c0392b; padding:2px 8px; border-radius:4px; font-size:12px; display:inline-block; margin:2px;">❌ complex de implementat</span>
                <span style="background:#FFE3E3; color:#c0392b; padding:2px 8px; border-radius:4px; font-size:12px; display:inline-block; margin:2px;">❌ necesită date despre utilizatori</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style="
            background: #FDF4FF;
            border-top: 4px solid #9C27B0;
            border-radius: 12px;
            padding: 18px 20px;
            line-height: 1.75;
            font-size: 14px;
            min-height: 260px;
        ">
            <div style="font-size:16px; font-weight:700; color:#1a1a2e; margin-bottom:10px;">
                🧮 Matrix Factorization
            </div>
            <div style="color:#333; margin-bottom:12px;">
                Descompune o matrice mare de tip utilizatori × melodii în factori latenți (preferințe ascunse), identificând modele de comportament pe care utilizatorii nu le exprimă explicit.
Această metodă permite realizarea unor recomandări foarte precise în sisteme cu un număr mare de utilizatori și interacțiuni.
            </div>
            <div style="color:#555; font-size:13px; margin-bottom:12px;">
                <b>Folosit de:</b> Spotify și Netflix la scară industrială
            </div>
            <div>
                <span style="background:#D3F9D8; color:#1a7431; padding:2px 8px; border-radius:4px; font-size:12px; display:inline-block; margin:2px;">✅ extrem de precis la scară mare</span>
                <span style="background:#FFE3E3; color:#c0392b; padding:2px 8px; border-radius:4px; font-size:12px; display:inline-block; margin:2px;">❌ extrem de complex</span>
                <span style="background:#FFE3E3; color:#c0392b; padding:2px 8px; border-radius:4px; font-size:12px; display:inline-block; margin:2px;">❌ necesită date masive</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # =========================
    # DE CE KNN
    # =========================

    st.markdown("""
    <div style="
        background: #EEF4FF;
        border: 2px solid #4C6EF5;
        border-radius: 12px;
        padding: 20px 24px;
        font-size: 14.5px;
        line-height: 1.85;
    ">
        <div style="font-size:16px; font-weight:700; color:#1a1a2e; margin-bottom:12px;">
            🎯 De ce folosim Content-Based Filtering cu KNN în acest proiect?
        </div>
        <div style="color:#333; margin-bottom:10px;">
            Setul nostru conține <b>caracteristici audio ale melodiilor</b> — energie, tempo, dispoziție —
            dar nu avem date despre comportamentul utilizatorilor.
            Singura metodă aplicabilă este <b>Content-Based Filtering</b>.
        </div>
        <div style="color:#333; margin-bottom:10px;">
            Algoritmul <b>K-Nearest Neighbors (KNN)</b> reprezintă fiecare melodie ca un punct
            într-un spațiu numeric și găsește cele mai apropiate K melodii față de cea aleasă.
            Simplu, rapid și ușor de interpretat.
        </div>
        <div style="color:#333;">
            <b>Un avantaj important:</b> poți vedea exact de ce a fost recomandată o melodie —
            energy 0.73 vs 0.71, tempo 168 vs 171. Transparență totală, fără cutie neagră.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # =========================
    # CONFIGURARE MODEL
    # =========================

    st.subheader("⚙️ Configurare model")
    st.write("Ajustează parametrii înainte de antrenare:")

    col1, col2 = st.columns([1, 1])

    with col1:

        metric = st.selectbox(
            "📐 Metrica de similaritate",
            ["cosine", "euclidean", "manhattan"],
            help="Cosine e recomandat pentru date audio normalizate."
        )

        n_neighbors = st.slider(
            "🎵 Câte recomandări vrei să primești?",
            min_value=3,
            max_value=15,
            value=5
        )

        algorithm = st.selectbox(
            "⚡ Algoritm de calcul",
            ["auto", "ball_tree", "kd_tree", "brute"],
            help="'auto' alege automat cel mai rapid algoritm pentru datele tale."
        )

    with col2:

        if metric == "cosine":
            bg = "#EEF4FF"
            border = "#4C6EF5"
            titlu = "📐 Cosine similarity"
            explicatie = "Compară <b>profilul sonor</b> al melodiilor, nu mărimea valorilor.<br><br>Dacă două melodii au același echilibru între energie, tempo și dispoziție — chiar dacă valorile diferă puțin — le consideră similare.<br><br><b>Recomandat pentru caracteristici audio normalizate.</b>"
        elif metric == "euclidean":
            bg = "#F3F0FF"
            border = "#7C3AED"
            titlu = "📏 Distanța Euclideană"
            explicatie = "Măsoară <b>distanța directă</b> dintre două melodii în spațiul numeric — ca distanța dintre două puncte pe o hartă, dar în mai multe dimensiuni.<br><br>Mai sensibilă la diferențele mari între valori — de aceea scalarea este importantă.<br><br><b>Funcționează bine după normalizarea datelor.</b>"
        else:
            bg = "#FFF8F0"
            border = "#F08C00"
            titlu = "📐 Distanța Manhattan"
            explicatie = "Calculează distanța <b>sumând diferențele</b> pe fiecare caracteristică în parte — ca și cum mergi pe o grilă, doar pe orizontală și verticală, niciodată diagonal.<br><br>Mai robustă când există valori extreme în date.<br><br><b>Bună alternativă la Euclidean.</b>"

        st.markdown(
            f'<div style="background:{bg}; border-left:5px solid {border}; padding:16px 18px; border-radius:10px; line-height:1.75; font-size:14px; color:#1a1a2e;"><b>{titlu}</b><br><br>{explicatie}</div>',
            unsafe_allow_html=True
        )

    # =========================
    # CASETA ALGORITM DE CALCUL
    # =========================

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    if algorithm == "auto":
        alg_titlu = "⚡ auto — alegere automată"
        alg_text = "Streamlit alege singur cel mai rapid algoritm în funcție de dimensiunea datelor. <b>Recomandat pentru majoritatea cazurilor</b> — nu trebuie să te gândești la nimic."
    elif algorithm == "ball_tree":
        alg_titlu = "🌳 ball_tree"
        alg_text = "Organizează datele într-o structură arborescentă de sfere concentrice. <b>Eficient pentru date cu multe caracteristici</b> și seturi mari. Funcționează cu orice metrică de distanță."
    elif algorithm == "kd_tree":
        alg_titlu = "📦 kd_tree"
        alg_text = "Împarte spațiul numeric în dreptunghiuri recursive pentru a găsi vecinii mai rapid. <b>Rapid pentru seturi mici cu puține caracteristici</b>, dar nu funcționează cu cosine similarity."
    else:
        alg_titlu = "🔍 brute"
        alg_text = "Compară melodia aleasă cu <b>fiecare altă melodie din dataset</b> una câte una. Cel mai lent, dar garantat corect. Util pentru debugging sau seturi mici de date."

    st.markdown(
        f'<div style="background:#FFFBEA; border-left:5px solid #F59F00; padding:16px 18px; border-radius:10px; line-height:1.75; font-size:14px; color:#1a1a2e;"><b>{alg_titlu}</b><br><br>{alg_text}</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # =========================
    # BUTON ANTRENARE
    # =========================

    st.subheader("🚀 Antrenează modelul")
    st.write("Apasă butonul de mai jos. Modelul calculează similaritatea dintre toate melodiile și devine gata să facă recomandări.")

    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #4C6EF5, #7C3AED);
        color: white;
        font-size: 16px;
        font-weight: 600;
        padding: 14px 0;
        border-radius: 10px;
        border: none;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    div.stButton > button:first-child:hover {
        opacity: 0.88;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("▶  Antrenează modelul", use_container_width=True):

        if "X_model" not in st.session_state:
            st.warning("⚠️ Mergi mai întâi la pagina **Preprocesare** și aplică transformările.")

        else:

            # =========================
            # VALIDARE COMPATIBILITATE
            # =========================

            if metric == "cosine" and algorithm in ["kd_tree", "ball_tree"]:
                st.error("""
    ⚠️ Algoritmul selectat nu este compatibil cu metrica aleasă.

    Pentru cosine similarity poți folosi doar:
    • auto
    • brute
    """)

                st.stop()

            # =========================
            # ANTRENARE MODEL
            # =========================

            with st.spinner("Se calculează similaritățile dintre melodii..."):

                model = NearestNeighbors(
                    metric=metric,
                    n_neighbors=n_neighbors + 1,
                    algorithm=algorithm
                )

                model.fit(st.session_state["X_model"])

                st.session_state["model"] = model
                st.session_state["metric"] = metric
                st.session_state["k"] = n_neighbors

            st.success("✅ Modelul e gata! Mergi la pagina **Recomandări** pentru a testa.")

            st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            col1.metric("🎵 Melodii procesate", f"{st.session_state['X_model'].shape[0]:,}")
            col2.metric("📊 Caracteristici folosite", st.session_state["X_model"].shape[1])
            col3.metric("🎯 Recomandări per melodie", n_neighbors)


if menu == "Recomandări":

    import urllib.parse
    import re

    st.markdown("""
    <h1 style="
    text-align:center;
    margin-bottom:5px;
    ">
    🎧 Recomandare melodii similare
    </h1>

    <hr style="
    height:2px;
    border:none;
    background:linear-gradient(90deg,#4C6EF5,#7C3AED);
    margin-top:5px;
    margin-bottom:30px;
    ">
    """, unsafe_allow_html=True)

    # =========================
    # VERIFICARE MODEL
    # =========================

    if "model" not in st.session_state:
        st.warning("Antrenează modelul mai întâi.")
        st.stop()

    model = st.session_state["model"]
    X_model = st.session_state["X_model"]
    df_clean = st.session_state["df_model"]

    # =========================
    # SELECTARE ARTIST
    # =========================

    artisti = sorted(df_clean["artists"].dropna().unique())

    col1, col2 = st.columns(2)

    with col1:
        artist_selectat = st.selectbox(
            "🎤 Selectează artist",
            artisti
        )

    with col2:
        melodii_artist = df_clean[
            df_clean["artists"].apply(
                lambda x: artist_selectat in [a.strip() for a in x.split(";")]
                if isinstance(x, str)
                else False
            )
        ]

        melodie_selectata = st.selectbox(
            "🎵 Selectează melodie",
            melodii_artist["track_name"].unique()
        )

    melodie_pos = melodii_artist[
        melodii_artist["track_name"] == melodie_selectata
    ].index[0]

    melodie = df_clean.loc[melodie_pos]

    # =========================
    # HERO CARD MELODIE
    # =========================

    dispozitie = (
        "😊 Fericită"
        if melodie["valence"] > 0.6
        else "😢 Tristă"
        if melodie["valence"] < 0.4
        else "😐 Neutră"
    )

    energie = (
        "⚡ Energie mare"
        if melodie["energy"] > 0.6
        else "🌙 Energie scăzută"
    )

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:

        st.markdown(f"""
        <div style="
        background:linear-gradient(90deg,#EEF4FF,#F3FFF3);
        border-radius:14px;
        padding:22px;
        font-size:15px;
        line-height:1.8;
        ">

        <h4>🎵 Melodie selectată</h4>

        <b>{melodie['track_name']}</b><br>
        {melodie['artists']}<br><br>

        🎸 Gen: {melodie['track_genre']}<br>
        ⭐ Popularitate: {melodie['popularity']} / 100<br>
        😊 Dispoziție: {dispozitie}<br>
        ⚡ Energie: {energie}<br>
        🥁 Tempo: {round(melodie['tempo'])} BPM

        </div>
        """, unsafe_allow_html=True)

    with col2:

        track_id = melodie["track_id"]

        st.markdown(f"""
        <iframe src="https://open.spotify.com/embed/track/{track_id}"
        width="100%" height="200"
        frameBorder="0"
        allow="autoplay; clipboard-write; encrypted-media"
        loading="lazy"
        style="border-radius:12px;">
        </iframe>
        """, unsafe_allow_html=True)

    # =========================
    # GENERARE RECOMANDARI
    # =========================

    st.markdown("---")

    if st.button("✨ Generează recomandări similare", use_container_width=True):

        input_vector = X_model[melodie_pos].reshape(1, -1)

        distances, indices = model.kneighbors(
            input_vector,
            n_neighbors=25   # luam mai multe ca sa eliminam duplicate
        )

        recomandari = []

        seen = set()

        for idx, dist in zip(indices[0], distances[0]):

            row = df_clean.iloc[idx]

            key = (row["track_name"], row["artists"])

            if key == (melodie["track_name"], melodie["artists"]):
                continue

            if key in seen:
                continue

            seen.add(key)

            recomandari.append((row, dist))

            if len(recomandari) == st.session_state["k"]:
                break

        st.subheader("🎯 Top recomandări pentru tine")

        # =========================
        # AFISARE CARDURI PREMIUM
        # =========================

        for i, (row, dist) in enumerate(recomandari):

            similaritate = round((1 - dist) * 100, 1)

            dispozitie = (
                "😊 Fericită"
                if row["valence"] > 0.6
                else "😢 Tristă"
                if row["valence"] < 0.4
                else "😐 Neutră"
            )

            energie = (
                "⚡ Mare"
                if row["energy"] > 0.6
                else "🌙 Mică"
            )

            col1, col2 = st.columns([2, 1])

            with col1:

                st.markdown(f"""
                <div style="
                background:linear-gradient(90deg,#EEF4FF,#F3FFF3);
                border-radius:14px;
                border:1px solid #E9ECEF;
                padding:18px;
                margin-bottom:12px;
                box-shadow:0px 2px 6px rgba(0,0,0,0.05);
                ">

                <b style="font-size:16px;">#{i+1} {row['track_name']}</b><br>
                {row['artists']}<br><br>

                ⭐ Similaritate: <b>{similaritate}%</b><br>
                🎸 Gen: {row['track_genre']}<br>
                😊 Dispoziție: {dispozitie}<br>
                ⚡ Energie: {energie}<br>
                🥁 Tempo: {round(row['tempo'])} BPM

                </div>
                """, unsafe_allow_html=True)

            with col2:

                track_id = row["track_id"]

                st.markdown(f"""
                <iframe src="https://open.spotify.com/embed/track/{track_id}"
                width="100%" height="200"
                frameBorder="0"
                allow="autoplay; clipboard-write; encrypted-media"
                loading="lazy"
                style="border-radius:10px;">
                </iframe>
                """, unsafe_allow_html=True)

if menu == "Acasă":
    st.markdown("---")
    # INTRO
    st.markdown("""
    <div style="
    background:#F0F4FF;
    border-left:6px solid #4C6EF5;
    padding:22px;
    border-radius:12px;
    font-size:16px;
    line-height:1.7;
    ">

    Planul de astăzi este să construim propriul nostru <b>sistem de recomandare muzicală</b>, inspirat de tehnologia din spatele Spotify sau YouTube.

    Vom folosi un set de date real de pe Kaggle, care cuprinde peste 100.000 de piese. Obiectivul nostru este să analizăm aceste date, să le pregătim și să antrenăm un algoritm care „înțelege” muzica prin cifre: energie, tempo, cât de potrivită e pentru dans (danceability) sau starea de spirit pe care o transmite (valence). La final, vom obține un instrument capabil să sugereze automat piese similare celor preferate de utilizator.

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🤔 Întrebări pentru voi")

    st.markdown("""
    <div style="
    background:#FFF9DB;
    border-left:6px solid #FAB005;
    padding:20px;
    border-radius:12px;
    font-size:15.5px;
    line-height:1.7;
    ">

    📱 Cum credeți că <b>Instagram</b> știe ce videoclipuri să vă sugereze în Reels?

    🛒 Cum credeți că aplicațiile online știu ce <b>reclame</b> să vă arate în funcție de interesele voastre?

    🎧 Cum credeți că <b>Spotify</b> reușește să recomande melodii asemănătoare celor pe care le ascultați?

    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🚀 Pașii pe care îi vom parcurge astăzi")


    # GRID 2x2 CARDURI

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div style="
        background:#E3FAFC;
        padding:22px;
        border-radius:14px;
        min-height:180px;
        ">

        <h4>📊 Exploratory Data Analysis (EDA)</h4>

        Vom analiza datasetul pentru a înțelege structura datelor,
        tipurile de variabile existente și relațiile dintre caracteristicile muzicale.

        </div>
        """, unsafe_allow_html=True)


        st.markdown("""
        <div style="
        background:#E6FCF5;
        padding:22px;
        border-radius:14px;
        min-height:180px;
        margin-top:20px;
        ">

        <h4>⚙️ Preprocesarea datelor</h4>

        Vom selecta variabilele relevante, verificăm dacă există valori lipsă
        și aducem datele la aceeași scară pentru a putea fi utilizate corect de algoritm.

        </div>
        """, unsafe_allow_html=True)


    with col2:

        st.markdown("""
        <div style="
        background:#FFF3BF;
        padding:22px;
        border-radius:14px;
        min-height:180px;
        ">

        <h4>🤖 Construirea modelului</h4>

        Vom construi algoritmul de recomandare care calculează similaritatea dintre melodii
        folosind caracteristicile audio ale acestora.

        </div>
        """, unsafe_allow_html=True)


        st.markdown("""
        <div style="
        background:#F3F0FF;
        padding:22px;
        border-radius:14px;
        min-height:180px;
        margin-top:20px;
        ">

        <h4>✅ Testarea recomandărilor</h4>

        Vom verifica dacă sistemul generează recomandări corecte
        și dacă melodiile sugerate sunt într-adevăr asemănătoare între ele.

        </div>
        """, unsafe_allow_html=True)