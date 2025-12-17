# Core
import streamlit as st
import pandas as pd
import numpy as np

# Vizualizare
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Statistica
from scipy import stats
from scipy.stats import skew, kurtosis

# Preprocesare
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

# Setări
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Proiect EDA cu Streamlit",
    layout="wide"
)

# =========================
# CSS
# =========================
st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 700;
    color: #1f4e79;
    margin-bottom: 5px;
}

.section-title {
    font-size: 24px;
    font-weight: 600;
    color: #1f4e79;
    margin-top: 30px;
}

.blue-line {
    border: none;
    height: 3px;
    background-color: #1f4e79;
    margin: 10px 0 20px 0;
}

.upload-label {
    font-size: 20px;
    font-weight: 600;
    color: #1f4e79;
}

.sidebar-title {
    font-size: 18px;
    font-weight: 700;
    color: #1f4e79;
    margin-top: 10px;
}

.sidebar-subtitle {
    font-size: 16px;
    font-weight: 600;
    color: #1f4e79;
    margin-top: 10px;
}

.missing-card {
    background-color: #f9fafb;
    border: 1px solid #d0d7de;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    text-align: center;
}

.missing-title {
    font-size: 16px;
    font-weight: 600;
    color: #1f4e79;
    margin-bottom: 5px;
}

.missing-percent {
    font-size: 14px;
    margin-bottom: 10px;
}

.info-blue {
    background-color: #e8f2ff;
    border-left: 6px solid #1f4e79;
    padding: 18px;
    border-radius: 8px;
    height: 100%;
}

.info-yellow {
    background-color: #fff6d6;
    border-left: 6px solid #f4d03f;
    padding: 18px;
    border-radius: 8px;
    height: 100%;
}

.equal-height {
    height: 420px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.stat-card {
    background-color: #f9fafb;
    border: 1px solid #d0d7de;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.stat-title {
    font-size: 16px;
    font-weight: 600;
    color: #1f4e79;
    margin-bottom: 8px;
}

.stat-value {
    font-size: 28px;
    font-weight: 700;
}

.metric-card {
    background-color: #f8fbff;
    border: 1px solid #d6e4f0;
    border-radius: 12px;
    padding: 20px 25px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.metric-left {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 18px;
    font-weight: 600;
    color: #1f4e79;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #1f4e79;
}

.home-card {
    border-radius: 14px;
    padding: 20px;
    color: #1f1f1f;
    min-height: 340px;
    box-shadow: 0 6px 14px rgba(0,0,0,0.08);
}

.home-card h3 {
    margin-top: 0;
    font-size: 20px;
    font-weight: 700;
}

.home-card ul {
    padding-left: 18px;
}

.eda-card {
    padding: 22px;
    border-radius: 18px;
    min-height: 320px;
    margin-bottom: 30px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
}

.card-blue { background-color: #eaf3fb; }
.card-green { background-color: #eaf7ef; }
.card-yellow { background-color: #fff6df; }
.card-purple { background-color: #f5effa; }
.card-orange { background-color: #fff0dc; }

.eda-card h3 {
    margin-bottom: 10px;
    color: #1f4e79;
}

.eda-card ul {
    padding-left: 18px;
}
</style>
""", unsafe_allow_html=True)



# =========================
# SIDEBAR
# =========================
def sidebar_navigation():
    st.sidebar.markdown("# 📊 Proiect EDA")
    st.sidebar.markdown("### Navigare pe cerințe")

    sections = [
        "Acasă",
        "C1 – Încărcare & Filtrare Date",
        "C2 – Analiză Generală",
        "C3 – Analiză Numerică",
        "C4 – Analiză Categorică",
        "C5 – Corelații & Outlieri"
    ]

    selected = st.sidebar.radio(
        "Selectează secțiunea:",
        sections
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Încărcare date, filtrare, analiză descriptivă "
        "și vizualizări interactive."
    )

    return selected


# =========================
# SESSION STATE – DATASET
# =========================
if "df" not in st.session_state:
    st.session_state.df = None


# =========================
# PAGINA ACASĂ
# =========================
def show_home():

    st.markdown(
        '<div class="main-title">📌 Tema EDA cu Streamlit</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-box">
    Această aplicație realizează o analiză exploratorie a datelor (EDA),
    conform cerințelor temei.

    <br><br>
    👉 Folosiți meniul din stânga pentru a naviga între cerințe.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="eda-card card-blue">
        <h3>📁 Cerința 1 – Încărcare & Filtrare</h3>
        <ul>
            <li>Încărcare fișier CSV / Excel</li>
            <li>Validare citire fișier</li>
            <li>Mesaj de confirmare</li>
            <li>Afișare primele rânduri</li>
            <li>Filtrare numerică (slidere)</li>
            <li>Filtrare categorică (multiselect)</li>
            <li>Rânduri înainte / după filtrare</li>
            <li>DataFrame filtrat</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="eda-card card-green">
        <h3>🔍 Cerința 2 – Cunoaștere date</h3>
        <ul>
            <li>Număr rânduri și coloane</li>
            <li>Tipuri de date pe coloană</li>
            <li>Identificare valori lipsă</li>
            <li>Procent valori lipsă</li>
            <li>Vizualizare valori lipsă</li>
            <li>Statistici descriptive</li>
            <li>Corectare valori lipsă</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="eda-card card-yellow">
        <h3>📊 Cerința 3 – Analiză numerică</h3>
        <ul>
            <li>Selectare variabilă numerică</li>
            <li>Histogramă interactivă</li>
            <li>Slider pentru bins</li>
            <li>Boxplot</li>
            <li>Medie, mediană, deviație</li>
            <li>Tratare outlieri</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("<div style='height: 35px'></div>", unsafe_allow_html=True)


    c4, c5 = st.columns(2)

    with c4:
        st.markdown("""
        <div class="eda-card card-purple">
        <h3>🏷️ Cerința 4 – Analiză categorică</h3>
        <ul>
            <li>Identificare coloane categorice</li>
            <li>Selectare variabilă</li>
            <li>Count plot (bar chart)</li>
            <li>Frecvențe absolute</li>
            <li>Procente</li>
            <li>Codificare categorii</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown("""
        <div class="eda-card card-orange">
        <h3>📈 Cerința 5 – Corelații & Outlieri</h3>
        <ul>
            <li>Matrice de corelație</li>
            <li>Heatmap corelații</li>
            <li>Scatter plot</li>
            <li>Coeficient Pearson</li>
            <li>Detecție outlieri (IQR)</li>
            <li>Vizualizare outlieri</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)



def show_cerinta_1():

    st.markdown('<div class="main-title">Încărcare & Filtrare Date</div>', unsafe_allow_html=True)
    st.markdown('<hr class="blue-line">', unsafe_allow_html=True)


    st.markdown('<div class="upload-label">📂 Alege un fișier CSV sau Excel</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state.df = df
            st.success("✅ Fișier încărcat și citit corect!")

        except Exception as e:
            st.error(f"❌ Eroare la citirea fișierului: {e}")
            return

    if st.session_state.get("df") is None:
        st.info("Te rog să încarci un fișier pentru a continua.")
        return

    df = st.session_state.df


    st.markdown('<hr class="blue-line">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Vizualizare date</div>', unsafe_allow_html=True)

    nr_randuri = st.slider(
        "Selectează numărul de rânduri afișate",
        min_value=5,
        max_value=min(100, len(df)),
        value=10
    )

    st.dataframe(df.head(nr_randuri), use_container_width=True)


    st.sidebar.markdown('<div class="sidebar-title">🧩 Filtrare date</div>', unsafe_allow_html=True)

    df_filtered = df.copy()

    # -------- DATE NUMERICE
    st.sidebar.markdown('<div class="sidebar-subtitle">🔢 Date numerice</div>', unsafe_allow_html=True)
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        min_val = float(df[col].min())
        max_val = float(df[col].max())

        selected_range = st.sidebar.slider(
            col,
            min_val,
            max_val,
            (min_val, max_val)
        )

        df_filtered = df_filtered[
            (df_filtered[col] >= selected_range[0]) &
            (df_filtered[col] <= selected_range[1])
        ]

    # -------- DATE NENUMERICE
    st.sidebar.markdown(
        '<div class="sidebar-subtitle">🧾 Date nenumerice</div>',
        unsafe_allow_html=True
    )

    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        # separator vizual între variabile
        st.sidebar.markdown(
            '<hr style="margin:10px 0;">',
            unsafe_allow_html=True
        )


        st.sidebar.markdown(f"**{col}**")


        search_text = st.sidebar.text_input(
            "",
            placeholder="Caută...",
            key=f"search_{col}"
        )


        values = df[col].dropna().unique().tolist()


        if search_text:
            values = [
                v for v in values
                if search_text.lower() in str(v).lower()
            ]


        selected = st.sidebar.multiselect(
            "",
            options=values,
            default=values,
            key=f"multi_{col}"
        )


        df_filtered = df_filtered[df_filtered[col].isin(selected)]

    st.session_state.df_filtered = df_filtered


    st.markdown('<hr class="blue-line">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📌 Filtrare date</div>', unsafe_allow_html=True)
    st.caption("Filtrarea se realizează din meniul din stânga.")

    st.write(f"🔢 Număr rânduri nefiltrate: **{df.shape[0]}**")
    st.write(f"🔢 Număr rânduri filtrate: **{df_filtered.shape[0]}**")

    st.dataframe(df_filtered, use_container_width=True)



def show_cerinta_2():
    if st.session_state.df is None:
        st.info("Te rog să încarci datele în C1 înainte de a continua.")
        return

    df = st.session_state.df


    st.markdown('<div class="main-title">Cunoaștere date</div>', unsafe_allow_html=True)



    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-left">
                    🔢 Număr rânduri
                </div>
                <div class="metric-value">
                    {df.shape[0]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_b:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-left">
                    🧱 Număr coloane
                </div>
                <div class="metric-value">
                    {df.shape[1]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<hr class="blue-line">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Tipuri de date pe coloane</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Selectează coloanele**")
        selected_cols = st.multiselect(
            "",
            options=df.columns.tolist(),
            default=df.columns.tolist()
        )

    with col2:
        if selected_cols:
            dtype_info = []
            for col in selected_cols:
                types = df[col].dropna().map(type).astype(str).unique()
                dtype_info.append({
                    "Coloană": col,
                    "Tipuri de date detectate": ", ".join(types)
                })

            st.dataframe(pd.DataFrame(dtype_info), use_container_width=True)

    st.markdown('<hr class="blue-line">', unsafe_allow_html=True)


    st.markdown('<div class="section-title">Valori lipsă</div>', unsafe_allow_html=True)

    missing_cols = df.columns[df.isnull().sum() > 0]

    if len(missing_cols) == 0:
        st.success("✅ Datasetul nu conține valori lipsă.")
    else:
        max_per_row = 4
        rows = [
            missing_cols[i:i + max_per_row]
            for i in range(0, len(missing_cols), max_per_row)
        ]

        for row in rows:

            cols = st.columns(max_per_row)

            for i in range(max_per_row):
                if i < len(row):
                    col_name = row[i]
                    missing_pct = df[col_name].isnull().mean() * 100

                    with cols[i]:
                        st.markdown(
                            f"""
                            <div class="missing-card">
                                <div class="missing-title">{col_name}</div>
                                <div class="missing-percent">
                                    {missing_pct:.2f}% valori lipsă
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        fig = px.pie(
                            values=[missing_pct, 100 - missing_pct],
                            names=["Lipsă", "Complet"],
                            hole=0.65,
                            color_discrete_sequence=["#f4d03f", "#1f4e79"]
                        )
                        fig.update_layout(
                            showlegend=False,
                            height=220,
                            margin=dict(t=0, b=0, l=0, r=0)
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"missing_donut_{col_name}"
                        )

                else:

                    with cols[i]:
                        st.empty()


    st.markdown('<div class="section-title">Vizualizare valori lipsă</div>', unsafe_allow_html=True)

    st.caption(
        "Această vizualizare permite identificarea rapidă a tiparelor de valori lipsă "
        "și a coloanelor problematice."
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    colours = ['#1f4e79', '#f4d03f']  # albastru = existent, galben = lipsă
    sns.heatmap(df.isnull(), cmap=sns.color_palette(colours), cbar=False, ax=ax)
    st.pyplot(fig)

    st.markdown('<hr class="blue-line">', unsafe_allow_html=True)


    st.markdown('<div class="section-title">Statistici descriptive (coloane numerice)</div>', unsafe_allow_html=True)

    stats_df = df.describe().T
    stats_df["median"] = df.median(numeric_only=True)

    st.dataframe(stats_df, use_container_width=True)


    st.markdown('<hr class="blue-line">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Corectarea valorilor lipsă</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Această etapă permite aplicarea unor metode de tratare a valorilor lipsă "
        "pe o copie a datasetului, fără a modifica datele originale."
    )


    cols_with_na = df.columns[df.isnull().sum() > 0].tolist()

    if not cols_with_na:
        st.success("Datasetul nu conține valori lipsă care să necesite corectare.")
        return


    selected_col = st.selectbox(
        "Selectează coloana pentru corectare:",
        cols_with_na
    )

    is_numeric = pd.api.types.is_numeric_dtype(df[selected_col])


    if is_numeric:
        method = st.radio(
            "Alege metoda de corectare (numeric):",
            ["Medie", "Mediană", "Mod", "Interpolare", "Elimină rânduri"]
        )
    else:
        method = st.radio(
            "Alege metoda de corectare (categoric):",
            ["Mod", "Elimină rânduri"]
        )

    apply_fix = st.button("Aplică corectarea")

    if apply_fix:
        df_copie = df.copy()

        if method == "Medie":
            df_copie[selected_col].fillna(df_copie[selected_col].mean(), inplace=True)

        elif method == "Mediană":
            df_copie[selected_col].fillna(df_copie[selected_col].median(), inplace=True)

        elif method == "Mod":
            mode_val = df_copie[selected_col].mode().iloc[0]
            df_copie[selected_col].fillna(mode_val, inplace=True)

        elif method == "Interpolare":
            df_copie[selected_col] = df_copie[selected_col].interpolate()

        elif method == "Elimină rânduri":
            df_copie = df_copie.dropna(subset=[selected_col])

        st.success(f"✅ Corectarea a fost aplicată folosind metoda: **{method}**")
        st.session_state.df_curatat = df_copie


        st.markdown('<hr class="blue-line">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Comparație date – înainte și după corectare</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Dataset inițial (primele 10 rânduri)")
            st.dataframe(df.head(10), use_container_width=True)

        with col2:
            st.markdown("### Dataset corectat (primele 10 rânduri)")
            st.dataframe(df_copie.head(10), use_container_width=True)

        # =========================
        # HEATMAP DUPĂ CORECTARE
        # =========================
        st.markdown('<hr class="blue-line">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Vizualizare valori lipsă după corectare</div>',
            unsafe_allow_html=True
        )

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.heatmap(
            df_copie.isnull(),
            cmap=sns.color_palette(['#1f4e79', '#f4d03f']),
            cbar=False,
            ax=ax2
        )
        st.pyplot(fig2)

# =========================
# Cerinta 3 – ANALIZĂ VARIABILE NUMERICE
# =========================
def show_cerinta_3():
    if st.session_state.get("df") is None:
        st.info("Te rog să încarci mai întâi un dataset în C1.")
        return

    df = st.session_state.df


    st.markdown(
        '<div class="main-title">Analiza distribuției variabilelor numerice</div>',
        unsafe_allow_html=True
    )
    st.markdown('<hr class="blue-line">', unsafe_allow_html=True)


    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        st.warning("Datasetul nu conține coloane numerice.")
        return

    selected_col = st.selectbox(
        "Selectează variabila numerică pentru analiză:",
        numeric_cols
    )

    data = df[selected_col].dropna()


    bins = st.slider(
        "Număr de bins pentru histogramă",
        min_value=10,
        max_value=100,
        value=30
    )

    # =========================
    # HISTOGRAMĂ
    # =========================
    left, right = st.columns([1, 2])

    with left:
        st.markdown(
            """
            <div class="info-blue equal-height">
            <strong>📊Histogramă – distribuția variabilei</strong><br><br>
            Histograma reprezintă distribuția valorilor unei variabile numerice
            prin gruparea acestora în intervale (bins).<br><br>

            • Un număr mic de bins evidențiază forma generală a distribuției.<br>
            • Un număr mare de bins permite observarea detaliilor fine și a
              eventualelor asimetrii.<br><br>

            Ajustarea acestui parametru influențează nivelul de granularitate
            al analizei vizuale.
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        fig_hist = px.histogram(
            data,
            x=selected_col,
            nbins=bins,
            color_discrete_sequence=["#1f4e79"]
        )
        fig_hist.update_layout(
            height=420,
            title=f"Distribuția valorilor – {selected_col}",
            title_x=0.5
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True,
            key=f"hist_{selected_col}"
        )


    st.markdown('<hr class="blue-line">', unsafe_allow_html=True)

    # =========================
    # BOXPLOT
    # =========================
    left, right = st.columns([2, 1])

    with left:
        fig_box = px.box(
            data,
            y=selected_col,
            color_discrete_sequence=["#f4d03f"]
        )
        fig_box.update_layout(
            height=420,
            title=f"Boxplot – {selected_col}",
            title_x=0.5
        )

        st.plotly_chart(
            fig_box,
            use_container_width=True,
            key=f"box_{selected_col}"
        )

    with right:
        st.markdown(
            """
            <div class="info-yellow equal-height">
            <strong>📦Interpretarea boxplot-ului</strong><br><br>
            Boxplot-ul oferă o sinteză statistică a distribuției datelor
            prin intermediul quartilelor.<br><br>

            • Linia centrală indică valoarea mediană.<br>
            • Cutia reprezintă intervalul interquartilic (Q1–Q3).<br>
            • Valorile extreme pot semnala prezența outlierilor.<br><br>

            Această reprezentare este utilă pentru evaluarea variabilității
            și a asimetriilor distribuției.
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown('<hr class="blue-line">', unsafe_allow_html=True)

    # =========================
    # STATISTICI
    # =========================
    st.markdown(
        '<div class="section-title">📐 Indicatori statistici</div>',
        unsafe_allow_html=True
    )

    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-title">📈 Medie</div>
                <div class="stat-value">{mean_val:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-title">📊 Mediană</div>
                <div class="stat-value">{median_val:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-title">📉 Deviație standard</div>
                <div class="stat-value">{std_val:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<hr class="blue-line">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">🔧 Corectări asupra variabilelor numerice</div>',
        unsafe_allow_html=True
    )

    method = st.radio(
        "Selectează metoda de corectare numerică:",
        [
            "Fără corectare",
            "Eliminare outlieri (IQR)",
            "Limitare outlieri (Winsorization)",
            "Standardizare (Z-score)",
            "Normalizare (Min-Max)"
        ]
    )

    apply_numeric_fix = st.button("Aplică metoda de corectare")

    if apply_numeric_fix:

        data_original = df[selected_col].dropna()
        data_corrected = data_original.copy()

        if method == "Fără corectare":
            st.info("Nu a fost aplicată nicio corectare asupra variabilei.")

        elif method == "Eliminare outlieri (IQR)":
            Q1 = data_corrected.quantile(0.25)
            Q3 = data_corrected.quantile(0.75)
            IQR = Q3 - Q1

            data_corrected = data_corrected[
                (data_corrected >= Q1 - 1.5 * IQR) &
                (data_corrected <= Q3 + 1.5 * IQR)
                ]

            st.success("Outlierii au fost eliminați folosind metoda IQR.")
            st.write(f"🔢 Număr valori inițiale: {len(data_original)}")
            st.write(f"🔢 Număr valori după eliminare: {len(data_corrected)}")

        elif method == "Limitare outlieri (Winsorization)":
            lower = data_corrected.quantile(0.05)
            upper = data_corrected.quantile(0.95)
            data_corrected = data_corrected.clip(lower, upper)

            st.success("Valorile extreme au fost limitate (winsorizare).")
            st.write(f"Interval aplicat: [{lower:.2f}, {upper:.2f}]")

        elif method == "Standardizare (Z-score)":
            data_corrected = (data_corrected - data_corrected.mean()) / data_corrected.std()

            st.success("Datele au fost standardizate (Z-score).")
            st.write(f"Medie după standardizare: {data_corrected.mean():.2f}")
            st.write(f"Deviație standard după standardizare: {data_corrected.std():.2f}")

        elif method == "Normalizare (Min-Max)":
            data_corrected = (data_corrected - data_corrected.min()) / (
                    data_corrected.max() - data_corrected.min()
            )

            st.success("Datele au fost normalizate în intervalul [0, 1].")
            st.write(
                f"Min: {data_corrected.min():.2f} | Max: {data_corrected.max():.2f}"
            )

        # =========================
        st.markdown('<hr class="blue-line">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">📊 Comparație vizuală</div>',
            unsafe_allow_html=True
        )

        col_l, col_r = st.columns(2)

        # =========================
        # CAZURI CU OUTLIERI → BOXPLOT
        # =========================
        if method in ["Eliminare outlieri (IQR)", "Limitare outlieri (Winsorization)"]:

            with col_l:
                st.markdown("**Boxplot – date inițiale**")
                fig_before = px.box(
                    data_original,
                    y=data_original,
                    color_discrete_sequence=["#1f4e79"]
                )
                st.plotly_chart(fig_before, use_container_width=True)

            with col_r:
                st.markdown("**Boxplot – după corectare**")
                fig_after = px.box(
                    data_corrected,
                    y=data_corrected,
                    color_discrete_sequence=["#27ae60"]
                )
                st.plotly_chart(fig_after, use_container_width=True)

            st.info(
                "Boxplot-ul evidențiază modificările asupra valorilor extreme și "
                "intervalului interquartilic în urma aplicării metodei."
            )

        # =========================
        # ALTE METODE → HISTOGRAMĂ
        # =========================
        else:
            with col_l:
                st.markdown("**Distribuția inițială**")
                fig_before = px.histogram(
                    data_original,
                    nbins=bins,
                    color_discrete_sequence=["#1f4e79"]
                )
                st.plotly_chart(fig_before, use_container_width=True)

            with col_r:
                st.markdown("**Distribuția după corectare**")
                fig_after = px.histogram(
                    data_corrected,
                    nbins=bins,
                    color_discrete_sequence=["#27ae60"]
                )
                st.plotly_chart(fig_after, use_container_width=True)



# =========================
# CERINȚA 4 – ANALIZA VARIABILELOR CATEGORICE
# =========================
def show_cerinta_4():
    # folosim datasetul inițial din C1
    if st.session_state.get("df") is None:
        st.info("Te rog să încarci mai întâi un dataset în C1.")
        return

    df = st.session_state.df


    st.markdown(
        '<div class="main-title">Analiza distribuției variabilelor categorice</div>',
        unsafe_allow_html=True
    )
    st.markdown('<hr class="blue-line">', unsafe_allow_html=True)


    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(categorical_cols) == 0:
        st.warning("Datasetul nu conține coloane categorice.")
        return


    st.markdown("### Selectarea variabilei categorice")
    selected_col = st.selectbox(
        "Alege variabila categorică pentru analiză:",
        categorical_cols
    )

    st.markdown('<hr class="blue-line">', unsafe_allow_html=True)


    freq_abs = df[selected_col].value_counts(dropna=False)
    freq_pct = df[selected_col].value_counts(normalize=True, dropna=False) * 100

    freq_df = pd.DataFrame({
        "Frecvență absolută": freq_abs,
        "Procent (%)": freq_pct.round(2)
    }).reset_index()

    freq_df.columns = [selected_col, "Frecvență absolută", "Procent (%)"]

    # =========================
    # COUNT PLOT
    # =========================
    left, right = st.columns([2, 1])

    with left:
        fig = px.bar(
            freq_df,
            x=selected_col,
            y="Frecvență absolută",
            color_discrete_sequence=["#1f4e79"]
        )

        fig.update_layout(
            height=420,
            title=f"Distribuția frecvențelor – {selected_col}",
            title_x=0.5,
            xaxis_title=selected_col,
            yaxis_title="Frecvență"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"cat_bar_{selected_col}"
        )

    with right:
        st.markdown(
            """
            <div class="info-blue equal-height">
            <strong>📊 Count plot – interpretare</strong><br><br>
            Graficul de tip bară evidențiază frecvența de apariție
            a fiecărei categorii din variabila selectată.<br><br>

            • Categoriile cu bare mai înalte sunt mai frecvente.<br>
            • Diferențele de înălțime indică dezechilibre în distribuție.<br><br>

            Această analiză este utilă pentru identificarea
            categoriilor dominante sau rare.
            </div>
            """,
            unsafe_allow_html=True
        )

    # =========================
    # TABEL FRECVENȚE
    # =========================
    st.markdown('<hr class="blue-line">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Frecvențe absolute și procente</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        freq_df,
        use_container_width=True
    )

    st.markdown('<hr class="blue-line">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">🔧 Tratarea categoriilor rare</div>',
        unsafe_allow_html=True
    )

    threshold = st.slider(
        "Prag minim de frecvență (%) pentru o categorie:",
        min_value=1,
        max_value=20,
        value=5
    )

    apply_grouping = st.button("Grupează categoriile rare")

    if apply_grouping:
        df_cat = df.copy()

        freq_pct_full = df_cat[selected_col].value_counts(normalize=True) * 100
        rare_categories = freq_pct_full[freq_pct_full < threshold].index

        df_cat[selected_col] = df_cat[selected_col].replace(
            rare_categories, "Other"
        )

        st.success("Categoriile rare au fost grupate în 'Other'.")

    st.markdown('<hr class="blue-line">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">🔢 Codificarea variabilei categorice</div>',
        unsafe_allow_html=True
    )

    encoding_method = st.radio(
        "Selectează metoda de codificare:",
        [
            "Fără codificare",
            "Label Encoding",
            "One-Hot Encoding"
        ]
    )

    apply_encoding = st.button("Aplică codificarea")

    if apply_encoding:
        df_encoded = df.copy()

        if encoding_method == "Label Encoding":
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            df_encoded[selected_col] = le.fit_transform(
                df_encoded[selected_col].astype(str)
            )

            st.success("Label Encoding a fost aplicat.")
            st.write("Mapare categorii → valori numerice:")
            mapping_df = pd.DataFrame({
                "Categorie": le.classes_,
                "Cod numeric": range(len(le.classes_))
            })
            st.dataframe(mapping_df, use_container_width=True)

        elif encoding_method == "One-Hot Encoding":
            df_encoded = pd.get_dummies(
                df_encoded,
                columns=[selected_col],
                prefix=selected_col
            )

            st.success("One-Hot Encoding a fost aplicat.")
            st.write("Structura datasetului după codificare:")
            st.dataframe(df_encoded.head(10), use_container_width=True)

    top_n = st.slider(
        "Număr de categorii afișate (Top N):",
        min_value=3,
        max_value=20,
        value=10
    )

    freq_df_top = freq_df.head(top_n)


# =========================
# CERINȚA 5 – PLACEHOLDER
# =========================

def show_cerinta_5():
    if st.session_state.get("df") is None:
        st.info("Te rog să încarci mai întâi un dataset în C1.")
        return

    df = st.session_state.df


    st.markdown(
        '<div class="main-title">Analiza corelațiilor și detecția valorilor anormale</div>',
        unsafe_allow_html=True
    )
    st.markdown('<hr class="blue-line">', unsafe_allow_html=True)


    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Sunt necesare cel puțin două coloane numerice.")
        return


    st.markdown(
        '<div class="section-title">📌 Matricea de corelație</div>',
        unsafe_allow_html=True
    )

    corr_matrix = df[numeric_cols].corr(method="pearson")

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        aspect="auto"
    )

    fig_corr.update_layout(
        height=500,
        title="Heatmap – coeficienți de corelație Pearson",
        title_x=0.5
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True,
        key="corr_heatmap"
    )

    st.info(
        "Heatmap-ul evidențiază relațiile liniare dintre variabilele numerice.\n\n"
        "• valori apropiate de **1** → corelație pozitivă puternică\n"
        "• valori apropiate de **-1** → corelație negativă puternică\n"
        "• valori apropiate de **0** → relație slabă sau inexistentă"
    )


    st.markdown('<hr class="blue-line">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">📈 Analiza relației dintre două variabile</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        var_x = st.selectbox("Variabila X:", numeric_cols, index=0)
    with col2:
        var_y = st.selectbox("Variabila Y:", numeric_cols, index=1)

    df_pair = df[[var_x, var_y]].dropna()

    pearson_corr = df_pair[var_x].corr(df_pair[var_y], method="pearson")

    fig_scatter = px.scatter(
        df_pair,
        x=var_x,
        y=var_y,
        color_discrete_sequence=["#1f4e79"]
    )

    fig_scatter.update_layout(
        height=420,
        title=f"Scatter plot – {var_x} vs {var_y}",
        title_x=0.5
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True,
        key="scatter_corr"
    )

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-title">Coeficient Pearson</div>
            <div class="stat-value">{pearson_corr:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown('<hr class="blue-line">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">🚨 Detecția valorilor anormale (IQR)</div>',
        unsafe_allow_html=True
    )

    outlier_summary = []

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]
        count_outliers = outliers.shape[0]
        percent_outliers = (count_outliers / df[col].dropna().shape[0]) * 100

        outlier_summary.append({
            "Coloană": col,
            "Număr outlieri": count_outliers,
            "Procent outlieri (%)": round(percent_outliers, 2)
        })

    outlier_df = pd.DataFrame(outlier_summary)

    st.dataframe(outlier_df, use_container_width=True)


    st.markdown('<hr class="blue-line">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">📊 Vizualizarea outlierilor</div>',
        unsafe_allow_html=True
    )

    selected_out_col = st.selectbox(
        "Selectează coloana pentru vizualizarea outlierilor:",
        numeric_cols
    )

    Q1 = df[selected_out_col].quantile(0.25)
    Q3 = df[selected_out_col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_out = df.copy()
    df_out["Outlier"] = (df_out[selected_out_col] < lower) | (df_out[selected_out_col] > upper)

    fig_out = px.scatter(
        df_out,
        y=selected_out_col,
        color="Outlier",
        color_discrete_map={True: "red", False: "#1f4e79"}
    )

    fig_out.update_layout(
        height=420,
        title=f"Outlieri detectați – {selected_out_col}",
        title_x=0.5
    )

    st.plotly_chart(
        fig_out,
        use_container_width=True,
        key="outlier_plot"
    )


selected_page = sidebar_navigation()

if selected_page == "Acasă":
    show_home()
elif selected_page == "C1 – Încărcare & Filtrare Date":
    show_cerinta_1()
elif selected_page == "C2 – Analiză Generală":
    show_cerinta_2()
elif selected_page == "C3 – Analiză Numerică":
    show_cerinta_3()
elif selected_page == "C4 – Analiză Categorică":
    show_cerinta_4()
elif selected_page == "C5 – Corelații & Outlieri":
    show_cerinta_5()

