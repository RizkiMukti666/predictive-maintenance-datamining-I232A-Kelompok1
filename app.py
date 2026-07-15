"""
==============================================================================
 SISTEM PREDIKSI KEGAGALAN MESIN (PREDICTIVE MAINTENANCE)
 Proyek Akhir Praktikum Data Mining 2026
------------------------------------------------------------------------------
 Algoritma : Decision Tree Classifier
 Dataset   : AI4I 2020 Predictive Maintenance Dataset
             (S. Matzka, UCI Machine Learning Repository)
==============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ============================================================================
# KONFIGURASI HALAMAN
# ============================================================================
st.set_page_config(
    page_title="Prediksi Kegagalan Mesin | Data Mining 2026",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# TEMA VISUAL — "Panel Kontrol Industri"
# ============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

/* ---- Basis tipografi ---- */
html, body, [class*="css"], .stMarkdown, p, span, div, label {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ---- Warna latar utama ---- */
.stApp {
    background: #F4F6F9;
}

/* ---- Hero panel (kepala aplikasi) ---- */
.hero {
    background: linear-gradient(135deg, #1A1D23 0%, #2C333E 100%);
    border-radius: 16px;
    padding: 30px 34px;
    margin-bottom: 8px;
    border: 1px solid #333B47;
    box-shadow: 0 8px 24px rgba(26,29,35,0.18);
}
.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #F5A623;
    margin: 0 0 10px 0;
    font-weight: 600;
}
.hero-title {
    font-size: 30px;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0 0 10px 0;
    line-height: 1.2;
}
.hero-sub {
    font-size: 15px;
    color: #AEB6C2;
    margin: 0;
    line-height: 1.6;
    max-width: 780px;
}
.hero-chips { margin-top: 18px; }
.chip {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    font-weight: 600;
    color: #DDE3EC;
    background: rgba(245,166,35,0.10);
    border: 1px solid rgba(245,166,35,0.35);
    padding: 5px 12px;
    border-radius: 20px;
    margin-right: 8px;
}

/* ---- Kartu statistik (readout gaya panel) ---- */
.stat-grid { display: flex; gap: 14px; flex-wrap: wrap; margin: 6px 0 4px 0; }
.stat-card {
    flex: 1;
    min-width: 150px;
    background: #FFFFFF;
    border: 1px solid #E3E7ED;
    border-left: 4px solid #3E6D9C;
    border-radius: 10px;
    padding: 16px 18px;
}
.stat-card.amber { border-left-color: #E8890C; }
.stat-card.green { border-left-color: #2E9E5B; }
.stat-card.red   { border-left-color: #D6453D; }
.stat-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; letter-spacing: 1px; text-transform: uppercase;
    color: #6B7482; margin: 0 0 6px 0; font-weight: 600;
}
.stat-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px; font-weight: 700; color: #1A1D23; margin: 0; line-height: 1;
}
.stat-note { font-size: 12px; color: #8A929E; margin: 6px 0 0 0; }

/* ---- Kotak penjelasan (callout) ---- */
.callout {
    border-radius: 10px;
    padding: 16px 18px;
    margin: 12px 0;
    font-size: 14.5px;
    line-height: 1.65;
    border: 1px solid;
}
.callout .c-title {
    font-weight: 700; display: block; margin-bottom: 6px; font-size: 14.5px;
}
.callout.info   { background:#EEF4FB; border-color:#CBDDF0; color:#26425F; }
.callout.info .c-title { color:#1E3A57; }
.callout.tip    { background:#FEF6E9; border-color:#F6D9A6; color:#7A5410; }
.callout.tip .c-title { color:#8A5B08; }
.callout.warn   { background:#FCEEEC; border-color:#F3C4BE; color:#8A2E27; }
.callout.warn .c-title { color:#A8352C; }
.callout.plain  { background:#F2F4F7; border-color:#E0E4EA; color:#3A424E; }
.callout.plain .c-title { color:#1A1D23; }

/* ---- Judul bagian ---- */
.section-head {
    font-size: 20px; font-weight: 700; color:#1A1D23;
    margin: 6px 0 2px 0; display:flex; align-items:center; gap:9px;
}
.section-desc { font-size: 14px; color:#6B7482; margin: 0 0 10px 0; line-height:1.55; }

/* ---- Metric bawaan Streamlit ---- */
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace; font-weight: 700; color:#1A1D23;
}
[data-testid="stMetricLabel"] { font-weight: 600; color:#6B7482; }

/* ---- Tab ---- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    font-weight: 600; font-size: 14px; padding: 8px 16px; border-radius: 8px 8px 0 0;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E3E7ED; }

/* ---- Tabel dataframe ---- */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* ---- Divider tipis ---- */
hr { border-color: #E3E7ED; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def callout(title, body, kind="info"):
    """Kotak penjelasan berwarna. kind: info | tip | warn | plain"""
    st.markdown(
        f'<div class="callout {kind}"><span class="c-title">{title}</span>{body}</div>',
        unsafe_allow_html=True,
    )


def section(title, desc=""):
    html = f'<div class="section-head">{title}</div>'
    if desc:
        html += f'<div class="section-desc">{desc}</div>'
    st.markdown(html, unsafe_allow_html=True)


# ============================================================================
# KONSTANTA & METADATA
# ============================================================================
ID_COLUMNS = ["UDI", "Product ID"]

# Kolom "bocor" (data leakage): kelima kolom ini adalah SUB-JENIS kegagalan yang
# menjadi penyebab langsung target 'Machine failure'. Nilainya baru diketahui
# SETELAH mesin gagal, sehingga tidak boleh dipakai sebagai fitur prediksi.
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

TARGET_COLUMN = "Machine failure"

FEATURE_INFO = {
    "Type": ("Varian Kualitas Produk",
             "Kelas kualitas produk yang sedang dikerjakan mesin: L (rendah), M (sedang), H (tinggi)."),
    "Air temperature [K]": ("Suhu Udara",
             "Suhu udara di sekitar mesin (satuan Kelvin). Suhu ~300 K setara ±27 °C."),
    "Process temperature [K]": ("Suhu Proses",
             "Suhu selama proses kerja mesin berlangsung (Kelvin). Biasanya lebih tinggi dari suhu udara."),
    "Rotational speed [rpm]": ("Kecepatan Putar",
             "Seberapa cepat mesin berputar, dalam rotasi per menit (rpm)."),
    "Torque [Nm]": ("Torsi",
             "Besarnya gaya putar yang dihasilkan mesin (Newton-meter). Torsi tinggi = beban kerja berat."),
    "Tool wear [min]": ("Keausan Alat",
             "Total waktu alat sudah dipakai (menit). Semakin lama, alat semakin aus dan rawan gagal."),
}


# ============================================================================
# FUNGSI DATA & MODEL
# ============================================================================
@st.cache_data
def load_default_data():
    try:
        df = pd.read_csv("ai4i2020.csv")
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        return None


def preprocess(df):
    drop_cols = [c for c in (ID_COLUMNS + LEAKAGE_COLUMNS) if c in df.columns]
    df_clean = df.drop(columns=drop_cols)
    if TARGET_COLUMN not in df_clean.columns:
        return None, None, None, drop_cols, None
    type_encoder = None
    if "Type" in df_clean.columns and df_clean["Type"].dtype == object:
        type_encoder = LabelEncoder()
        df_clean["Type"] = type_encoder.fit_transform(df_clean["Type"])
    X = df_clean.drop(columns=[TARGET_COLUMN]).select_dtypes(include=[np.number])
    y = df_clean[TARGET_COLUMN]
    return X, y, df_clean, drop_cols, type_encoder


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### 🏭 Panel Kontrol")
    st.caption("Atur data dan parameter model di sini.")

    uploaded_file = st.file_uploader("📁 Upload Dataset (.csv)", type=["csv"])

    st.markdown("---")
    st.markdown("#### ⚙️ Parameter Model")
    max_depth = st.slider(
        "Kedalaman Pohon (max_depth)", 2, 20, 10, 1,
        help="Seberapa banyak lapis pertanyaan yang boleh ditanyakan pohon keputusan. "
             "Terlalu dalam = model bisa 'menghafal' data (overfitting).",
    )
    test_size = st.slider(
        "Porsi Data Uji", 0.10, 0.40, 0.20, 0.05,
        help="Berapa bagian data yang disisihkan untuk menguji model. "
             "0.20 = 20% untuk ujian, 80% untuk latihan.",
    )
    use_balanced = st.checkbox(
        "Seimbangkan perhatian antar kelas",
        value=False,
        help="Aktifkan agar model lebih fokus pada kasus 'Gagal' yang jumlahnya sedikit.",
    )

    st.markdown("---")
    st.markdown("#### 📘 Apa ini?")
    st.caption(
        "Aplikasi ini menebak apakah sebuah mesin pabrik akan **rusak/gagal** atau "
        "**tetap normal**, hanya dengan membaca data sensornya (suhu, kecepatan putar, "
        "torsi, dan usia alat). Cocok untuk *predictive maintenance* — merawat mesin "
        "**sebelum** rusak, bukan sesudah."
    )


# ============================================================================
# MEMUAT DATA
# ============================================================================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    data_source = f"file unggahan Anda ({uploaded_file.name})"
else:
    df = load_default_data()
    data_source = "dataset contoh bawaan (AI4I 2020)"

# ============================================================================
# HERO
# ============================================================================
st.markdown(
    """
    <div class="hero">
      <p class="hero-eyebrow">Predictive Maintenance · Data Mining 2026</p>
      <p class="hero-title">Sistem Prediksi Kegagalan Mesin Industri</p>
      <p class="hero-sub">
        Aplikasi ini mempelajari ribuan catatan kondisi mesin, lalu belajar mengenali
        pola kapan sebuah mesin cenderung <b>gagal</b>. Tujuannya sederhana: mendeteksi
        gejala kerusakan lebih awal, agar perbaikan bisa dijadwalkan sebelum mesin
        benar-benar berhenti dan mengganggu produksi.
      </p>
      <div class="hero-chips">
        <span class="chip">ALGORITMA · DECISION TREE</span>
        <span class="chip">10.000 DATA MESIN</span>
        <span class="chip">6 SENSOR</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if df is None:
    st.error(
        "Data belum tersedia. Silakan upload file CSV di panel kiri, atau pastikan "
        "file `ai4i2020.csv` ada di folder aplikasi."
    )
    st.stop()

st.caption(f"✓ Data dimuat dari {data_source} — **{df.shape[0]:,} baris × {df.shape[1]} kolom**")

# Preprocessing awal
X, y, df_clean, dropped_cols, type_encoder = preprocess(df)
if X is None:
    st.error(f"Dataset harus punya kolom target bernama '{TARGET_COLUMN}'.")
    st.stop()

# Latih model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)
clf = DecisionTreeClassifier(
    random_state=42, max_depth=max_depth,
    class_weight="balanced" if use_balanced else None,
)
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
baseline_acc = (y_test == 0).mean()
report_dict = classification_report(
    y_test, y_test_pred, target_names=["Normal", "Gagal"], output_dict=True
)

# ============================================================================
# TAB NAVIGASI
# ============================================================================
tab_home, tab_data, tab_eda, tab_model, tab_predict = st.tabs([
    "🏠 Beranda", "📋 Data", "📊 Eksplorasi", "🌳 Model & Hasil", "🔮 Coba Prediksi"
])

# ---------------------------------------------------------------------------
# TAB HOME — penjelasan untuk orang awam
# ---------------------------------------------------------------------------
with tab_home:
    section("🎯 Apa yang dikerjakan aplikasi ini?",
            "Penjelasan singkat tanpa istilah teknis.")
    callout(
        "Ceritanya begini",
        "Bayangkan sebuah pabrik punya banyak mesin. Kalau satu mesin tiba-tiba rusak "
        "di tengah produksi, kerugiannya besar: produksi berhenti, biaya perbaikan "
        "mendadak melonjak, dan target meleset. <b>Aplikasi ini bertugas seperti "
        "'dokter mesin'</b> — ia membaca tanda-tanda vital mesin (suhu, kecepatan "
        "putar, torsi, usia alat) lalu memberi peringatan dini: <i>“mesin ini "
        "berisiko gagal, sebaiknya diperiksa.”</i>",
        "info",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        callout("1️⃣ Mesin belajar dari pengalaman",
                "Komputer diberi 10.000 catatan kondisi mesin masa lalu — lengkap dengan "
                "info mana yang akhirnya gagal. Dari sini ia belajar polanya.", "plain")
    with c2:
        callout("2️⃣ Menemukan pola tersembunyi",
                "Model menemukan sendiri aturan seperti: <i>“kalau torsi sangat tinggi "
                "DAN alat sudah aus, mesin cenderung gagal.”</i>", "plain")
    with c3:
        callout("3️⃣ Memprediksi mesin baru",
                "Saat diberi data mesin baru yang belum pernah dilihat, model menebak: "
                "Normal atau Gagal — beserta tingkat keyakinannya.", "plain")

    section("🌳 Kenapa disebut 'Pohon Keputusan' (Decision Tree)?",
            "Algoritma inti aplikasi ini, dijelaskan dengan analogi sederhana.")
    callout(
        "Seperti daftar pertanyaan bercabang",
        "Decision Tree bekerja persis seperti cara kita menebak sesuatu lewat "
        "pertanyaan ya/tidak yang berurutan. Contohnya seperti dokter: "
        "<i>“Apakah suhunya tinggi? → Ya. Apakah torsinya juga tinggi? → Ya. "
        "Apakah alat sudah lama dipakai? → Ya.”</i> → kesimpulan: <b>berisiko gagal.</b> "
        "Setiap pertanyaan menyaring kemungkinan sampai model yakin dengan jawabannya. "
        "Di bawah ini contoh nyata pohon keputusan yang dibuat model (disederhanakan "
        "3 lapis agar mudah dibaca):",
        "tip",
    )

    # Pohon ilustratif dangkal (max_depth=3) agar terbaca
    illustr = DecisionTreeClassifier(random_state=42, max_depth=3)
    illustr.fit(X_train, y_train)
    fig_tree, ax_tree = plt.subplots(figsize=(13, 6))
    plot_tree(
        illustr, feature_names=list(X.columns), class_names=["Normal", "Gagal"],
        filled=True, rounded=True, fontsize=9, ax=ax_tree, impurity=False, proportion=True,
    )
    ax_tree.set_title("Contoh alur berpikir model (versi sederhana 3 lapis)",
                      fontsize=12, fontweight="bold")
    st.pyplot(fig_tree)
    st.caption(
        "Cara membaca: mulai dari kotak paling atas. Ikuti panah KIRI bila kondisi "
        "terpenuhi (benar), panah KANAN bila tidak. Warna oranye = cenderung 'Gagal', "
        "biru = cenderung 'Normal'. Kotak paling bawah adalah kesimpulan akhir."
    )

    section("📌 Ringkasan hasil aplikasi ini")
    r1, r2, r3, r4 = st.columns(4)
    r1.markdown(f'<div class="stat-card green"><p class="stat-label">Akurasi Uji</p>'
                f'<p class="stat-value">{test_acc*100:.1f}%</p>'
                f'<p class="stat-note">tebakan benar pada data ujian</p></div>', unsafe_allow_html=True)
    r2.markdown(f'<div class="stat-card"><p class="stat-label">Data Dipelajari</p>'
                f'<p class="stat-value">{len(X_train):,}</p>'
                f'<p class="stat-note">catatan mesin untuk latihan</p></div>', unsafe_allow_html=True)
    r3.markdown(f'<div class="stat-card amber"><p class="stat-label">Sensor Dipakai</p>'
                f'<p class="stat-value">{len(X.columns)}</p>'
                f'<p class="stat-note">faktor kondisi mesin</p></div>', unsafe_allow_html=True)
    kegagalan_terdeteksi = report_dict["Gagal"]["recall"] * 100
    r4.markdown(f'<div class="stat-card red"><p class="stat-label">Kegagalan Terdeteksi</p>'
                f'<p class="stat-value">{kegagalan_terdeteksi:.0f}%</p>'
                f'<p class="stat-note">dari total mesin yg gagal</p></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB DATA
# ---------------------------------------------------------------------------
with tab_data:
    section("📋 Mengenal Datanya",
            "Sebelum menebak, kita lihat dulu bahan mentahnya. Setiap baris = satu mesin "
            "pada satu waktu; setiap kolom = satu informasi tentang mesin itu.")

    a, b, c = st.columns(3)
    a.metric("Jumlah Baris (catatan mesin)", f"{df.shape[0]:,}")
    b.metric("Jumlah Kolom (informasi)", df.shape[1])
    c.metric("Data Kosong/Hilang", int(df.isnull().sum().sum()))

    callout("Kabar baik: data ini bersih",
            "Tidak ada nilai yang hilang (0 data kosong), jadi kita tidak perlu repot "
            "menambal data. Ini kondisi ideal untuk membangun model.", "info")

    st.markdown("**Cuplikan 15 baris pertama:**")
    st.dataframe(df.head(15), use_container_width=True)

    section("🎯 Yang ingin ditebak: kolom 'Machine failure'")
    class_counts = df[TARGET_COLUMN].value_counts().sort_index()
    total = class_counts.sum()
    normal_n = int(class_counts.get(0, 0))
    gagal_n = int(class_counts.get(1, 0))
    gagal_pct = gagal_n / total * 100

    cc1, cc2 = st.columns([1.1, 1])
    with cc1:
        st.markdown(
            f'<div class="stat-grid">'
            f'<div class="stat-card green"><p class="stat-label">Mesin Normal</p>'
            f'<p class="stat-value">{normal_n:,}</p><p class="stat-note">{100-gagal_pct:.1f}% dari data</p></div>'
            f'<div class="stat-card red"><p class="stat-label">Mesin Gagal</p>'
            f'<p class="stat-value">{gagal_n:,}</p><p class="stat-note">{gagal_pct:.1f}% dari data</p></div>'
            f'</div>', unsafe_allow_html=True)
        callout(
            "Datanya 'timpang' (imbalanced) — dan ini penting!",
            f"Hanya <b>{gagal_pct:.1f}%</b> mesin yang benar-benar gagal. Ibarat mencari "
            "jarum di tumpukan jerami. Akibatnya, model yang asal menebak 'Normal' terus "
            f"pun sudah 'benar' {100-gagal_pct:.1f}% — padahal tidak berguna sama sekali. "
            "Itulah kenapa nanti kita <b>tidak cukup melihat akurasi saja</b>, tapi juga "
            "seberapa jago model menangkap kasus 'Gagal' yang langka itu.",
            "warn",
        )
    with cc2:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.bar(["Normal", "Gagal"], [normal_n, gagal_n], color=["#2E9E5B", "#D6453D"])
        ax.set_ylabel("Jumlah Mesin")
        ax.set_title("Perbandingan Jumlah Mesin")
        for i, v in enumerate([normal_n, gagal_n]):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontweight="bold")
        st.pyplot(fig)

# ---------------------------------------------------------------------------
# TAB EDA
# ---------------------------------------------------------------------------
with tab_eda:
    section("📊 Menyelidiki Data (Exploratory Data Analysis)",
            "Tahap 'kenalan' dengan data: kita cari tahu sensor mana yang berbeda "
            "nyata antara mesin normal dan mesin yang gagal.")

    section("🧹 Kolom yang sengaja dibuang sebelum melatih model")
    callout(
        "Kenapa ada kolom yang dibuang?",
        "Tidak semua kolom boleh dipakai. Ada dua jenis yang kami keluarkan:", "plain")
    id_dropped = [c for c in dropped_cols if c in ID_COLUMNS]
    leak_dropped = [c for c in dropped_cols if c in LEAKAGE_COLUMNS]
    colx, coly = st.columns(2)
    with colx:
        callout("🆔 Nomor identitas",
                f"<b>{', '.join(id_dropped)}</b><br>Cuma nomor urut/ID mesin — tidak "
                "menggambarkan kondisi mesin, jadi tak berguna untuk menebak.", "plain")
    with coly:
        callout("⚠️ Data 'bocor' (kecurangan tak sengaja)",
                f"<b>{', '.join(leak_dropped)}</b><br>Ini rincian jenis kerusakan yang "
                "baru diketahui <i>setelah</i> mesin gagal. Kalau dipakai, model seperti "
                "'menyontek jawaban' — curang dan tidak realistis di dunia nyata.", "warn")

    st.markdown(f"**✅ Sensor yang akhirnya dipakai ({len(X.columns)}):** "
                f"{', '.join(X.columns)}")
    with st.expander("📖 Arti tiap sensor (klik untuk buka)"):
        for feat in X.columns:
            if feat in FEATURE_INFO:
                nama, desc = FEATURE_INFO[feat]
                st.markdown(f"- **{feat}** — {nama}: {desc}")

    section("📈 Bandingkan sensor: mesin Normal vs mesin Gagal",
            "Pilih satu sensor. Kalau dua kurva (hijau & merah) terpisah jelas, berarti "
            "sensor itu bagus untuk membedakan mesin gagal dari yang normal.")
    numeric_feats = [c for c in X.columns if c != "Type"]
    selected_feat = st.selectbox("Pilih sensor untuk dilihat:", numeric_feats)
    fig, ax = plt.subplots(figsize=(9, 4))
    for cls, label, color in [(0, "Normal", "#2E9E5B"), (1, "Gagal", "#D6453D")]:
        mask = y == cls
        if mask.sum() > 0:
            sns.kdeplot(df_clean.loc[mask, selected_feat], label=label, fill=True,
                        alpha=0.35, ax=ax, color=color, linewidth=2)
    ax.set_xlabel(FEATURE_INFO.get(selected_feat, (selected_feat, ""))[0] + f"  ({selected_feat})")
    ax.set_ylabel("Kepadatan")
    ax.set_title(f"Sebaran '{selected_feat}': Normal vs Gagal")
    ax.legend()
    st.pyplot(fig)
    callout("Cara membaca grafik ini",
            "Sumbu datar = nilai sensor. Gunung <span style='color:#2E9E5B'><b>hijau</b></span> "
            "= mesin normal, gunung <span style='color:#D6453D'><b>merah</b></span> = mesin gagal. "
            "Semakin dua gunung ini <b>tidak bertumpuk</b>, semakin mudah model membedakan "
            "keduanya lewat sensor tersebut.", "info")

    section("🔗 Hubungan antar sensor (matriks korelasi)",
            "Angka mendekati +1 atau -1 = dua sensor bergerak sangat berkaitan. "
            "Mendekati 0 = hampir tak berhubungan.")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax2,
                linewidths=0.5, linecolor="white")
    st.pyplot(fig2)

# ---------------------------------------------------------------------------
# TAB MODEL
# ---------------------------------------------------------------------------
with tab_model:
    section("🌳 Seberapa Pintar Model Kita?",
            "Model sudah selesai belajar. Sekarang kita uji dengan data yang belum pernah "
            "ia lihat — seperti ujian bagi siswa.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Nilai saat Latihan", f"{train_acc*100:.2f}%",
              help="Akurasi pada data yang dipakai belajar.")
    m2.metric("Nilai saat Ujian", f"{test_acc*100:.2f}%",
              help="Akurasi pada data baru — ini yang paling penting.")
    m3.metric("Tebak asal 'Normal'", f"{baseline_acc*100:.2f}%",
              help="Skor kalau model malas & menebak Normal terus.")

    if train_acc >= 0.80 and test_acc >= 0.85:
        callout("✅ Lolos kriteria 'Sangat Baik'",
                f"Nilai latihan {train_acc*100:.1f}% (syarat ≥80%) dan nilai ujian "
                f"{test_acc*100:.1f}% (syarat ≥85%). Keduanya terlampaui.", "info")
    else:
        callout("⚠️ Belum memenuhi target",
                "Coba ubah 'Kedalaman Pohon' di panel kiri untuk memperbaiki hasil.", "warn")

    callout(
        "Jangan tertipu angka akurasi yang tinggi!",
        f"Akurasi {test_acc*100:.1f}% terdengar hebat, tapi menebak 'Normal' terus saja "
        f"sudah dapat {baseline_acc*100:.1f}%. Jadi nilai sesungguhnya model ada pada "
        "kemampuannya <b>menangkap mesin yang benar-benar gagal</b> — kita ukur ini "
        "lewat <i>Recall</i> dan <i>Precision</i> di bawah.",
        "tip",
    )

    st.markdown("")
    colL, colR = st.columns(2)
    with colL:
        st.markdown("**🎯 Tabel Tebakan Benar & Salah (Confusion Matrix)**")
        cm = confusion_matrix(y_test, y_test_pred)
        fig3, ax3 = plt.subplots(figsize=(5, 4.2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Normal", "Gagal"], yticklabels=["Normal", "Gagal"],
                    ax=ax3, linewidths=1, linecolor="white", cbar=False,
                    annot_kws={"size": 15, "weight": "bold"})
        ax3.set_xlabel("Tebakan Model")
        ax3.set_ylabel("Kenyataan")
        st.pyplot(fig3)
        tn, fp, fn, tp = cm.ravel()
        callout("Membaca tabel ini",
                f"↖️ <b>{tn}</b> mesin normal ditebak benar · "
                f"↘️ <b>{tp}</b> mesin gagal ditebak benar (bagus!) · "
                f"<b>{fn}</b> mesin gagal <i>terlewat</i> (ditebak normal) · "
                f"<b>{fp}</b> alarm palsu (normal dikira gagal).", "plain")
    with colR:
        st.markdown("**⭐ Sensor Paling Menentukan (Feature Importance)**")
        importance_df = pd.DataFrame({
            "Fitur": X.columns, "Importance": clf.feature_importances_
        }).sort_values("Importance", ascending=True)
        fig4, ax4 = plt.subplots(figsize=(5, 4.2))
        bars = ax4.barh(importance_df["Fitur"], importance_df["Importance"], color="#E8890C")
        ax4.set_xlabel("Tingkat Kepentingan")
        st.pyplot(fig4)
        top_feat = importance_df.iloc[-1]["Fitur"]
        callout("Artinya apa?",
                f"Sensor <b>{top_feat}</b> adalah faktor yang paling sering dipakai model "
                "untuk mengambil keputusan. Semakin panjang batangnya, semakin besar "
                "perannya dalam menentukan gagal/tidaknya mesin.", "plain")

    section("📋 Rapor Lengkap Model")
    callout(
        "Arti 3 istilah penting (bahasa sederhana)",
        "<b>Precision</b> = dari semua yang <i>ditebak gagal</i>, berapa % yang benar gagal "
        "(mengukur seberapa sering alarm palsu). &nbsp;•&nbsp; "
        "<b>Recall</b> = dari semua yang <i>benar-benar gagal</i>, berapa % yang berhasil "
        "ketangkap (mengukur berapa kerusakan yang lolos). &nbsp;•&nbsp; "
        "<b>F1-score</b> = nilai gabungan keduanya.",
        "info",
    )
    report_df = pd.DataFrame(report_dict).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)
    recall_gagal = report_dict["Gagal"]["recall"]
    prec_gagal = report_dict["Gagal"]["precision"]
    callout(
        "Kesimpulan kualitas model",
        f"Model berhasil menangkap <b>{recall_gagal*100:.0f}%</b> dari seluruh mesin yang "
        f"benar-benar gagal (recall), dan <b>{prec_gagal*100:.0f}%</b> dari alarm 'gagal' "
        "yang dibunyikannya memang tepat (precision). Untuk data yang sangat timpang, "
        "ini hasil yang kuat. 💡 Coba centang <i>'Seimbangkan perhatian antar kelas'</i> "
        "di panel kiri untuk melihat bagaimana angka recall bisa berubah.",
        "tip",
    )

# ---------------------------------------------------------------------------
# TAB PREDIKSI
# ---------------------------------------------------------------------------
with tab_predict:
    section("🔮 Coba Sendiri: Prediksi Kondisi Mesin",
            "Atur nilai sensor di bawah, lalu tekan tombol. Model akan langsung menebak "
            "apakah mesin dengan kondisi itu berisiko gagal.")

    callout("Tips mencoba",
            "Coba naikkan <b>Torsi</b> dan <b>Keausan Alat</b> ke nilai tinggi sekaligus, "
            "lalu lihat bagaimana prediksi berubah. Kombinasi beban berat + alat aus "
            "biasanya paling berisiko.", "tip")

    input_values = {}
    cols = st.columns(2)
    for i, feature in enumerate(X.columns):
        col = cols[i % 2]
        nama = FEATURE_INFO.get(feature, (feature, ""))[0]
        if feature == "Type" and type_encoder is not None:
            label = col.selectbox(f"{nama} ({feature})", options=list(type_encoder.classes_))
            input_values[feature] = type_encoder.transform([label])[0]
        else:
            mn, mx = float(X[feature].min()), float(X[feature].max())
            mean_val = float(X[feature].mean())
            input_values[feature] = col.slider(f"{nama} — {feature}", mn, mx, mean_val)

    st.markdown("")
    if st.button("🔍  PREDIKSI SEKARANG", type="primary", use_container_width=True):
        input_df = pd.DataFrame([input_values])[X.columns]
        pred = clf.predict(input_df)[0]
        proba = clf.predict_proba(input_df)[0]

        st.markdown("")
        if pred == 1:
            st.markdown(
                f'<div class="callout warn" style="text-align:center;font-size:16px;">'
                f'<span class="c-title" style="font-size:22px;">⚠️ MESIN BERISIKO GAGAL</span>'
                f'Tingkat keyakinan model: <b>{proba[1]*100:.1f}%</b><br>'
                f'Rekomendasi: jadwalkan pemeriksaan / pemeliharaan preventif segera.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="callout info" style="text-align:center;font-size:16px;'
                f'background:#EAF6EF;border-color:#B8E0C8;color:#1E5B37;">'
                f'<span class="c-title" style="font-size:22px;color:#227C46;">✅ MESIN NORMAL</span>'
                f'Tingkat keyakinan model: <b>{proba[0]*100:.1f}%</b><br>'
                f'Mesin diperkirakan beroperasi normal pada kondisi ini.</div>',
                unsafe_allow_html=True)

        cpa, cpb = st.columns([1, 1])
        with cpa:
            proba_df = pd.DataFrame({"Status": ["Normal", "Gagal"],
                                     "Keyakinan (%)": (proba * 100).round(1)})
            st.dataframe(proba_df, use_container_width=True, hide_index=True)
        with cpb:
            figp, axp = plt.subplots(figsize=(4.5, 2.6))
            axp.barh(["Normal", "Gagal"], proba, color=["#2E9E5B", "#D6453D"])
            axp.set_xlim(0, 1)
            axp.set_xlabel("Probabilitas")
            for i, v in enumerate(proba):
                axp.text(v, i, f" {v*100:.1f}%", va="center", fontweight="bold")
            st.pyplot(figp)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption(
    "Proyek Akhir Praktikum Data Mining 2026 · Algoritma: Decision Tree Classifier · "
    "Dataset: AI4I 2020 Predictive Maintenance (S. Matzka, UCI Machine Learning Repository, "
    "CC BY 4.0)"
)
