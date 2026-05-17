import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import time

# =============================================================================
# KONFIGURASI HALAMAN & INJEKSI CSS
# =============================================================================
st.set_page_config(
    page_title="Schrödinger Equation Simulator",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_css():
    st.markdown("""
    <style>
        .header-container {
            background-color: #00699c;
            color: #ffffff;
            padding: 1.8rem 2rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .header-container h1 { margin: 0; font-size: 2.1rem; font-weight: 600; }
        .header-container p { margin: 0.35rem 0 0; font-size: 0.95rem; opacity: 0.95; }
        
        .card-container {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 3px 8px rgba(0,0,0,0.08);
        }
        .card-container h3 { margin-top: 0; color: #0f4c75; border-bottom: 2px solid #00699c; padding-bottom: 0.5rem; }
        
        .footer-container {
            text-align: center;
            padding: 1.2rem 0;
            margin-top: 2.5rem;
            font-size: 0.85rem;
            color: #5a6268;
            border-top: 1px solid #dee2e6;
        }
        .metric-card { background: #f8fafc; border-radius: 8px; padding: 1rem; text-align: center; }
        .stSlider > div > div > div > div { background-color: #00699c; }
        .stButton>button { background-color: #00699c; color: white; border-radius: 6px; }
        .stButton>button:hover { background-color: #005f8a; }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# FUNGSI FISIKA & KOMPUTASI NUMERIK
# =============================================================================
def generate_potential(x, v_type, params):
    """Membangkitkan array potensial berdasarkan jenis dan parameter."""
    if v_type == "Sumur Potensial Tak Hingga":
        V = np.zeros_like(x)
        V[0], V[-1] = 1e5, 1e5
        return V
    elif v_type == "Barrier Potensial":
        V = np.zeros_like(x)
        w, h, c = params.get('width', 1.0), params.get('height', 5.0), params.get('center', 0.0)
        mask = np.abs(x - c) < w / 2
        V[mask] = h
        return V
    elif v_type == "Osilator Harmonik":
        omega = params.get('omega', 1.0)
        return 0.5 * omega**2 * x**2
    elif v_type == "Custom":
        a = params.get('a', 0.5)
        b = params.get('b', 0.1)
        return a * x**2 + b * x**4
    return np.zeros_like(x)

def solve_time_independent_schrodinger(x, V, mass, num_states=5):
    """Menyelesaikan TISE 1D menggunakan metode beda hingga & sparse eigensolver."""
    hbar = 1.0
    dx = x[1] - x[0]
    N = len(x)
    
    T_factor = hbar**2 / (2 * mass * dx**2)
    T = diags([1, -2, 1], [-1, 0, 1], shape=(N, N)) * T_factor
    V_mat = diags([V], [0])
    H = T + V_mat
    
    energies, states = eigsh(H, k=num_states, which='SM')
    
    # PERBAIKAN: np.trapz diganti menjadi np.trapezoid untuk kompatibilitas NumPy ≥2.0
    for i in range(states.shape[1]):
        norm_factor = np.sqrt(np.trapezoid(np.abs(states[:, i])**2, x))
        states[:, i] /= norm_factor
        
    return energies, states

def evolve_wavefunction(x, states, energies, coefficients, t, hbar=1.0):
    """Evolusi waktu menggunakan dekomposisi spektral."""
    psi_t = np.zeros(len(x), dtype=complex)
    for n in range(len(energies)):
        psi_t += coefficients[n] * states[:, n] * np.exp(-1j * energies[n] * t / hbar)
    return psi_t

# =============================================================================
# ANTARMUKA STREAMLIT
# =============================================================================
def main():
    inject_custom_css()
    
    # HEADER
    st.markdown("""
    <div class="header-container">
        <h1>Schrödinger Equation Simulator</h1>
        <p>Dikembangkan oleh Felix Marcellino Henrikus, S.Si.</p>
        <p>Program Studi Magister Sains Data, UKSW Salatiga</p>
        <p>Untuk digunakan dalam pembelajaran Fisika Kuantum di S1 Fisika, UKSW Salatiga</p>
    </div>
    """, unsafe_allow_html=True)

    # SIDEBAR: INPUT PARAMETER
    st.sidebar.header("⚙️ Parameter Sistem")
    v_type = st.sidebar.selectbox(
        "Jenis Potensial",
        ["Sumur Potensial Tak Hingga", "Barrier Potensial", "Osilator Harmonik", "Custom"]
    )
    
    x_min = st.sidebar.number_input("Batas Kiri Domain (x_min)", -10.0, 0.0, -5.0, step=0.5)
    x_max = st.sidebar.number_input("Batas Kanan Domain (x_max)", 0.0, 10.0, 5.0, step=0.5)
    grid_points = st.sidebar.slider("Resolusi Grid (N)", 100, 800, 300, step=50)
    mass = st.sidebar.number_input("Massa Partikel (m)", 0.1, 5.0, 1.0, step=0.1)
    num_states = st.sidebar.slider("Jumlah Eigenstate yang Dihitung", 3, 8, 5)
    
    params = {}
    if v_type == "Barrier Potensial":
        params['width'] = st.sidebar.number_input("Lebar Barrier", 0.5, 5.0, 1.0)
        params['height'] = st.sidebar.number_input("Tinggi Barrier (V₀)", 0.0, 50.0, 5.0)
        params['center'] = st.sidebar.number_input("Pusat Barrier", x_min, x_max, 0.0)
    elif v_type == "Osilator Harmonik":
        params['omega'] = st.sidebar.number_input("Frekuensi Sudut (ω)", 0.1, 5.0, 1.0)
    elif v_type == "Custom":
        params['a'] = st.sidebar.number_input("Koefisien a (x²)", 0.0, 10.0, 0.5)
        params['b'] = st.sidebar.number_input("Koefisien b (x⁴)", 0.0, 5.0, 0.1)
        
    x = np.linspace(x_min, x_max, grid_points)
    V = generate_potential(x, v_type, params)
    
    # TOMBOL EKSEKUSI
    if st.sidebar.button("🚀 Hitung & Visualisasi Sistem"):
        with st.spinner("Melakukan komputasi numerik dan penyelesaian persamaan Schrödinger..."):
            energies, states = solve_time_independent_schrodinger(x, V, mass, num_states)
            
            c0 = np.zeros(num_states, dtype=complex)
            c0[0] = 1.0
            
            # --- KARTU 1: POTENSIAL & EIGENSTATE ---
            st.markdown('<div class="card-container"><h3>📐 Potensial dan Fungsi Gelombang (Eigenstate)</h3></div>', unsafe_allow_html=True)
            fig_wave = go.Figure()
            fig_wave.add_trace(go.Scatter(x=x, y=V, mode='lines', name='V(x)', line=dict(color='#e74c3c', dash='dash')))
            for i in range(num_states):
                offset = energies[i]
                fig_wave.add_trace(go.Scatter(x=x, y=states[:, i].real + offset, mode='lines', name=f'ψ_{i} (Bagian Real)'))
            fig_wave.update_layout(
                title="Visualisasi Eigenstate terhadap Potensial",
                xaxis_title="Posisi (x)", yaxis_title="Energi / Amplitudo",
                legend_title="State", hovermode="x unified"
            )
            st.plotly_chart(fig_wave, use_container_width=True)
            
            # --- KARTU 2: DENSITAS PROBABILITAS ---
            st.markdown('<div class="card-container"><h3>📊 Densitas Probabilitas |ψ(x)|²</h3></div>', unsafe_allow_html=True)
            fig_prob = go.Figure()
            for i in range(num_states):
                fig_prob.add_trace(go.Scatter(x=x, y=np.abs(states[:, i])**2, mode='lines', name=f'n={i}'))
            fig_prob.update_layout(
                title="Distribusi Probabilitas Partikel",
                xaxis_title="Posisi (x)", yaxis_title="Probabilitas |ψ|²",
                hovermode="x unified"
            )
            st.plotly_chart(fig_prob, use_container_width=True)
            
            # --- KARTU 3: LEVEL ENERGI ---
            st.markdown('<div class="card-container"><h3>⚡ Spektrum Level Energi</h3></div>', unsafe_allow_html=True)
            energy_df = st.DataFrame({
                "Keadaan Kuantum (n)": [f"n={i}" for i in range(num_states)],
                "Energi Eigenvalue (Eₙ)": np.round(energies, 4)
            })
            st.dataframe(energy_df, use_container_width=True, hide_index=True)
            
            # --- KARTU 4: EVOLUSI WAKTU & ANIMASI ---
            st.markdown('<div class="card-container"><h3>⏱️ Evolusi Waktu Paket Gelombang</h3></div>', unsafe_allow_html=True)
            
            st.markdown("Atur komposisi superposisi keadaan kuantum:")
            col1, col2, col3 = st.columns(3)
            with col1: c0[0] = st.number_input("Koefisien ψ₀", 0.0, 1.0, 1.0, step=0.1, format="%.2f")
            with col2: c0[1] = st.number_input("Koefisien ψ₁", 0.0, 1.0, 0.0, step=0.1, format="%.2f")
            with col3: c0[2] = st.number_input("Koefisien ψ₂", 0.0, 1.0, 0.0, step=0.1, format="%.2f")
            
            norm_c = np.sqrt(np.sum(np.abs(c0[:3])**2))
            if norm_c > 0:
                c0[:3] /= norm_c
                
            if st.button("▶️ Jalankan Animasi Evolusi"):
                placeholder = st.empty()
                max_time = 20.0
                frames = 60
                for t in np.linspace(0, max_time, frames):
                    psi_t = evolve_wavefunction(x, states, energies, c0, t)
                    prob_t = np.abs(psi_t)**2
                    
                    fig_ev = go.Figure()
                    fig_ev.add_trace(go.Scatter(x=x, y=prob_t, mode='lines', name='|ψ(x,t)|²', line=dict(color='#00699c')))
                    fig_ev.add_trace(go.Scatter(x=x, y=V/np.max(np.abs(states[0])**2)*0.5, mode='lines', name='V(x) (skala)', line=dict(color='#e74c3c', dash='dash')))
                    fig_ev.update_layout(
                        title=f"Dinamika Probabilitas |ψ(x,t)|² | t = {t:.2f}",
                        xaxis_title="Posisi (x)", yaxis_title="Densitas Probabilitas",
                        yaxis_range=[0, np.max(np.abs(states[0])**2)*2.2],
                        hovermode="x unified"
                    )
                    placeholder.plotly_chart(fig_ev, use_container_width=True)
                    time.sleep(0.05)
                    
            t_static = st.slider("Eksplorasi Waktu Statis (t)", 0.0, 20.0, 0.0, 0.05)
            psi_static = evolve_wavefunction(x, states, energies, c0, t_static)
            prob_static = np.abs(psi_static)**2
            fig_static = go.Figure()
            fig_static.add_trace(go.Scatter(x=x, y=prob_static, mode='lines', name='|ψ(x,t)|²', line=dict(color='#00699c')))
            fig_static.update_layout(
                title=f"Probabilitas pada t = {t_static:.2f}",
                xaxis_title="Posisi (x)", yaxis_title="Probabilitas", hovermode="x unified"
            )
            st.plotly_chart(fig_static, use_container_width=True)
            
            # --- KARTU 5: VALIDASI NORMALISASI ---
            st.markdown('<div class="card-container"><h3>✅ Validasi Normalisasi & Konsistensi Numerik</h3></div>', unsafe_allow_html=True)
            # PERBAIKAN: np.trapz diganti menjadi np.trapezoid
            norm_val = np.trapezoid(prob_static, x)
            st.markdown(f"""
            <div class="metric-card">
                <strong>Hasil Integrasi:</strong> ∫ |ψ(x,t)|² dx = <span style="color:#00699c; font-weight:bold;">{norm_val:.6f}</span>
            </div>
            <p style="margin-top:0.5rem; font-size:0.9rem; color:#555;">
            Nilai yang mendekati <strong>1.000000</strong> mengindikasikan bahwa fungsi gelombang terormalisasi secara konsisten sepanjang evolusi waktu, sesuai dengan postulat konservasi probabilitas dalam mekanika kuantum.
            </p>
            """, unsafe_allow_html=True)
            
    # FOOTER
    st.markdown('<div class="footer-container">© 2026 - Felix Marcellino Henrikus, S.Si. - UKSW Salatiga</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
