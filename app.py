import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

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
    
    # Sesuai permintaan: pola [1, -2, 1] dipertahankan
    T_factor = hbar**2 / (2 * mass * dx**2)
    T = diags([1, -2, 1], [-1, 0, 1], shape=(N, N)) * T_factor
    
    V_mat = diags([V], [0])
    H = T + V_mat
    
    # PERBAIKAN KONVERGENSI: 
    # ncv ditingkatkan untuk stabilitas Lanczos, maxiter & tol disesuaikan untuk potensial halus (HO)
    try:
        energies, states = eigsh(
            H, k=num_states, which='SM', 
            ncv=2*num_states+10, maxiter=2000, tol=1e-6
        )
    except ArpackNoConvergence:
        # Fallback dengan parameter lebih longgar jika konvergensi tetap gagal
        energies, states = eigsh(
            H, k=num_states, which='SM', 
            ncv=3*num_states, maxiter=5000, tol=1e-4
        )
    
    # Normalisasi eigenstate: ∫|ψ|² dx = 1
    for i in range(states.shape[1]):
        norm_factor = np.sqrt(np.trapezoid(np.abs(states[:, i])**2, x))
        states[:, i] /= norm_factor
        
    return energies, states

# =============================================================================
# ANTARMUKA STREAMLIT
# =============================================================================
def main():
    inject_custom_css()
    
    st.markdown("""
    <div class="header-container">
        <h1>Schrödinger Equation Simulator</h1>
        <p>Dikembangkan oleh Felix Marcellino Henrikus, S.Si.</p>
        <p>Program Studi Magister Sains Data, UKSW Salatiga</p>
        <p>Untuk digunakan dalam pembelajaran Fisika Kuantum di S1 Fisika, UKSW Salatiga</p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.header("️ Parameter Sistem")
    v_type = st.sidebar.selectbox(
        "Jenis Potensial",
        ["Sumur Potensial Tak Hingga", "Barrier Potensial", "Osilator Harmonik", "Custom"]
    )
    
    x_min = st.sidebar.number_input("Batas Kiri Domain (x_min)", -10.0, 0.0, -5.0, step=0.5, key="x_min")
    x_max = st.sidebar.number_input("Batas Kanan Domain (x_max)", 0.0, 10.0, 5.0, step=0.5, key="x_max")
    grid_points = st.sidebar.slider("Resolusi Grid (N)", 100, 800, 300, step=50, key="grid")
    mass = st.sidebar.number_input("Massa Partikel (m)", 0.1, 5.0, 1.0, step=0.1, key="mass")
    num_states = st.sidebar.slider("Jumlah Eigenstate yang Dihitung", 3, 8, 5, key="num_states")
    
    params = {}
    if v_type == "Barrier Potensial":
        params['width'] = st.sidebar.number_input("Lebar Barrier", 0.5, 5.0, 1.0, key="width")
        params['height'] = st.sidebar.number_input("Tinggi Barrier (V₀)", 0.0, 50.0, 5.0, key="height")
        params['center'] = st.sidebar.number_input("Pusat Barrier", x_min, x_max, 0.0, key="center")
    elif v_type == "Osilator Harmonik":
        params['omega'] = st.sidebar.number_input("Frekuensi Sudut (ω)", 0.1, 5.0, 1.0, key="omega")
    elif v_type == "Custom":
        params['a'] = st.sidebar.number_input("Koefisien a (x²)", 0.0, 10.0, 0.5, key="a")
        params['b'] = st.sidebar.number_input("Koefisien b (x⁴)", 0.0, 5.0, 0.1, key="b")
        
    x = np.linspace(x_min, x_max, grid_points)
    V = generate_potential(x, v_type, params)
    
    if st.sidebar.button("🚀 Hitung & Visualisasi Sistem"):
        with st.spinner("Melakukan komputasi numerik dan penyelesaian persamaan Schrödinger..."):
            energies, states = solve_time_independent_schrodinger(x, V, mass, num_states)
            
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#d35400']
            
            # --- KARTU 1: POTENSIAL & EIGENSTATE ---
            st.markdown('<div class="card-container"><h3>📐 Potensial dan Fungsi Gelombang (Eigenstate)</h3></div>', unsafe_allow_html=True)
            
            V_display = V.copy()
            if v_type == "Sumur Potensial Tak Hingga":
                max_energy = np.max(np.abs(energies)) if len(energies) > 0 else 10
                V_display = np.zeros_like(x)
                V_display[0] = max_energy * 1.2
                V_display[-1] = max_energy * 1.2
            
            fig_wave = go.Figure()
            fig_wave.add_trace(go.Scatter(x=x, y=V_display, mode='lines', name='V(x)', line=dict(color='#e74c3c', dash='dash', width=2)))
            
            for i in range(num_states):
                offset = energies[i]
                psi_scaled = states[:, i].real * 2 + offset
                fig_wave.add_trace(go.Scatter(x=x, y=psi_scaled, mode='lines', name=f'ψ_{i} + E_{i}', line=dict(color=colors[i], width=1.5)))
                
            fig_wave.update_layout(
                title="Visualisasi Eigenstate terhadap Potensial",
                xaxis_title="Posisi (x)", yaxis_title="Amplitudo",
                legend_title="State", hovermode="x unified", height=500
            )
            st.plotly_chart(fig_wave, use_container_width=True, key="wave_chart")
            
            # --- KARTU 2: DENSITAS PROBABILITAS ---
            st.markdown('<div class="card-container"><h3>📊 Densitas Probabilitas |ψ(x)|²</h3></div>', unsafe_allow_html=True)
            fig_prob = go.Figure()
            for i in range(num_states):
                fig_prob.add_trace(go.Scatter(x=x, y=np.abs(states[:, i])**2, mode='lines', name=f'n={i}', line=dict(color=colors[i])))
            fig_prob.update_layout(
                title="Distribusi Probabilitas Partikel",
                xaxis_title="Posisi (x)", yaxis_title="Probabilitas |ψ|²",
                hovermode="x unified", height=400
            )
            st.plotly_chart(fig_prob, use_container_width=True, key="prob_chart")
            
            # --- KARTU 3: LEVEL ENERGI ---
            st.markdown('<div class="card-container"><h3>⚡ Spektrum Level Energi</h3></div>', unsafe_allow_html=True)
            
            energy_df = pd.DataFrame({
                "Keadaan Kuantum (n)": [f"n={i}" for i in range(num_states)],
                "Energi Eigenvalue (Eₙ)": np.round(energies, 4)
            })
            st.dataframe(energy_df, use_container_width=True)
            
            fig_energy = go.Figure()
            fig_energy.add_trace(go.Scatter(
                x=[0] * num_states,
                y=energies,
                mode='markers+text',
                marker=dict(size=15, color=colors[:num_states]),
                text=[f"E_{i} = {energies[i]:.4f}" for i in range(num_states)],
                textposition="middle right",
                name="Level Energi"
            ))
            fig_energy.update_layout(
                title="Diagram Level Energi",
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis_title="Energi",
                height=300,
                showlegend=False,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_energy, use_container_width=True, key="energy_chart")
                       
    st.markdown('<div class="footer-container">© 2026 - Felix Marcellino Henrikus, S.Si. - UKSW Salatiga</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
