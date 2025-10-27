import streamlit as st

# --- Page Config ---
st.set_page_config(page_title="EcoScnce | Welcome", page_icon="🌿", layout="wide")

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    body {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #e6f7f1, #c8e6f9);
        color: #003300;
    }
    /* Header */
    .topnav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 60px;
        background-color: rgba(255,255,255,0.9);
        border-bottom: 2px solid #b0d8c4;
        position: sticky;
        top: 0;
        z-index: 10;
    }
    .nav-links a {
        text-decoration: none;
        color: #006644;
        font-weight: 600;
        margin-right: 25px;
    }
    .nav-links a:hover {
        color: #004d33;
    }
    .logo {
        font-size: 22px;
        font-weight: 700;
        color: #00796b;
    }
    /* Hero Section */
    .hero {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 70px 60px;
    }
    .hero-text {
        width: 55%;
    }
    .hero-text h1 {
        font-size: 56px;
        font-weight: 700;
        color: #004d33;
    }
    .hero-text p {
        font-size: 18px;
        color: #003d26;
        margin-bottom: 25px;
    }
    .hero-image img {
        width: 420px;
        height: auto;
        border-radius: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .get-started {
        background-color: #009966;
        color: white;
        padding: 14px 32px;
        border-radius: 30px;
        font-size: 18px;
        text-decoration: none;
        font-weight: 600;
    }
    .get-started:hover {
        background-color: #007a52;
    }
    /* Info Grid */
    .info-grid {
        padding: 60px;
        background-color: #f6fffb;
        text-align: center;
    }
    .grid-title {
        font-size: 30px;
        font-weight: 700;
        color: #004d33;
        margin-bottom: 40px;
    }
    .grid-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 30px;
    }
    .grid-item {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    .grid-item img {
        width: 100%;
        border-radius: 10px;
    }
    .grid-item p {
        margin-top: 10px;
        font-weight: 600;
        color: #006644;
    }
    /* Chart Section */
    .chart-preview {
        background-color: #e6f7f1;
        padding: 60px;
        text-align: center;
    }
    .chart-preview h2 {
        color: #004d33;
        margin-bottom: 20px;
    }
    /* About Section */
    .about {
        background-color: #f2fff8;
        padding: 60px;
        text-align: center;
    }
    .about h2 {
        color: #004d33;
        font-size: 28px;
        margin-bottom: 15px;
    }
    .about p {
        font-size: 17px;
        color: #004d33;
        width: 70%;
        margin: 0 auto;
    }
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        background-color: #004d33;
        color: white;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="topnav">
    <div class="nav-links">
        <a href="#">Home</a>
        <a href="#about">About</a>
    </div>
    <div class="logo">🌿 EcoScnce</div>
</div>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
<div class="hero">
    <div class="hero-text">
        <h1>Welcome to EcoScnce</h1>
        <p>Empowering sustainability decisions with intelligent insights. 
        Discover how your products impact our planet and explore eco-friendly alternatives with ease.</p>
    </div>
    <div class="hero-image">
        <img src="https://images.unsplash.com/photo-1503264116251-35a269479413?auto=format&fit=crop&w=600&q=80" alt="Nature image" style="border-radius: 20px;">
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")  # spacing

# --- Button to switch to predictor page ---
if st.button(" Get Started"):
    st.switch_page("./pages/1_Predictor_Page.py")
  
# --- Info Grid Section ---
st.markdown("""
<div class="info-grid">
    <div class="grid-title">Why Choose EcoScnce?</div>
    <div class="grid-container">
        <div class="grid-item">
            <img src="https://images.unsplash.com/photo-1611854494547-1c3b6c2745d5?auto=format&fit=crop&w=400&q=60" />
            <p>Promotes Eco Awareness</p>
        </div>
        <div class="grid-item">
            <img src="https://images.unsplash.com/photo-1549880338-65ddcdfd017b?auto=format&fit=crop&w=400&q=60" />
            <p>Find Sustainable Products</p>
        </div>
        <div class="grid-item">
            <img src="https://images.unsplash.com/photo-1581574203170-2dc15d62f682?auto=format&fit=crop&w=400&q=60" />
            <p>AI-Powered Insights</p>
        </div>
        <div class="grid-item">
            <img src="https://images.unsplash.com/photo-1523978591478-c753949ff840?auto=format&fit=crop&w=400&q=60" />
            <p>Alternative Recommendations</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Chart Preview ---
st.markdown("""
<div class="chart-preview">
    <h2>🔍 See the Sustainability Score & Alternatives</h2>
    <p>Analyze your product’s sustainability level and discover eco-friendly substitutes that make a real difference.</p>
</div>
""", unsafe_allow_html=True)

# --- About Section ---
st.markdown("""
<div class="about" id="about">
    <h2>About EcoScnce</h2>
    <p>EcoScnce is a sustainability prediction system designed to encourage greener product choices. 
    Using intelligent data and AI models, it helps users identify eco-friendly products, 
    understand sustainability scores, and find better alternatives to make a positive environmental impact.</p>
</div>
""", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class="footer">
    © 2025 EcoScnce | All rights reserved.
</div>
""", unsafe_allow_html=True)
