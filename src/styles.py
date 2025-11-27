"""
Custom CSS Styles for Premium Analytics Dashboard
Bright, vibrant colors for excellent visibility
"""

def get_custom_css():
    return """
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        
        /* Global font */
        * {
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Main App Background - Deep gradient */
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        }
        
        /* All text - Bright white for maximum visibility */
        .stApp, .stApp p, .stApp span, .stApp div, .stApp label {
            color: #FFFFFF !important;
        }
        
        /* Headers - Bright gradient text */
        h1, h2, h3, h4, h5, h6 {
            color: #00D9FF !important;
            font-weight: 700 !important;
            text-shadow: 0 0 20px rgba(0, 217, 255, 0.5);
        }
        
        /* Gradient text effect */
        .gradient-text {
            background: linear-gradient(135deg, #00D9FF 0%, #7B2FFF 50%, #FF2E63 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3rem !important;
            font-weight: 800 !important;
            animation: glow 2s ease-in-out infinite alternate;
        }
        
        @keyframes glow {
            from { filter: drop-shadow(0 0 10px #00D9FF); }
            to { filter: drop-shadow(0 0 20px #FF2E63); }
        }
        
        /* Tabs - Bright colors */
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 10px;
            backdrop-filter: blur(10px);
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #00D9FF !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
            border-radius: 10px;
            padding: 12px 24px;
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(0, 217, 255, 0.2);
            transform: translateY(-2px);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #00D9FF 0%, #7B2FFF 100%) !important;
            color: #FFFFFF !important;
            box-shadow: 0 5px 20px rgba(0, 217, 255, 0.4);
        }
        
        /* Buttons - Bright and vibrant */
        .stButton > button {
            background: linear-gradient(135deg, #FF2E63 0%, #FF6B9D 100%);
            color: #FFFFFF !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
            border: none;
            border-radius: 12px;
            padding: 12px 32px;
            box-shadow: 0 5px 20px rgba(255, 46, 99, 0.4);
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #FF6B9D 0%, #FF2E63 100%);
            transform: translateY(-3px);
            box-shadow: 0 8px 30px rgba(255, 46, 99, 0.6);
        }
        
        /* Input fields - White background with Bright Blue text */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > div,
        .stTextArea > div > div > textarea,
        .stNumberInput > div > div > input {
            background: #FFFFFF !important;
            color: #0044FF !important; /* Bright Blue */
            border: 2px solid #00D9FF !important;
            border-radius: 10px !important;
            font-size: 1rem !important;
            padding: 12px !important;
            font-weight: 700 !important;
        }
        
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > div:focus,
        .stTextArea > div > div > textarea:focus,
        .stNumberInput > div > div > input:focus {
            border-color: #FF2E63 !important;
            box-shadow: 0 0 15px rgba(255, 46, 99, 0.5) !important;
            color: #FF2E63 !important; /* Hot Pink on focus */
        }
        
        /* Selectbox dropdown items */
        .stSelectbox > div > div > ul > li {
            color: #0044FF !important;
            background: #FFFFFF !important;
        }
        
        /* Metric cards - Bright glassmorphism */
        .stMetric {
            background: linear-gradient(135deg, rgba(0, 217, 255, 0.2) 0%, rgba(123, 47, 255, 0.2) 100%);
            padding: 20px;
            border-radius: 15px;
            border: 2px solid rgba(0, 217, 255, 0.3);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 217, 255, 0.2);
        }
        
        .stMetric label {
            color: #00D9FF !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
        }
        
        .stMetric [data-testid="stMetricValue"] {
            color: #FFFFFF !important;
            font-size: 2rem !important;
            font-weight: 800 !important;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
        }
        
        .stMetric [data-testid="stMetricDelta"] {
            color: #7FFF00 !important;
            font-weight: 600 !important;
        }
        
        /* Dataframes - Bright theme */
        .stDataFrame {
            border: 2px solid #00D9FF;
            border-radius: 10px;
            overflow: hidden;
        }
        
        /* File uploader - Bright */
        .stFileUploader {
            background: rgba(0, 217, 255, 0.1);
            border: 2px dashed #00D9FF;
            border-radius: 15px;
            padding: 30px;
        }
        
        .stFileUploader label {
            color: #00D9FF !important;
            font-weight: 600 !important;
            font-size: 1.2rem !important;
        }
        
        /* Progress bar - Bright gradient */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #00D9FF 0%, #7B2FFF 50%, #FF2E63 100%);
        }
        
        /* Sidebar - Bright */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e 0%, #0f3460 100%);
            border-right: 2px solid #00D9FF;
        }
        
        [data-testid="stSidebar"] * {
            color: #FFFFFF !important;
        }
        
        /* Success/Info/Warning/Error messages - Bright */
        .stSuccess {
            background: rgba(127, 255, 0, 0.2) !important;
            color: #7FFF00 !important;
            border-left: 5px solid #7FFF00 !important;
            font-weight: 600 !important;
        }
        
        .stInfo {
            background: rgba(0, 217, 255, 0.2) !important;
            color: #00D9FF !important;
            border-left: 5px solid #00D9FF !important;
            font-weight: 600 !important;
        }
        
        .stWarning {
            background: rgba(255, 215, 0, 0.2) !important;
            color: #FFD700 !important;
            border-left: 5px solid #FFD700 !important;
            font-weight: 600 !important;
        }
        
        .stError {
            background: rgba(255, 46, 99, 0.2) !important;
            color: #FF2E63 !important;
            border-left: 5px solid #FF2E63 !important;
            font-weight: 600 !important;
        }
        
        /* Expander - Bright */
        .streamlit-expanderHeader {
            background: rgba(0, 217, 255, 0.1) !important;
            color: #00D9FF !important;
            font-weight: 600 !important;
            border: 2px solid #00D9FF !important;
            border-radius: 10px !important;
        }
        
        /* Radio buttons - Bright */
        .stRadio > label {
            color: #00D9FF !important;
            font-weight: 600 !important;
        }
        
        /* Checkbox - Bright */
        .stCheckbox > label {
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }
        
        /* Spinner - Bright */
        .stSpinner > div {
            border-top-color: #00D9FF !important;
        }
        
        /* Links - Bright cyan */
        a {
            color: #00D9FF !important;
            font-weight: 600 !important;
            text-decoration: none !important;
        }
        
        a:hover {
            color: #FF2E63 !important;
            text-shadow: 0 0 10px rgba(255, 46, 99, 0.5);
        }
        
        /* Scrollbar - Bright */
        ::-webkit-scrollbar {
            width: 12px;
            height: 12px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.2);
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #00D9FF 0%, #7B2FFF 100%);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #7B2FFF 0%, #FF2E63 100%);
        }
    </style>
    """
