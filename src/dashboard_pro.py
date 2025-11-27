import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.auth import AuthManager
from src.eda import (create_distribution_analysis, create_time_series_decomposition,
                     create_correlation_heatmap, create_temporal_patterns, create_statistical_summary)
from src.data.loader import load_data, standardize_columns
from src.data.cleaning import clean_timestamps, handle_duplicates, impute_missing, clean_numeric
from src.features.engineering import prepare_ml_features
from src.models.baseline import train_predict_prophet
from src.models.sarima import train_sarima
from src.evaluation.metrics import calculate_metrics
from src.styles import get_custom_css

# Page config
st.set_page_config(
    page_title="Border Crossing Analytics Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Apply Custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Initialize auth manager
auth = AuthManager()

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None

# ============================================================================
# AUTHENTICATION PAGES
# ============================================================================

def show_login_page():
    """Display login page."""
    st.markdown('<h1 class="gradient-text">🛂 Border Crossing Analytics Pro</h1>', unsafe_allow_html=True)
    st.markdown("### **Premium Forecasting Platform**")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["🔐 Login", "📝 Register", "👑 Admin"])
        
        with tab1:
            st.subheader("Login to Your Account")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("🚀 Login", use_container_width=True):
                if username and password:
                    result = auth.login(username, password)
                    if result['success']:
                        st.session_state.logged_in = True
                        st.session_state.user_data = result['user_data']
                        st.session_state.is_admin = False
                        st.success(result['message'])
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.warning("Please enter username and password")
        
        with tab2:
            st.subheader("Create New Account")
            
            reg_username = st.text_input("Username", key="reg_username")
            reg_email = st.text_input("Email", key="reg_email")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_password_confirm = st.text_input("Confirm Password", type="password", key="reg_password_confirm")
            
            user_type = st.radio("Account Type", ["👨‍🎓 Student (FREE)", "👔 Professional ($10.99/month)"])
            
            if "Student" in user_type:
                st.info("📚 Student accounts are **FREE** with verification!")
                college_name = st.text_input("College/University Name")
                college_email = st.text_input("College Email (.edu or student email)")
                college_id = st.file_uploader("Upload College ID (Image)", type=['jpg', 'jpeg', 'png', 'pdf'])
                
                st.markdown("""
                **Student Verification Requirements:**
                - ✅ Valid college/university email address
                - ✅ College ID card or student ID
                - ✅ Verification typically takes 24-48 hours
                """)
            else:
                st.info("💼 Professional Plan: $10.99/month (7-day free trial)")
                college_name = None
                college_email = None
                college_id = None
            
            if st.button("✨ Create Account", use_container_width=True):
                if not all([reg_username, reg_email, reg_password, reg_password_confirm]):
                    st.error("Please fill in all fields")
                elif reg_password != reg_password_confirm:
                    st.error("Passwords do not match")
                else:
                    # Determine user type
                    is_student = "Student" in user_type
                    email_to_use = college_email if is_student else reg_email
                    
                    # Save college ID if provided
                    college_id_path = None
                    if college_id and is_student:
                        id_dir = Path("user_data/college_ids")
                        id_dir.mkdir(parents=True, exist_ok=True)
                        college_id_path = id_dir / f"{reg_username}_{college_id.name}"
                        with open(college_id_path, "wb") as f:
                            f.write(college_id.getbuffer())
                    
                    result = auth.register_user(
                        username=reg_username,
                        email=email_to_use,
                        password=reg_password,
                        user_type='student' if is_student else 'regular',
                        college_name=college_name,
                        college_id_path=college_id_path
                    )
                    
                    if result['success']:
                        st.success(result['message'])
                        st.balloons()
                    else:
                        st.error(result['message'])
        
        with tab3:
            st.subheader("👑 Admin Access")
            st.warning("⚠️ Authorized Personnel Only")
            
            admin_username = st.text_input("Admin Username", key="admin_username")
            admin_password = st.text_input("Admin Password", type="password", key="admin_password")
            
            if st.button("🔓 Admin Login", use_container_width=True):
                if admin_username and admin_password:
                    result = auth.admin_login(admin_username, admin_password)
                    if result['success']:
                        st.session_state.logged_in = True
                        st.session_state.user_data = result['user_data']
                        st.session_state.is_admin = True
                        st.success(result['message'])
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.warning("Please enter admin credentials")


def show_admin_panel():
    """Display admin panel with full access to everything."""
    st.markdown('<h1 class="gradient-text">👑 Admin Control Panel</h1>', unsafe_allow_html=True)
    st.markdown("### **Full System Access - All Features Unlocked**")
    
    # Sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("**👑 ADMIN MODE**")
        st.success("✅ Unlimited Access")
        st.info("🔓 All Features Available")
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.user_data = None
            st.session_state.is_admin = False
            st.rerun()
    
    # Admin tabs - ALL 8 TABS
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 System Dashboard",
        "👥 User Management",
        "✅ Student Verification",
        "💰 Subscriptions",
        "📂 Data Upload",
        "📊 EDA",
        "🤖 Model Training",
        "📈 Results"
    ])
    
    # TAB 1: System Dashboard
    with tab1:
        st.header("📊 System Overview")
        
        stats = auth.get_user_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👥 Total Users", stats['total_users'])
        col2.metric("🎓 Students", stats['students'])
        col3.metric("💼 Professionals", stats['professionals'])
        col4.metric("✅ Active Subs", stats['active_subscriptions'])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🚫 Blocked Users", stats['blocked_users'])
        col2.metric("⏳ Pending Verifications", stats['pending_verifications'])
        col3.metric("📈 Total Revenue", f"${stats['active_subscriptions'] * 10.99:.2f}/mo")
    
    # TAB 2: User Management
    with tab2:
        st.header("👥 User Management")
        
        users = auth.get_all_users()
        
        if users:
            user_list = []
            for username, data in users.items():
                user_list.append({
                    'Username': username,
                    'Email': data['email'],
                    'Type': data['user_type'],
                    'Status': data['subscription_status'],
                    'Blocked': data.get('blocked', False),
                    'Registered': data['registration_date'][:10]
                })
            
            df_users = pd.DataFrame(user_list)
            st.dataframe(df_users, use_container_width=True)
            
            st.subheader("User Actions")
            col1, col2 = st.columns(2)
            
            with col1:
                selected_user = st.selectbox("Select User", list(users.keys()))
                
                if st.button("🚫 Block User", use_container_width=True):
                    if auth.block_user(selected_user):
                        st.success(f"Blocked {selected_user}")
                        st.rerun()
                
                if st.button("✅ Unblock User", use_container_width=True):
                    if auth.unblock_user(selected_user):
                        st.success(f"Unblocked {selected_user}")
                        st.rerun()
            
            with col2:
                if st.button("🗑️ Delete User", use_container_width=True):
                    if auth.delete_user(selected_user):
                        st.success(f"Deleted {selected_user}")
                        st.rerun()
                
                new_status = st.selectbox("Change Subscription", 
                                         ['free', 'trial', 'active', 'expired'])
                if st.button("💰 Update Subscription", use_container_width=True):
                    if auth.update_user_subscription(selected_user, new_status):
                        st.success(f"Updated {selected_user} to {new_status}")
                        st.rerun()
        else:
            st.info("No users registered yet")
    
    # TAB 3: Student Verification
    with tab3:
        st.header("✅ Student Verification")
        
        users = auth.get_all_users()
        pending_students = {k: v for k, v in users.items() 
                          if v['user_type'] == 'student' and not v.get('college_id_verified', False)}
        
        if pending_students:
            for username, data in pending_students.items():
                with st.expander(f"🎓 {username} - {data['college_name']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Email:** {data['email']}")
                        st.write(f"**College:** {data['college_name']}")
                        st.write(f"**Registered:** {data['registration_date'][:10]}")
                    
                    with col2:
                        if data.get('college_id_path'):
                            st.write(f"**ID Path:** {data['college_id_path']}")
                            if Path(data['college_id_path']).exists():
                                st.image(str(data['college_id_path']), width=300)
                    
                    if st.button(f"✅ Verify {username}", key=f"verify_{username}"):
                        if auth.verify_student_id(username):
                            st.success(f"Verified {username}")
                            st.rerun()
        else:
            st.info("No pending student verifications")
    
    # TAB 4: Subscription Management
    with tab4:
        st.header("💰 Subscription Management")
        
        users = auth.get_all_users()
        active_subs = {k: v for k, v in users.items() 
                      if v.get('subscription_status') == 'active'}
        
        if active_subs:
            sub_list = []
            for username, data in active_subs.items():
                sub_list.append({
                    'Username': username,
                    'Type': data['user_type'],
                    'Start Date': data.get('subscription_date', 'N/A')[:10],
                    'Monthly Fee': '$10.99' if data['user_type'] == 'regular' else 'FREE'
                })
            
            df_subs = pd.DataFrame(sub_list)
            st.dataframe(df_subs, use_container_width=True)
            
            total_revenue = len([u for u in active_subs.values() if u['user_type'] == 'regular']) * 10.99
            st.metric("💵 Monthly Revenue", f"${total_revenue:.2f}")
        else:
            st.info("No active subscriptions")
    
    # TAB 5: Data Upload (Full Access)
    with tab5:
        st.header("📂 Data Upload & Processing")
        
        uploaded_file = st.file_uploader("Upload Border Crossing Data (CSV/Excel)", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                with st.spinner("Processing data..."):
                    temp_path = Path("data/raw") / uploaded_file.name
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    df_raw = load_data(temp_path)
                    st.success(f"✅ Loaded {len(df_raw):,} rows")
                    
                    progress_bar = st.progress(0)
                    df_clean = standardize_columns(df_raw)
                    progress_bar.progress(33)
                    
                    if 'date' in df_clean.columns:
                        df_clean = clean_timestamps(df_clean, date_col='date')
                    if 'value' in df_clean.columns:
                        df_clean = clean_numeric(df_clean, col='value')
                    progress_bar.progress(66)
                    
                    df_clean = handle_duplicates(df_clean)
                    df_clean = impute_missing(df_clean)
                    progress_bar.progress(100)
                    
                    st.session_state.df = df_clean
                    st.success("✅ Data Pipeline Complete!")
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")
        
        elif Path("data/processed/cleaned_data.csv").exists():
            st.info("ℹ️ Using existing processed data")
            if st.session_state.df is None:
                st.session_state.df = pd.read_csv("data/processed/cleaned_data.csv")
                if 'timestamp' in st.session_state.df.columns:
                    st.session_state.df['timestamp'] = pd.to_datetime(st.session_state.df['timestamp'])
                if 'value' in st.session_state.df.columns:
                    st.session_state.df = clean_numeric(st.session_state.df, col='value')
        
        if st.session_state.df is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📊 Total Rows", f"{len(st.session_state.df):,}")
            col2.metric("📅 From", f"{st.session_state.df['timestamp'].min().date()}")
            col3.metric("📅 To", f"{st.session_state.df['timestamp'].max().date()}")
            col4.metric("📈 Total Volume", f"{st.session_state.df['value'].sum()/1e6:.1f}M")
    
    # TAB 6: EDA (Full Access)
    with tab6:
        st.header("📊 Exploratory Data Analysis")
        
        if st.session_state.df is not None:
            df = st.session_state.df.copy()
            df_agg = df.groupby('timestamp')['value'].sum().reset_index()
            
            # Statistical Summary
            st.subheader("📈 Statistical Summary")
            stats_df = create_statistical_summary(df_agg)
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(stats_df.iloc[:6], use_container_width=True)
            with col2:
                st.dataframe(stats_df.iloc[6:], use_container_width=True)
            
            # Distribution Analysis
            st.subheader("📊 Distribution Analysis")
            fig_dist = create_distribution_analysis(df_agg)
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Temporal Patterns
            st.subheader("🕐 Temporal Patterns")
            fig_temporal = create_temporal_patterns(df_agg)
            st.plotly_chart(fig_temporal, use_container_width=True)
            
            # Time Series Decomposition
            st.subheader("📉 Time Series Decomposition")
            with st.spinner("Performing seasonal decomposition..."):
                try:
                    fig_decomp = create_time_series_decomposition(df_agg)
                    st.plotly_chart(fig_decomp, use_container_width=True)
                except Exception as e:
                    st.warning(f"Decomposition requires more data: {e}")
            
            # Correlation Analysis
            st.subheader("🔗 Feature Correlation")
            df_ml = prepare_ml_features(df_agg)
            fig_corr = create_correlation_heatmap(df_ml)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("⚠️ Please upload data first")
    
    # TAB 7: Model Training (Full Access)
    with tab7:
        st.header("🤖 Model Training & Forecasting")
        
        if st.session_state.df is not None:
            df_agg = st.session_state.df.groupby('timestamp')['value'].sum().reset_index()
            df_agg['year'] = df_agg['timestamp'].dt.year
            
            train_data = df_agg[df_agg['year'] <= 2024]
            test_data = df_agg[df_agg['year'] == 2025]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🎓 Training Data (≤2024)", f"{len(train_data):,} days")
            col2.metric("🧪 Test Data (2025)", f"{len(test_data):,} days")
            if len(train_data) + len(test_data) > 0:
                col3.metric("📊 Split Ratio", f"{len(train_data)/(len(train_data)+len(test_data))*100:.1f}%")
            
            if len(test_data) > 0:
                if st.button("🚀 Train All Models", use_container_width=True):
                    with st.spinner("Training models..."):
                        results = {}
                        y_test_actual = test_data['value'].values
                        test_dates = test_data['timestamp'].values
                        
                        df_ml = prepare_ml_features(df_agg)
                        train_ml = df_ml[df_ml['year'] <= 2024].drop('year', axis=1)
                        test_ml = df_ml[df_ml['year'] == 2025].drop('year', axis=1)
                        
                        st.write("📊 Training Prophet...")
                        results['Prophet'] = train_predict_prophet(train_data, len(test_data))['forecast'].tail(len(test_data))['yhat'].values
                        
                        st.write("📈 Training SARIMA...")
                        try:
                            results['SARIMA'] = train_sarima(train_data, test_data)['test_pred']
                        except:
                            st.warning("SARIMA training skipped")
                        
                        st.write("🌲 Training XGBoost...")
                        from xgboost import XGBRegressor
                        feature_cols = [col for col in train_ml.columns if col not in ['value', 'timestamp']]
                        xgb = XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, random_state=42)
                        xgb.fit(train_ml[feature_cols], train_ml['value'])
                        results['XGBoost'] = xgb.predict(test_ml[feature_cols])
                        
                        st.write("💡 Training LightGBM...")
                        from lightgbm import LGBMRegressor
                        lgbm = LGBMRegressor(n_estimators=500, num_leaves=128, learning_rate=0.03, random_state=42, verbose=-1)
                        lgbm.fit(train_ml[feature_cols], train_ml['value'])
                        results['LightGBM'] = lgbm.predict(test_ml[feature_cols])
                        
                        st.session_state.comparison_results = {
                            'actual': y_test_actual,
                            'dates': test_dates,
                            'predictions': results
                        }
                        
                        st.success("✅ All models trained!")
            else:
                st.warning("⚠️ No 2025 data for testing")
        else:
            st.warning("⚠️ Upload data first")
    
    # TAB 8: Results (Full Access)
    with tab8:
        st.header("📈 Results & Model Comparison")
        
        if st.session_state.comparison_results:
            results = st.session_state.comparison_results
            
            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results['dates'], y=results['actual'],
                name='Actual', mode='lines+markers',
                line=dict(color='#FFD700', width=4)
            ))
            
            colors = {'Prophet': '#FF6B6B', 'SARIMA': '#4ECDC4', 'XGBoost': '#45B7D1', 'LightGBM': '#96CEB4'}
            for model_name, preds in results['predictions'].items():
                min_len = min(len(results['actual']), len(preds))
                fig.add_trace(go.Scatter(
                    x=results['dates'][:min_len], y=preds[:min_len],
                    name=model_name, mode='lines',
                    line=dict(color=colors.get(model_name, '#888'), width=2)
                ))
            
            fig.update_layout(
                title="Model Predictions vs Actual (2025)",
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            st.subheader("📊 Performance Metrics")
            metrics_data = []
            for model_name, preds in results['predictions'].items():
                min_len = min(len(results['actual']), len(preds))
                if min_len > 0:
                    metrics = calculate_metrics(results['actual'][:min_len], preds[:min_len])
                    metrics['Model'] = model_name
                    metrics['Accuracy (%)'] = (1 - metrics['MAPE']/100) * 100
                    metrics_data.append(metrics)
            
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(
                df_metrics[['Model', 'R²', 'RMSE', 'MAE', 'MAPE', 'Accuracy (%)']].style.format({
                    'R²': '{:.4f}', 'RMSE': '{:.2f}', 'MAE': '{:.2f}',
                    'MAPE': '{:.2f}%', 'Accuracy (%)': '{:.2f}%'
                }).highlight_max(axis=0, subset=['R²', 'Accuracy (%)'], color='lightgreen')
                .highlight_min(axis=0, subset=['RMSE', 'MAE', 'MAPE'], color='lightgreen'),
                use_container_width=True
            )
            
            best_model = df_metrics.loc[df_metrics['R²'].idxmax(), 'Model']
            best_r2 = df_metrics['R²'].max()
            best_acc = df_metrics['Accuracy (%)'].max()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🏆 Best Model", best_model)
            col2.metric("📈 Best R²", f"{best_r2:.4f}")
            col3.metric("✅ Best Accuracy", f"{best_acc:.2f}%")
            col4.metric("📊 Models", len(df_metrics))
        else:
            st.info("ℹ️ Train models first")



def show_dashboard():
    """Display main dashboard for logged-in users."""
    user = st.session_state.user_data
    
    # Header
    st.markdown('<h1 class="gradient-text">🛂 Border Crossing Analytics Pro</h1>', unsafe_allow_html=True)
    st.markdown(f"### Welcome, **{user['username']}** {'🎓' if user['user_type'] == 'student' else '💼'}")
    
    # Sidebar user info
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"**Account:** {user['user_type'].title()}")
        st.markdown(f"**Status:** {user['subscription_status'].title()}")
        if user['user_type'] == 'student':
            st.success("✅ FREE Student Account")
        elif user['subscription_status'] == 'trial':
            st.info("🎁 Trial Active")
        elif user['subscription_status'] == 'active':
            st.success("✅ Premium Active")
        
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.user_data = None
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📂 Data Upload",
        "📊 Exploratory Data Analysis",
        "🤖 Model Training & Forecasting",
        "📈 Results & Comparison"
    ])
    
    # TAB 1: Data Upload
    with tab1:
        st.header("📂 Data Upload & Processing")
        
        uploaded_file = st.file_uploader("Upload Border Crossing Data (CSV/Excel)", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                with st.spinner("Processing data..."):
                    temp_path = Path("data/raw") / uploaded_file.name
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    df_raw = load_data(temp_path)
                    st.success(f"✅ Loaded {len(df_raw):,} rows")
                    
                    progress_bar = st.progress(0)
                    df_clean = standardize_columns(df_raw)
                    progress_bar.progress(33)
                    
                    if 'date' in df_clean.columns:
                        df_clean = clean_timestamps(df_clean, date_col='date')
                    if 'value' in df_clean.columns:
                        df_clean = clean_numeric(df_clean, col='value')
                    progress_bar.progress(66)
                    
                    df_clean = handle_duplicates(df_clean)
                    df_clean = impute_missing(df_clean)
                    progress_bar.progress(100)
                    
                    st.session_state.df = df_clean
                    st.success("✅ Data Pipeline Complete!")
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")
        
        elif Path("data/processed/cleaned_data.csv").exists():
            st.info("ℹ️ Using existing processed data")
            if st.session_state.df is None:
                st.session_state.df = pd.read_csv("data/processed/cleaned_data.csv")
                if 'timestamp' in st.session_state.df.columns:
                    st.session_state.df['timestamp'] = pd.to_datetime(st.session_state.df['timestamp'])
                if 'value' in st.session_state.df.columns:
                    st.session_state.df = clean_numeric(st.session_state.df, col='value')
        
        if st.session_state.df is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📊 Total Rows", f"{len(st.session_state.df):,}")
            col2.metric("📅 From", f"{st.session_state.df['timestamp'].min().date()}")
            col3.metric("📅 To", f"{st.session_state.df['timestamp'].max().date()}")
            col4.metric("📈 Total Volume", f"{st.session_state.df['value'].sum()/1e6:.1f}M")
    
    # TAB 2: EDA
    with tab2:
        st.header("📊 Exploratory Data Analysis")
        
        if st.session_state.df is not None:
            df = st.session_state.df.copy()
            df_agg = df.groupby('timestamp')['value'].sum().reset_index()
            
            # Statistical Summary
            st.subheader("📈 Statistical Summary")
            stats_df = create_statistical_summary(df_agg)
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(stats_df.iloc[:6], use_container_width=True)
            with col2:
                st.dataframe(stats_df.iloc[6:], use_container_width=True)
            
            # Distribution Analysis
            st.subheader("📊 Distribution Analysis")
            fig_dist = create_distribution_analysis(df_agg)
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Temporal Patterns
            st.subheader("🕐 Temporal Patterns")
            fig_temporal = create_temporal_patterns(df_agg)
            st.plotly_chart(fig_temporal, use_container_width=True)
            
            # Time Series Decomposition
            st.subheader("📉 Time Series Decomposition")
            with st.spinner("Performing seasonal decomposition..."):
                try:
                    fig_decomp = create_time_series_decomposition(df_agg)
                    st.plotly_chart(fig_decomp, use_container_width=True)
                except Exception as e:
                    st.warning(f"Decomposition requires more data: {e}")
            
            # Correlation Analysis
            st.subheader("🔗 Feature Correlation")
            df_ml = prepare_ml_features(df_agg)
            fig_corr = create_correlation_heatmap(df_ml)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("⚠️ Please upload data first")
    
    # TAB 3: Model Training
    with tab3:
        st.header("🤖 Model Training & Forecasting")
        
        if st.session_state.df is not None:
            df_agg = st.session_state.df.groupby('timestamp')['value'].sum().reset_index()
            df_agg['year'] = df_agg['timestamp'].dt.year
            
            train_data = df_agg[df_agg['year'] <= 2024]
            test_data = df_agg[df_agg['year'] == 2025]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🎓 Training Data (≤2024)", f"{len(train_data):,} days")
            col2.metric("🧪 Test Data (2025)", f"{len(test_data):,} days")
            if len(train_data) + len(test_data) > 0:
                col3.metric("📊 Split Ratio", f"{len(train_data)/(len(train_data)+len(test_data))*100:.1f}%")
            
            if len(test_data) > 0:
                if st.button("🚀 Train All Models", use_container_width=True):
                    with st.spinner("Training models..."):
                        results = {}
                        y_test_actual = test_data['value'].values
                        test_dates = test_data['timestamp'].values
                        
                        df_ml = prepare_ml_features(df_agg)
                        train_ml = df_ml[df_ml['year'] <= 2024].drop('year', axis=1)
                        test_ml = df_ml[df_ml['year'] == 2025].drop('year', axis=1)
                        
                        st.write("📊 Training Prophet...")
                        results['Prophet'] = train_predict_prophet(train_data, len(test_data))['forecast'].tail(len(test_data))['yhat'].values
                        
                        st.write("📈 Training SARIMA...")
                        try:
                            results['SARIMA'] = train_sarima(train_data, test_data)['test_pred']
                        except:
                            st.warning("SARIMA training skipped")
                        
                        st.write("🌲 Training XGBoost...")
                        from xgboost import XGBRegressor
                        feature_cols = [col for col in train_ml.columns if col not in ['value', 'timestamp']]
                        xgb = XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, random_state=42)
                        xgb.fit(train_ml[feature_cols], train_ml['value'])
                        results['XGBoost'] = xgb.predict(test_ml[feature_cols])
                        
                        st.write("💡 Training LightGBM...")
                        from lightgbm import LGBMRegressor
                        lgbm = LGBMRegressor(n_estimators=500, num_leaves=128, learning_rate=0.03, random_state=42, verbose=-1)
                        lgbm.fit(train_ml[feature_cols], train_ml['value'])
                        results['LightGBM'] = lgbm.predict(test_ml[feature_cols])
                        
                        st.session_state.comparison_results = {
                            'actual': y_test_actual,
                            'dates': test_dates,
                            'predictions': results
                        }
                        
                        st.success("✅ All models trained!")
            else:
                st.warning("⚠️ No 2025 data for testing")
        else:
            st.warning("⚠️ Upload data first")
    
    # TAB 4: Results
    with tab4:
        st.header("📈 Results & Model Comparison")
        
        if st.session_state.comparison_results:
            results = st.session_state.comparison_results
            
            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results['dates'], y=results['actual'],
                name='Actual', mode='lines+markers',
                line=dict(color='#FFD700', width=4)
            ))
            
            colors = {'Prophet': '#FF6B6B', 'SARIMA': '#4ECDC4', 'XGBoost': '#45B7D1', 'LightGBM': '#96CEB4'}
            for model_name, preds in results['predictions'].items():
                min_len = min(len(results['actual']), len(preds))
                fig.add_trace(go.Scatter(
                    x=results['dates'][:min_len], y=preds[:min_len],
                    name=model_name, mode='lines',
                    line=dict(color=colors.get(model_name, '#888'), width=2)
                ))
            
            fig.update_layout(
                title="Model Predictions vs Actual (2025)",
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            st.subheader("📊 Performance Metrics")
            metrics_data = []
            for model_name, preds in results['predictions'].items():
                min_len = min(len(results['actual']), len(preds))
                if min_len > 0:
                    metrics = calculate_metrics(results['actual'][:min_len], preds[:min_len])
                    metrics['Model'] = model_name
                    metrics['Accuracy (%)'] = (1 - metrics['MAPE']/100) * 100
                    metrics_data.append(metrics)
            
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(
                df_metrics[['Model', 'R²', 'RMSE', 'MAE', 'MAPE', 'Accuracy (%)']].style.format({
                    'R²': '{:.4f}', 'RMSE': '{:.2f}', 'MAE': '{:.2f}',
                    'MAPE': '{:.2f}%', 'Accuracy (%)': '{:.2f}%'
                }).highlight_max(axis=0, subset=['R²', 'Accuracy (%)'], color='lightgreen')
                .highlight_min(axis=0, subset=['RMSE', 'MAE', 'MAPE'], color='lightgreen'),
                use_container_width=True
            )
            
            best_model = df_metrics.loc[df_metrics['R²'].idxmax(), 'Model']
            best_r2 = df_metrics['R²'].max()
            best_acc = df_metrics['Accuracy (%)'].max()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🏆 Best Model", best_model)
            col2.metric("📈 Best R²", f"{best_r2:.4f}")
            col3.metric("✅ Best Accuracy", f"{best_acc:.2f}%")
            col4.metric("📊 Models", len(df_metrics))
        else:
            st.info("ℹ️ Train models first")

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None

# Main app logic
if not st.session_state.logged_in:
    show_login_page()
elif st.session_state.is_admin:
    show_admin_panel()
else:
    show_dashboard()
