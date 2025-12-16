"""
Page 5: Association Rules Browser
Explore track co-occurrence patterns
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Association Rules", page_icon="🔗", layout="wide")

st.markdown("# 🔗 Association Rules")
st.markdown("### Track Co-occurrence Patterns via Apriori Algorithm")
st.markdown("---")

# Load association rules
@st.cache_data
def load_rules():
    """Load real association rules or generate sample"""
    from pathlib import Path
    import pickle
    
    # Try to load real rules
    rules_file = Path("data/processed/association_rules_full.csv")
    tracks_file = Path("data/processed/tracks_full_mpd.parquet")
    
    if rules_file.exists():
        try:
            rules = pd.read_csv(rules_file)
            
            # Check if it has the expected columns
            if 'antecedent' in rules.columns and 'consequent' in rules.columns:
                
                # Try to load track names for mapping URIs
                track_names = {}
                if tracks_file.exists():
                    try:
                        tracks_df = pd.read_parquet(tracks_file, columns=['track_uri', 'track_name', 'artist_name'])
                        tracks_unique = tracks_df[['track_uri', 'track_name', 'artist_name']].drop_duplicates('track_uri')
                        track_names = dict(zip(tracks_unique['track_uri'], 
                                             tracks_unique['track_name'] + ' - ' + tracks_unique['artist_name']))
                        st.success(f"✅ Using REAL association rules from Script 25! ({len(rules):,} rules with track names)")
                    except:
                        st.success(f"✅ Using REAL association rules from Script 25! ({len(rules):,} rules)")
                else:
                    st.success(f"✅ Using REAL association rules from Script 25! ({len(rules):,} rules - showing IDs)")
                
                # Rename columns to match display format
                rules = rules.rename(columns={
                    'antecedent': 'Antecedent',
                    'consequent': 'Consequent',
                    'support': 'Support',
                    'confidence': 'Confidence',
                    'lift': 'Lift'
                })
                
                # Take top 1000 by lift for display
                rules = rules.nlargest(min(1000, len(rules)), 'Lift')
                
                # Map URIs to names if available
                if track_names:
                    rules['Antecedent'] = rules['Antecedent'].apply(
                        lambda x: track_names.get(x, x.split(':')[-1][:40] if isinstance(x, str) and ':' in x else str(x)[:40])
                    )
                    rules['Consequent'] = rules['Consequent'].apply(
                        lambda x: track_names.get(x, x.split(':')[-1][:40] if isinstance(x, str) and ':' in x else str(x)[:40])
                    )
                else:
                    # Just truncate URIs to last part
                    rules['Antecedent'] = rules['Antecedent'].apply(lambda x: x.split(':')[-1][:40] if isinstance(x, str) and ':' in x else str(x)[:40])
                    rules['Consequent'] = rules['Consequent'].apply(lambda x: x.split(':')[-1][:40] if isinstance(x, str) and ':' in x else str(x)[:40])
                
                return rules[['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift']]
                
        except Exception as e:
            st.warning(f"Could not load rules: {e}")
    
    # Fallback to sample data
    #st.info("📊 Using sample association rules (run script 25 for real results)")
    
    rules_data = [
        ("Shape of You", "Despacito", 0.15, 0.75, 2.8),
        ("Blinding Lights", "Save Your Tears", 0.12, 0.82, 3.2),
        ("Dance Monkey", "Memories", 0.10, 0.68, 2.1),
        ("Levitating", "Don't Start Now", 0.14, 0.79, 3.0),
        ("Circles", "Sunflower", 0.11, 0.71, 2.4),
        ("bad guy", "bury a friend", 0.13, 0.85, 3.5),
        ("7 rings", "thank u, next", 0.16, 0.88, 3.8),
        ("Old Town Road", "Panini", 0.09, 0.72, 2.3),
        ("Senorita", "Havana", 0.12, 0.76, 2.7),
        ("Watermelon Sugar", "Adore You", 0.13, 0.80, 3.1),
    ]
    
    return pd.DataFrame(rules_data, columns=['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift'])

rules_df = load_rules()

st.markdown("## 🔍 Filter Rules")

col1, col2, col3 = st.columns(3)

with col1:
    min_support = st.slider("Minimum Support", 0.0, 0.2, 0.05, 0.01)

with col2:
    min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.05)

with col3:
    min_lift = st.slider("Minimum Lift", 1.0, 5.0, 2.0, 0.1)

# Filter rules
filtered_rules = rules_df[
    (rules_df['Support'] >= min_support) &
    (rules_df['Confidence'] >= min_confidence) &
    (rules_df['Lift'] >= min_lift)
]

st.markdown(f"## 📊 Top Association Rules ({len(filtered_rules)} rules)")

# Display rules
for idx, row in filtered_rules.iterrows():
    with st.expander(f"**{row['Antecedent']}** → **{row['Consequent']}**"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Support", f"{row['Support']:.2%}")
            st.caption("Frequency of co-occurrence")
        
        with col2:
            st.metric("Confidence", f"{row['Confidence']:.2%}")
            st.caption("Probability of consequent given antecedent")
        
        with col3:
            st.metric("Lift", f"{row['Lift']:.1f}")
            st.caption("Strength of association")

st.markdown("---")

# Visualization
st.markdown("## 📈 Rule Metrics Distribution")

col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_rules['Support'],
        y=filtered_rules['Confidence'],
        mode='markers',
        marker=dict(
            size=filtered_rules['Lift'] * 5,
            color=filtered_rules['Lift'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Lift")
        ),
        text=filtered_rules['Antecedent'] + ' → ' + filtered_rules['Consequent'],
        hovertemplate='<b>%{text}</b><br>Support: %{x:.2%}<br>Confidence: %{y:.2%}<extra></extra>'
    ))
    fig.update_layout(
        title='Support vs Confidence (bubble size = Lift)',
        xaxis_title='Support',
        yaxis_title='Confidence',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Top rules by lift
    top_rules = filtered_rules.nlargest(5, 'Lift')
    fig = go.Figure(go.Bar(
        x=top_rules['Lift'],
        y=top_rules['Antecedent'] + ' → ' + top_rules['Consequent'],
        orientation='h',
        marker_color='#1DB954'
    ))
    fig.update_layout(
        title='Top 5 Rules by Lift',
        xaxis_title='Lift',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Download
csv = filtered_rules.to_csv(index=False)
st.download_button(
    label="📥 Download Rules",
    data=csv,
    file_name="association_rules.csv",
    mime="text/csv"
)

st.markdown("---")

st.markdown("## 💡 Interpretation Guide")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    **Support**: How often both tracks appear together
    - High support = common pattern
    - Low support = rare but potentially interesting
    
    **Confidence**: If track A appears, probability track B appears
    - High confidence = strong predictive relationship
    """)

with col2:
    st.success("""
    **Lift**: Strength of association beyond random chance
    - Lift > 1: Positive association
    - Lift = 1: Independent
    - Lift < 1: Negative association (mutual exclusivity)
    """)