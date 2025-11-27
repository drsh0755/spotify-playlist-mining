"""
Page 7: Project Timeline
Development milestones and progress tracking
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Timeline", page_icon="⏱️", layout="wide")

st.markdown("# ⏱️ Project Timeline")
st.markdown("### Development Milestones and Progress")
st.markdown("---")

# Project phases with durations instead of dates
phases = [
    {
        'phase': 'Phase 1: Data Preparation',
        'duration_hours': 40,
        'tasks': [
            '✅ Data acquisition and exploration',
            '✅ Preprocessing pipeline development',
            '✅ Feature extraction',
            '✅ Co-occurrence matrix construction'
        ],
        'status': 'Completed',
        'scripts': '01-24'
    },
    {
        'phase': 'Phase 2: Core Models',
        'duration_hours': 60,
        'tasks': [
            '✅ Association rule mining',
            '✅ K-means clustering',
            '✅ Baseline recommendation systems',
            '✅ Model evaluation framework'
        ],
        'status': 'Completed',
        'scripts': '25-31'
    },
    {
        'phase': 'Phase 3: Advanced Models',
        'duration_hours': 50,
        'tasks': [
            '✅ SVD matrix factorization',
            '✅ Neural network recommender',
            '✅ Hybrid ensemble system',
            '✅ Predictive models'
        ],
        'status': 'Completed',
        'scripts': '32-35'
    },
    {
        'phase': 'Phase 4: Visualization & Dashboard',
        'duration_hours': 20,
        'tasks': [
            '✅ Figure generation (17 figures)',
            '✅ Dashboard development',
            '⏳ Final report writing',
            '⏳ Presentation preparation'
        ],
        'status': 'In Progress',
        'scripts': '36-42'
    }
]

# Phase 4 reminder banner
st.info("""
📝 **Phase 4 Update Reminder**

When Phase 4 is complete, remember to update:
1. Change status to 'Completed'
2. Update duration_hours (currently estimated at 20h)
3. Mark all tasks as ✅ 
4. Update this in: `dashboard/pages/7_Timeline.py`

**Current tasks:**
- ✅ Figure generation (DONE)
- ✅ Dashboard development (DONE)
- ⏳ Final report writing (TODO)
- ⏳ Presentation preparation (TODO)
""")

st.markdown("---")

# Timeline visualization - Duration bar chart
st.markdown("## ⏱️ Project Effort Distribution")

fig = go.Figure()

colors = {
    'Completed': '#1DB954',
    'In Progress': '#FFB020',
    'Planned': '#808080'
}

# Create horizontal bar chart showing duration
phase_names = [p['phase'] for p in phases]
durations = [p['duration_hours'] for p in phases]
phase_colors = [colors[p['status']] for p in phases]

fig.add_trace(go.Bar(
    x=durations,
    y=phase_names,
    orientation='h',
    marker=dict(color=phase_colors),
    text=[f"{d}h" for d in durations],
    textposition='inside',
    textfont=dict(size=14, color='white', weight='bold'),
    hovertemplate='<b>%{y}</b><br>Duration: %{x} hours<extra></extra>'
))

fig.update_layout(
    title='Time Investment by Project Phase',
    xaxis_title='Hours of Work',
    yaxis_title='Phase',
    height=400,
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# Summary stats
total_hours = sum(durations)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Project Hours", f"{total_hours}h")
with col2:
    st.metric("Average per Phase", f"{total_hours/len(phases):.0f}h")
with col3:
    st.metric("Equivalent Weeks", f"{total_hours/40:.1f}")

st.markdown("---")

# Phase details
st.markdown("## 📋 Phase Details")

for phase in phases:
    status_emoji = "✅" if phase['status'] == 'Completed' else "⏳"
    
    with st.expander(f"{status_emoji} **{phase['phase']}** ({phase['duration_hours']} hours)"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Status:** {phase['status']}")
            st.markdown(f"**Scripts:** {phase['scripts']}")
            st.markdown("**Tasks:**")
            for task in phase['tasks']:
                st.markdown(f"- {task}")
        
        with col2:
            st.metric("Time Investment", f"{phase['duration_hours']}h")
            if phase['status'] == 'Completed':
                st.success("Complete")
            else:
                st.warning("In Progress")

st.markdown("---")

# Key milestones
st.markdown("## 🎯 Key Milestones")

milestones = pd.DataFrame({
    'Milestone': [
        'Project Started',
        'Data Pipeline Complete',
        'Core Models Implemented',
        'Advanced Models Complete',
        'Visualization Complete',
        'Dashboard & Report'
    ],
    'Phase': ['Phase 1', 'Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 4'],
    'Status': ['✅', '✅', '✅', '✅', '✅', '⏳']
})

col1, col2, col3 = st.columns(3)

completed = (milestones['Status'] == '✅').sum()
total = len(milestones)
progress = completed / total

with col1:
    st.metric("Milestones Completed", f"{completed}/{total}")

with col2:
    st.metric("Overall Progress", f"{progress:.0%}")

with col3:
    hours_completed = sum([p['duration_hours'] for p in phases if p['status'] == 'Completed'])
    st.metric("Hours Invested", f"{hours_completed}h")

st.dataframe(milestones, use_container_width=True, hide_index=True)

st.markdown("---")

# Computational performance
st.markdown("## ⚡ Computational Performance")

comp_data = pd.DataFrame({
    'Script': ['Data Processing', 'Association Rules', 'Clustering', 
               'Recommendations', 'SVD Training', 'Neural Network', 'Visualization'],
    'Runtime (hours)': [2.5, 3.0, 1.5, 5.0, 2.0, 3.5, 0.1],
    'Memory (GB)': [8, 12, 16, 20, 24, 32, 4],
    'CPU Cores': [8, 8, 8, 8, 8, 8, 4]
})

col1, col2 = st.columns(2)

with col1:
    fig = go.Figure(go.Bar(
        x=comp_data['Script'],
        y=comp_data['Runtime (hours)'],
        marker_color='#1DB954',
        text=comp_data['Runtime (hours)'].apply(lambda x: f'{x:.1f}h'),
        textposition='outside'
    ))
    fig.update_layout(
        title='Script Runtime',
        xaxis_title='Script',
        yaxis_title='Hours',
        height=400,
        xaxis={'tickangle': -45}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = go.Figure(go.Bar(
        x=comp_data['Script'],
        y=comp_data['Memory (GB)'],
        marker_color='#FF6B6B',
        text=comp_data['Memory (GB)'].apply(lambda x: f'{x}GB'),
        textposition='outside'
    ))
    fig.update_layout(
        title='Memory Usage',
        xaxis_title='Script',
        yaxis_title='GB',
        height=400,
        xaxis={'tickangle': -45}
    )
    st.plotly_chart(fig, use_container_width=True)

# Total stats
total_runtime = comp_data['Runtime (hours)'].sum()
max_memory = comp_data['Memory (GB)'].max()

st.info(f"""
**Total Computational Resources:**
- Total Runtime: {total_runtime:.1f} hours
- Peak Memory: {max_memory} GB
- Hardware: M4 MacBook Air (32GB RAM, 10-core CPU)
- Infrastructure: Local processing (transitioned from AWS EC2)
""")

st.markdown("---")

# Project statistics
st.markdown("## 📊 Project Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Scripts Written", "42")

with col2:
    st.metric("Lines of Code", "8,500+")

with col3:
    st.metric("Figures Generated", "17")

with col4:
    st.metric("Models Evaluated", "7")

st.markdown("---")

# Technologies used
st.markdown("## 🛠️ Technologies & Tools")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Core Libraries")
    st.markdown("""
    - pandas
    - numpy
    - scikit-learn
    - scipy
    """)

with col2:
    st.markdown("### ML Frameworks")
    st.markdown("""
    - Surprise (SVD)
    - TensorFlow
    - mlxtend (Apriori)
    - implicit
    """)

with col3:
    st.markdown("### Visualization")
    st.markdown("""
    - matplotlib
    - plotly
    - seaborn
    - streamlit
    """)

st.markdown("---")

# Future work
st.markdown("## 🚀 Future Enhancements")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Potential Improvements")
    st.info("""
    - Real-time recommendation API
    - A/B testing framework
    - User feedback integration
    - Multi-modal features (audio analysis)
    - Temporal pattern modeling
    """)

with col2:
    st.markdown("### Scalability")
    st.success("""
    - Distributed processing (Spark)
    - Online learning algorithms
    - Incremental model updates
    - Caching layer for predictions
    - Model compression for mobile
    """)

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: #777; padding: 2rem;'>
    <p><strong>Total Project Duration:</strong> 170 hours (4.25 work-weeks)</p>
    <p><strong>CSCI 6443 Data Mining • Fall 2024</strong></p>
</div>
""", unsafe_allow_html=True)