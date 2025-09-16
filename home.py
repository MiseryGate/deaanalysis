import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import the enhanced DEA analyzer
try:
    from dea_algorithm import DEAAnalyzer, DEAResults
except ImportError:
    st.error("Please ensure the enhanced dea_algorithm.py file is available in your directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Enhanced DEA Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Enhanced DEA Analysis with CRS/VRS, Slacks, and Recommendations")
st.markdown("*Advanced Data Envelopment Analysis with comprehensive insights and visualizations*")

# Sidebar for parameters
st.sidebar.header("⚙️ Analysis Parameters")

# File upload
uploaded_file = st.file_uploader(
    "📁 Upload CSV file", 
    type=["csv"],
    help="Upload a CSV file with DMU data. Use semicolon (;) or comma (,) as separator."
)

if uploaded_file:
    # Try different separators
    try:
        df = pd.read_csv(uploaded_file, sep=";", engine="python").dropna()
        if len(df.columns) == 1:  # Probably wrong separator
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=",", engine="python").dropna()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Data preview
    with st.expander("📋 Data Preview", expanded=True):
        st.write(f"**Dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Basic statistics
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Numeric Columns:**")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            st.write(numeric_cols)
        with col2:
            st.write("**Data Types:**")
            st.write(df.dtypes.to_dict())

    # Column selection
    st.sidebar.subheader("📊 Column Selection")
    dmu_col = st.sidebar.selectbox(
        "Select DMU identifier column", 
        df.columns,
        help="Choose the column that identifies each Decision Making Unit"
    )
    
    available_cols = [c for c in df.columns if c != dmu_col]
    input_cols = st.sidebar.multiselect(
        "Select input columns", 
        available_cols,
        help="Choose columns representing inputs (resources consumed)"
    )
    
    remaining_cols = [c for c in available_cols if c not in input_cols]
    output_cols = st.sidebar.multiselect(
        "Select output columns", 
        remaining_cols,
        help="Choose columns representing outputs (products/services produced)"
    )

    # Analysis parameters
    st.sidebar.subheader("🔧 Model Configuration")
    orientation = st.sidebar.radio(
        "Select orientation", 
        ["input", "output"],
        help="Input-oriented: minimize inputs keeping outputs fixed. Output-oriented: maximize outputs keeping inputs fixed."
    )
    
    rts = st.sidebar.radio(
        "Select returns to scale", 
        ["CRS", "VRS"],
        help="CRS: Constant Returns to Scale. VRS: Variable Returns to Scale."
    )
    
    slack_correction = st.sidebar.checkbox(
        "Apply slack correction", 
        value=True,
        help="Use two-stage approach to maximize slacks for efficient units"
    )
    
    tolerance = st.sidebar.number_input(
        "Numerical tolerance", 
        value=1e-6, 
        format="%.1e",
        help="Tolerance for numerical computations"
    )

    # Validation and analysis
    if input_cols and output_cols:
        # Validate data
        dmus = df[dmu_col].astype(str).tolist()
        
        try:
            X = df[input_cols].values.astype(float)
            Y = df[output_cols].values.astype(float)
            
            # Check for non-positive values
            if np.any(X <= 0) or np.any(Y <= 0):
                st.error("❌ All input and output values must be positive. Please check your data.")
                st.stop()
                
        except ValueError as e:
            st.error(f"❌ Error converting data to numeric: {e}")
            st.stop()

        # Run DEA Analysis
        if st.button("🚀 Run DEA Analysis", type="primary", use_container_width=True):
            with st.spinner("Running DEA analysis..."):
                try:
                    # Initialize analyzer
                    analyzer = DEAAnalyzer(tolerance=tolerance)
                    
                    # Perform analysis
                    results = analyzer.analyze(
                        X=X, 
                        Y=Y, 
                        orientation=orientation, 
                        rts=rts, 
                        slack_correction=slack_correction
                    )
                    
                    # Store results in session state
                    st.session_state['dea_results'] = results
                    st.session_state['dmu_names'] = dmus
                    st.session_state['input_cols'] = input_cols
                    st.session_state['output_cols'] = output_cols
                    st.session_state['analysis_params'] = {
                        'orientation': orientation,
                        'rts': rts,
                        'slack_correction': slack_correction
                    }
                    
                    st.success("✅ Analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error in DEA analysis: {e}")
                    st.stop()

    # Display results if available
    if 'dea_results' in st.session_state:
        results = st.session_state['dea_results']
        dmus = st.session_state['dmu_names']
        input_cols = st.session_state['input_cols']
        output_cols = st.session_state['output_cols']
        params = st.session_state['analysis_params']
        
        # Summary Statistics
        st.header("📈 Summary Statistics")
        stats = DEAAnalyzer().summary_statistics(results)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Efficiency", f"{stats['mean_efficiency']:.3f}")
        with col2:
            st.metric("Efficient Units", f"{stats['efficient_units']}/{stats['total_units']}")
        with col3:
            st.metric("Min Efficiency", f"{stats['min_efficiency']:.3f}")
        with col4:
            st.metric("Max Efficiency", f"{stats['max_efficiency']:.3f}")
        
        # Main Results Table
        st.header("🎯 DEA Results")
        results_df = results.to_dataframe(dmus)
        
        # Add efficiency classification
        def classify_efficiency(score, status):
            if status == "Efficient":
                return "🟢 Efficient"
            elif status == "Weakly Efficient":
                return "🟡 Weakly Efficient" 
            else:
                return "🔴 Inefficient"
        
        results_df['Classification'] = results_df.apply(
            lambda row: classify_efficiency(row['Efficiency_Score'], row['Status']), axis=1
        )
        
        # Display with formatting
        st.dataframe(
            results_df[['DMU', 'Efficiency_Score', 'Classification', 'Status', 'Peers']].style.format({
                'Efficiency_Score': '{:.4f}'
            }),
            use_container_width=True
        )
        
        # Visualizations
        st.header("📊 Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Efficiency Distribution", "Slacks Analysis", "Peer Network", "Detailed Analysis"])
        
        with tab1:
            # Efficiency histogram
            fig = px.histogram(
                results_df, 
                x='Efficiency_Score', 
                nbins=20,
                title="Distribution of Efficiency Scores",
                labels={'Efficiency_Score': 'Efficiency Score', 'count': 'Number of DMUs'}
            )
            fig.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="Efficiency Frontier")
            st.plotly_chart(fig, use_container_width=True)
            
            # Efficiency bar chart
            fig2 = px.bar(
                results_df.sort_values('Efficiency_Score'), 
                x='DMU', 
                y='Efficiency_Score',
                color='Status',
                title="Efficiency Scores by DMU",
                color_discrete_map={
                    'Efficient': '#2E8B57',
                    'Weakly Efficient': '#FFD700', 
                    'Inefficient': '#DC143C'
                }
            )
            fig2.add_hline(y=1.0, line_dash="dash", line_color="black", annotation_text="Efficiency Frontier")
            fig2.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            # Slacks analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Input Slacks")
                input_slacks_df = pd.DataFrame(
                    results.input_slacks, 
                    columns=input_cols, 
                    index=dmus
                )
                st.dataframe(input_slacks_df.style.format('{:.4f}'), use_container_width=True)
                
                # Input slacks heatmap
                if len(input_cols) > 1:
                    fig = px.imshow(
                        input_slacks_df.T, 
                        title="Input Slacks Heatmap",
                        aspect="auto",
                        color_continuous_scale="Reds"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Output Slacks")
                output_slacks_df = pd.DataFrame(
                    results.output_slacks, 
                    columns=output_cols, 
                    index=dmus
                )
                st.dataframe(output_slacks_df.style.format('{:.4f}'), use_container_width=True)
                
                # Output slacks heatmap
                if len(output_cols) > 1:
                    fig = px.imshow(
                        output_slacks_df.T, 
                        title="Output Slacks Heatmap",
                        aspect="auto",
                        color_continuous_scale="Blues"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Peer analysis
            st.subheader("Peer Weights Matrix")
            lambdas_df = pd.DataFrame(
                results.lambdas, 
                index=dmus, 
                columns=dmus
            )
            
            # Show only non-zero weights
            lambdas_filtered = lambdas_df.copy()
            lambdas_filtered[lambdas_filtered < tolerance] = 0
            
            st.dataframe(lambdas_filtered.style.format('{:.4f}'), use_container_width=True)
            
            # Peer frequency
            peer_counts = {}
            for peers in results.peers:
                for peer in peers:
                    peer_name = dmus[peer]
                    peer_counts[peer_name] = peer_counts.get(peer_name, 0) + 1
            
            if peer_counts:
                peer_df = pd.DataFrame(list(peer_counts.items()), columns=['DMU', 'Frequency'])
                peer_df = peer_df.sort_values('Frequency', ascending=False)
                
                fig = px.bar(
                    peer_df.head(10), 
                    x='DMU', 
                    y='Frequency',
                    title="Most Frequent Benchmark Units"
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Detailed analysis for selected DMU
            selected_dmu = st.selectbox("Select DMU for detailed analysis", dmus)
            dmu_idx = dmus.index(selected_dmu)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Analysis for {selected_dmu}")
                st.write(f"**Efficiency Score:** {results.scores[dmu_idx]:.4f}")
                st.write(f"**Status:** {results.status[dmu_idx]}")
                
                if results.peers[dmu_idx]:
                    st.write("**Benchmark Peers:**")
                    peer_info = []
                    for peer_idx, weight in zip(results.peers[dmu_idx], results.peer_weights[dmu_idx]):
                        peer_info.append(f"- {dmus[peer_idx]} (weight: {weight:.4f})")
                    st.write("\n".join(peer_info))
                else:
                    st.write("**Status:** Self-benchmark (efficient)")
            
            with col2:
                # Slacks for selected DMU
                st.subheader("Slacks Analysis")
                
                input_slack = results.input_slacks[dmu_idx]
                output_slack = results.output_slacks[dmu_idx]
                
                if np.sum(input_slack) > tolerance:
                    st.write("**Input Excesses:**")
                    for i, slack in enumerate(input_slack):
                        if slack > tolerance:
                            st.write(f"- {input_cols[i]}: {slack:.4f}")
                
                if np.sum(output_slack) > tolerance:
                    st.write("**Output Shortfalls:**")
                    for i, slack in enumerate(output_slack):
                        if slack > tolerance:
                            st.write(f"- {output_cols[i]}: {slack:.4f}")
                
                if np.sum(input_slack) <= tolerance and np.sum(output_slack) <= tolerance:
                    st.write("✅ No slacks detected")
        
        # Enhanced Recommendations
        st.header("💡 Actionable Recommendations")
        
        recommendations = []
        for i, dmu in enumerate(dmus):
            rec_parts = []
            
            # Efficiency recommendation
            if results.status[i] == "Inefficient":
                if params['orientation'] == "input":
                    reduction = (1 - results.scores[i]) * 100
                    rec_parts.append(f"🔻 Reduce all inputs by {reduction:.1f}%")
                else:
                    increase = (1/results.scores[i] - 1) * 100
                    rec_parts.append(f"🔺 Increase all outputs by {increase:.1f}%")
            
            # Input slack recommendations
            input_slack = results.input_slacks[i]
            for j, slack in enumerate(input_slack):
                if slack > tolerance:
                    rec_parts.append(f"➖ Reduce {input_cols[j]} by {slack:.3f}")
            
            # Output slack recommendations  
            output_slack = results.output_slacks[i]
            for j, slack in enumerate(output_slack):
                if slack > tolerance:
                    rec_parts.append(f"➕ Increase {output_cols[j]} by {slack:.3f}")
            
            # Peer recommendations
            if results.peers[i]:
                peer_names = [dmus[p] for p in results.peers[i]]
                rec_parts.append(f"🎯 Learn from: {', '.join(peer_names)}")
            
            recommendations.append({
                "DMU": dmu,
                "Status": results.status[i],
                "Efficiency": f"{results.scores[i]:.4f}",
                "Recommendations": " | ".join(rec_parts) if rec_parts else "✅ Already efficient"
            })
        
        recs_df = pd.DataFrame(recommendations)
        st.dataframe(recs_df, use_container_width=True)
        
        # Interpretation Guide
        with st.expander("📖 Interpretation Guide", expanded=False):
            st.markdown("""
            ### Understanding DEA Results
            
            **Efficiency Scores:**
            - **Score = 1.0**: Unit is efficient (on the production frontier)
            - **Score < 1.0** (input-oriented): Unit should reduce inputs proportionally
            - **Score > 1.0** (output-oriented): Unit should increase outputs proportionally
            
            **Status Classification:**
            - **🟢 Efficient**: No improvements needed
            - **🟡 Weakly Efficient**: Efficient but has slacks (room for improvement)
            - **🔴 Inefficient**: Needs proportional input/output adjustments
            
            **Slacks:**
            - **Input Slacks**: Excess inputs that can be reduced
            - **Output Slacks**: Output shortfalls that should be increased
            
            **Peers:**
            - **Benchmark units**: Best-practice DMUs for comparison
            - **Weights**: Importance of each benchmark in the reference set
            """)
        
        # Export Results
        st.header("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Excel export
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                results_df.to_excel(writer, sheet_name="DEA_Results", index=False)
                input_slacks_df.to_excel(writer, sheet_name="Input_Slacks")
                output_slacks_df.to_excel(writer, sheet_name="Output_Slacks")
                lambdas_df.to_excel(writer, sheet_name="Peer_Weights")
                recs_df.to_excel(writer, sheet_name="Recommendations", index=False)
                
                # Add summary sheet
                summary_df = pd.DataFrame([stats]).T
                summary_df.columns = ['Value']
                summary_df.to_excel(writer, sheet_name="Summary_Statistics")
            
            excel_data = output.getvalue()
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_data,
                file_name=f"DEA_Analysis_{params['orientation']}_{params['rts']}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col2:
            # CSV export (main results)
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV Results",
                data=csv_data,
                file_name=f"DEA_Results_{params['orientation']}_{params['rts']}.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload a CSV file to begin the DEA analysis")
    
    with st.expander("📋 Data Format Requirements", expanded=True):
        st.markdown("""
        ### Required Data Format:
        
        1. **CSV file** with semicolon (;) or comma (,) separators
        2. **One column** for DMU identifiers (names/codes)
        3. **Multiple columns** for inputs (resources consumed)
        4. **Multiple columns** for outputs (products/services produced)
        5. **All input and output values must be positive**
        
        ### Example Data Structure:
        ```
        DMU_Name;Input1;Input2;Output1;Output2
        Hospital_A;100;50;200;150
        Hospital_B;120;60;180;140
        ...
        ```
        
        ### Recommendations:
        - Ensure at least 3×(inputs + outputs) DMUs for reliable results
        - Remove any rows with missing or zero values
        - Use meaningful column names for better interpretation
        """)
    
    # Sample data download
    sample_data = pd.DataFrame({
        'DMU': [f'Unit_{i+1}' for i in range(10)],
        'Labor': np.random.randint(50, 200, 10),
        'Capital': np.random.randint(100, 500, 10),
        'Revenue': np.random.randint(200, 800, 10),
        'Customers': np.random.randint(100, 300, 10)
    })
    
    csv_sample = sample_data.to_csv(index=False, sep=';')
    st.download_button(
        label="📋 Download Sample Data",
        data=csv_sample,
        file_name="sample_dea_data.csv",
        mime="text/csv"
    )