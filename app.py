import streamlit as st
from PIL import Image
import numpy as np
import sys
import os
import io
import zipfile
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from segmentation import LandCoverSegmenter
from cost_map import CostMapGenerator
from pathfinding import AStarPathfinder
from visualization import RouteVisualizer
from image_collection import SatelliteImageCollector

st.set_page_config(
    page_title="AI Highway Route Planner",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: #0E1117;
    }
    
    /* Rounded Cards */
    .card {
        background: linear-gradient(135deg, #1C1F26 0%, #252930 100%);
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.5);
        border: 1px solid #2A2E37;
        margin: 20px 0;
    }
    
    /* Page Title */
    .big-title {
        font-size: 42px;
        font-weight: 700;
        color: #E4ECFF;
        text-align: center;
        margin-bottom: 8px;
        text-shadow: 0 2px 10px rgba(100, 200, 255, 0.3);
    }
    
    .subtitle {
        font-size: 16px;
        color: #A0AEC0;
        text-align: center;
        margin-bottom: 30px;
        font-weight: 400;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 24px;
        font-weight: 600;
        color: #64B5F6;
        margin: 25px 0 15px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #2A2E37;
    }
    
    /* Step Indicator */
    .step-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 15px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d24 0%, #13161c 100%);
        border-right: 1px solid #2A2E37;
    }
    
    /* Navigation Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 24px;
        font-size: 15px;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        margin: 4px 0;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #64B5F6;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #A0AEC0;
        font-size: 14px;
    }
    
    /* Success/Info Messages */
    .stSuccess {
        background: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4CAF50;
        border-radius: 8px;
        padding: 12px;
    }
    
    .stInfo {
        background: rgba(33, 150, 243, 0.1);
        border-left: 4px solid #2196F3;
        border-radius: 8px;
        padding: 12px;
    }
    
    .stWarning {
        background: rgba(255, 152, 0, 0.1);
        border-left: 4px solid #FF9800;
        border-radius: 8px;
        padding: 12px;
    }
    
    /* Images */
    img {
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }
    
    /* Download buttons */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%);
        color: white;
        border-radius: 10px;
        padding: 12px 20px;
        font-weight: 600;
        border: none;
        box-shadow: 0 3px 10px rgba(0, 180, 216, 0.4);
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 180, 216, 0.6);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: #1C1F26;
        border-radius: 12px;
        padding: 20px;
        border: 2px dashed #2A2E37;
    }
    
    /* About Box */
    .about-box {
        background: rgba(100, 181, 246, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin-top: 20px;
        border: 1px solid rgba(100, 181, 246, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'upload'
if 'original_image' not in st.session_state:
    st.session_state.original_image = None

# Sidebar Navigation
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #64B5F6; font-size: 28px;'>🛣️ Route Planner</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #718096; font-size: 13px; margin-bottom: 30px;'>AI-Powered Highway Optimization</p>", unsafe_allow_html=True)
    
    st.markdown("### 📍 Navigation")
    
    pages = [
        ('upload', '📤 Upload Image', '1'),
        ('segment', '🔲 Land Segmentation', '2'),
        ('cost', '💰 Cost Map', '3'),
        ('route', '🗺️ Route Planning', '4'),
        ('results', '📊 Final Results', '5')
    ]
    
    for key, label, num in pages:
        if st.button(f"Step {num}: {label}", key=f"nav_{key}"):
            st.session_state.page = key
    
    st.markdown("---")
    
    st.markdown("""
    <div class="about-box">
        <h4 style='color: #64B5F6; margin: 0 0 10px 0;'>ℹ️ About</h4>
        <p style='color: #A0AEC0; font-size: 13px; margin: 0; line-height: 1.6;'>
        This platform uses advanced AI algorithms to analyze satellite imagery and calculate optimal highway routes based on terrain costs and environmental factors.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============= PAGE: UPLOAD IMAGE =============
if st.session_state.page == 'upload':
    st.markdown('<p class="step-badge">STEP 1 OF 5</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="big-title">📤 Upload Satellite Image</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Begin by uploading a satellite image of the area you want to analyze</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">🖼️ Image Upload</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a satellite image (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a high-resolution satellite image for analysis"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.session_state.original_image = image
        
        st.success("✅ Image uploaded successfully!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="section-header">📷 Preview</p>', unsafe_allow_html=True)
            st.image(image, use_container_width=True, caption="Uploaded Satellite Image")
        
        with col2:
            st.markdown('<p class="section-header">📊 Image Information</p>', unsafe_allow_html=True)
            st.metric("Width", f"{image.size[0]} px")
            st.metric("Height", f"{image.size[1]} px")
            st.metric("Format", image.format)
            st.metric("Mode", image.mode)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if st.button("➡️ Proceed to Land Segmentation", use_container_width=True):
            st.session_state.page = 'segment'
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        st.info("👆 Please upload a satellite image to get started")
        st.markdown("</div>", unsafe_allow_html=True)

# ============= PAGE: SEGMENTATION =============
elif st.session_state.page == 'segment':
    st.markdown('<p class="step-badge">STEP 2 OF 5</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="big-title">🔲 Land Cover Segmentation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered classification of terrain types including forests, water bodies, urban areas, and more</p>', unsafe_allow_html=True)
    
    if st.session_state.original_image is None:
        st.warning("⚠️ Please upload an image first")
        if st.button("⬅️ Go to Upload", use_container_width=True):
            st.session_state.page = 'upload'
            st.rerun()
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-header">⚙️ Segmentation Settings</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox(
                "🔧 Segmentation Method",
                ['rule_based', 'unet', 'deeplabv3'],
                help="Choose the AI model for land classification"
            )
        
        with col2:
            model_path = None
            if method != 'rule_based':
                model_path = st.text_input(
                    "📁 Model Path (Optional)",
                    help="Path to pre-trained model weights"
                )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if st.button("🚀 Run Segmentation", use_container_width=True):
            with st.spinner("🔄 Processing image... This may take a moment"):
                segmenter = LandCoverSegmenter(method=method, model_path=model_path if model_path else None)
                mask, colored_mask = segmenter.segment_image(np.array(st.session_state.original_image))
                
                st.session_state.segmentation_mask = mask
                st.session_state.colored_segmentation = Image.fromarray(colored_mask)
            
            st.success("✅ Segmentation completed successfully!")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if 'colored_segmentation' in st.session_state:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="section-header">🎨 Segmentation Results</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📷 Original Image**")
                st.image(st.session_state.original_image, use_container_width=True)
            
            with col2:
                st.markdown("**🗺️ Segmented Land Cover**")
                st.image(st.session_state.colored_segmentation, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="section-header">🏷️ Land Cover Legend</p>', unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            legend = [
                ("💧 Water", "#0000FF"),
                ("🌲 Forest", "#228B22"),
                ("🏙️ Urban", "#808080"),
                ("🏜️ Barren", "#D2B48C"),
                ("🛣️ Road", "#000000")
            ]
            
            for col, (label, color) in zip([col1, col2, col3, col4, col5], legend):
                with col:
                    st.markdown(f"""
                    <div style='background: {color}; padding: 15px; border-radius: 10px; text-align: center; color: white; font-weight: 600;'>
                        {label}
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if st.button("➡️ Proceed to Cost Map", use_container_width=True):
                st.session_state.page = 'cost'
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

# ============= PAGE: COST MAP =============
elif st.session_state.page == 'cost':
    st.markdown('<p class="step-badge">STEP 3 OF 5</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="big-title">💰 Cost Map Generation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Calculate construction costs based on terrain difficulty and environmental factors</p>', unsafe_allow_html=True)
    
    if 'segmentation_mask' not in st.session_state:
        st.warning("⚠️ Please complete segmentation first")
        if st.button("⬅️ Go to Segmentation", use_container_width=True):
            st.session_state.page = 'segment'
            st.rerun()
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-header">💵 Terrain Cost Configuration</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: rgba(100, 181, 246, 0.1); padding: 16px; border-radius: 10px; margin-bottom: 20px;'>
            <p style='color: #A0AEC0; margin: 0; font-size: 14px;'>
            💡 <strong>Cost Values:</strong> Higher values indicate more difficult/expensive terrain to build through.
            These costs will influence the optimal route calculation.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            water_cost = st.number_input("💧 Water Cost", value=1000, min_value=1, step=50)
            forest_cost = st.number_input("🌲 Forest Cost", value=500, min_value=1, step=50)
            urban_cost = st.number_input("🏙️ Urban Cost", value=200, min_value=1, step=50)
        
        with col2:
            barren_cost = st.number_input("🏜️ Barren Cost", value=100, min_value=1, step=50)
            road_cost = st.number_input("🛣️ Road Cost", value=50, min_value=1, step=50)
        
        terrain_costs = {
            0: water_cost,
            1: forest_cost,
            2: urban_cost,
            3: barren_cost,
            4: road_cost
        }
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if st.button("🚀 Generate Cost Map", use_container_width=True):
            with st.spinner("🔄 Calculating terrain costs..."):
                cost_gen = CostMapGenerator(terrain_costs=terrain_costs)
                cost_map = cost_gen.generate_cost_map(st.session_state.segmentation_mask)
                st.session_state.cost_map = cost_map
                st.session_state.terrain_costs = terrain_costs
            
            st.success("✅ Cost map generated successfully!")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if 'cost_map' in st.session_state:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="section-header">📊 Cost Map Visualization</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🗺️ Segmented Terrain**")
                st.image(st.session_state.colored_segmentation, use_container_width=True)
            
            with col2:
                st.markdown("**💰 Cost Heatmap**")
                fig, ax = plt.subplots(figsize=(10, 10))
                im = ax.imshow(st.session_state.cost_map, cmap='hot')
                ax.set_title("Construction Cost Map", fontsize=16, color='white')
                ax.axis('off')
                plt.colorbar(im, ax=ax, label='Cost Value')
                fig.patch.set_facecolor('#0E1117')
                st.pyplot(fig)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="section-header">📈 Cost Statistics</p>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Minimum Cost", f"{st.session_state.cost_map.min():.0f}")
            col2.metric("Maximum Cost", f"{st.session_state.cost_map.max():.0f}")
            col3.metric("Average Cost", f"{st.session_state.cost_map.mean():.0f}")
            col4.metric("Total Pixels", f"{st.session_state.cost_map.size:,}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if st.button("➡️ Proceed to Route Planning", use_container_width=True):
                st.session_state.page = 'route'
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

# ============= PAGE: ROUTE PLANNING =============
elif st.session_state.page == 'route':
    st.markdown('<p class="step-badge">STEP 4 OF 5</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="big-title">🗺️ Optimal Route Planning</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Use advanced A* pathfinding algorithm to calculate the most cost-effective highway route</p>', unsafe_allow_html=True)
    
    if 'cost_map' not in st.session_state:
        st.warning("⚠️ Please generate cost map first")
        if st.button("⬅️ Go to Cost Map", use_container_width=True):
            st.session_state.page = 'cost'
            st.rerun()
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-header">📍 Route Configuration</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: rgba(100, 181, 246, 0.1); padding: 16px; border-radius: 10px; margin-bottom: 20px;'>
            <p style='color: #A0AEC0; margin: 0; font-size: 14px;'>
            🎯 <strong>Instructions:</strong> Set start and end coordinates for your highway route.
            The AI will calculate the optimal path considering terrain costs.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🚩 Start Point**")
            start_x = st.number_input("X Coordinate", value=50, min_value=0, max_value=st.session_state.cost_map.shape[1]-1, key="start_x")
            start_y = st.number_input("Y Coordinate", value=50, min_value=0, max_value=st.session_state.cost_map.shape[0]-1, key="start_y")
        
        with col2:
            st.markdown("**🏁 End Point**")
            end_x = st.number_input("X Coordinate", value=st.session_state.cost_map.shape[1]-50, min_value=0, max_value=st.session_state.cost_map.shape[1]-1, key="end_x")
            end_y = st.number_input("Y Coordinate", value=st.session_state.cost_map.shape[0]-50, min_value=0, max_value=st.session_state.cost_map.shape[0]-1, key="end_y")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if st.button("🚀 Calculate Optimal Route", use_container_width=True):
            with st.spinner("🔄 Finding optimal path using A* algorithm..."):
                pathfinder = AStarPathfinder()
                path = pathfinder.find_path(
                    st.session_state.cost_map,
                    (start_y, start_x),
                    (end_y, end_x)
                )
                
                if path:
                    st.session_state.path = path
                    st.session_state.start_point = (start_y, start_x)
                    st.session_state.end_point = (end_y, end_x)
                    
                    visualizer = RouteVisualizer()
                    route_img = visualizer.visualize_route(
                        np.array(st.session_state.original_image),
                        path,
                        (start_y, start_x),
                        (end_y, end_x)
                    )
                    st.session_state.route_image = Image.fromarray(route_img)
                    
                    st.success(f"✅ Optimal route found! Path length: {len(path)} pixels")
                else:
                    st.error("❌ No valid path found. Try different start/end points.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if 'route_image' in st.session_state:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="section-header">🛣️ Route Visualization</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**💰 Cost Map with Route**")
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(st.session_state.cost_map, cmap='hot', alpha=0.7)
                path_array = np.array(st.session_state.path)
                ax.plot(path_array[:, 1], path_array[:, 0], 'cyan', linewidth=3, label='Optimal Route')
                ax.plot(st.session_state.start_point[1], st.session_state.start_point[0], 'go', markersize=15, label='Start')
                ax.plot(st.session_state.end_point[1], st.session_state.end_point[0], 'ro', markersize=15, label='End')
                ax.legend()
                ax.axis('off')
                fig.patch.set_facecolor('#0E1117')
                st.pyplot(fig)
            
            with col2:
                st.markdown("**🗺️ Route on Satellite Image**")
                st.image(st.session_state.route_image, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="section-header">📊 Route Statistics</p>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            path_length = len(st.session_state.path)
            total_cost = sum(st.session_state.cost_map[p[0], p[1]] for p in st.session_state.path)
            avg_cost = total_cost / path_length if path_length > 0 else 0
            
            col1.metric("Path Length", f"{path_length} px")
            col2.metric("Total Cost", f"{total_cost:,.0f}")
            col3.metric("Average Cost/px", f"{avg_cost:.0f}")
            col4.metric("Efficiency", f"{(1000/avg_cost)*100:.1f}%")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if st.button("➡️ View Final Results", use_container_width=True):
                st.session_state.page = 'results'
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

# ============= PAGE: RESULTS =============
elif st.session_state.page == 'results':
    st.markdown('<p class="step-badge">STEP 5 OF 5</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="big-title">📊 Final Results & Export</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Review comprehensive analysis results and download all generated outputs</p>', unsafe_allow_html=True)
    
    if 'route_image' not in st.session_state:
        st.warning("⚠️ Please complete route planning first")
        if st.button("⬅️ Go to Route Planning", use_container_width=True):
            st.session_state.page = 'route'
            st.rerun()
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-header">🖼️ Complete Analysis Overview</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📷 Original Satellite Image**")
            st.image(st.session_state.original_image, use_container_width=True)
            
            st.markdown("**🗺️ Land Cover Segmentation**")
            st.image(st.session_state.colored_segmentation, use_container_width=True)
        
        with col2:
            st.markdown("**💰 Cost Map Analysis**")
            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(st.session_state.cost_map, cmap='hot')
            ax.axis('off')
            plt.colorbar(im, ax=ax)
            fig.patch.set_facecolor('#0E1117')
            st.pyplot(fig)
            
            st.markdown("**🛣️ Optimal Highway Route**")
            st.image(st.session_state.route_image, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-header">📈 Project Summary</p>', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        path_length = len(st.session_state.path)
        total_cost = sum(st.session_state.cost_map[p[0], p[1]] for p in st.session_state.path)
        avg_cost = total_cost / path_length if path_length > 0 else 0
        
        col1.metric("🗺️ Image Size", f"{st.session_state.original_image.size[0]}×{st.session_state.original_image.size[1]}")
        col2.metric("🛣️ Route Length", f"{path_length} px")
        col3.metric("💰 Total Cost", f"{total_cost:,.0f}")
        col4.metric("📊 Avg Cost", f"{avg_cost:.0f}")
        col5.metric("✅ Efficiency", f"{(1000/avg_cost)*100:.1f}%")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-header">💾 Download Results</p>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            buf1 = io.BytesIO()
            st.session_state.colored_segmentation.save(buf1, format='PNG')
            st.download_button(
                "📥 Segmentation",
                buf1.getvalue(),
                "segmentation.png",
                "image/png",
                use_container_width=True
            )
        
        with col2:
            buf2 = io.BytesIO()
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(st.session_state.cost_map, cmap='hot')
            ax.axis('off')
            fig.savefig(buf2, format='png', bbox_inches='tight', facecolor='#0E1117')
            plt.close(fig)
            st.download_button(
                "📥 Cost Map",
                buf2.getvalue(),
                "cost_map.png",
                "image/png",
                use_container_width=True
            )
        
        with col3:
            buf3 = io.BytesIO()
            st.session_state.route_image.save(buf3, format='PNG')
            st.download_button(
                "📥 Route",
                buf3.getvalue(),
                "route.png",
                "image/png",
                use_container_width=True
            )
        
        with col4:
            # Create ZIP with all results
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, 'w') as zip_file:
                # Original
                buf_orig = io.BytesIO()
                st.session_state.original_image.save(buf_orig, format='PNG')
                zip_file.writestr('original.png', buf_orig.getvalue())
                
                # Segmentation
                zip_file.writestr('segmentation.png', buf1.getvalue())
                
                # Cost map
                zip_file.writestr('cost_map.png', buf2.getvalue())
                
                # Route
                zip_file.writestr('route.png', buf3.getvalue())
            
            st.download_button(
                "📦 All Files (ZIP)",
                zip_buf.getvalue(),
                "highway_route_results.zip",
                "application/zip",
                use_container_width=True
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.success("🎉 Analysis Complete! You can download individual files or all results as a ZIP archive.")
        
        if st.button("🔄 Start New Analysis", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key != 'page':
                    del st.session_state[key]
            st.session_state.page = 'upload'
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
