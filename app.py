import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
from folium.plugins import MiniMap, MeasureControl
import plotly.express as px
import streamlit.components.v1 as components

# ==============================================
# 页面配置与全局样式
# ==============================================
st.set_page_config(
    page_title="船迹智导平台",
    page_icon="🛥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入高级CSS样式
st.markdown("""
<style>
:root {
    --primary-dark: #005f73;
    --primary-main: #0a9396;
    --primary-light: #94d2bd;
    --secondary-dark: #bb3e03;
    --secondary-main: #ee9b00;
    --bg-default: #f8f9fa;
    --bg-paper: #ffffff;
    --text-primary: #212529;
    --text-secondary: #495057;
}

/* 现代字体系统 */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&family=Roboto+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans SC', sans-serif;
    background-color: var(--bg-default);
}

/* 标题栏改造 */
.header-gradient {
    background: linear-gradient(135deg, var(--primary-dark), var(--primary-main)) !important;
    color: white !important;
    padding: 1.8rem 2rem !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px rgba(0, 95, 115, 0.2) !important;
    margin-bottom: 2rem !important;
    border: none !important;
    z-index: 1000 !important;
}

/* 侧边栏专业设计 */
section[data-testid="stSidebar"] > div {
    background: var(--bg-paper) !important;
    border-right: 1px solid rgba(0, 0, 0, 0.05) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.08) !important;
    padding-top: 2rem !important;
}

/* 卡片式组件 */
.card {
    background: var(--bg-paper) !important;
    border-radius: 16px !important;
    padding: 1.8rem !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06) !important;
    border: 1px solid rgba(0, 0, 0, 0.03) !important;
    margin-bottom: 1.5rem !important;
}

/* 高级指标卡片 */
.metric-card {
    background: var(--bg-paper) !important;
    border-left: 4px solid var(--primary-main) !important;
    padding: 1.2rem !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
}
.metric-card:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1) !important;
}

/* 表格美化 */
.dataframe-container {
    border-radius: 12px !important;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05) !important;
    overflow: hidden !important;
}

/* 按钮增强 */
.stButton>button {
    border-radius: 8px !important;
    padding: 0.6rem 1.8rem !important;
    transition: all 0.3s ease !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    background-color: var(--primary-main) !important;
    color: white !important;
    border: none !important;
}
.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
}

/* 分隔线增强 */
.divider {
    margin: 2.5rem 0 !important;
    border-top: 1px solid rgba(0, 0, 0, 0.1) !important;
    opacity: 0.5 !important;
}
</style>
""", unsafe_allow_html=True)


# ==============================================
# 数据加载函数 (优化版)
# ==============================================
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("Predicted trajectory of Test(预测结果）.xlsx")

        # 列名标准化处理
        df.columns = df.columns.str.strip()
        col_mapping = {
            'is_predicted(=1:Predictions Trajectory;=0:original)': 'is_predicted',
            'ship_id': 'ship_id',
            'voyage_id': 'voyage_id',
            'lon': 'lon',
            'lat': 'lat'
        }
        df = df.rename(columns=col_mapping)

        # 确保必要列存在
        required_columns = ['ship_id', 'voyage_id', 'lon', 'lat']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"❌ 数据错误：缺少必要列 '{col}'")
                return None, None

        # 处理预测标识列
        if 'is_predicted' not in df.columns:
            st.warning("⚠️ 未检测到预测标识列，所有数据将视为原始轨迹")
            df['is_predicted'] = 0

        # 分组处理
        grouped = {}
        ship_voyage_pairs = []
        for (ship_id, voyage_id), group in df.groupby(['ship_id', 'voyage_id']):
            ship_id = str(ship_id)
            voyage_id = str(voyage_id)

            # 提取轨迹点并确保数据质量
            original = group[group['is_predicted'] == 0][['lon', 'lat']].dropna().values.tolist()
            predicted = group[group['is_predicted'] == 1][['lon', 'lat']].dropna().values.tolist()

            grouped[(ship_id, voyage_id)] = {
                'original': original,
                'predicted': predicted,
                'stats': {
                    'original_count': len(original),
                    'predicted_count': len(predicted)
                }
            }
            ship_voyage_pairs.append((ship_id, voyage_id))

        return grouped, ship_voyage_pairs

    except Exception as e:
        st.error(f"❌ 数据加载失败: {str(e)}")
        return None, None


# ==============================================
# 高级地图绘制 (专业版)
# ==============================================
def create_professional_map(actual, predicted, ship_id, voyage_id,
                            show_predicted=True, show_markers=True,
                            map_style="简洁版"):
    """创建带有专业元素的高级地图"""
    # 地图样式选择
    tile_layers = {
        "简洁版": "cartodbpositron",
        "卫星图": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "暗黑模式": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
    }

    # 创建地图实例
    m = folium.Map(
        location=[30, 120],  # 默认位置
        zoom_start=5,
        tiles=tile_layers.get(map_style, "cartodbpositron"),
        control_scale=True
    )

    # 添加其他备用底图
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        attr='CartoDB Dark',
        name='暗黑模式',
        control=False
    ).add_to(m)

    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri卫星图',
        name='卫星影像',
        control=False
    ).add_to(m)

    # 绘制真实轨迹（始终显示）
    if len(actual) > 1:
        folium.PolyLine(
            [point[::-1] for point in actual],
            color='#3182bd',
            weight=4.5,
            opacity=0.9,
            tooltip=f"真实轨迹 | {ship_id}"
        ).add_to(m)

        # 根据筛选条件决定是否显示标记点
        if show_markers:
            for i, point in enumerate(actual):
                folium.CircleMarker(
                    location=point[::-1],
                    radius=3 if i not in [0, len(actual) - 1] else 5,
                    color='#3182bd',
                    fill=True,
                    fill_opacity=0.7
                ).add_to(m)

    # 根据筛选条件决定是否显示预测轨迹
    if show_predicted and len(predicted) > 1:
        folium.PolyLine(
            [point[::-1] for point in predicted],
            color='#e6550d',
            weight=4.5,
            opacity=0.8,
            dash_array='10',
            tooltip=f"预测轨迹 | {ship_id}"
        ).add_to(m)

        if show_markers:
            folium.CircleMarker(
                location=predicted[-1][::-1],
                radius=6,
                color='#e6550d',
                fill=True,
                fill_opacity=0.9
            ).add_to(m)

    # 添加地图控件
    MiniMap(toggle_display=True, position="bottomleft").add_to(m)
    MeasureControl(position="topright").add_to(m)
    folium.LayerControl(position="topright").add_to(m)

    # 自动调整视图范围
    all_points = [point[::-1] for point in (actual + (predicted if show_predicted else []))]
    if len(all_points) > 1:
        m.fit_bounds([
            [min(p[0] for p in all_points), min(p[1] for p in all_points)],
            [max(p[0] for p in all_points), max(p[1] for p in all_points)]
        ])

    return m


# ==============================================
# 辅助可视化组件
# ==============================================



# ==============================================
# 主应用界面
# ==============================================
def main():
    # 标题区域
    with st.container():
        st.markdown("""
        <div class="header-gradient">
            <h1 style="margin:0; font-weight:700; color: white;">🛥️ 船迹智导|AIS-NavigaTransformer</h1>
            <p style="margin:0; opacity:0.9; font-size:1.1rem;">船舶轨迹智能分析与预测平台</p>
        </div>
        """, unsafe_allow_html=True)

    # 加载数据
    with st.spinner("🔄 正在加载航行数据..."):
        data, available_pairs = load_data()

    if data is None:
        st.stop()

    # ================= 侧边栏控制区 =================
    with st.sidebar:
        st.markdown("""
        <div style="margin-bottom:2rem;">
            <h3 style="color:var(--primary-main);margin-bottom:0.5rem;">⚓ 航行控制台</h3>
            <div style="height:2px;background: linear-gradient(135deg, #004d61, #00c9a7);"></div>
        </div>
        """, unsafe_allow_html=True)

        # 船舶选择器
        if available_pairs:
            ship_options = sorted(list(set(pair[0] for pair in available_pairs)))
            ship_id = st.selectbox(
                "选择船舶ID",
                options=ship_options,
                key="ship_select"
            )

            voyage_options = sorted(list(set(
                pair[1] for pair in available_pairs if pair[0] == ship_id
            )))
            voyage_id = st.selectbox(
                "选择航次ID",
                options=voyage_options,
                key="voyage_select"
            )


        else:
            st.warning("⚠️ 未检测到有效航行数据")
            st.stop()

    # ================= 主显示区 =================
    key = (ship_id, voyage_id)
    if key not in data:
        st.error(f"⚠️ 未找到船舶 {ship_id} 航次 {voyage_id} 的数据")
        st.stop()

    trajectory_data = data[key]

    # 主布局
    col1, col2 = st.columns([2.5, 1], gap="large")

    with col1:
        # 地图卡片
        with st.container():
            st.markdown("### 🌊 轨迹可视化")
            folium_static(
                create_professional_map(
                    trajectory_data['original'],
                    trajectory_data['predicted'],
                    ship_id,
                    voyage_id,
                    show_predicted=st.session_state.get("show_predicted", True),
                    show_markers=st.session_state.get("show_markers", True),
                    map_style=st.session_state.get("map_style", "简洁版")
                ),
                width=800,
                height=600
            )

    with col2:
        # 信息面板
        with st.container():
            st.markdown("### 📊 航行指标")

            # 指标卡片组
            cols = st.columns(2)
            with cols[0]:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="margin-top:0;color:var(--primary-main);">船舶ID</h3>
                    <p style="font-size:1.4rem;font-weight:500;">{}</p>
                </div>
                """.format(ship_id), unsafe_allow_html=True)

            with cols[1]:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="margin-top:0;color:var(--primary-main);">航次ID</h3>
                    <p style="font-size:1.4rem;font-weight:500;">{}</p>
                </div>
                """.format(voyage_id), unsafe_allow_html=True)

            # 统计卡片
            st.markdown("""
            <div class="metric-card">
                <div style="display:flex;justify-content:space-between;">
                    <div>
                        <h3 style="margin-top:0;color:var(--primary-main);">原始轨迹点</h3>
                        <p style="font-size:1.8rem;font-weight:700;color:#3182bd;">{}</p>
                    </div>
                    <div>
                        <h3 style="margin-top:0;color:var(--primary-main);">预测轨迹点</h3>
                        <p style="font-size:1.8rem;font-weight:700;color:#e6550d;">{}</p>
                    </div>
                </div>
            </div>
            """.format(
                len(trajectory_data['original']),
                len(trajectory_data['predicted'])
            ), unsafe_allow_html=True)



    # ================= 数据展示区 =================
    st.markdown("---")

    tabs = st.tabs(["📋 原始轨迹数据", "🔮 预测轨迹数据"])

    with tabs[0]:
        if trajectory_data['original']:
            st.dataframe(
                pd.DataFrame(trajectory_data['original'], columns=["经度", "纬度"]),
                height=300,
                use_container_width=True
            )
        else:
            st.warning("无原始轨迹数据")

    with tabs[1]:
        if trajectory_data['predicted']:
            st.dataframe(
                pd.DataFrame(trajectory_data['predicted'], columns=["经度", "纬度"]),
                height=300,
                use_container_width=True
            )
        else:
            st.warning("无预测轨迹数据")


# 运行应用
if __name__ == "__main__":
    main()