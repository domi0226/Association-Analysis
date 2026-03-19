import pandas as pd
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import matplotlib.pyplot as plt

# 頁面配置
st.set_page_config(page_title="圖書關聯分析系統 V2", layout="wide")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Mac 字體，Windows 可改 Microsoft JhengHei
plt.rcParams['axes.unicode_minus'] = False

st.title("萬用關聯規則分析系統")

# --- 1. 資料處理快取 ---
@st.cache_data
def process_data(uploaded_file, target_col):
    df = pd.read_excel(uploaded_file)
    # 處理交易項目：轉字串 -> 分隔 -> 去空格
    transactions = df[target_col].apply(lambda x: [i.strip() for i in str(x).split(",")])
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)
    return df, df_encoded

# --- 2. 視覺化函數 ---
def draw_relationship_graph(rules_df, target_item):
    mask = rules_df['antecedents'].apply(lambda x: target_item in x) | \
           rules_df['consequents'].apply(lambda x: target_item in x)
    plot_rules = rules_df[mask].sort_values("lift", ascending=False).head(10)

    if plot_rules.empty:
        st.warning(f"目前篩選條件下，找不到與『{target_item}』相關的規則。")
        return

    G = nx.DiGraph()
    for _, row in plot_rules.iterrows():
        ante = ", ".join(list(row['antecedents']))
        cons = ", ".join(list(row['consequents']))
        G.add_edge(ante, cons, weight=row['lift'])

    fig, ax = plt.subplots(figsize=(10, 6))
    pos = nx.spring_layout(G, k=1.2)
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="#4B9CD3", alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=2, edge_color="#B1B1B1", arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=9, font_family='Arial Unicode MS')
    
    edge_labels = nx.get_edge_attributes(G, 'weight')
    formatted_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_labels, font_size=8)
    plt.title(f"『{target_item}』的關聯推薦網絡 (數字為 Lift)")
    st.pyplot(fig)

# --- 3. 檔案上傳與預覽 ---
uploaded_file = st.file_uploader("請上傳 Excel 檔案", type=["xlsx", "xls"])

if uploaded_file:
    # 讀取並直接顯示預覽 (解決你不見的問題)
    df_preview = pd.read_excel(uploaded_file)
    st.write("### 原始資料預覽 (前 5 筆)")
    st.dataframe(df_preview.head(), use_container_width=True)

    # 欄位選擇
    all_cols = df_preview.columns.tolist()
    default_idx = all_cols.index("items") if "items" in all_cols else 0
    target_col = st.selectbox("請選擇包含『交易項目』的欄位", all_cols, index=default_idx)

    # 側邊欄參數
    st.sidebar.header("演算法參數")
    min_sup = st.sidebar.slider("最小支持度 (Support)", 0.01, 0.5, 0.05)
    min_conf = st.sidebar.slider("最小信心度 (Confidence)", 0.1, 1.0, 0.5)
    min_lift = st.sidebar.slider("最小提升度 (Lift)", 1.0, 10.0, 1.2)

    if st.button("開始執行分析"):
        df, df_encoded = process_data(uploaded_file, target_col)
        frequent_itemsets = apriori(df_encoded, min_support=min_sup, use_colnames=True)
        
        if not frequent_itemsets.empty:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
            rules = rules[rules["confidence"] >= min_conf]
            st.session_state['rules_data'] = rules
            st.success(f"分析完成！找到 {len(rules)} 條初始規則。")
        else:
            st.session_state['rules_data'] = None
            st.error("找不到任何規則，請嘗試調低 Support 或 Lift。")

    # --- 4. 結果分頁展示 ---
    if 'rules_data' in st.session_state and st.session_state['rules_data'] is not None:
        rules = st.session_state['rules_data']
        
        tab1, tab2 = st.tabs(["數據報表與篩選", "關聯視覺化圖表"])

        with tab1:
            st.subheader("進階篩選器")
            c1, c2 = st.columns(2)
            all_items = sorted(list(set().union(*rules["antecedents"]).union(*rules["consequents"])))
            
            with c1:
                sel_ante = st.multiselect("前項 (Antecedents) 包含：", all_items)
                max_ante = st.number_input("前項最大數量：", 1, 10, 5)
            with c2:
                sel_cons = st.multiselect("後項 (Consequents) 包含：", all_items)
                max_cons = st.number_input("後項最大數量：", 1, 10, 5)

            # 篩選邏輯
            mask = (rules["antecedents"].apply(len) <= max_ante) & (rules["consequents"].apply(len) <= max_cons)
            if sel_ante: mask &= rules["antecedents"].apply(lambda x: any(i in x for i in sel_ante))
            if sel_cons: mask &= rules["consequents"].apply(lambda x: any(i in x for i in sel_cons))
            
            filtered_df = rules[mask].copy()
            
            if not filtered_df.empty:
                # 整理顯示格式
                disp_df = filtered_df.copy()
                disp_df["antecedents"] = disp_df["antecedents"].apply(lambda x: ', '.join(list(x)))
                disp_df["consequents"] = disp_df["consequents"].apply(lambda x: ', '.join(list(x)))
                st.dataframe(disp_df.sort_values("lift", ascending=False), use_container_width=True)
                
                # 下載按鈕
                csv = disp_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("下載篩選後的 CSV", csv, "rules_results.csv", "text/csv")
            else:
                st.warning("目前的篩選條件下沒有結果。")

        with tab2:
            st.subheader("書名關聯圖")
            target_book = st.selectbox("選擇一本書查看推薦網絡：", all_items)
            if target_book:
                draw_relationship_graph(rules, target_book)
else:
    st.info("請上傳 Excel 檔案以開始分析。")
