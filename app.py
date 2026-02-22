import streamlit as st
import BigDataAnalysis_Package as bda_cnn
import BigDataAnalysis_Package_RF as bda_rf
import joblib
import tempfile
from pathlib import Path
import pandas as pd
@st.cache_resource
def load_resources():
    scaler_xa_dis = joblib.load('final_models_cnn/Xa_dis_scaler.pkl.pth')
    scaler_xa_con = joblib.load('final_models_cnn/Xa_con_scaler.pkl.pth')
    scaler_ya_dis = joblib.load('final_models_cnn/Ya_dis_scaler.pkl.pth')
    scaler_ya_con = joblib.load('final_models_cnn/Ya_con_scaler.pkl.pth')
    Xa_model_dis_path = 'final_models_cnn/Xa_dis.pth'
    Xa_model_con_path = 'final_models_cnn/Xa_con.pth'
    Ya_model_dis_path = 'final_models_cnn/Ya_dis.pth'
    Ya_model_con_path = 'final_models_cnn/Ya_con.pth'
    param_path = 'final_models_rf'
    return scaler_xa_dis, scaler_xa_con, scaler_ya_dis, scaler_ya_con, Xa_model_dis_path, Xa_model_con_path, Ya_model_dis_path, Ya_model_con_path, param_path
scaler_xa_dis, scaler_xa_con, scaler_ya_dis, scaler_ya_con, Xa_model_dis_path, Xa_model_con_path, Ya_model_dis_path, Ya_model_con_path, param_path = load_resources()

def render_cnn_con(out_list):
    val = float(out_list[0]) if out_list else None
    if val is None:
        st.warning("CNN 連續型輸出為空")
        return
    st.metric("CNN 連續型預測值", f"{val:.2f}")

def render_cnn_dis(out_list, label_xy):
    if not out_list:
        st.warning("CNN 離散型輸出為空")
        return
    if label_xy == "X":
        normal = 80
    else:
        normal = 260
    raw = out_list[0]
    if raw != normal:
        is_abn = True
    else:
        is_abn = False

    if is_abn:
        st.error("判定：異常")
    else:
        st.success("判定：正常")

    st.caption(f"模型輸出：{raw}")

def render_rf_df(out_df: pd.DataFrame, label_xy: str):
    if out_df is None or out_df.empty:
        st.warning("RF 輸出為空")
        return

    decision = str(out_df["decision"].iloc[0])
    d = decision.strip().lower()
    is_abn = d in ["Abnormal", "異常", "0", "true", "yes"]

    if is_abn:
        st.error("判定：異常")
    else:
        st.success("判定：正常")

    if "HI_mean" in out_df.columns:
        st.metric("HI_mean", f"{float(out_df['HI_mean'].iloc[0]):.2f}")

    if label_xy.upper() == "X":
        extra_cols = ["HI_65", "HI_95", "HI_130"]
    else:
        extra_cols = ["HI_220", "HI_300", "HI_380"]

    cols_to_show = []
    for c in ["HI_mean", "decision"] + extra_cols:
        if c in out_df.columns:
            cols_to_show.append(c)

    st.subheader("指標明細")
    show_df = out_df[cols_to_show].copy()

    show_df["decision"] = show_df["decision"].replace({"Abnormal": "異常", "Normal": "正常"})
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    with st.expander("查看完整輸出 DataFrame"):
        st.dataframe(out_df, use_container_width=True)

st.set_page_config(page_title="震動預測小幫手", layout="centered")

st.markdown("""
<style>
/* 頁面背景柔和 */
.stApp {
    background: #fbfaf7;
}

/* 置中的卡片容器 */
.card {
    max-width: 760px;
    margin: 30px auto;
    background: #ffffff;
    padding: 28px 34px 22px 34px;
    border-radius: 18px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.06);
    border: 1px solid #f0f0f0;
}

/* 內層灰色區塊 */
.panel {
    background: #f6f6f6;
    padding: 18px 18px 8px 18px;
    border-radius: 10px;
    border: 1px solid #eaeaea;
}

/* 標題 */
.title {
    text-align:center;
    color:#5a7f3b;
    font-weight: 800;
    font-size: 34px;
    margin-bottom: 18px;
}

/* 調整按鈕顏色（Streamlit 版本可能略有差異） */
div.stButton > button {
    background-color: #5a7f3b;
    color: white;
    border-radius: 10px;
    border: 0px;
    padding: 8px 18px;
    font-weight: 700;
}
div.stButton > button:hover {
    background-color: #4c6f31;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="title">震動預測小幫手</div>', unsafe_allow_html=True)

st.markdown('<div class="panel">', unsafe_allow_html=True)
model_type = st.selectbox("模型選擇：", ["CNN", "Random Forest (RF)"])



source = st.selectbox(
        "資料來源：",
        ["Xa - 水平傳動軸馬達側", "Ya - 垂直傳動軸馬達側"]
    )

if model_type == "CNN":
    task = st.radio("預測任務：", ["連續型", "離散型"], horizontal=True)
else:
    task = "離散型"
    st.info("Random Forest 僅支援離散型（正常/異常）")

with st.form("predict_form", clear_on_submit=False):
    uploaded = st.file_uploader("上傳測試檔（.txt）", type=["txt"], accept_multiple_files=True)
    submitted = st.form_submit_button("開始預測")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
result_box = st.container(border=True)
if submitted:
    if not uploaded:
        with result_box:
            st.warning("請先上傳 .txt 測試檔。")
    else:
        is_folder = len(uploaded) > 1
        tmpdir_obj = None
        if is_folder:
            tmpdir_obj = tempfile.TemporaryDirectory()
            tmp_path = tmpdir_obj.name

            for f in uploaded:
                (Path(tmp_path) / f.name).write_bytes(f.getvalue())
        else:
            f = uploaded[0]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                tmp.write(f.getvalue())
                tmp_path = tmp.name
        label_xy = "X" if source.startswith("X") else "Y"
        output = None
        out_df = None
        if label_xy == "X":
            if model_type == 'CNN' and task == '連續型':
                output, valid_keys, err = bda_cnn.predict_con(tmp_path, scaler_xa_con, model_path = Xa_model_con_path, label = 'X', folder = is_folder)
            if model_type == 'CNN' and task == '離散型':
                output, valid_keys, err = bda_cnn.predict_dis(tmp_path, scaler_xa_dis, model_path = Xa_model_dis_path, label = 'X', folder = is_folder)
            if model_type == 'Random Forest (RF)':
                out_df = bda_rf.predict_by_rf(tmp_path , param_path, fold = is_folder)
        else:
            if model_type == 'CNN' and task == '連續型':
                output, valid_keys, err = bda_cnn.predict_con(tmp_path, scaler_ya_con, model_path = Ya_model_con_path, label = 'Y', folder = is_folder)
            if model_type == 'CNN' and task == '離散型':
                output, valid_keys, err = bda_cnn.predict_dis(tmp_path, scaler_ya_dis, model_path = Ya_model_dis_path, label = 'Y', folder = is_folder)
            if model_type == 'Random Forest (RF)':
                out_df = bda_rf.predict_by_rf(tmp_path , param_path, fold = is_folder, group_prefix = "Y")
        with result_box:
            st.subheader("預測結果")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**模式：** {'資料夾（多檔）' if is_folder else '單檔'}")

            with col2:
                st.markdown(f"**模型：** {model_type}")
            if not is_folder:
                st.markdown(f"**檔名：** {uploaded[0].name}")
            else:
                st.markdown(f"**總檔案數：** {len(uploaded)}")

            if model_type == "CNN":
                if err:
                    st.error(err)
                else:
                    if not is_folder:
                        if task == "連續型":
                            render_cnn_con(output)
                        else:
                            render_cnn_dis(output, label_xy)
                    else:
                        clean_keys = [k.replace("data_root_", "") for k in valid_keys]
                        all_keys = [f.name for f in uploaded]
                        dropped_keys = [k for k in all_keys if k not in set(clean_keys)]
                        df = pd.DataFrame({"file":  clean_keys, "pred": output})
                        st.dataframe(df, use_container_width=True)
                        if task == "離散型":
                            ref_label = 80 if label_xy == "X" else 260
                            df["decision"] = df["pred"].apply(lambda x: "Abnormal" if int(x) != ref_label else "Normal")
                            abn_files = df.loc[df["decision"] == "Abnormal", "file"].tolist()
                            if abn_files:
                                st.metric("異常數", len(abn_files))
                                st.write("異常檔案：")
                                st.code("\n".join(abn_files))
                            else:
                                st.write("預測檔案均為正常")
                        if dropped_keys:
                            st.metric("資料不足被忽略數", len(dropped_keys))
                            st.write("資料不足被忽略的檔案：")
                            st.code("\n".join(dropped_keys))
            else:
                out2 = out_df.reset_index(drop=True)
                if is_folder and len(out2) == len(uploaded):
                    out2.insert(0, "file", [f.name for f in uploaded])

                if "decision" in out2.columns:
                    abn_mask = out2["decision"].astype(str).str.lower().isin(["abnormal", "異常"])
                    st.metric("異常數", int(abn_mask.sum()))
                    if "file" in out2.columns:
                        abn_files = out2.loc[abn_mask, "file"].tolist()
                        if abn_files:
                            st.write("異常檔案：")
                            st.code("\n".join(abn_files))
                st.dataframe(out2, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)