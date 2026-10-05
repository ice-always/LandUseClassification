from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import UnidentifiedImageError

from model import load_model, pixel_hash, predict, read_image

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
RUNS = ASSETS / "runs"
CLASS_NAMES = ["Agriculture", "Nature_Forest", "Water", "Residential", "Industrial_Infrastructure"]
CLASS_TH = ["เกษตรกรรม", "ธรรมชาติและป่า", "แหล่งน้ำ", "ที่อยู่อาศัย", "อุตสาหกรรมและโครงสร้างพื้นฐาน"]
ARCH_NAMES = {"resnet18": "ResNet18", "mobilenet_v3_large": "MobileNetV3-Large", "efficientnet_b0": "EfficientNet-B0"}
PAGES = ["ภาพรวมโครงการ", "เปรียบเทียบโมเดล", "กราฟและผลทดสอบ", "อัปโหลดภาพทำนาย"]

st.set_page_config(page_title="Land Use Classification", page_icon="🌍", layout="wide")


@st.cache_data
def read_csv(relative_path):
    return pd.read_csv(ASSETS / relative_path)


@st.cache_data
def read_json(relative_path):
    return json.loads((ASSETS / relative_path).read_text(encoding="utf-8"))


@st.cache_data
def known_images():
    frame = read_csv("manifest.csv")
    return {row.pixel_sha256: {"source": row.source, "split": row.split, "class_name": row.class_name}
            for row in frame.itertuples()}


@st.cache_resource
def selected_model():
    return load_model(ASSETS / "landuse_selected_model.pt")


def run_label(architecture, attention):
    return ARCH_NAMES[architecture] + (" + spatial attention" if attention else " baseline")


def comparison_table():
    frame = read_csv("comparison.csv").copy()
    frame["attention"] = frame.attention.astype(str).str.lower().eq("true")
    means = frame[frame.source != "All"].groupby(["architecture", "attention"]).macro_f1.mean()
    overall = frame[frame.source == "All"].copy()
    overall["selection_f1"] = [means.loc[(r.architecture, r.attention)] for r in overall.itertuples()]
    overall["model"] = [run_label(r.architecture, r.attention) for r in overall.itertuples()]
    overall["run"] = [f"{r.architecture}_{'attention' if r.attention else 'baseline'}_seed{r.seed}"
                      for r in overall.itertuples()]
    return overall.sort_values("selection_f1", ascending=False)


def class_table():
    return pd.DataFrame({"Class ID": range(5), "Class": CLASS_NAMES, "ความหมาย": CLASS_TH})


def csv_download(label, frame, filename):
    st.download_button(label, frame.to_csv(index=False).encode("utf-8-sig"),
                       file_name=filename, mime="text/csv")


def show_confusion(relative_path, title):
    cm = pd.read_csv(ASSETS / relative_path, index_col=0)
    fig = px.imshow(cm.to_numpy(), x=cm.columns, y=cm.index,
                    text_auto=True, color_continuous_scale="Blues", aspect="auto",
                    labels={"x": "Predicted class", "y": "True class", "color": "Images"}, title=title)
    fig.update_layout(height=450)
    st.plotly_chart(fig, width="stretch")
    st.caption("แถว = คลาสจริง · คอลัมน์ = คลาสที่ทำนาย · ช่องทแยง = ทำนายถูก")


def overview():
    st.header("ภาพรวมโครงการ")
    st.write("จำแนกภาพการใช้ที่ดินเป็น 5 กลุ่ม และเปรียบเทียบ CNN 3 สถาปัตยกรรม "
             "ทั้งแบบ baseline และแบบเพิ่ม spatial self-attention รวม 6 แบบ")
    metrics = read_csv("test_metrics.csv")
    result = metrics[metrics.source == "All"].iloc[0]
    cols = st.columns(4)
    cols[0].metric("ภาพทั้งหมด", "28,000")
    cols[1].metric("รูปแบบที่ทดลอง", "6")
    cols[2].metric("Test accuracy", f"{result.accuracy:.2%}")
    cols[3].metric("Test Macro-F1", f"{result.macro_f1:.4f}")
    st.success("โมเดลที่เลือก: EfficientNet-B0 baseline · checkpoint จาก epoch 12")
    st.caption("เลือกจากค่าเฉลี่ย Macro-F1 ของ EuroSAT และ UC Merced บน validation ก่อนประเมิน test")
    left, right = st.columns(2)
    with left:
        st.subheader("คลาสที่ทำนาย")
        st.dataframe(class_table(), hide_index=True, width="stretch")
    with right:
        st.subheader("จำนวนภาพแต่ละชุด")
        manifest = read_csv("manifest.csv")
        counts = manifest.groupby(["source", "split"]).size().unstack(fill_value=0)
        counts = counts.reindex(columns=["train", "val", "test"])
        counts.columns = ["Train", "Validation", "Test"]
        st.dataframe(counts, width="stretch")
        st.caption("Train 19,599 · Validation 4,200 · Test 4,201 ภาพ")
    counts = read_csv("manifest.csv").groupby(["class_name", "source"]).size().reset_index(name="Images")
    counts["class_name"] = pd.Categorical(counts.class_name, categories=CLASS_NAMES, ordered=True)
    fig = px.bar(counts.sort_values("class_name"), x="class_name", y="Images", color="source",
                 title="จำนวนภาพแต่ละคลาส แยกตาม dataset", labels={"class_name": "Class", "source": "Dataset"})
    st.plotly_chart(fig, width="stretch")
    st.subheader("การเตรียมข้อมูลและการฝึก")
    st.write("ใช้ split เดียวกันทุกแบบ ตรวจไฟล์เสียและภาพซ้ำ ปรับภาพเป็น 224×224 และ normalize ตาม ImageNet "
             "ทำ augmentation เฉพาะ train และเพิ่มโอกาสสุ่มกลุ่ม source×class ที่มีภาพน้อย")
    st.write("ฝึกหัวใหม่ 2 epochs แล้ว fine-tune บล็อกท้ายอีกไม่เกิน 12 epochs "
             "ใช้ early stopping และเก็บ checkpoint ที่คะแนน validation ดีที่สุด")
    st.subheader("การนำไปใช้และข้อจำกัด")
    st.write("ใช้คัดกรองและจัดหมวดหมู่ภาพก่อนนำไปวิเคราะห์ GIS ต่อ ผลลัพธ์เป็นหนึ่งคลาสต่อภาพ "
             "ยังระบุตำแหน่งแต่ละพื้นที่ในภาพหรือพยากรณ์น้ำท่วมไม่ได้")
    st.caption("ทดลองหนึ่ง seed (42) · แบ่งข้อมูลระดับภาพ ไม่ได้แยกพื้นที่ทางภูมิศาสตร์ · "
               "resize ไม่ทำให้ความละเอียดหรือขนาดพื้นที่จริงของสอง dataset เท่ากัน")


def compare_models():
    st.header("เปรียบเทียบโมเดลบน validation")
    data = comparison_table()
    table = pd.DataFrame({
        "โมเดล": data.model,
        "Accuracy (%)": data.accuracy * 100,
        "Macro-F1 รวม": data.macro_f1,
        "เฉลี่ย Macro-F1 แยก dataset": data.selection_f1,
        "Best epoch": data.best_epoch,
        "Epochs ที่ฝึก": data.epochs_run,
        "พารามิเตอร์ (ล้าน)": data.parameters / 1e6,
        "เวลาฝึก (นาที)": data.training_minutes,
    })
    st.dataframe(table.round(4), hide_index=True, width="stretch")
    st.caption("เรียงตามเกณฑ์เลือก: ค่าเฉลี่ย Macro-F1 ของแต่ละ dataset ให้น้ำหนัก EuroSAT และ UC Merced เท่ากัน")
    csv_download("ดาวน์โหลดตารางเปรียบเทียบ", table, "validation_comparison.csv")
    source = st.selectbox("แสดงคะแนนของ dataset", ["All", "EuroSAT", "UCMerced"])
    subset = read_csv("comparison.csv").query("source == @source").copy()
    subset["Architecture"] = subset.architecture.map(ARCH_NAMES)
    subset["Variant"] = subset.attention.astype(str).str.lower().map({"false": "Baseline", "true": "+ spatial attention"})
    subset["Accuracy (%)"] = subset.accuracy * 100
    fig = px.bar(subset, x="Architecture", y="Accuracy (%)", color="Variant", barmode="group",
                 text_auto=".2f", title=f"Validation accuracy: {source}")
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, width="stretch")
    st.info("Attention ที่เพิ่มไม่ได้เพิ่มเกณฑ์เลือกทั้ง 3 คู่ใน seed นี้ แม้ ResNet18 accuracy รวมเพิ่มเล็กน้อย "
            "ยังสรุปไม่ได้ว่า attention ไม่ดีในทุกกรณี")
    st.caption("MobileNetV3 และ EfficientNet มี SE อยู่แล้ว คำว่า baseline หมายถึงไม่เพิ่ม spatial self-attention ของเรา")
    with st.expander("Fine-tuning และกติกาการเปรียบเทียบ"):
        st.write("ใช้ pretrained ImageNet และหัวใหม่ขนาด 128 มิติเหมือนกันทุกแบบ "
                 "ควบคุม split, preprocessing, seed, batch size และสูตรฝึก แต่สถาปัตยกรรมและจำนวนพารามิเตอร์ต่างกัน")
        st.dataframe(pd.DataFrame({"โมเดล": list(ARCH_NAMES.values()),
                                  "บล็อกที่ fine-tune": ["layer4", "features[15:17]", "features[7:9]"]}),
                     hide_index=True, width="stretch")
        st.write("Batch 32 · AdamW · LR หัวเริ่ม 0.001 แล้วลดเป็น 0.001/3 ช่วง fine-tune · "
                 "LR backbone 0.0001 · weight decay 0.0001 · dropout 0.30 · label smoothing 0.05")
        st.write("ตั้งสูงสุด 14 epochs ตามงบการทดลอง หยุดเมื่อเกณฑ์ validation ไม่ดีขึ้น 4 epochs "
                 "และลด LR เมื่อคะแนนคงที่ ไม่ได้พิสูจน์ว่า 14 เป็นจำนวนที่ดีที่สุด")
    with st.expander("ผล ViT จากงานเดิมของสมาชิก"):
        st.write("ViT-B/16 เป็นผลอ้างอิงจาก notebook ของมิก สูตรฝึกต่างจากการทดลอง CNN รอบนี้ "
                 "และไม่มีน้ำหนัก ViT สำหรับประเมินซ้ำ จึงไม่ได้จัดอันดับร่วมกับทั้ง 6 แบบ")


def training_and_test():
    st.header("กราฟการฝึกและผลทดสอบ")
    data = comparison_table()
    labels = dict(zip(data.run, data.model))
    run = st.selectbox("เลือกผลการฝึก", data.run.tolist(), format_func=lambda value: labels[value])
    history = read_csv(f"runs/{run}/history.csv")
    best = read_json(f"runs/{run}/best_epoch.json")["epoch"]
    subset_n = len(read_csv(f"runs/{run}/train_evaluation_subset.csv"))
    st.caption(f"{labels[run]} · Best epoch {best} · ฝึกทั้งหมด {len(history)} epochs")
    left, right = st.columns(2)
    for col, columns, title in [(left, ["train_eval_loss", "val_loss"], "Loss: clean evaluation"),
                                (right, ["train_eval_accuracy", "val_accuracy"], "Accuracy: clean evaluation")]:
        with col:
            chart = history[["epoch"] + columns].melt(id_vars="epoch", var_name="Set", value_name="Value")
            chart["Set"] = chart.Set.map({columns[0]: "Train subset", columns[1]: "Validation"})
            fig = px.line(chart, x="epoch", y="Value", color="Set", markers=True, title=title)
            fig.add_vline(x=best, line_dash="dot", line_color="gray")
            fig.update_layout(height=350)
            fig.update_yaxes(rangemode="tozero")
            st.plotly_chart(fig, width="stretch")
    st.caption(f"Train ในกราฟคือชุดตัวอย่างคงที่ {subset_n:,} ภาพ ไม่ใช่ train ทั้งหมด "
               "ทั้งสองเส้นประเมินโดยไม่มี augmentation และใช้ plain cross-entropy เหมือนกัน")
    source = st.selectbox("Validation confusion matrix", ["All", "EuroSAT", "UCMerced"])
    show_confusion(f"runs/{run}/validation_{source}_confusion.csv", f"Validation: {labels[run]} / {source}")
    with st.expander("ตัวอย่างที่ทำนายผิดบน validation"):
        st.image(str(RUNS / run / "validation_errors.png"), width="stretch")
        errors = read_csv(f"runs/{run}/validation_errors.csv")
        st.dataframe(errors[["source", "class_name", "pred", "confidence"]].head(10), hide_index=True, width="stretch")
        st.caption("pred เป็น Class ID ตามตารางหน้าแรก; confidence เป็น softmax score ไม่ใช่โอกาสถูกที่สอบเทียบแล้ว")
    st.divider()
    st.subheader("Final test: EfficientNet-B0 baseline เท่านั้น")
    st.caption("ผล test นี้เป็นของโมเดลที่เลือก ไม่เปลี่ยนตามเมนูผลการฝึกด้านบน")
    metrics = read_csv("test_metrics.csv")
    shown = metrics.rename(columns={"source": "Dataset", "n": "Images", "accuracy": "Accuracy", "macro_f1": "Macro-F1"})
    st.dataframe(shown, hide_index=True, width="stretch")
    test_source = st.selectbox("Test confusion matrix", ["All", "EuroSAT", "UCMerced"])
    show_confusion(f"test/test_{test_source}_confusion.csv", f"Final test: {test_source}")
    report = read_json(f"test/test_{test_source}_report.json")
    st.subheader("Precision / Recall / F1 รายคลาส")
    st.dataframe(pd.DataFrame({name: report[name] for name in CLASS_NAMES}).T.round(4), width="stretch")
    st.subheader("ตัวอย่างภาพที่ทำนายผิดบน test")
    st.image(str(ASSETS / "test/test_errors.png"), width="stretch")
    st.caption("Test ผิด 65 จาก 4,201 ภาพ มีทั้งกรณีเกษตรกับธรรมชาติคล้ายกันและภาพที่มีหลายองค์ประกอบ")
    st.info("โมเดลที่เลือก: train subset accuracy 100%, validation 98.67%, test 98.45% "
            "อาจมี overfitting เล็กน้อยบน split นี้ ยังไม่รับประกันผลกับพื้นที่ใหม่")


def upload_prediction():
    st.header("อัปโหลดภาพทำนาย")
    st.write("ใช้ EfficientNet-B0 baseline ที่ฝึกแล้ว ทำนายสดด้วย CPU ได้โดยไม่ต้องโหลด dataset หรือฝึกซ้ำ")
    uploaded = st.file_uploader("เลือกภาพ RGB / ภาพถ่ายทางอากาศ", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"])
    examples = read_json("examples.json")
    example = st.selectbox("หรือเลือกภาพตัวอย่างจาก test", [None] + list(range(len(examples))),
                           format_func=lambda index: "ไม่ใช้ภาพตัวอย่าง" if index is None else examples[index]["label"])
    raw = None
    true_class = None
    if uploaded is not None:
        raw = uploaded.getvalue()
        name = uploaded.name
    elif example is not None:
        item = examples[example]
        raw = (ASSETS / "examples" / item["filename"]).read_bytes()
        name = item["filename"]
        true_class = item["class_name"]
    else:
        st.info("อัปโหลดภาพใหม่เพื่อทดสอบ unseen หรือเลือกภาพ test เพื่อสาธิตการโหลดโมเดล")
        return
    try:
        image = read_image(raw)
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("อ่านภาพไม่ได้ กรุณาเลือกไฟล์ภาพที่เปิดได้จริง")
        return
    identity = pixel_hash(image)
    match = known_images().get(identity)
    if match:
        st.info(f"ภาพตรงกับ dataset เดิม: {match['source']} / {match['split']} "
                "จึงเป็นภาพสาธิตจากข้อมูลเดิม ไม่ใช่ unseen")
        true_class = match["class_name"]
    else:
        st.caption("ไม่พบภาพซ้ำแบบตรงกันใน dataset เดิม การตรวจนี้ไม่ยืนยันว่าเป็นสถานที่ใหม่หรือไม่มีภาพใกล้เคียง")
    left, right = st.columns([1, 2])
    with left:
        st.image(image, caption=f"{name} · {image.width}×{image.height} pixels", width=300)
        st.caption("โมเดล resize เป็น 224×224 และ normalize เหมือนตอนฝึก")
    expected = st.selectbox("คลาสจริง (ถ้าทราบ)", ["ไม่ระบุ"] + CLASS_NAMES,
                            index=CLASS_NAMES.index(true_class) + 1 if true_class in CLASS_NAMES else 0)
    result_key = f"prediction_{identity}"
    if st.button("ทำนายภาพ", type="primary"):
        try:
            with st.spinner("กำลังโหลดโมเดลและทำนาย..."):
                model, transform, package = selected_model()
                st.session_state[result_key] = predict(model, transform, image).tolist()
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            st.error(f"โหลดหรือทำนายโมเดลไม่สำเร็จ: {error}")
            return
    if result_key not in st.session_state:
        return
    scores = st.session_state[result_key]
    class_id = max(range(5), key=lambda index: scores[index])
    with right:
        st.success(f"Class {class_id}: {CLASS_NAMES[class_id]} — {CLASS_TH[class_id]}")
        st.metric("Softmax score ของคลาสที่เลือก", f"{scores[class_id]:.2%}")
        if expected != "ไม่ระบุ":
            if CLASS_NAMES[class_id] == expected:
                st.success(f"ตรงกับคลาสจริงที่ระบุ: {expected}")
            else:
                st.warning(f"ไม่ตรงกับคลาสจริงที่ระบุ: {expected}")
        frame = pd.DataFrame({"Class ID": range(5), "Class": CLASS_NAMES, "ความหมาย": CLASS_TH,
                              "Softmax score (%)": [score * 100 for score in scores]})
        frame = frame[["Class ID", "Class", "Softmax score (%)", "ความหมาย"]]
        st.dataframe(frame.round(4), hide_index=True, width="stretch")
        fig = px.bar(frame, x="Softmax score (%)", y="Class", orientation="h", text_auto=".2f")
        fig.update_xaxes(range=[0, 100])
        fig.update_layout(height=300, yaxis={"categoryorder": "array", "categoryarray": CLASS_NAMES[::-1]})
        st.plotly_chart(fig, width="stretch")
        csv_download("ดาวน์โหลดผลทำนาย", frame, "prediction_scores.csv")
    st.caption("Softmax score ไม่ใช่ accuracy หรือโอกาสถูกที่สอบเทียบแล้ว "
               "ภาพนอกกลุ่มทั้ง 5 ก็ยังถูกบังคับให้เลือกหนึ่งคลาส เช่นภาพคนหรือสัตว์")


st.sidebar.title("Land Use Classification")
page = st.sidebar.radio("เลือกหน้า", PAGES)
st.sidebar.caption("060243413 Applied Machine Learning\nKMUTNB · Seed 42")
st.sidebar.caption("โมเดลที่ใช้งาน: EfficientNet-B0 baseline")
st.title("Land Use Classification")
if not (ASSETS / "comparison.csv").exists():
    st.error("ไม่พบโฟลเดอร์ assets กรุณาแตก ZIP ทั้งโฟลเดอร์ก่อนเปิดแอป")
    st.stop()
if page == PAGES[0]:
    overview()
elif page == PAGES[1]:
    compare_models()
elif page == PAGES[2]:
    training_and_test()
else:
    upload_prediction()
