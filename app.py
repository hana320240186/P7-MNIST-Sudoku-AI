"""P7 Sudoku AI Lab - simple Streamlit dashboard.
Run: streamlit run app.py
"""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import streamlit as st

from p7.sudoku import make_demo, find_conflicts, valid_grid, solve_sudoku
from p7.search import correct_grid

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "mnist_digit_model.keras"
METRICS_PATH = ROOT / "results" / "training_metrics.json"
CM_PATH = ROOT / "results" / "confusion_matrix.csv"

st.set_page_config(page_title="P7 | Sudoku AI Lab", page_icon="🧩", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1450px; padding-top: 1.3rem; padding-bottom: 3rem;}
.hero {padding: 1.6rem 1.8rem; border-radius: 20px; color: white;
       background: linear-gradient(115deg,#101c3a,#254c82 58%,#5578c8); margin-bottom: 1rem;}
.hero h1 {color:white; font-size:2.1rem; margin:0 0 .25rem 0;}
.hero p {color:#e6edff; margin:0; font-size:1rem;}
.panel {border:1px solid rgba(128,145,170,.28); border-radius:16px; padding:1rem 1.1rem; margin:.35rem 0 .8rem 0;}
.kicker {font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; opacity:.7; font-weight:700;}
div[data-testid="stMetric"] {border:1px solid rgba(128,145,170,.25); padding:.7rem .9rem; border-radius:14px;}
.stButton>button {border-radius:11px; font-weight:650; min-height:2.7rem;}
</style>
<div class="hero">
  <div class="kicker">P7 · Machine Learning + Search + CSP</div>
  <h1>🧩 Sudoku AI Lab</h1>
  <p>Train on real MNIST digits, inspect the puzzle before correction, then compare the result after Search + CSP.</p>
</div>
""", unsafe_allow_html=True)


def grid_df(grid):
    rows = []
    for r in range(9):
        rows.append([grid[r * 9 + c] if grid[r * 9 + c] else "·" for c in range(9)])
    return pd.DataFrame(rows, index=[f"{r+1}" for r in range(9)], columns=[str(c+1) for c in range(9)])


def manual_probabilities(grid, confidence):
    result = []
    other = (1.0 - confidence) / 8.0
    for value in grid:
        if value == 0:
            result.append({d: 1 / 9 for d in range(1, 10)})
        else:
            result.append({d: confidence if d == value else other for d in range(1, 10)})
    return result


def demo_probabilities(grid, solution, changed):
    changed_map = {cell: (wrong, correct) for cell, wrong, correct in changed}
    result = []
    for cell, value in enumerate(grid):
        if value == 0:
            result.append({d: 1 / 9 for d in range(1, 10)})
        elif cell in changed_map:
            wrong, correct = changed_map[cell]
            p = {d: 0.01 for d in range(1, 10)}
            p[wrong] = 0.60
            p[correct] = 0.32
            result.append(p)
        else:
            p = {d: 0.0125 for d in range(1, 10)}
            p[value] = 0.90
            result.append(p)
    return result


def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


def get_model():
    if not MODEL_PATH.exists():
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)


# Sidebar: project controls
with st.sidebar:
    st.header("⚙️ Experiment setup")
    mode = st.radio("Puzzle input", ["Generated example", "Enter puzzle manually"])
    method = st.selectbox("Correction method", ["UCS", "Greedy", "A*"], help="Choose one search strategy. You can also compare all three below.")
    max_changes = st.slider("Maximum corrections", 1, 3, 2)
    compare = st.checkbox("Compare all three methods", value=True)
    st.divider()
    st.caption("UCS: lowest total correction cost.\n\nGreedy: chooses the state with the smallest conflict estimate.\n\nA*: combines correction cost + conflict estimate.")

st.markdown("### 1. Real-data training status")
metrics = load_metrics()
if metrics:
    a, b, c, d = st.columns(4)
    a.metric("MNIST test accuracy", f"{metrics['test_accuracy'] * 100:.2f}%")
    b.metric("Training images", f"{metrics['train_images_after_zero_filter']:,}")
    c.metric("Validation images", f"{metrics['validation_images_after_zero_filter']:,}")
    d.metric("Test images", f"{metrics['test_images_after_zero_filter']:,}")
    st.caption("These values are read from results/training_metrics.json after train_mnist.py has trained and evaluated the CNN on real MNIST images. Zero is excluded; the original test split is not used for training.")
    if CM_PATH.exists():
        with st.expander("View MNIST confusion matrix"):
            cm = pd.read_csv(CM_PATH, header=None)
            cm.index = [str(i) for i in range(1, 10)]
            cm.columns = [str(i) for i in range(1, 10)]
            st.dataframe(cm, use_container_width=True)
else:
    st.warning("The real MNIST model has not been trained yet. In Colab, run the training cell first. The Sudoku demo still works, but MNIST accuracy will appear after training.")

with st.expander("Test the trained MNIST digit classifier", expanded=False):
    st.write("Upload a **single digit crop** (not a complete Sudoku photo). The CNN predicts one digit from 1 to 9.")
    digit_file = st.file_uploader("Upload one handwritten digit", type=["png", "jpg", "jpeg"], key="digit_crop")
    if digit_file is not None:
        from PIL import Image, ImageOps
        image = Image.open(digit_file).convert("L")
        st.image(image, caption="Uploaded digit", width=150)
        if st.button("Classify with trained MNIST CNN"):
            model = get_model()
            if model is None:
                st.error("Train the model first by running train_mnist.py in Colab.")
            else:
                image = ImageOps.fit(image, (20, 20)).resize((28, 28))
                arr = np.asarray(image).astype("float32")
                # MNIST digits are dark on a light background. Invert if the crop is dark overall.
                if arr.mean() < 127:
                    arr = 255 - arr
                arr = arr / 255.0
                arr = arr[None, ..., None]
                probs = model.predict(arr, verbose=0)[0]
                digit = int(np.argmax(probs)) + 1
                st.success(f"Predicted digit: {digit} · confidence: {float(np.max(probs)) * 100:.1f}%")
                st.bar_chart(pd.DataFrame({"Probability": probs}, index=list(range(1, 10))))

st.divider()
st.markdown("### 2. Choose the Sudoku puzzle")
if mode == "Generated example":
    left, mid, right = st.columns(3)
    with left:
        seed = st.number_input("Random seed", 0, 99999, 21)
    with mid:
        givens = st.slider("Number of filled cells", 28, 45, 36)
    with right:
        mistakes = st.slider("Injected reading mistakes", 1, 3, 2)
    st.info("This reproducible demo starts from a valid Sudoku, hides cells, then injects a few wrong readings. It lets you see the correction method clearly. The generated correction probabilities are demonstration values, not CNN predictions.")
    if st.button("🎲 Generate puzzle", type="primary"):
        puzzle, solution, changed = make_demo(int(seed), int(givens), int(mistakes))
        st.session_state["grid"] = puzzle
        st.session_state["solution_reference"] = solution
        st.session_state["changed"] = changed
        st.session_state["probs"] = demo_probabilities(puzzle, solution, changed)
        st.session_state["source"] = f"Generated example · seed {seed} · {givens} clues · {len(changed)} wrong readings"
        st.session_state["ready"] = True
else:
    default = "530070000\n600195000\n098000060\n800060003\n400803001\n700020006\n060000280\n000419005\n000080079"
    raw = st.text_area("Enter 9 rows; use 0 or . for empty cells", value=default, height=190)
    confidence = st.slider("Confidence of typed digits", 0.55, 0.99, 0.80, 0.01)
    if st.button("Use this puzzle", type="primary"):
        chars = [ch for ch in raw if ch.isdigit() or ch == "."]
        if len(chars) != 81:
            st.error(f"Enter exactly 81 cells. Current count: {len(chars)}")
        else:
            grid = [0 if ch in "0." else int(ch) for ch in chars]
            if any(v < 0 or v > 9 for v in grid):
                st.error("Use only digits 1-9, zero, and dot.")
            else:
                st.session_state["grid"] = grid
                st.session_state["solution_reference"] = None
                st.session_state["changed"] = []
                st.session_state["probs"] = manual_probabilities(grid, confidence)
                st.session_state["source"] = "Manually entered grid · probabilities estimated from the confidence slider"
                st.session_state["ready"] = True

if st.session_state.get("ready"):
    grid = st.session_state["grid"]
    probs = st.session_state["probs"]
    conflicts = find_conflicts(grid)
    st.divider()
    st.markdown("## 3. BEFORE — before applying Search")
    st.caption(st.session_state.get("source", ""))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filled cells", sum(v != 0 for v in grid), "/ 81")
    m2.metric("Empty cells", sum(v == 0 for v in grid))
    m3.metric("Duplicate conflicts", len(conflicts))
    m4.metric("Input status", "Consistent" if valid_grid(grid) else "Conflicts found")
    st.dataframe(grid_df(grid), use_container_width=True, height=300)
    with st.expander("Show conflict details", expanded=bool(conflicts)):
        if conflicts:
            for a, b in conflicts:
                ra, ca = divmod(a, 9)
                rb, cb = divmod(b, 9)
                st.write(f"Digit **{grid[a]}** is repeated at (row {ra+1}, col {ca+1}) and (row {rb+1}, col {cb+1}).")
        else:
            st.success("No duplicate digits in rows, columns, or 3×3 boxes.")

    st.markdown("## 4. APPLY — run correction method")
    methods = ["ucs", "greedy", "astar"] if compare else [{"UCS":"ucs", "Greedy":"greedy", "A*":"astar"}[method]]
    labels = {"ucs": "UCS", "greedy": "Greedy", "astar": "A*"}
    if st.button("🚀 Apply method and compare", type="primary", use_container_width=True):
        results = {}
        progress = st.progress(0)
        for i, name in enumerate(methods):
            started = time.perf_counter()
            result = correct_grid(grid, probs, name, max_changes=max_changes)
            result["seconds"] = time.perf_counter() - started
            results[name] = result
            progress.progress((i + 1) / len(methods))
        st.session_state["results"] = results
        st.session_state["selected_result"] = methods[0]

    results = st.session_state.get("results")
    if results:
        st.markdown("## 5. AFTER — after applying Search + CSP")
        summary = []
        for name, result in results.items():
            summary.append({
                "Method": labels[name],
                "Solved": "Yes" if result["success"] else "No",
                "Corrections": result["changes"],
                "Nodes expanded": result["expanded"],
                "Nodes generated": result["generated"],
                "Time (sec)": round(result["seconds"], 4),
                "Total cost": round(result["total_cost"], 4) if result["total_cost"] is not None else "—",
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
        selected = st.selectbox("Show detailed result", methods, format_func=lambda x: labels[x], key="selected_method")
        result = results[selected]
        if result["success"]:
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Final conflicts", len(find_conflicts(result["corrected_grid"])))
            x2.metric("Corrections made", result["changes"])
            x3.metric("Search nodes", result["expanded"])
            x4.metric("Runtime", f"{result['seconds']:.4f}s")
            before, after = st.columns(2)
            with before:
                st.markdown("**Before correction**")
                st.dataframe(grid_df(grid), use_container_width=True, height=300)
            with after:
                st.markdown("**After correction**")
                st.dataframe(grid_df(result["corrected_grid"]), use_container_width=True, height=300)
            st.markdown("**Correction log**")
            if result["log"]:
                log_rows = []
                for cell, old, new, cost in result["log"]:
                    r, c = divmod(cell, 9)
                    log_rows.append({"Cell": f"Row {r+1}, Col {c+1}", "Read digit": old, "New digit": new, "Correction cost": round(cost, 5)})
                st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
                st.download_button("Download correction log (CSV)", pd.DataFrame(log_rows).to_csv(index=False).encode(), "correction_log.csv", "text/csv")
            else:
                st.info("No corrections were needed for this method.")
            st.markdown("**Final solved Sudoku**")
            st.dataframe(grid_df(result["solution"]), use_container_width=True, height=300)
            st.success("CSP Backtracking found a valid completion for the corrected grid.")
        else:
            st.error("No solution found within the selected correction limit. Try increasing the limit or use a different puzzle.")
    else:
        st.caption("Click **Apply method and compare** to see the after view and method statistics.")

st.divider()
st.caption("Project note: the CNN is trained on real MNIST digit images. The Sudoku demo is generated for repeatable method comparison. The single-digit upload tests the trained CNN; full Sudoku-photo grid extraction is not included in this beginner-friendly Colab version.")
