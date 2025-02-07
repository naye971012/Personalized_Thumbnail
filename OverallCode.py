import streamlit as st
import torch
import numpy as np
import pandas as pd
import os
import json
import requests
from PIL import Image
from io import BytesIO
import sys
import textwrap  # 텍스트 감싸기를 위해 추가
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# --------------------------------------------------------------------------------
# Paths and Constants
# --------------------------------------------------------------------------------
ABS_PATH = "."
API_KEY = os.getenv("API_KEY")

MODEL_PATH = f"{ABS_PATH}/src/SASRec/ml-1m_default/SASRec.epoch=1000.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth"
MOVIE_METADATA_PATH = f"{ABS_PATH}/src/SASRec/ml-1m_original_data/movies.dat"
MOVIE_DETAILDATA_PATH = f"{ABS_PATH}/src/movies_metadata.csv"
NUM_USERS, NUM_ITEMS = 6040, 3416
MAX_SEQ_LEN = 200

# Replace this with a real poster base URL if different
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"

sys.path.append(f"{ABS_PATH}/src/CreatiLayout")
sys.path.append(f"{ABS_PATH}/src/SASRec")

from src.CreatiLayout.utils.bbox_visualization import bbox_visualization, scale_boxes
from src.CreatiLayout.src.models.transformer_sd3_SiamLayout import SiamLayoutSD3Transformer2DModel
from src.CreatiLayout.src.pipeline.pipeline_CreatiLayout import CreatiLayoutSD3Pipeline

# --------------------------------------------------------------------------------
# SASRec model and utility imports
# --------------------------------------------------------------------------------
from src.SASRec.model import SASRec
from src.SASRec.utils import build_index

# --------------------------------------------------------------------------------
# Default GPT system instruction for layout design
# --------------------------------------------------------------------------------
default_system_instruction = (
    "You are an assistant that designs layouts for movie thumbnails. "
    "The layout should have between three and five objects. Each object should be described "
    "with (object name, a short feature, [x1, y1, x2, y2]). "
    "Additionally, include one line that describes the overall mood/background "
    "of the entire image and also provide a 'title' for the movie thumbnail. "
    "The objects must be bigger than the half of entire image. "    
    "Make sure to consider the user's layout mood preferences inferred from their previously watched movies, "
    "and let the design reflect the genres or moods present in those movies. "
    "Do not include any specific or proprietary names in the 'object_name' or 'feature' descriptions. "
    "They must be simple, generic descriptions without brand or character names. "
    "The 'overall_mood' should begin with a single style descriptor keyword (e.g. 'realistic', 'animated') "
    "followed by a short descriptive phrase that references and integrates the 'object_name' and 'feature' details. "
    "Coordinates should be between 0 and 1. Output in valid JSON format with the schema:\n"
    "{\n"
    '  "title": "...",\n'
    '  "layout": {\n'
    '    "objects": [\n'
    '      {\n'
    '        "object_name": "...",\n'
    '        "feature": "...",\n'
    '        "bbox": [x1, y1, x2, y2]\n'
    "      }, ...\n"
    "    ],\n"
    '    "overall_mood": "..." \n'
    "  }\n"
    "}\n"
)


# --------------------------------------------------------------------------------
# Functions to load SASRec model and metadata
# --------------------------------------------------------------------------------
def load_model(model_path, num_users, num_items, args):
    model = SASRec(num_users, num_items, args)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

def load_metadata_for_sasrec(data_path, detailadata_path):
    """
    Reads lines from data_path in the format: "item_id::title::genre".
    Creates a small DataFrame with columns: [item_id, title, genre].
    Merges with detailed metadata from detailadata_path (CSV) on "title", then
    sorts by 'item_id' and creates a dictionary that maps item_id to row index.
    
    Returns:
        merged_df (pd.DataFrame): Sorted DataFrame with columns:
            [item_id, title, genre, original_title, overview, belongs_to_collection]
        item_id2index (dict): A dictionary mapping item_id -> row index in merged_df
    """

    # Load the detailed metadata CSV
    movie_metadata = pd.read_csv(detailadata_path)

    # 1) Parse the file at data_path
    records = []
    with open(data_path, 'r', encoding='latin-1') as file:
        for line in file:
            parts = line.strip().split("::")
            if len(parts) == 3:
                item_id, title, genre = parts
                if int(item_id) >= 3400:
                    continue
                title = title.split("(")[0][:-1]
                records.append([int(item_id), title, genre])

    df_small = pd.DataFrame(records, columns=["item_id", "title", "genre"])

    print("df_small len", len(df_small))

    # 2) Merge on "title" (left join)
    desired_cols = ["title", "original_title", "overview", "belongs_to_collection"]
    existing_cols = [c for c in desired_cols if c in movie_metadata.columns]

    merged_df = pd.merge(
        df_small,
        movie_metadata[existing_cols],
        on="title",
        how="left"
    )

    # 3) Fill missing values with defaults
    if "original_title" in merged_df.columns:
        merged_df["original_title"] = merged_df["original_title"].fillna(merged_df["title"])
    else:
        merged_df["original_title"] = merged_df["title"]

    if "overview" in merged_df.columns:
        merged_df["overview"] = merged_df["overview"].fillna("No overview.")
    else:
        merged_df["overview"] = "No overview."

    if "belongs_to_collection" in merged_df.columns:
        merged_df["belongs_to_collection"] = merged_df["belongs_to_collection"].fillna("")
    else:
        merged_df["belongs_to_collection"] = ""

    # 5) Create the item_id2index mapping
    item_id2index = {
        row["item_id"]: idx
        for idx, row in merged_df.iterrows()
    }

    # 6) Return the final DataFrame and the dictionary
    return merged_df[[
        "item_id",
        "title",
        "genre",
        "original_title",
        "overview",
        "belongs_to_collection"
    ]], item_id2index

# --------------------------------------------------------------------------------
# GPT-based layout generation
# --------------------------------------------------------------------------------
def get_layout_from_gpt(api_key: str, scenario: str, system_instruction: str) -> dict:
    client = OpenAI(api_key=api_key)
    messages = [
        {
            "role": "system",
            "content": system_instruction
        },
        {
            "role": "user",
            "content": (
                "[Instruction] Please create a layout for a movie thumbnail.\n\n"
                "The product of the bounding box's width and height must be greater than 0.16.\n"
                f"Here is the scenario:\n{scenario}"
            )
        },
    ]

    # Change "gpt-4o-mini" to a valid model if necessary
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    layout_response = response.choices[0].message.content

    start_index = layout_response.find('{')
    end_index = layout_response.rfind('}')
    if start_index != -1 and end_index != -1 and start_index < end_index:
        layout_response = layout_response[start_index:end_index + 1]

    try:
        layout_data = json.loads(layout_response)
    except json.JSONDecodeError:
        raise ValueError("The GPT response is not valid JSON:\n" + layout_response)

    return layout_data

# --------------------------------------------------------------------------------
# Load CreatiLayout models (Stable Diffusion + Layout Transformer)
# --------------------------------------------------------------------------------
@st.cache_resource
def load_creatilaout_models(device):
    transformer_additional_kwargs = dict(attention_type="layout", strict=True)
    transformer = SiamLayoutSD3Transformer2DModel.from_pretrained(
        "HuiZhang0812/CreatiLayout",
        subfolder="transformer",
        torch_dtype=torch.float16,
        **transformer_additional_kwargs
    )
    pipe = CreatiLayoutSD3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        transformer=transformer,
        torch_dtype=torch.float16
    )
    pipe = pipe.to(device)
    return pipe

# --------------------------------------------------------------------------------
# Diffusion generation with step-by-step callbacks
# --------------------------------------------------------------------------------
def generate_image(pipe, params, image_placeholder):
    def on_step_end(pipeline, step, t, callback_kwargs):
        latents = callback_kwargs["latents"]
        if step % 5 == 0 and step > 0:
            latents_for_decode = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
            image_torch = pipeline.vae.decode(latents_for_decode, return_dict=False)[0]
            image_pil = pipeline.image_processor.postprocess(image_torch, output_type="pil")
            image_placeholder.image(
                image_pil, 
                caption=f"Step {step} / {params['num_inference_steps']}",
                use_container_width=True
            )
        return {"latents": latents}

    with torch.no_grad():
        output = pipe(
            prompt=params['prompt'],
            num_inference_steps=params['num_inference_steps'],
            guidance_scale=params['guidance_scale'],
            bbox_phrases=params['region_caption_list'],
            bbox_raw=params['region_bboxes_list'],
            height=params['height'],
            width=params['width'],
            callback_on_step_end=on_step_end,
            callback_on_step_end_tensor_inputs=["latents"]
        )

    final_image = output.images[0]
    image_placeholder.image(final_image, caption="Final Generation", use_container_width=True)
    return final_image

# --------------------------------------------------------------------------------
# Save and visualize final image with bounding boxes
# --------------------------------------------------------------------------------
def save_and_visualize_image(image, params):
    os.makedirs(params['img_save_root'], exist_ok=True)
    os.makedirs(params['img_with_layout_save_root'], exist_ok=True)

    img_path = os.path.join(params['img_save_root'], f"{params['filename']}.png")
    image.save(img_path)

    numeric_labels = [f"({i+1})" for i in range(len(params['region_caption_list']))]
    show_input = {
        "boxes": scale_boxes(params['region_bboxes_list'], params['width'], params['height']),
        "labels": numeric_labels
    }

    white_image = Image.new('RGB', (params['width'], params['height']), color='white')
    bbox_vis_img = bbox_visualization(white_image.copy(), show_input)
    image_with_bbox = bbox_visualization(image.copy(), show_input)

    total_width = params['width'] * 2
    total_height = params['height']
    new_image = Image.new('RGB', (total_width, total_height))
    new_image.paste(bbox_vis_img, (0, 0))
    new_image.paste(image_with_bbox, (params['width'], 0))

    img_with_layout_save_name = os.path.join(
        params['img_with_layout_save_root'], f"{params['filename']}_layout.png"
    )
    new_image.save(img_with_layout_save_name)

    st.image([new_image], caption=["Layout Visualization"], use_container_width=True)

    st.markdown("### Object Legend")
    for idx, label in enumerate(numeric_labels):
        st.markdown(f"""
            <span style="font-size: 16px; font-weight: bold; color: #4CAF50;">{label}</span> → 
            <span style="font-size: 16px; font-weight: bold;">{params['region_caption_list'][idx]}</span>
        """, unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# Preview layout on a white background
# --------------------------------------------------------------------------------
def visualize_layout_on_white(layout_data, preview_width=512, preview_height=512):
    white_image = Image.new('RGB', (preview_width, preview_height), color='white')
    objects = layout_data["layout"]["objects"]
    scaled_bboxes = scale_boxes(
        [obj["bbox"] for obj in objects],
        preview_width,
        preview_height
    )
    numeric_labels = [f"({i+1})" for i in range(len(objects))]
    show_input = {
        "boxes": scaled_bboxes,
        "labels": numeric_labels
    }
    vis_img = bbox_visualization(white_image, show_input)
    return vis_img, numeric_labels

# --------------------------------------------------------------------------------
# Main Streamlit App
# --------------------------------------------------------------------------------
def main():
    st.title("🎬 Movie Recommendation + Thumbnail Layout & Diffusion")

    # -------------------------------------------------------------------------
    # Part A: Load SASRec model and Data
    # -------------------------------------------------------------------------
    st.subheader("1) Movie Recommendation (SASRec)")

    class Args:
        hidden_units = 50
        num_blocks = 2
        num_heads = 1
        dropout_rate = 0.2
        maxlen = MAX_SEQ_LEN
        device = "cpu"

    args = Args()
    model = load_model(MODEL_PATH, NUM_USERS, NUM_ITEMS, args)

    # IMPORTANT: Adjust this function to load your real movie DataFrame
    movies_df, item_id2index = load_metadata_for_sasrec(MOVIE_METADATA_PATH, MOVIE_DETAILDATA_PATH)

    # -------------------------------------------------------------------------
    # Instead of numerical IDs, let user select movies by title with multiselect
    # -------------------------------------------------------------------------
    all_titles = sorted(movies_df["original_title"].unique().tolist())
    watched_titles = st.multiselect(
        "Select watched movies (autocomplete)",
        all_titles
    )
    top_k = st.slider("Number of recommendations:", 1, 10, 1)

    # -------------------------------------------------------------------------
    # Recommendation Step
    # -------------------------------------------------------------------------
    if st.button("Get Recommendations"):
        if not watched_titles:
            st.warning("Please select at least one movie.")
        else:
            # Convert selected titles to item IDs
            watched_rows = movies_df[movies_df["original_title"].isin(watched_titles)]
            watched_item_ids = watched_rows["item_id"].tolist()

            # Show small poster images + details of watched movies
            st.markdown("### Watched Movies")
            for _, row in watched_rows.iterrows():
                col_watch_l, col_watch_r = st.columns([1, 4])
                with col_watch_l:
                    if row["belongs_to_collection"]:
                        try:
                            collection_data = json.loads(row["belongs_to_collection"].replace("'", "\""))
                            poster_path = POSTER_BASE_URL + collection_data.get("poster_path", "")
                            response = requests.get(poster_path)
                            if response.status_code == 200:
                                watched_img = Image.open(BytesIO(response.content))
                                st.image(watched_img, width=80)
                            else:
                                st.write("No Poster")
                        except:
                            st.write("No Poster")
                    else:
                        st.write("No Poster")

                with col_watch_r:
                    st.write(f"**{row['original_title']}** (ID: {row['item_id']})")
                    st.caption(f"Genre: {row.get('genre', 'N/A')}")
                    st.write(f"Overview: {row.get('overview', 'N/A')}")

            # Prepare input sequence for SASRec
            if len(watched_item_ids) > MAX_SEQ_LEN:
                watched_item_ids = watched_item_ids[-MAX_SEQ_LEN:]
            input_sequence = [0] * (MAX_SEQ_LEN - len(watched_item_ids)) + watched_item_ids

            # Perform prediction
            all_item_indices = torch.arange(1, NUM_ITEMS + 1).unsqueeze(0)
            with torch.no_grad():
                logits = model.predict(
                    user_ids=None,
                    log_seqs=torch.tensor([input_sequence]),
                    item_indices=all_item_indices
                )

            scores, top_items = torch.topk(logits.squeeze(0), top_k)
            scores = scores.tolist()
            top_items = top_items.tolist()

            # Retrieve recommended movies from the DataFrame
            recommended_movies = []
            for item_id in top_items:
                movie_info = movies_df[movies_df["item_id"] == item_id]
                if not movie_info.empty:
                    recommended_movies.append(movie_info.iloc[0])

            if recommended_movies:
                st.markdown("### Recommended Movies")
                for i, (movie_, score) in enumerate(zip(recommended_movies, scores)):
                    col_rec_l, col_rec_r = st.columns([1, 4])
                    with col_rec_l:
                        if movie_["belongs_to_collection"]:
                            try:
                                collection_data = json.loads(movie_["belongs_to_collection"].replace("'", "\""))
                                poster_path = POSTER_BASE_URL + collection_data.get("poster_path", "")
                                response = requests.get(poster_path)
                                if response.status_code == 200:
                                    rec_img = Image.open(BytesIO(response.content))
                                    st.image(rec_img, width=80)
                                else:
                                    st.write("No Poster")
                            except:
                                st.write("No Poster")
                        else:
                            st.write("No Poster")

                    with col_rec_r:
                        st.markdown(f"**{i+1}. {movie_['original_title']}**")
                        st.caption(f"Genre: {movie_['genre']}")
                        st.write(f"Overview: {movie_['overview']}")
                        st.write(f"**Score**: {score:.4f} | **item_id**: {movie_['item_id']}")

                # Automatically fill scenario text area with recommended movies' overview
                from collections import Counter
                watched_genres = watched_rows['genre'].dropna().tolist()
                if watched_genres:
                    genre_counter = Counter(watched_genres)
                    common_genres = ", ".join([genre for genre, count in genre_counter.most_common(3)])
                    mood_preference_text = f"User's watched movie genres: {common_genres}. The layout design should evoke a mood that reflects these genres."
                else:
                    mood_preference_text = ""
                
                recommended_overviews = "\n\n".join(
                    [f"{rm['original_title']}: {rm.get('overview', 'No overview')}" 
                     for rm in recommended_movies]
                )
                scenario_text = (mood_preference_text + "\n\n" + recommended_overviews) if mood_preference_text else recommended_overviews
                st.session_state["scenario_text"] = scenario_text.strip()
                # Store watched and recommended movies for process summary
                st.session_state["watched_titles"] = watched_titles
                st.session_state["recommended_movies"] = [dict(movie_) for movie_ in recommended_movies]
            else:
                st.warning("No recommendations found.")

    st.write("---")

    # -------------------------------------------------------------------------
    # Part B: Layout Generation + Diffusion
    # -------------------------------------------------------------------------
    st.subheader("2) ChatGPT Layout Generation → Editing → Image Generation")

    # GPT System Instruction
    with st.expander("GPT System Instruction (Editable)", expanded=False):
        system_instruction_input = st.text_area(
            "GPT System Instruction",
            default_system_instruction,
            height=300
        )

    # OpenAI API Key
    api_key = st.text_input("OpenAI API Key", value=API_KEY, type="password")

    # Scenario (auto-filled from recommended results if present)
    default_scenario = st.session_state.get("scenario_text", "No scenario. Please get recommendations first.")
    scenario = st.text_area("Scenario (based on recommended movie overviews and your watched movies)", default_scenario)

    # Store layout_data in session state
    if "layout_data" not in st.session_state:
        st.session_state["layout_data"] = None

    # Generate Layout from GPT
    if st.button("1) Generate Layout"):
        if not api_key:
            st.warning("Please provide an OpenAI API Key.")
        else:
            with st.spinner("Requesting layout from GPT..."):
                layout_data = get_layout_from_gpt(api_key, scenario, system_instruction_input)
                st.session_state["layout_data"] = layout_data
            st.success("Layout generated successfully!")
            with st.expander("GPT Layout JSON", expanded=False):
                st.json(st.session_state["layout_data"])

    # Preview & Edit Layout
    if st.session_state["layout_data"]:
        st.markdown("### Layout Preview & Editing")
        layout_preview, numeric_labels = visualize_layout_on_white(st.session_state["layout_data"])
        st.image(layout_preview, caption="Layout Preview (White Background)", use_container_width=True)

        st.markdown("**[Current Layout: Object Legend]**")
        for i, obj in enumerate(st.session_state["layout_data"]["layout"]["objects"]):
            st.markdown(f"""
                <span style="font-size: 16px; font-weight: bold; color: #4CAF50;">{numeric_labels[i]}</span> → 
                <span style="font-size: 16px; font-weight: bold;">{obj['object_name']}</span> / 
                <span style="font-size: 14px; color: #757575;">{obj['feature']}</span>
            """, unsafe_allow_html=True)

        layout_data = st.session_state["layout_data"]
        title = st.text_input("Thumbnail Title", layout_data["title"])
        overall_mood = st.text_input("Overall Mood/Background", layout_data["layout"]["overall_mood"])

        updated_objects = []
        for idx, obj in enumerate(layout_data["layout"]["objects"]):
            st.markdown(f"**[Object {idx+1}]**  {numeric_labels[idx]}")
            col_left, col_right = st.columns([2, 2])
            with col_left:
                object_name = st.text_input(
                    f"Object Name #{idx+1}",
                    obj["object_name"],
                    key=f"object_name_{idx}"
                )
                feature = st.text_input(
                    f"Feature #{idx+1}",
                    obj["feature"],
                    key=f"feature_{idx}"
                )
            with col_right:
                x1 = st.number_input(
                    f"x1 (#{idx+1})",
                    0.0, 1.0,
                    value=float(obj["bbox"][0]),
                    step=0.01,
                    key=f"x1_{idx}"
                )
                y1 = st.number_input(
                    f"y1 (#{idx+1})",
                    0.0, 1.0,
                    value=float(obj["bbox"][1]),
                    step=0.01,
                    key=f"y1_{idx}"
                )
                x2 = st.number_input(
                    f"x2 (#{idx+1})",
                    0.0, 1.0,
                    value=float(obj["bbox"][2]),
                    step=0.01,
                    key=f"x2_{idx}"
                )
                y2 = st.number_input(
                    f"y2 (#{idx+1})",
                    0.0, 1.0,
                    value=float(obj["bbox"][3]),
                    step=0.01,
                    key=f"y2_{idx}"
                )
            updated_objects.append({
                "object_name": object_name,
                "feature": feature,
                "bbox": [x1, y1, x2, y2]
            })

        if st.button("Apply Layout Updates"):
            st.session_state["layout_data"]["title"] = title
            st.session_state["layout_data"]["layout"]["overall_mood"] = overall_mood
            st.session_state["layout_data"]["layout"]["objects"] = updated_objects

            st.success("Layout updated!")
            layout_preview, numeric_labels2 = visualize_layout_on_white(st.session_state["layout_data"])
            st.image(layout_preview, caption="Updated Layout Preview", use_container_width=True)

    st.write("---")
    # -------------------------------------------------------------------------
    # Load CreatiLayout model and generate the final image
    # -------------------------------------------------------------------------
    st.subheader("3) Generate the Thumbnail Image (Diffusion)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with st.spinner("Loading CreatiLayout pipeline..."):
        pipe = load_creatilaout_models(device)
    st.success("Model loaded successfully!")

    st.markdown("### Diffusion Parameters")
    num_inference_steps = st.number_input("num_inference_steps", min_value=1, max_value=200, value=50)
    guidance_scale = st.slider("guidance_scale", min_value=0.0, max_value=20.0, value=7.5)
    height = st.number_input("height (px)", min_value=64, max_value=2048, value=512)
    width = st.number_input("width (px)", min_value=64, max_value=2048, value=512)
    filename = st.text_input("Output filename", "movie_thumbnail")

    img_save_root = os.path.join("output", "images")
    img_with_layout_save_root = os.path.join("output", "images_with_layout")

    if st.button("Generate Image"):
        if not st.session_state["layout_data"]:
            st.warning("Please generate or load a layout first.")
        else:
            layout_data = st.session_state["layout_data"]
            global_prompt = f"{layout_data['title']} / {layout_data['layout']['overall_mood']}"

            region_caption_list = []
            region_bboxes_list = []
            for obj in layout_data["layout"]["objects"]:
                region_caption_list.append(f"{obj['object_name']} {obj['feature']}")
                region_bboxes_list.append(obj["bbox"])

            params = {
                "prompt": global_prompt,
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(guidance_scale),
                "region_caption_list": region_caption_list,
                "region_bboxes_list": region_bboxes_list,
                "height": int(height),
                "width": int(width),
                "img_save_root": img_save_root,
                "img_with_layout_save_root": img_with_layout_save_root,
                "filename": filename,
            }

            image_placeholder = st.empty()
            with st.spinner("Generating image..."):
                final_image = generate_image(pipe, params, image_placeholder)
                save_and_visualize_image(final_image, params)
            st.success("Thumbnail image generated and saved!")
            st.session_state["final_image_params"] = params

    st.write("---")
    st.markdown("**Result Paths**")
    st.write(f"- Generated images are saved in: `{img_save_root}`.")
    st.write(f"- Images with bounding box layout are saved in: `{img_with_layout_save_root}`.")

    # --------------------------------------------------------------------------------
    # [추가] 5) Process Summary Image 생성
    # --------------------------------------------------------------------------------
    st.write("---")
    st.subheader("5) Process Summary Image")
    if st.button("Generate Process Summary Image"):
        # 수집된 프로세스 정보
        watched = st.session_state.get("watched_titles", "Not captured")
        recommended = st.session_state.get("recommended_movies", "Not captured")
        scenario = st.session_state.get("scenario_text", "Not captured")
        layout_data = st.session_state.get("layout_data", "Not captured")
        if layout_data != "Not captured" and isinstance(layout_data, dict):
            global_prompt = f"{layout_data.get('title', '')} / {layout_data.get('layout', {}).get('overall_mood', '')}"
        else:
            global_prompt = "Not captured"

        # 텍스트 블록 구성
        text_lines = []
        text_lines.append("Process Summary")
        text_lines.append("")
        text_lines.append("Watched Movies:")
        if isinstance(watched, list):
            for movie in watched:
                text_lines.append(" - " + movie)
        else:
            text_lines.append(" " + str(watched))
        text_lines.append("")
        text_lines.append("Recommended Movies:")
        if isinstance(recommended, list):
            for movie in recommended:
                text_lines.append(" - " + movie.get("original_title", "Unknown"))
        else:
            text_lines.append(" " + str(recommended))
        text_lines.append("")
        text_lines.append("Scenario:")
        text_lines.append(scenario)
        text_lines.append("")
        text_lines.append("Layout:")
        if layout_data != "Not captured" and isinstance(layout_data, dict):
            text_lines.append(" Title: " + layout_data.get("title", ""))
            layout_inner = layout_data.get("layout", {})
            text_lines.append(" Overall Mood: " + layout_inner.get("overall_mood", ""))
            text_lines.append(" Objects:")
            for obj in layout_inner.get("objects", []):
                text_lines.append("  - " + f"{obj.get('object_name', '')} / {obj.get('feature', '')} / BBox: {obj.get('bbox', '')}")
        else:
            text_lines.append(" " + str(layout_data))
        text_lines.append("")
        text_lines.append("Global Prompt:")
        text_lines.append(global_prompt)

        summary_text = "\n".join(text_lines)

        # 최종 생성 이미지 로드 (이미 저장된 파일)
        final_image_path = os.path.join(img_save_root, f"{filename}.png")
        if os.path.exists(final_image_path):
            final_img = Image.open(final_image_path)
        else:
            st.error("Final generated image not found!")
            final_img = None

        # composite 이미지의 기준 너비와 높이 (final_img가 있으면 해당 값, 아니면 기본값)
        composite_width = final_img.width if final_img else 512
        composite_height = final_img.height if final_img else 512

        # 텍스트 블록 이미지를 생성 (PIL의 ImageDraw 사용)
        from PIL import ImageDraw, ImageFont
        try:
            font = ImageFont.truetype("arial.ttf", size=16)
        except:
            font = ImageFont.load_default()

        # 각 줄을 composite_width에 맞게 감싸기 (평균 문자폭 약 7px 기준)
        wrapped_lines = []
        for line in text_lines:
            max_chars = composite_width // 7
            wrapped = textwrap.fill(line, width=max_chars)
            wrapped_lines.extend(wrapped.split("\n"))

        # 기존 font.getsize 대신 getbbox를 사용 (높이 계산)
        line_height = (font.getbbox("A")[3] - font.getbbox("A")[1]) + 4
        text_block_height = line_height * len(wrapped_lines) + 20  # padding 포함
        text_img = Image.new("RGB", (composite_width, text_block_height), color="white")
        draw_text = ImageDraw.Draw(text_img)
        y_text = 10
        for l in wrapped_lines:
            draw_text.text((10, y_text), l, fill="black", font=font)
            y_text += line_height

        # --- 상단 행: 좌측에는 description (text_img), 우측에는 최종 생성 이미지 (final_img) 배치 ---
        if final_img:
            top_row_height = max(text_img.height, final_img.height)
            top_row_width = composite_width * 2  # 두 이미지를 좌우로 배치하므로
            top_row = Image.new("RGB", (top_row_width, top_row_height), color="white")
            # 왼쪽: 텍스트 블록
            top_row.paste(text_img, (0, 0))
            # 오른쪽: 최종 생성 이미지
            top_row.paste(final_img, (composite_width, 0))
        else:
            top_row = text_img  # fallback

        layout_composite_path = os.path.join(img_with_layout_save_root, f"{filename}_layout.png")
        if os.path.exists(layout_composite_path):
            layout_composite_img = Image.open(layout_composite_path)
            # 전체 너비(상단 row와 동일한 composite_width*2)로 조정하고, 높이는 composite_height 유지
            layout_composite_img = layout_composite_img.resize((composite_width * 2, composite_height))
        else:
            layout_composite_img = Image.new("RGB", (composite_width * 2, composite_height), color="gray")
        bottom_row = layout_composite_img
        # --- 최종 composite: 상단 행과 하단 행을 세로로 결합 ---
        final_composite_width = max(top_row.width, bottom_row.width)
        final_composite_height = top_row.height + bottom_row.height
        composite_img = Image.new("RGB", (final_composite_width, final_composite_height), color="white")
        composite_img.paste(top_row, (0, 0))
        composite_img.paste(bottom_row, (0, top_row.height))

        # 합친 이미지 저장 및 다운로드 버튼 생성
        composite_filename = f"process_summary_{filename}.png"
        composite_img.save(composite_filename)
        st.image(composite_img, caption="Process Summary Image", use_container_width =True)

        buf = BytesIO()
        composite_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(
            "Download Process Summary Image",
            data=byte_im,
            file_name=composite_filename,
            mime="image/png"
        )

        #public_url = f"publicpath/{composite_filename}"
        
        # import qrcode
        # qr = qrcode.QRCode(
        #     version=1,
        #     error_correction=qrcode.constants.ERROR_CORRECT_L,
        #     box_size=10,
        #     border=4,
        # )
        # qr.add_data(public_url)
        # qr.make(fit=True)
        # qr_img = qr.make_image(fill_color="black", back_color="white")
        # buf = BytesIO()
        # qr_img.save(buf, format="PNG")
        # qr_img_bytes = buf.getvalue()
        # st.image(qr_img_bytes, caption="Scan this QR code to access the Process Summary Image", use_column_width=True)

if __name__ == "__main__":
    main()
