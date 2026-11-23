
import streamlit as st
from annotate import annotate_image
from pathlib import Path
import tempfile

st.set_page_config(
    page_title="visionTag",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 visionTag")
st.subheader(
    "Computer Vision Based Smart Image Annotation System"
)

st.write(
    "Upload an image and let AI automatically "
    "detect and annotate objects."
)

uploaded_file = st.file_uploader(
    "Upload your image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Original Image",
        use_container_width=True
    )

    if st.button("Start Smart Annotation"):

        with st.spinner("Detecting objects..."):

            with tempfile.TemporaryDirectory() as temp_dir:

                input_path = Path(temp_dir) / uploaded_file.name

                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                image_path, label_path = annotate_image(
                    input_path,
                    Path(temp_dir) / "outputs",
                    confidence
                )

                st.success("Annotation completed!")

                st.subheader("Annotated Image")

                st.image(
                    str(image_path),
                    use_container_width=True
                )

                col1, col2 = st.columns(2)

                with col1:
                    with open(image_path, "rb") as f:
                        st.download_button(
                            "Download Annotated Image",
                            f.read(),
                            file_name="annotated_image.jpg",
                            mime="image/jpeg"
                        )

                with col2:
                    with open(label_path, "rb") as f:
                        st.download_button(
                            "Download Annotation File",
                            f.read(),
                            file_name="annotations.txt",
                            mime="text/plain"
                        )

