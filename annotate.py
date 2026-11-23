
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import argparse

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")


def annotate_image(image_path, output_dir="outputs",
                   confidence=0.25):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Perform object detection
    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False
    )

    result = results[0]

    # Draw bounding boxes
    annotated_array = result.plot()
    annotated_image = Image.fromarray(
        annotated_array[:, :, ::-1]
    )

    # Save annotated image
    image_output = output_dir / "annotated_image.jpg"
    annotated_image.save(image_output)

    # Save YOLO format annotations
    label_output = output_dir / "annotations.txt"

    with open(label_output, "w") as file:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height

                file.write(
                    f"{class_id} {x_center:.6f} "
                    f"{y_center:.6f} {box_width:.6f} "
                    f"{box_height:.6f}\n"
                )

    print("Image annotation completed!")
    print("Detected objects:", len(result.boxes))
    print("Annotated image:", image_output)
    print("Annotation file:", label_output)

    return image_output, label_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--conf", type=float, default=0.25)

    args = parser.parse_args()

    annotate_image(
        args.image,
        args.output,
        args.conf
    )

