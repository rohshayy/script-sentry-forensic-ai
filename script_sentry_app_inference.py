import torch
from PIL import Image
from torchvision import transforms
from script_sentry_model import ScriptSentry
import torch.nn.functional as F
import os


def process_image(image_path):
    img = Image.open(image_path).convert('L').resize((28, 28))
    img_tensor = transforms.ToTensor()(img)
    img_tensor = transforms.Normalize((0.5,), (0.5,))(img_tensor).unsqueeze(0)
    return img_tensor


def run_multi_anchor_detection(anchor_paths, suspect_path):
    # 1. Load Model
    model = ScriptSentry()
    model.load_state_dict(torch.load("scriptsentry_v1.pth", map_location=torch.device('cpu')))
    model.eval()

    # 2. Extract Average Stylometric Fingerprint from multiple Anchors
    anchor_vectors = []
    print(f"Enrolling Reference Samples...")

    for path in anchor_paths:
        if os.path.exists(path):
            img = process_image(path)
            with torch.no_grad():
                vector = model(img, return_features=True)
                anchor_vectors.append(vector)
        else:
            print(f"Warning: {path} not found. Skipping.")

    if not anchor_vectors:
        print("Error: No reference images found!")
        return

    # Calculate the MEAN vector (The "Average" Style)
    mean_anchor_vector = torch.mean(torch.stack(anchor_vectors), dim=0)

    # 3. Process Suspect
    suspect_img = process_image(suspect_path)
    with torch.no_grad():
        suspect_vector = model(suspect_img, return_features=True)
        output = model(suspect_img)
        prob = torch.nn.functional.softmax(output, dim=1)
        conf, suspect_pred = torch.max(prob, 1)

    # 4. Calculate Similarity against the Mean
    similarity = F.cosine_similarity(mean_anchor_vector, suspect_vector)

    print(f"\n" + "=" * 45)
    print(f"   MULTI-ANCHOR FORENSIC REPORT")
    print(f"=" * 45)
    print(f"SAMPLES ENROLLED : {len(anchor_vectors)}")
    print(f"SUSPECT IDENTITY : Class {suspect_pred.item()}")
    print(f"STYLE SIMILARITY : {similarity.item() * 100:.2f}%")
    print(f"-" * 45)

    # STRICT LOGIC (92% threshold)
    if similarity.item() > 0.92:
        print("FINAL RESULT: ✅ VERIFIED GENUINE")
    else:
        print("FINAL RESULT: ❌ FORGERY ALERT")
        print("Reason: Style deviates from average user profile.")
    print(f"=" * 45)


if __name__ == "__main__":
    # To use this, create 3 handwritten samples of your letter:
    # real_1.png, real_2.png, real_3.png
    my_anchors = ["real_1.png", "real_2.png", "real_3.png"]
    suspect = "test_image.png"

    try:
        run_multi_anchor_detection(my_anchors, suspect)
    except FileNotFoundError as e:
        print(f"Error: {e}")