"""Download 16 high-quality realistic test photos across the 6 evaluation categories."""
import os
import requests

DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")
os.makedirs(DEST, exist_ok=True)

IMAGES = {
    # 1. People & Activities
    "1_person_walking.jpg": "https://picsum.photos/id/64/600/400",
    "1_person_reading.jpg": "https://picsum.photos/id/338/600/400",
    "1_person_skating.jpg": "https://picsum.photos/id/193/600/400",

    # 2. Animals
    "2_animal_dog.jpg": "https://picsum.photos/id/237/600/400",
    "2_animal_pug.jpg": "https://picsum.photos/id/1025/600/400",
    "2_animal_bear.jpg": "https://picsum.photos/id/1084/600/400",

    # 3. Outdoor & Nature Scenes
    "3_scene_forest_mountains.jpg": "https://picsum.photos/id/10/600/400",
    "3_scene_beach_rocks.jpg": "https://picsum.photos/id/1015/600/400",
    "3_scene_desert_road.jpg": "https://picsum.photos/id/1018/600/400",

    # 4. Food & Everyday Objects
    "4_food_berries.jpg": "https://picsum.photos/id/1080/600/400",
    "4_object_vintage_car.jpg": "https://picsum.photos/id/133/600/400",
    "4_object_laptop_workspace.jpg": "https://picsum.photos/id/180/600/400",

    # 5. Complex / Crowded Scenes
    "5_complex_cityscape.jpg": "https://picsum.photos/id/1011/600/400",
    "5_complex_train_station.jpg": "https://picsum.photos/id/1033/600/400",

    # 6. Unusual / Abstract / Challenging
    "6_unusual_architecture.jpg": "https://picsum.photos/id/1076/600/400",
    "6_unusual_fog_trees.jpg": "https://picsum.photos/id/1044/600/400",
}

print(f"Downloading {len(IMAGES)} test photos across 6 categories...")
for filename, url in IMAGES.items():
    out_path = os.path.join(DEST, filename)
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(resp.content)
            print(f"  [OK] {filename} ({len(resp.content)//1024} KB)")
        else:
            print(f"  [HTTP {resp.status_code}] {filename}")
    except Exception as e:
        print(f"  [Error] {filename}: {e}")

print("Done! All images saved to:", DEST)
