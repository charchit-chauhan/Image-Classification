import json
import random

# -----------------------------
# Paths
# -----------------------------
TRAIN_JSON = "data/annotations/instances_train2017.json"
VAL_JSON = "data/annotations/instances_val2017.json"

TARGET_CLASSES = ["person", "car", "dog", "bicycle", "bus"]

TRAIN_LIMIT = 1000
VAL_LIMIT = 200


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


train_data = load_json(TRAIN_JSON)
val_data = load_json(VAL_JSON)

# -----------------------------------
# Get category IDs
# -----------------------------------
category_ids = {}

for category in train_data["categories"]:
    if category["name"] in TARGET_CLASSES:
        category_ids[category["id"]] = category["name"]

print("Selected Categories:")
for k, v in category_ids.items():
    print(k, "->", v)


# -----------------------------------
# Collect image IDs
# -----------------------------------
def collect_image_ids(dataset):

    image_ids = set()

    for ann in dataset["annotations"]:

        if ann["category_id"] in category_ids:
            image_ids.add(ann["image_id"])

    return list(image_ids)


train_image_ids = collect_image_ids(train_data)
val_image_ids = collect_image_ids(val_data)

print("\nTotal matching training images :", len(train_image_ids))
print("Total matching validation images :", len(val_image_ids))


random.seed(42)

random.shuffle(train_image_ids)
random.shuffle(val_image_ids)

train_subset = train_image_ids[:TRAIN_LIMIT]
val_subset = val_image_ids[:VAL_LIMIT]

print("\nSelected training images :", len(train_subset))
print("Selected validation images :", len(val_subset))
# -----------------------------------
# Map image IDs to filenames
# -----------------------------------

def get_filenames(dataset, selected_ids):

    id_set = set(selected_ids)

    filenames = []

    for image in dataset["images"]:

        if image["id"] in id_set:
            filenames.append(image["file_name"])

    return filenames


train_filenames = get_filenames(train_data, train_subset)
val_filenames = get_filenames(val_data, val_subset)

print("\nTraining filenames :", len(train_filenames))
print("Validation filenames :", len(val_filenames))


# Save filenames to text files
with open("train_files.txt", "w") as f:
    for name in train_filenames:
        f.write(name + "\n")

with open("val_files.txt", "w") as f:
    for name in val_filenames:
        f.write(name + "\n")

print("\nSaved train_files.txt")
print("Saved val_files.txt")